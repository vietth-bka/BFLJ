import os
import sys
import math
import argparse
import pickle
import cv2

import numpy as np
from tqdm import tqdm
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, '../lib')
sys.path.insert(0, '../model')
# from data.CrowdHuman import CrowdHuman
from data.CrowdHuman_json import ArbitraryImage
from utils import misc_utils, nms_utils
from evaluate import compute_JI, compute_APMR
from evaluate import compute_MMR
from det_oprs.bbox_opr import Pointlist_dis, matcher
# from test_bfj import inference_bfj


def eval_your_img(args, config, network):
    # model_path
    saveDir = os.path.join('../model', args.model_dir, config.model_dir)
    evalDir = os.path.join('../model', args.model_dir, config.eval_dir)
    misc_utils.ensure_dir(evalDir)
    if 'pth' not in args.resume_weights:
        model_file = os.path.join(saveDir, 
                'dump-{}{}.pth'.format(args.resume_weights, args.loss))
    else:
        model_file = args.resume_weights
    print('model_file:', model_file)
    assert os.path.exists(model_file)
    # get devices
    str_devices = args.devices
    devices = misc_utils.device_parser(str_devices)
    # load data      
    dataset = ArbitraryImage(config, False, args.img_path)
    match_result = inference_bfj(config=config, args=args, network=network, model_file=model_file, device=devices[0], dataset=dataset)
    # print(match_result)

    img = cv2.imread(args.img_path)
    bboxes = [i['bbox'] for i in match_result]
    b_score = [i['score'] for i in match_result]
    faces = [i['f_bbox'] for i in match_result]
    f_score = [i['f_score'] for i in match_result]
    lmks = [i['f_lmk'] for i in match_result]
    bboxes = torch.tensor(bboxes, dtype=torch.int64)
    faces = torch.tensor(faces, dtype=torch.int64)
    # lmks = torch.tensor(lmks)
    bboxes[:, 2:4] += bboxes[:, :2]
    faces[:, 2:4] += faces[:, :2]
    thickness = 3
    print('total faces:', (faces[:,0]!=0).sum())

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    YELLOW = (0, 255, 255)
    color = [(255, 153, 255), RED, BLACK, WHITE, GREEN, YELLOW, BLUE, CYAN, MAGENTA]
    color = color*len(bboxes)

    for i in range(len(bboxes)):
        ID_color = i%8
        if faces[i].sum() == 0:
            body_color = (255, 128, 0)    
        else:        
            body_color = color[ID_color]
            img = cv2.rectangle(img, tuple([faces[i][0].item(), faces[i][1].item()]), 
                            tuple([faces[i][2].item(), faces[i][3].item()]), 
                            color[ID_color], thickness)        
            landmark = lmks[i]        
            landmark = [int(j) for j in landmark]
            for j in range(5):
                img = cv2.circle(img, tuple([int(landmark[2*j]), int(landmark[2*j+1])]), 1, color[ID_color], 2)

        img = cv2.rectangle(img, tuple([bboxes[i][0].item(), bboxes[i][1].item()]), 
                            tuple([bboxes[i][2].item(), bboxes[i][3].item()]), 
                            body_color, thickness)                
        # img = cv2.putText(img, str(b_score[i]), (bboxes[i][0].item(), bboxes[i][1].item()), cv2.FONT_HERSHEY_SIMPLEX, 
        #                   1, body_color, 2, cv2.LINE_AA)
    print(img.shape)
    cv2.imwrite('pred.png', img)

def inference_bfj(config, args, network, model_file, device, dataset=None, start=None, end=None, result_queue=None, result_queue_match=None):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.multiprocessing.set_sharing_strategy('file_system')
    # init model
    net = network()
    net.cuda(device)
    net = net.eval()
    check_point = torch.load(model_file)
    net.load_state_dict(check_point['state_dict'])
    # init data    
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
    print(len(data_iter))
    # data_iter = CrowdHuman(config, False)
    # inference
    for image, gt_boxes, im_info, ID, image_id in data_iter:
        print(image.shape)
        pred_boxes, pred_lmks, pred_emb, class_num = net(image.cuda(device), im_info.cuda(device))
        print('pred_boxes:',pred_boxes.shape)   #1592x8
        scale = im_info[0, 2]
        if config.test_nms_method == 'set_nms':
            assert pred_boxes.shape[-1] > 6, "Not EMD Network! Using normal_nms instead."
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
            top_k = pred_boxes.shape[-1] // 6
            n = pred_boxes.shape[0]
            pred_boxes = pred_boxes.reshape(-1, 6)
            idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1)
            pred_boxes = np.hstack((pred_boxes, idents))
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            keep = nms_utils.set_cpu_nms(pred_boxes, 0.5)
            pred_boxes = pred_boxes[keep]
        elif config.test_nms_method == 'normal_nms':
            assert pred_boxes.shape[-1] % 8 == 0, "Prediction dim Error!"
            pred_boxes = pred_boxes.reshape(-1, 8)
            pred_lmks = pred_lmks.reshape(-1, 11)
            pred_emb = pred_emb.reshape(-1, 32)
            keep = pred_boxes[:, 6] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            pred_lmks = pred_lmks[keep]
            pred_emb = pred_emb[keep]
            result = []
            result_lmk = []
            result_emb = []
            for classid in range(class_num):
                keep = pred_boxes[:, 7] == (classid + 1)
                class_boxes = pred_boxes[keep]
                class_lmks = pred_lmks[keep]
                class_emb = pred_emb[keep]
                keep = nms_utils.cpu_nms(class_boxes, 0.3)#config.test_nms)
                class_boxes = class_boxes[keep]
                class_lmks = class_lmks[keep]
                class_emb = class_emb[keep]
                result.append(class_boxes)
                result_lmk.append(class_lmks)
                result_emb.append(class_emb)
            pred_boxes = np.vstack(result)
            pred_lmks = np.vstack(result_lmk)
            pred_emb = np.vstack(result_emb)
        elif config.test_nms_method == 'none':
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
            pred_boxes = pred_boxes.reshape(-1, 6)
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
        else:
            raise ValueError('Unknown NMS method.')
        #if pred_boxes.shape[0] > config.detection_per_image and \
        #    config.test_nms_method != 'none':
        #    order = np.argsort(-pred_boxes[:, 4])
        #    order = order[:config.detection_per_image]
        #    pred_boxes = pred_boxes[order]
        # recovery the scale
        pred_boxes[:, :6] /= scale
        pred_boxes[:, 2:4] -= pred_boxes[:, :2]
        pred_lmks[:, :10] /= scale
        # gt_boxes = gt_boxes[0].numpy()
        # gt_boxes[:, 2:4] -= gt_boxes[:, :2]
        print('pred_boxes:', pred_boxes.shape)
        # print(pred_boxes)        
        match_result = match_body_face_bflj(args, pred_boxes, pred_lmks, pred_emb, image_id)

    return match_result

def match_body_face_bflj(args, pred_boxes, pred_lmks, pred_emb, image_id):
    """
    new matching method with landmarks
    """
    keep_body = pred_boxes[:, 7] == 1
    keep_face = pred_boxes[:, 7] == 2
    body_boxes = pred_boxes[keep_body]
    body_embs = pred_emb[keep_body]
    face_boxes = pred_boxes[keep_face]
    face_embs = pred_emb[keep_face]
    lmks_info = pred_lmks[keep_face]
    wof_flag=False
    # print('keep_body:', keep_face)
    if len(face_boxes) == 0:
        wof_flag = True
    base_body_boxes = body_boxes[:, :4]
    base_body_scores = body_boxes[:, 6]
    base_body_hooks = body_boxes[:, 4:6]
    # print('base_body_scores:', base_body_scores)

    base_face_boxes = face_boxes[:, :4]
    base_face_scores = face_boxes[:, 6]
    base_face_hooks = face_boxes[:, 4:6]
    base_lmks_info = lmks_info
    # print('base_face_scores:', base_face_scores)

    inds_conf_base_body = (base_body_scores > 0.3).nonzero()
    if not inds_conf_base_body[0].size:
        inds_conf_base_body = np.argmax(base_body_scores)[None]
        wof_flag = True
    inds_conf_base_face = (base_face_scores > 0.3).nonzero()
    if not inds_conf_base_face[0].size and (not wof_flag):
        inds_conf_base_face = np.argmax(base_face_scores)[None]
        wof_flag = True

    base_body_boxes = base_body_boxes[inds_conf_base_body]
    # print('base_body_boxes:', base_body_boxes.shape)
    base_body_hooks = base_body_hooks[inds_conf_base_body]
    base_body_scores = base_body_scores[inds_conf_base_body]
    base_body_embeddings = body_embs[inds_conf_base_body]
    body_norm = np.linalg.norm(base_body_embeddings, axis=-1, keepdims=True) #torch.norm(base_body_embeddings, dim=-1).unsqueeze(-1)
    nbase_body_embeddings = base_body_embeddings/body_norm

    if not wof_flag:
        base_face_boxes = base_face_boxes[inds_conf_base_face]
        base_face_scores = base_face_scores[inds_conf_base_face]
        base_face_hooks = base_face_hooks[inds_conf_base_face]
        base_lmks_info = base_lmks_info[inds_conf_base_face]
        base_face_embeddings = face_embs[inds_conf_base_face]
        face_norm = np.linalg.norm(base_face_embeddings, axis=-1, keepdims=True)
        nbase_face_embeddings = base_face_embeddings/face_norm

    if wof_flag:
        face_boxes = np.zeros_like(base_body_boxes)
        face_scores = np.zeros_like(base_body_scores)
        face_lmks = np.zeros((base_body_boxes.shape[0], 11))
    else:
        
        score_matrix = (base_face_scores[:, None] + base_body_scores) / 2

        distance_matrix = Pointlist_dis(base_face_hooks, base_body_hooks, base_body_boxes)
        distance_matrix_max = np.max(distance_matrix, axis=0)
        distance_matrix = distance_matrix / distance_matrix_max        
        # embedding_matrix = np.sqrt(np.square(base_face_embeddings[:, None] - base_body_embeddings).sum(-1))        
        embedding_matrix = np.matmul(nbase_face_embeddings, nbase_body_embeddings.transpose())        
        # print('embedding_matrix:', embedding_matrix)
        embedding_matrix = 1 - embedding_matrix.clip(-1,1)
        embedding_matrix = np.exp(embedding_matrix/2)#1/math.e #!!!
        embedding_matrix_max = np.max(embedding_matrix, axis=0)
        embedding_matrix = embedding_matrix / embedding_matrix_max        
        match_merge_matrix = distance_matrix * score_matrix * score_matrix + embedding_matrix * (1 - score_matrix * score_matrix)
        match_merge_matrix = np.exp(-match_merge_matrix)
        #------Old version-------
        # print('match_merge_matrix:', match_merge_matrix.shape)
        # matched_vals = np.max(match_merge_matrix, axis=0)
        # matched_indices = np.argmax(match_merge_matrix, axis=0)
        # print('matched_indices:', matched_indices)
        # ignore_indices = (matched_vals < args.cut_off).nonzero()
        # print('ignore_indices:', ignore_indices)
        # dummy_tensor = np.array([0.0, 0.0, 0.0, 0.0])

        # face_boxes = base_face_boxes[matched_indices]
        # face_scores = base_face_scores[matched_indices]
        # face_lmks = base_lmks_info[matched_indices]
        # if ignore_indices[0].size:
        #     face_boxes[ignore_indices] = dummy_tensor
        #     face_scores[ignore_indices] = 0
        #     face_lmks[ignore_indices] *= 0
        #-------new udates------
        row_ind, col_ind, matched_vals = linear_assignment(match_merge_matrix.transpose())                
        face_boxes = np.zeros((base_body_boxes.shape[0], 4))
        face_scores = np.zeros((base_body_boxes.shape[0]))
        face_lmks = np.zeros((base_body_boxes.shape[0], 11))
        
        face_boxes[row_ind] = base_face_boxes[col_ind]        
        face_scores[row_ind] = base_face_scores[col_ind]
        face_lmks[row_ind] = base_lmks_info[col_ind]

        # print('matched_vals:', matched_vals)
        ignore_indices = row_ind[matched_vals < args.cut_off]
        # print('ignore_indices:', ignore_indices)
        dummy_tensor = np.array([0.0, 0.0, 0.0, 0.0])
        # exit(-1)
        if ignore_indices.shape[0]:
            face_boxes[ignore_indices] = dummy_tensor
            face_scores[ignore_indices] = 0
            face_lmks[ignore_indices] *= 0
        #--------end---------
        
    bodylist = np.hstack((base_body_boxes, base_body_scores[:, None]))
    facelist = np.hstack((face_boxes, face_scores[:, None]))
    lmklist = face_lmks
    result = []
    for body, face, lmk in zip(bodylist, facelist, lmklist):
        body = body.tolist()
        face = face.tolist()
        lmk = lmk.tolist()
        content = {
            'image_id': int(image_id),
            'category_id': 1,
            'bbox':[round(i, 1) for i in body[:4]],
            'score':round(float(body[4]), 5),
            'f_bbox':[round(i, 1) for i in face[:4]],
            'f_score':round(float(face[4]), 5),
            'f_lmk':[round(i, 1) for i in lmk[:10]],
            'lmk_score:':round(float(lmk[10]), 5)
        }
        result.append(content)
    return result

def linear_assignment(matrix):
    #make sure the number of row equals to the number of body
    # b x f    
    row_ind, col_ind = linear_sum_assignment(1-matrix)
    # print(row_ind, col_ind)
    assign_vals = matrix[row_ind, col_ind]
    return row_ind, col_ind, assign_vals


def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)    
    parser.add_argument('--config', '-c', default=None,required=True,type=str)
    parser.add_argument('--resume_weights', '-r', default=None, required=True, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    parser.add_argument('--cut_off', '-co', default=0.96, type=float)
    parser.add_argument('--loss','-l', default='cse', type=str)
    parser.add_argument('--img_path', '-p', default='../../data/CrowdHuman/CrowdHuman_val/Images/273271,2b3eb0002dbca786.jpg', type=str)
    os.environ['NCCL_IB_DISABLE'] = '1'
    args = parser.parse_args()
    # import libs
    model_root_dir = os.path.join('../model/', args.model_dir)
    sys.path.insert(0, model_root_dir)
    
    if args.config == 'bfj':
        from config_bfj import config
    else:
        raise Exception('Error - only support for bfj.')
    
    if config.network == 'bfj':
        from network_bfj import Network
    else:
        raise Exception('Error - only support for bfj.')
    eval_your_img(args, config, Network)

if __name__ == '__main__':
    run_test()