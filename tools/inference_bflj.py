import os
import sys
import math
import argparse
import pickle
import cv2

import numpy as np
from tqdm import tqdm
import torch
from torch.multiprocessing import Queue, Process

sys.path.insert(0, '../lib')
sys.path.insert(0, '../model')
# from data.CrowdHuman import CrowdHuman
from data.CrowdHuman_json import CrowdHuman
from utils import misc_utils, nms_utils
from evaluate import compute_JI, compute_APMR
from evaluate import compute_MMR
from det_oprs.bbox_opr import Pointlist_dis, matcher
from scipy.optimize import linear_sum_assignment

MAX_VAL = 8e6
cut_off = 0.96
print('cut_off:', cut_off)

def eval_all(args, config, network):
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
    selected_id = args.random
    crowdhuman = CrowdHuman(config, if_train=False)
    crowdhuman.records = crowdhuman.records[selected_id:selected_id+1]

    # multiprocessing 
    match_result = inference_bfj(config, args, network, model_file, devices[0], crowdhuman, selected_id, selected_id+1)    

    result = crowdhuman.records[0]
    img_name = result['ID']
    img = cv2.imread('../../data/CrowdHuman/CrowdHuman_val/Images/'+img_name)
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
    thickness = 2
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
    # cv2.imshow('testbox', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
        print(pred_boxes)

        from inference_one_img import match_body_face_bflj
        match_result = match_body_face_bflj(args, pred_boxes, pred_lmks, pred_emb, image_id)  
        # match_result = match_body_face_bfj(pred_boxes, pred_lmks, pred_emb, image_id)

    return match_result

def match_body_face_bfj_old(pred_boxes, pred_lmks, pred_emb, image_id):
    keep_body = pred_boxes[:, 7] == 1
    keep_face = pred_boxes[:, 7] == 2
    body_boxes = pred_boxes[keep_body]
    body_embs = pred_emb[keep_body]
    face_boxes = pred_boxes[keep_face]
    face_embs = pred_emb[keep_face]
    lmks_info = pred_lmks[keep_face]
    wof_flag=False

    if len(face_boxes) == 0:
        wof_flag = True
    base_body_boxes = body_boxes[:, :4]
    base_body_scores = body_boxes[:, 6]
    base_body_hooks = body_boxes[:, 4:6]

    base_face_boxes = face_boxes[:, :4]
    base_face_scores = face_boxes[:, 6]
    base_face_hooks = face_boxes[:, 4:6]
    base_lmks_info = lmks_info

    inds_conf_base_body = (base_body_scores > 0.5).nonzero()    #0.5
    if not inds_conf_base_body[0].size:
        inds_conf_base_body = np.argmax(base_body_scores)[None]
        wof_flag = True
    inds_conf_base_face = (base_face_scores > 0.3).nonzero()
    if not inds_conf_base_face[0].size and (not wof_flag):
        inds_conf_base_face = np.argmax(base_face_scores)[None]
        wof_flag = True

    base_body_boxes = base_body_boxes[inds_conf_base_body]
    base_body_hooks = base_body_hooks[inds_conf_base_body]
    base_body_scores = base_body_scores[inds_conf_base_body]
    base_body_embeddings = body_embs[inds_conf_base_body]

    if not wof_flag:
        base_face_boxes = base_face_boxes[inds_conf_base_face]
        base_face_scores = base_face_scores[inds_conf_base_face]
        base_face_hooks = base_face_hooks[inds_conf_base_face]
        base_lmks_info = base_lmks_info[inds_conf_base_face]
        base_face_embeddings = face_embs[inds_conf_base_face]

    if wof_flag:
        face_boxes = np.zeros_like(base_body_boxes)
        face_scores = np.zeros_like(base_body_scores)
        face_lmks = np.zeros((base_body_boxes.shape[0], 11)) #np.zeros_like(base_body_scores)
    else:
        
        score_matrix = (base_face_scores[:, None] + base_body_scores) / 2

        distance_matrix = Pointlist_dis(base_face_hooks, base_body_hooks, base_body_boxes)
        embedding_matrix = np.sqrt(np.square(base_face_embeddings[:, None] - base_body_embeddings).sum(-1))
        distance_matrix_max = np.max(distance_matrix, axis=0)
        distance_matrix = distance_matrix / distance_matrix_max
        embedding_matrix_max = np.max(embedding_matrix, axis=0)
        embedding_matrix = embedding_matrix / embedding_matrix_max
        match_merge_matrix = distance_matrix * score_matrix * score_matrix + embedding_matrix * (1 - score_matrix * score_matrix)
        match_merge_matrix = np.exp(-match_merge_matrix)
        matched_vals = np.max(match_merge_matrix, axis=0)
        matched_indices = np.argmax(match_merge_matrix, axis=0)
        ignore_indices = (matched_vals < cut_off).nonzero()

        dummy_tensor = np.array([0.0, 0.0, 0.0, 0.0])

        face_boxes = base_face_boxes[matched_indices]
        face_scores = base_face_scores[matched_indices]
        face_lmks = base_lmks_info[matched_indices]
        if ignore_indices[0].size:
            face_boxes[ignore_indices] = dummy_tensor
            face_scores[ignore_indices] = 0
            face_lmks[ignore_indices] *= 0
    bodylist = np.hstack((base_body_boxes, base_body_scores[:, None]))
    facelist = np.hstack((face_boxes, face_scores[:, None]))
    lmklist = face_lmks#np.hstack((face_lmks, lmks_scores[:, None]))
    print('bodylist:', bodylist.shape)
    print('facelist:', facelist.shape)
    print('lmklist:', lmklist.shape)
    assert lmklist.shape[1] == 11
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

def match_body_face_pos(pred_boxes, image_id):
    keep_body = pred_boxes[:, 5] == 1
    keep_face = pred_boxes[:, 5] == 2
    body_boxes = pred_boxes[keep_body]
    face_boxes = pred_boxes[keep_face]
    wof_flag=False

    if len(face_boxes) == 0:
        wof_flag = True
    base_body_boxes = body_boxes[:, :4]
    base_body_scores = body_boxes[:, 4]

    base_face_boxes = face_boxes[:, :4]
    base_face_scores = face_boxes[:, 4]

    inds_conf_base_body = (base_body_scores > 0.8).nonzero()
    if not inds_conf_base_body[0].size:
        inds_conf_base_body = np.argmax(base_body_scores)[None]
        wof_flag = True
    inds_conf_base_face = (base_face_scores > 0.3).nonzero()
    if not inds_conf_base_face[0].size and (not wof_flag):
        inds_conf_base_face = np.argmax(base_face_scores)[None]
        wof_flag = True

    base_body_boxes = base_body_boxes[inds_conf_base_body]
    base_body_scores = base_body_scores[inds_conf_base_body]

    if not wof_flag:
        base_face_boxes = base_face_boxes[inds_conf_base_face]
        base_face_scores = base_face_scores[inds_conf_base_face]

    if wof_flag:
        face_boxes = np.zeros_like(base_body_boxes)
        face_scores = np.zeros_like(base_body_scores)
    else:
        body_face_distance_matrix = cal_body_face_distance_matrix(base_body_boxes, base_face_boxes)
        base_body_boxes_filter = []
        base_body_scores_filter = []
        base_face_boxes_filter = []
        base_face_scores_filter = []
        body_row_idxs, face_col_idxs = linear_sum_assignment(body_face_distance_matrix)
        for body_idx in body_row_idxs:
            f_idx = np.where(body_row_idxs == body_idx)[0][0]
            col_face_idx = face_col_idxs[f_idx]

            if body_face_distance_matrix[body_idx, col_face_idx] != MAX_VAL:
        # for body_idx in body_row_idxs:
        #     f_idx = np.where(body_row_idxs == body_idx)[0][0]
        #     col_face_idx = face_col_idxs[f_idx]
        #     if body_face_distance_matrix[body_idx, col_face_idx] != MAX_VAL:
                base_body_boxes_filter.append(base_body_boxes[body_idx])
                base_body_scores_filter.append(base_body_scores[body_idx])
                base_face_boxes_filter.append(base_face_boxes[col_face_idx])
                base_face_scores_filter.append(base_face_scores[col_face_idx])
        if base_body_boxes_filter == []:
            face_boxes = np.zeros_like(base_body_boxes)
            face_scores = np.zeros_like(base_body_scores)
            wof_flag = True
        else:
            base_body_boxes = np.vstack(base_body_boxes_filter)
            base_body_scores = np.hstack(base_body_scores_filter)
            face_boxes = np.vstack(base_face_boxes_filter)
            face_scores = np.hstack(base_face_scores_filter)

    bodylist = np.hstack((base_body_boxes, base_body_scores[:, None]))
    facelist = np.hstack((face_boxes, face_scores[:, None]))
    result = []
    for body, face in zip(bodylist, facelist):
        body = body.tolist()
        face = face.tolist()
        content = {
            'image_id': int(image_id),
            'category_id': 1,
            'bbox':[round(i, 1) for i in body[:4]],
            'score':round(float(body[4]), 5),
            'f_bbox':[round(i, 1) for i in face[:4]],
            'f_score':round(float(face[4]), 5)
        }
        result.append(content)
    return result

def cal_body_face_distance_matrix(body_boxes, face_boxes):
    body_boxes_nums = len(body_boxes)
    face_boxes_nums = len(face_boxes)
    body_face_distance_matrix = np.zeros((body_boxes_nums, face_boxes_nums))
    for body_idx in range(body_boxes_nums):
        body_box = body_boxes[body_idx]
        for face_idx in range(face_boxes_nums):
            face_box = face_boxes[face_idx]
            face_iou_in_body = one_side_iou(face_box, body_box)
            if face_iou_in_body > 0.2:
                body_face_distance_matrix[body_idx, face_idx] = 1 / face_iou_in_body
            else:
                body_face_distance_matrix[body_idx, face_idx] = MAX_VAL

    return body_face_distance_matrix

def one_side_iou(box1, box2):
    # 1. to corner box
    # box1[2:4] = box1[0:2] + box1[2:4]
    # box2[2:4] = box2[0:2] + box2[2:4]
    x1 = max(box1[0], box2[0])
    x2 = min(box1[2] + box1[0], box2[2] + box2[0])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3] + box1[1], box2[3] + box2[1])

    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    # a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a1 = box1[2] * box1[3]

    iou = intersection / a1  # intersection over box 1
    return iou

def boxes_dump(boxes, embs=None):
    # print('boxes:',boxes.shape)
    # print('embs:',embs.shape)
    if boxes.shape[-1] >= 8: # v2 or v3
        if embs is not None:
            result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                   'score':round(float(box[6]), 5),
                   'tag':int(box[7]),
                   'emb':emb.tolist()} for box, emb in zip(boxes, embs)]
        else:
            result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                    'score':round(float(box[6]), 5),
                    'tag':int(box[7])} for box in boxes]
    elif boxes.shape[-1] == 7:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5]),
                   'proposal_num':int(box[6])} for box in boxes]
    elif boxes.shape[-1] == 6: # v1
        result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5])} for box in boxes]
    elif boxes.shape[-1] == 5:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'tag':int(box[4])} for box in boxes]
    else:
        raise ValueError('Unknown box dim.')
    return result

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--config', '-c', default=None,required=True,type=str)
    parser.add_argument('--resume_weights', '-r', default=None, required=True, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    parser.add_argument('--random', '-n', default=0, type=int)
    parser.add_argument('--cut_off', '-co', default=0.96, type=float)
    parser.add_argument('--loss', '-l', default='cse', type=str)
    os.environ['NCCL_IB_DISABLE'] = '1'
    args = parser.parse_args()
    # import libs
    model_root_dir = os.path.join('../model/', args.model_dir)
    sys.path.insert(0, model_root_dir)
    if args.config == 'pos':
        from config_pos import config
    elif args.config == 'bfj':
        from config_bfj import config
    else:
        raise Exception('Error - only support for bfj or pos.')

    if config.network == 'pos':
        from network_pos import Network
    elif config.network == 'bfj':
        from network_bfj import Network
    else:
        raise Exception('Error - only support for bfj or pos.')
    eval_all(args, config, Network)

if __name__ == '__main__':
    run_test()

