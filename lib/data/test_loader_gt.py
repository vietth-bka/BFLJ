#%%
import os, sys
import cv2
import torch
import numpy as np
import json
from tqdm import tqdm
import cv2
import insightface
sys.path.append('../..')
# from utils import misc_utils


# anno = json.load(open('', 'r'))

numid = 0
def load_gt(dict_input, key_name, key_box, class_names, network='bfj'):
    assert key_name in dict_input
    if len(dict_input[key_name]) < 1:
        return np.empty([0, 5])
    else:
        assert key_box in dict_input[key_name][0]
    bbox = []
    global numid
    real_lmks = np.array([10, 10, 30, 10, 20, 20, 10, 30, 30, 30])
    fake_lmks = np.ones(10)*-1
    for rb in dict_input[key_name]:
        if rb['tag'] in class_names:
            tag = class_names.index(rb['tag'])
        else:
            tag = -1
        if 'extra' in rb:
            if 'ignore' in rb['extra']:
                if rb['extra']['ignore'] != 0:
                    tag = -1
        if network == 'pos':
            bbox.append(np.hstack((rb[key_box], tag)))
            if rb['fbox'][2] > 1:
                if tag != -1:
                    tag_f = 2
                    bbox.append(np.hstack((rb['fbox'], tag_f)))
        elif network == 'bfj':
            # bbox.append(np.hstack((rb[key_box], rb['head_center'], tag, numid)))
            if rb['fbox'][2] > 1:
                if tag != -1:
                    tag_f = 2
                    bbox.append(np.hstack((rb['fbox'], rb['head_center'], tag_f, numid, real_lmks)))
                    bbox.append(np.hstack((rb[key_box], rb['head_center'], tag, numid, real_lmks)))
            
            else:
                bbox.append(np.hstack((rb[key_box], rb['head_center'], tag, numid, fake_lmks)))

        else:
            raise Exception('Error - only support for bfj or pos.')
        numid += 1
    if len(bbox) == 0:
        return np.empty([0, 5])
    bboxes = np.vstack(bbox).astype(np.float64)
    return bboxes


class COCO(object):
    def __init__(self, gt_json, dt_json = None, if_have_face = False):
        self.anno = json.load(open(gt_json, 'r'))

        print('len images:', len(self.anno['images']))
        print('len annotations:', len(self.anno['annotations']))
        print('len categories:', len(self.anno['categories']))
        print(self.anno.keys())
        print('images:', self.anno['images'][0])
        print('annotations:', self.anno['annotations'][0])
        print('categories:', self.anno['categories'])

        self.dt_json = dt_json
        self._image_id()
        self._category_id()
        self.del_image_id2()
        self._anno_process(if_have_face)

    def _image_id(self):
        self.image_id = {}
        self.image_wh = {}
        for _, content in enumerate(self.anno['images']):
            self.image_id[content['id']] = content['file_name']
        
        print('Done _image_id', len(self.image_id.keys()))

    def del_image_id(self):
        self.image_id_new = {}
        for key, value in tqdm(self.image_id.items()):
            for _, result_ann in enumerate(self.anno['annotations']):
                if result_ann['image_id'] == key:
                    if result_ann['ignore'] == 0:
                        self.image_id_new[key] = value
                        break
        
        print('Done del_image_id', len(self.image_id_new))
    
    def del_image_id2(self):
        self.image_id_new = {}
        for result_ann in self.anno['annotations']:
            if result_ann['ignore'] == 0:
                if result_ann['image_id'] not in self.image_id_new and result_ann['image_id'] in self.image_id:
                    self.image_id_new[result_ann['image_id']] = self.image_id[result_ann['image_id']]
        
        print('len image_id_new:', len(self.image_id_new))


    def _category_id(self):
        self.category_id = {}
        for idx, category_id_name_dict in enumerate(self.anno['categories']):
            self.category_id[category_id_name_dict['id']] = category_id_name_dict['name']
        print(self.category_id)
        
    def _anno_process(self, if_have_face = False):
        self.annotation_id = {}
        self.annotation_face_id = {}
        for idx, result_ann in enumerate(self.anno['annotations']):
            result_attr_dict = {
                "bbox": result_ann['bbox'],
                "fbox": result_ann['f_bbox'],
                "head_center": [result_ann["h_bbox"][0]+result_ann["h_bbox"][2]/2, result_ann["h_bbox"][1]+result_ann["h_bbox"][3]/2],
                "tag": self.category_id[result_ann['category_id']],
                "extra": {'ignore': result_ann['ignore']}
            }
            if result_ann['category_id'] == 1:
                if result_ann['image_id'] not in self.annotation_id.keys():
                    self.annotation_id[result_ann['image_id']] = []
                    self.annotation_id[result_ann['image_id']].append(result_attr_dict)
                else:
                    self.annotation_id[result_ann['image_id']].append((result_attr_dict))
            else:
                if if_have_face:
                    if result_ann['image_id'] not in self.annotation_face_id.keys():
                        self.annotation_face_id[result_ann['image_id']] = []
                        self.annotation_face_id[result_ann['image_id']].append(result_attr_dict)
                    else:
                        self.annotation_face_id[result_ann['image_id']].append((result_attr_dict))
    
    def process_gt(self, if_have_face = False):
        result = []
        result_face = []
        for image_id, image_name in self.image_id_new.items():
            if image_id not in self.annotation_id.keys():
                continue
            image_result = {
                "ID": image_name,
                "image_id": image_id,
                "gtboxes": self.annotation_id[image_id]
            }
            result.append(image_result)
        if if_have_face:
            for image_id, image_name in self.image_id.items():
                if image_id not in self.annotation_face_id.keys():
                    gtboxes = []
                else:
                    gtboxes = self.annotation_face_id[image_id]
                image_result = {
                    "ID": image_name,
                    "image_id": image_id,
                    "gtboxes": gtboxes
                }
                result_face.append(image_result)
        return result


def draw_box_v2(result, ind, ignore):
    result = result[ind]
    # print('result:', result)
    img_name = result['ID']
    img = cv2.imread('../../../data/CrownHuman/CrowdHuman_train/Images/'+img_name)
    print('Num bboxes:', len(result['gtboxes']))
    bboxes = [i['bbox'] for i in result['gtboxes']]
    faces = [i['fbox'] for i in result['gtboxes']]
    ignores = [i['extra']['ignore'] for i in result['gtboxes']]
    if ignore:
        bboxes = torch.tensor(bboxes)
        faces = torch.tensor(faces)    
        ignores = torch.tensor(ignores)
        ig_ids = ignores > 0
        bboxes[ig_ids] *= 0
        faces[ig_ids] *= 0
    # print(faces)
    bboxes[:, 2:4] += bboxes[:, :2]
    nofaces_inds = faces[:,2]==1        
    if nofaces_inds.sum() != 0:
        faces[nofaces_inds, :] *= 0
    faces[:, 2:4] += faces[:, :2]
    # print(bboxes.shape)

    _, lmks, _ = detect_face(img)
    all_lmks, sus, drop_lmk_f = matching_face_lmks_v2(lmks, faces)
    new_result = insert_lmks(all_lmks, result)
    # print('new_result:', new_result)

    thickness = 1
    color = [(255, 128, 0), (255, 153, 255), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)]
    color = color*len(bboxes)
    for i in range(len(bboxes)):
        start_point = tuple(bboxes[i][:2])
        end_point = tuple(bboxes[i][2:4])
        img = cv2.rectangle(img, start_point, end_point, color[i], thickness)
        img = cv2.rectangle(img, tuple(faces[i][:2]), tuple(faces[i][2:4]), color[i], thickness)
        try:
            landmark = all_lmks[i]
            landmark = [list(j) for j in landmark]
            for it in landmark:
                img = cv2.circle(img, tuple([int(it[0]), int(it[1])]), 1, color[i], 2)
        except:
            print('No lmks found')
            pass

    print('Total lmks:', len(lmks))
    # print(lmks)
    if len(lmks) != 0:    
        for lmk in lmks:
            lmk = list(lmk)
            lmk = [list(i) for i in lmk]        
            for i in lmk:            
                img = cv2.circle(img, tuple([int(i[0]), int(i[1])]), 1, (255, 0, 0), 1)

    cv2.imwrite('gt.png', img)
    # cv2.imshow('testbox', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if sus:
        return new_result, ind, drop_lmk_f
    else:
        return new_result, None, None


def detect_face(img):
    # img = cv2.imread('../../../data/CrownHuman/CrowdHuman_train/Images/'+img_name)
    k_scale = max(1, max(img.shape)/1024)    
    resized_img = cv2.resize(img, (int(img.shape[1]/k_scale), int(img.shape[0]/k_scale)))
    # print(f'Need to resize from {img.shape} to',resized_img.shape, img_name)
    bboxes, landmarks = model.detect(resized_img, threshold=0.8, scale=1.0)
    del resized_img
    landmarks = [landmark*k_scale for landmark in landmarks]    
    return bboxes, landmarks, img

def matching_face_lmks_v2(lmks, faces):
    sus = False
    lmks = torch.tensor(lmks)
    if lmks.dim() < 3:
        return torch.zeros(faces.shape[0], 5, 2), sus, None
    lmks_ctr = lmks.mean(1)
    faces_ctr = torch.cat([((faces[:,0]+faces[:,2])/2)[:,None], ((faces[:,1]+faces[:,3])/2)[:,None]], dim=1)
    # print('lmks:', lmks_ctr.shape, 'faces:', faces_ctr.shape)
    distance_matrix = faces_ctr[:, None] - lmks_ctr[None, :]
    # print('distance_matrix:', distance_matrix.shape)
    distance_matrix = torch.sqrt(torch.square(distance_matrix).sum(-1))
    # print('distance_matrix:', distance_matrix.shape)
    _, inds = torch.min(distance_matrix, dim=-1)    
    # check whether the lmks are inside the face boxes
    pos_inds = (lmks_ctr[inds][:,0]>faces[:,0])*(lmks_ctr[inds][:,0]<faces[:,2])*\
               (lmks_ctr[inds][:,1]>faces[:,1])*(lmks_ctr[inds][:,1]<faces[:,3])
    # print('Check id:', len(inds[pos_inds]), len(inds[pos_inds].unique()))
    
    # assert len(inds[pos_inds]) == len(inds[pos_inds].unique()), 'Overlaped inds'
    # remove lmks belonging to 2 or more gt faces
    ovl_mask = inds[pos_inds].unique()[:,None]==inds[pos_inds]
    ovl_cnt = ovl_mask.long().sum(-1)
    ovl_vals_ids = inds[pos_inds].unique()[ovl_cnt>1]
    print(ovl_vals_ids, ovl_vals_ids.shape[-1]==0)
    if ovl_vals_ids.shape[-1] != 0:
        sus = True
        ovl_mask2 = inds[pos_inds][:,None] == ovl_vals_ids
        if ovl_mask2.dim() == 1:
            ovl_vals = ovl_mask2
        else:
            ovl_vals = ovl_mask2.sum(-1).bool()
        print('ovl_vals:', ovl_vals)        
        lmks[inds[pos_inds][ovl_vals]] *= 0
        print('lmks:', lmks[inds][pos_inds][ovl_vals])
        print('faces:', faces[pos_inds][ovl_vals])
    else:
        ovl_vals = False
    all_lmks = lmks[inds]
    all_lmks[~pos_inds] *= 0
    # all_lmks[pos_inds][ovl_vals] *= torch.tensor(0)    
    # pos_lmks 
    print('num pos:', (all_lmks[:,2, 0]>0).sum().item())    
    assert all_lmks.shape[0] == faces.shape[0], 'Lmks and faces dont have the same shape !'

    return all_lmks, sus, faces[pos_inds][ovl_vals]


def insert_lmks(lmks, result):
    new_result = result
    for id, info in enumerate(new_result['gtboxes']):
        landmark = [j.item() for i in lmks[id] for j in i]
        info['lmk'] = landmark
    
    return new_result


if __name__ =='__main__':
    from tqdm import tqdm
    import pickle

    coco = COCO('../../../data/CrownHuman/Annotations/instances_train_full_bhf_new.json')
    image_id = coco.image_id
    anno = coco.anno
    result = coco.process_gt(True)
    print('Len results:', len(result))

    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=0, nms=0.1)

    new_results=[]
    sus_names = []
    sus_faces = []
    for i, r in enumerate(tqdm(result)):
        new_result, ind, sus_face = draw_box_v2(result, i, True)
        new_results.append(new_result)
        if ind is not None:
            sus_names.append(ind)
            sus_faces.append(sus_face)            
        pickle.dump(new_results, open('new_data.pkl', 'wb'))
        with open('sus_name.txt','w') as f:
            if len(sus_names) > 0:
                for n, faces in zip(sus_names, sus_faces):
                    f.write('Box_id: '+str(n) + '\n')
                    faces = list(faces)
                    faces = [list(face) for face in faces]
                    # print(faces)
                    for face in faces:
                        face = [iface.item() for iface in face]
                        f.write(str(face)+'\n')
        print('Removed lmks:', len(sus_names))
        print('r_id:', i)
        # break
        
    #----------------
    # draw_box_v2(result[2308], True)
    # draw_box_v2(result[torch.randint(0, 4370, (1,1)).item()], True)
    
    #----------------
    # user_in = ' '#input('Enter your order!:  ')
    # while(user_in==' '):
    #     r_id = torch.randint(0, 4370, (1,1)).item()
    #     print('r_id:', r_id)
    #     draw_box_v2(result, r_id, True)
    #     user_in = input('Stop or next?:  ')
    # detect_face('282555,d9dc6000e9cd678f.jpg')