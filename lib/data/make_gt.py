# from test_loader_gt import COCO
import cv2
import torch
import pickle

with open('CrHu_train15k.pkl', 'rb') as f:
    data = pickle.load(f)

def draw_box_v2(result, ignore):    
    print('result:', result)
    img_name = result['ID']
    img = cv2.imread('../../../data/CrownHuman/CrowdHuman_train/Images/'+img_name)
    print('Num bboxes:', len(result['gtboxes']))
    bboxes = [i['bbox'] for i in result['gtboxes']]
    faces = [i['fbox'] for i in result['gtboxes']]
    lmks = [i['lmk'] for i in result['gtboxes']]
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
    print(bboxes.shape)

    # _, lmks, _ = detect_face(img_name)
    # all_lmks, sus = matching_face_lmks_v2(lmks, faces)
    # new_result = insert_lmks(all_lmks, result)
    # print('new_result:', new_result)

    thickness = 1
    color = [(255, 128, 0), (255, 153, 255), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)]
    color = color*len(bboxes)
    for i in range(len(bboxes)):
        start_point = tuple(bboxes[i][:2])
        end_point = tuple(bboxes[i][2:4])
        img = cv2.rectangle(img, start_point, end_point, color[i], thickness)
        img = cv2.rectangle(img, tuple(faces[i][:2]), tuple(faces[i][2:4]), color[i], thickness)

        landmark = lmks[i]
        
        for j in range(5):
            j = 4
            img = cv2.circle(img, tuple([int(landmark[j*2]), int(landmark[j*2+1])]), 1, color[i], 2)

    print('Total lmks:', len(lmks))
    print('Pos lmks:', (torch.tensor(lmks)[:,2]>0).sum().item())
    # print(lmks)
    # if len(lmks) != 0:    
    #     for lmk in lmks:
    #         lmk = list(lmk)
    #         lmk = [list(i) for i in lmk]        
    #         for i in lmk:            
    #             img = cv2.circle(img, tuple([int(i[0]), int(i[1])]), 1, (255, 0, 0), 1)

    cv2.imwrite('gt2.png', img)
    cv2.imshow('testbox', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print('Len data:', len(data))
user_in = input('Put number here ')
user_in = int(user_in)
draw_box_v2(data[user_in], True)