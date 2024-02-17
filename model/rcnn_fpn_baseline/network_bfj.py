import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('../lib/')
sys.path.append('../model/rcnn_fpn_baseline/')

from config_bfj import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from module.rpn import RPN
from layers.pooler import roi_pooler
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.bbox_opr import bbox_transform_inv_opr_v2, bbox_transform_inv_opr_lmk
from det_oprs.fpn_roi_target import fpn_roi_target_bfj
from det_oprs.loss_opr import softmax_loss, smooth_l1_loss, embedding_loss, embedding_loss2, angular_loss, embedding_loss_cse
from det_oprs.utils import get_padded_tensor

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 2, 6)
        self.RPN = RPN(config.rpn_channel)
        self.RCNN = RCNN()

    def forward(self, image, im_info, gt_boxes=None):
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)
        if self.training:
            return self._forward_train(image, im_info, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        loss_dict = {}
        fpn_fms = self.FPN(image)        
        # fpn_fms stride: 64,32,16,8,4, p6->p2
        rpn_rois, loss_dict_rpn = self.RPN(fpn_fms, im_info, gt_boxes)        
        rcnn_rois, rcnn_lmks, rcnn_labels, rcnn_tags, rcnn_ious, rcnn_centers, rcnn_bbox_targets = fpn_roi_target_bfj(
                rpn_rois, im_info, gt_boxes, top_k=1)
        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois, rcnn_lmks,
                rcnn_labels, rcnn_tags, rcnn_ious, rcnn_centers, rcnn_bbox_targets)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms = self.FPN(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        # print('rpn_rois:', rpn_rois.shape, ', im_info:',im_info)
        pred_bbox, pred_lmk, pred_emb, num_classes = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox.cpu().detach(), pred_lmk.cpu().detach(), pred_emb.cpu().detach(), num_classes

class LandmarkHead(nn.Module):
    def __init__(self, inchannels=1024, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv1d(inchannels,num_anchors*10,kernel_size=1\
                            , stride=1,padding=0)
        nn.init.normal_(self.conv1x1.weight, std=0.001)
        nn.init.constant_(self.conv1x1.bias, 0)
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.view(out.shape[0], -1, 10)

        return out

class Landmark_Cls(nn.Module):
    def __init__(self, inchannels=1024, num_classes=2):
        super(Landmark_Cls, self).__init__()
        self.conv1x1 = nn.Conv1d(inchannels,num_classes,kernel_size=1\
                            , stride=1,padding=0)
        nn.init.normal_(self.conv1x1.weight, std=0.001)
        nn.init.constant_(self.conv1x1.bias, 0)
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.view(-1, 2)

        return out

class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # roi head
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(256*7*7, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        
        # for l in [self.fc1, self.fc2]:
        for l in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        # box predictor
        self.pred_cls = nn.Linear(1024, config.num_classes)
        self.pred_delta = nn.Linear(1024, config.num_classes * 4) # refer to 2 boxes - face and body, choose one at training, keep two at inference
        self.pred_pos = nn.Linear(1024, config.num_classes * 2)
        self.pred_emb = nn.Linear(1024, 32)
        # self.pred_landmarks = LandmarkHead(1024, config.num_classes)
        self.pred_landmarks = nn.Linear(1024, config.num_classes * 10)
        self.pred_lmk_cls = nn.Linear(1024, 2)
        # self.pred_lmk_cls = Landmark_Cls(1024, 2)
        
        for l in [self.pred_cls]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        for l in [self.pred_delta]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)
        for l in [self.pred_pos]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)
        for l in [self.pred_emb]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.pred_landmarks.weight, std=0.001)
        nn.init.constant_(self.pred_landmarks.bias, 0)
        nn.init.normal_(self.pred_lmk_cls.weight, std=0.01)
        nn.init.constant_(self.pred_lmk_cls.bias, 0)

    def forward(self, fpn_fms, rcnn_rois, rcnn_lmks=None, labels=None, tags=None, ious=None, centers=None, bbox_targets_dis=None):
        # input p2-p5
        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]
        pool_features = roi_pooler(fpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")
        flatten_feature = torch.flatten(pool_features, start_dim=1)
        flatten_feature_1 = F.relu_(self.fc1(flatten_feature))        
        flatten_feature_1 = F.relu_(self.fc2(flatten_feature_1))
        emb_feature = F.relu_(self.fc3(flatten_feature))
        emb_feature = F.relu_(self.fc4(emb_feature))
        # print('emb_feature:', emb_feature.shape)
        pred_cls = self.pred_cls(flatten_feature_1)
        pred_delta = self.pred_delta(flatten_feature_1)
        pred_pos = self.pred_pos(flatten_feature_1)
        # pred_lmk = self.pred_landmarks(flatten_feature_1[...,None])
        pred_lmk = self.pred_landmarks(flatten_feature_1)
        pred_lmk_clone = pred_lmk.clone()
        pred_lmk = pred_lmk.reshape(-1, config.num_classes, 10)
        pred_emb = self.pred_emb(emb_feature)
        # pred_lmk_cls = self.pred_lmk_cls(flatten_feature_1[...,None])
        pred_lmk_cls = self.pred_lmk_cls(flatten_feature_1)
        soft_lmk_cls = F.softmax(pred_lmk_cls, dim=-1)
        # print('pred_lmk_cls:', pred_lmk_cls.shape)
        if self.training:
            loss_dict = {}
            # loss for regression
            labels = labels.long().flatten()
            tags = tags.long().flatten()
            ious = ious.flatten()
            centers = centers.flatten(1)
            fg_masks = labels > 0
            valid_masks = labels >= 0
            # multi class
            pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
            pred_delta1 = pred_delta.clone()
            pred_pos = pred_pos.reshape(-1, config.num_classes, 2)
            fg_gt_classes = labels[fg_masks]
            pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
            pred_pos = pred_pos[fg_masks, fg_gt_classes, :]
            pred_regression = torch.cat((pred_delta, pred_pos), dim=-1)            
            localization_loss = smooth_l1_loss(
                pred_regression,
                bbox_targets_dis[fg_masks, :6],
                config.rcnn_smooth_l1_beta)

            # loss for face landmarks  
            ''' Create mask of boxes with available landmarks. We only annotate landmarks
            for the face boxes and the corresponding body boxes.'''            
            lmk_masks = (rcnn_lmks[:,-1]>0) * fg_masks
            skip_lmk = True if lmk_masks.sum() == 0 else False
            if not skip_lmk:
                pred_lmk = pred_lmk[lmk_masks, labels[lmk_masks], :]
                # assert lmk_masks.sum() >= len(tags_face)
                # print('rcnn_lmks:', rcnn_lmks[lmk_masks])                
                location_lmk_loss = smooth_l1_loss(
                    pred_lmk,
                    bbox_targets_dis[lmk_masks, 6:],
                    config.rcnn_smooth_l1_beta
                )
                loss_rcnn_lmk = location_lmk_loss.sum()/pred_lmk.numel()

                # loss for lmks classification
                # loss_lmk_cls = softmax_loss(pred_lmk_cls, lmk_masks.long(), num_classes=2)
                # loss_lmk_cls = loss_lmk_cls.sum()/ pred_lmk_cls.shape[0]
                loss_lmk_cls = softmax_loss(pred_lmk_cls[lmk_masks], lmk_masks[lmk_masks>0].long(), num_classes=2)
                loss_lmk_cls = loss_lmk_cls.sum()/ loss_lmk_cls.shape[0]

            # loss for classification
            ##objectness_loss = softmax_loss(pred_cls[fg_masks], labels[fg_masks], num_classes=config.num_classes-1)
            objectness_loss = softmax_loss(pred_cls, labels, num_classes=config.num_classes)
            objectness_loss = objectness_loss * valid_masks
            normalizer = 1.0 / valid_masks.sum().item()
            loss_rcnn_loc = localization_loss.sum() * normalizer
            loss_rcnn_cls = objectness_loss.sum() * normalizer

            # loss for embedding
            loss_rcnn_emb = self.pull_push_loss(labels, tags, ious, centers, pred_emb)
            
            # loss for angular
            class_num = pred_cls.shape[-1] - 1
            base_rois = rcnn_rois[:, 1:5].reshape(-1, 4)[fg_masks]
            pred_regression = pred_regression.reshape(-1, 6)
            gt_bbox = bbox_targets_dis[fg_masks, :6].reshape(-1, 6)
            pred_bbox = restore_bbox(base_rois, pred_regression, True)
            gt_bbox = restore_bbox(base_rois, gt_bbox, True)
            pos_regression = pred_bbox[:, 4:]
            pos_body_targets = gt_bbox[:, :2]
            pos_targets = gt_bbox[:, 4:]
            pos_regression = pos_regression - pos_body_targets
            pos_targets = pos_targets - pos_body_targets
            angular_loss_pos = angular_loss(pos_regression, pos_targets) * normalizer

            # loss for angular with lmk
            if not skip_lmk:
                base_rois_lmks = rcnn_rois[:, 1:5].reshape(-1, 4)[lmk_masks]
                pred_delta1 = pred_delta1[lmk_masks, labels[lmk_masks], :2]
                # print('**:', pred_delta1.shape, pred_lmk.shape)
                pred_offset_lmk = torch.cat((pred_delta1, pred_lmk), dim= 1)
                pred_pos_lmk = restore_bbox(base_rois_lmks, pred_offset_lmk, True, True)
                # Need to find the center lmk point
                pred_pos_lmk_x = pred_pos_lmk[:, 2:12:2].mean(-1)
                pred_pos_lmk_y = pred_pos_lmk[:, 3:13:2].mean(-1)
                pred_cen_lmk = torch.cat((pred_pos_lmk_x.unsqueeze(-1), pred_pos_lmk_y.unsqueeze(-1)), dim=1)
                gt_offset_lmk = torch.cat((bbox_targets_dis[lmk_masks, :2], bbox_targets_dis[lmk_masks, 6:]), dim=1)
                gt_lmk = restore_bbox(base_rois_lmks, gt_offset_lmk, True, True)
                lmk_target_x = gt_lmk[:, 2:12:2].mean(-1)
                lmk_target_y = gt_lmk[:, 3:13:2].mean(-1)
                target_cen_lmk = torch.cat((lmk_target_x.unsqueeze(-1), lmk_target_y.unsqueeze(-1)), dim=1)
                target_vector = target_cen_lmk - gt_lmk[:, :2]
                pred_vector = pred_cen_lmk - gt_lmk[:, :2]
                angular_loss_lmk = angular_loss(target_vector, pred_vector) / target_vector.shape[0]                
                loss_dict['loss_lmk'] = 2*loss_rcnn_lmk + loss_lmk_cls + 1.5*angular_loss_lmk
            
            loss_dict['loss_rcnn_loc'] = loss_rcnn_loc * 2 + angular_loss_pos 
            loss_dict['loss_rcnn_cls'] = loss_rcnn_cls
            loss_dict['loss_rcnn_emb'] = loss_rcnn_emb
            return loss_dict
        else:
            class_num = pred_cls.shape[-1] - 1
            tag = torch.arange(class_num).type_as(pred_cls)+1            
            tag = tag.repeat(pred_cls.shape[0], 1).reshape(-1,1)
            # print('tag:', tag.shape)
            pred_scores = F.softmax(pred_cls, dim=-1)[:, 1:].reshape(-1, 1)
            # print('pred_scores:', pred_scores.shape)
            pred_delta = pred_delta[:, 4:].reshape(-1, 4)
            # print('pred_delta', pred_delta.shape)
            pred_offset_lmk = pred_lmk_clone[:, 10:].reshape(-1, 10)
            pred_offset_lmk = torch.cat((pred_delta[:, :2], pred_offset_lmk), dim=1)
            pred_pos = pred_pos[:, 2:].reshape(-1, 2)
            base_rois = rcnn_rois[:, 1:5].repeat(1, class_num).reshape(-1, 4)
            pred_six = torch.cat((pred_delta, pred_pos), dim=-1)    #1592*6
            pred_lmks = restore_bbox(base_rois, pred_offset_lmk, True, True)[:, 2:]
            soft_lmk_cls = (soft_lmk_cls[:,1][:,None].repeat(1,2)).reshape(-1,1)
            pred_lmks = torch.cat((pred_lmks, soft_lmk_cls), dim=1)
            assert pred_lmks.shape[1] == 11
            pred_bbox = restore_bbox(base_rois, pred_six, True)     #1592*6            
            pred_bbox = torch.cat([pred_bbox, pred_scores, tag], axis=1)    #1592*8            
            pred_emb = pred_emb.repeat(1, 2).reshape(-1, 32)
            return pred_bbox, pred_lmks, pred_emb, class_num
    
    def pull_push_loss(self, labels, tags, ious, centers, embedding_pred):

        # embeddings
        embedding_pred = embedding_pred.squeeze()

        # sample bodies
        sampled_body_inds_subset = torch.nonzero((labels == 1) & (tags != -1), as_tuple=False).squeeze(1)
        embedding_pred_body = embedding_pred[sampled_body_inds_subset]
        tags_body = tags[sampled_body_inds_subset]
        ious_body = ious[sampled_body_inds_subset]
        centers_body = centers[sampled_body_inds_subset]
        tags_unique = tags_body.unique() # eg. 24

        if len(tags_body) == 0:
            return 0

        body = tags_body == tags_unique[:, None]    # 24x77
        # print('body:', body.shape, 'tags_unique:', tags_unique.shape, 'tags_body:', tags_body.shape)
        # print('embedding_pred_body:', embedding_pred_body.shape)
        topk = 3
        if len(tags_body) < topk:
            pad = nn.ZeroPad2d(padding=(0, topk - len(tags_body)))
            ious_body = pad(ious_body)

            pad = nn.ZeroPad2d(padding=(0, 0, 0, topk - len(tags_body)))
            embedding_pred_body = pad(embedding_pred_body)
            centers_body = pad(centers_body)

            pad = nn.ZeroPad2d(padding=(0, topk - len(tags_body)))
            body = pad(body)

        body_embedding_post = torch.where(body.unsqueeze(2), embedding_pred_body, torch.tensor(0.0, device=body.device)) 
        # print('body_embedding_post:', body_embedding_post.shape) #24x77x32
        body_centers = torch.where(body.unsqueeze(2), centers_body, torch.tensor(0.0, device=body.device))  #24x77x3
        # print('body_centers:', body_centers.shape)
        body_ious_post = torch.where(body, ious_body, torch.tensor(0.0, device=body.device))    #24x77

        body_ious_post_vals, body_ious_post_ids = body_ious_post.topk(topk, dim=1)  #24x3, 24x3
        body_ious_post_vals_max, body_ious_post_ids_max = body_ious_post.max(dim=1) #24, 24
        body_ious_post_ids = torch.where(body_ious_post_vals != 0, body_ious_post_ids, body_ious_post_ids_max[:, None])
        #24x3        
        body_embedding_post_new = torch.cat([i[j].unsqueeze(0) for i,j in zip(body_embedding_post, body_ious_post_ids)], dim=0)
        # print('body_embedding_post_new:', body_embedding_post_new.shape)    #24x3x32
        body_centers_new = torch.cat([i[j].unsqueeze(0) for i,j in zip(body_centers, body_ious_post_ids)], dim=0)   
        # print('body_centers_new:', body_centers_new.shape)  #24x3x3

        # sample faces
        sampled_face_inds_subset = torch.nonzero((labels == 2) & (tags != -1), as_tuple=False).squeeze(1)
        embedding_pred_face = embedding_pred[sampled_face_inds_subset]
        tags_face = tags[sampled_face_inds_subset]
        ious_face = ious[sampled_face_inds_subset]
        centers_face = centers[sampled_face_inds_subset]

        face = tags_face == tags_unique[:, None]    #eg: 26x29

        if len(tags_face) > 0:
            if len(tags_face) < topk:
                pad = nn.ZeroPad2d(padding=(0, topk - len(tags_face)))
                ious_face = pad(ious_face)

                pad = nn.ZeroPad2d(padding=(0, 0, 0, topk - len(tags_face)))
                embedding_pred_face = pad(embedding_pred_face)
                centers_face = pad(centers_face)

                pad = nn.ZeroPad2d(padding=(0,topk - len(tags_face)))
                face = pad(face)

            face_embedding_post = torch.where(face.unsqueeze(2), embedding_pred_face, torch.tensor(0.0, device=face.device))
            # print('face_embedding_post:', face_embedding_post.shape)    #24x29x32
            face_centers = torch.where(face.unsqueeze(2), centers_face, torch.tensor(0.0, device=face.device))
            # print('face_centers:', face_centers.shape)  #24x29x3
            face_ious_post = torch.where(face, ious_face, torch.tensor(0.0, device=face.device))    #24x29

            face_ious_post_vals, face_ious_post_ids = face_ious_post.topk(topk, dim=1)  #24x3
            face_ious_post_vals_max, face_ious_post_ids_max = face_ious_post.max(dim=1) #24
            face_ious_post_ids = torch.where(face_ious_post_vals != 0, face_ious_post_ids, face_ious_post_ids_max[:, None])
            #24x3

            face_embedding_post_new = torch.cat([i[j].unsqueeze(0) for i,j in zip(face_embedding_post, face_ious_post_ids)], dim=0)
            #24x3x32
            face_centers_new = torch.cat([i[j].unsqueeze(0) for i,j in zip(face_centers, face_ious_post_ids)], dim=0)
            #24x3x3

            face_embedding_post_new = torch.where(face_embedding_post_new != 0, face_embedding_post_new, body_embedding_post_new)
            # nofaces_box_ind = torch.abs(face_centers_new).sum((-1,-2)) == 0
            # print('nofaces_box_ind:', nofaces_box_ind.sum())
            face_centers_new = torch.where(face_centers_new != 0, face_centers_new, body_centers_new)
        else:
            face_embedding_post_new = body_embedding_post_new
            face_centers_new = body_centers_new
        
        assert (body_embedding_post_new.shape[1] == topk) and (face_embedding_post_new.shape[1] == topk), 'wrong shape 1'
        assert (body_embedding_post_new.shape[2] == 32) and (face_embedding_post_new.shape[2] == 32), 'wrong shape 2'
        assert (body_centers_new.shape[1]==topk) and (face_centers_new.shape[1]==topk), 'wrong shape 1'
        assert (body_centers_new.shape[2]==3) and (face_centers_new.shape[2]==3), 'wrong shape 2'
        assert body_embedding_post_new.shape[0] == face_embedding_post_new.shape[0], 'wrong shape 0'

        if config.loss_type == 'cse':
            pull_loss, push_loss = embedding_loss_cse(body_embedding_post_new, face_embedding_post_new, body_centers_new, face_centers_new, topk, 0.00, 32.0)
        elif config.loss_type == 'cons_min':
            pull_loss, push_loss = embedding_loss(body_embedding_post_new, face_embedding_post_new, body_centers_new, face_centers_new, topk)
        elif config.loss_type == 'cons_cos':
            pull_loss, push_loss = embedding_loss2(body_embedding_post_new, face_embedding_post_new, body_centers_new, face_centers_new, topk)
        else:
            print('No suppored loss type !!!')
        aeloss = pull_loss * 0.1 + push_loss * 0.1 #0.15

        return aeloss

def restore_bbox(rois, deltas, unnormalize=True, lmks=False):
    if unnormalize:        
        if not lmks:
            std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
            mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
        else:
            std_opr = torch.tensor(config.bbox_12_normalize_stds[None, :]).type_as(deltas)
            mean_opr = torch.tensor(config.bbox_12_normalize_means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    if not lmks:
        pred_bbox = bbox_transform_inv_opr_v2(rois, deltas)
    else:
        pred_bbox = bbox_transform_inv_opr_lmk(rois, deltas)
    return pred_bbox
