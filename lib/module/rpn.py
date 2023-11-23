import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config_bfj import config
from det_oprs.anchors_generator import AnchorGenerator
from det_oprs.find_top_rpn_proposals import find_top_rpn_proposals
from det_oprs.fpn_anchor_target import fpn_anchor_target, fpn_rpn_reshape
from det_oprs.loss_opr import softmax_loss, smooth_l1_loss

class RPN(nn.Module):
    def __init__(self, rpn_channel = 256):
        super().__init__()
        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,     #32
            config.anchor_aspect_ratios, #[1, 2, 3]
            config.anchor_base_scale)    #[1]
        self.rpn_conv = nn.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = nn.Conv2d(rpn_channel, config.num_cell_anchors * 2, kernel_size=1, stride=1)
        self.rpn_bbox_offsets = nn.Conv2d(rpn_channel, config.num_cell_anchors * 4, kernel_size=1, stride=1)

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features, im_info, boxes=None):
        # prediction
        pred_cls_score_list = []
        pred_bbox_offsets_list = []
        for x in features:
            t = F.relu(self.rpn_conv(x))
            # print('fpn features:', x.shape)
            # print('self.rpn_cls_score(t):', self.rpn_cls_score(t).shape)
            pred_cls_score_list.append(self.rpn_cls_score(t))
            # print('self.rpn_bbox_offsets(t):', self.rpn_bbox_offsets(t).shape)
            pred_bbox_offsets_list.append(self.rpn_bbox_offsets(t))
        # get anchors
        all_anchors_list = []
        # stride: 64,32,16,8,4 p6->p2
        base_stride = 4
        off_stride = 2**(len(features)-1) # 16
        for fm in features:
            layer_anchors = self.anchors_generator(fm, base_stride, off_stride) #generate anchor boxes
            off_stride = off_stride // 2 # reduce anchor size as the receiptive field decreased
            all_anchors_list.append(layer_anchors)
        # sample from the predictions
        rpn_rois = find_top_rpn_proposals(
                self.training, pred_bbox_offsets_list, pred_cls_score_list,
                all_anchors_list, im_info) # use nms to find the proposal regions
        # print('rpn_rois:', rpn_rois.shape)
        rpn_rois = rpn_rois.type_as(features[0])
        if self.training:
            rpn_labels, rpn_bbox_targets = fpn_anchor_target(
                    boxes, im_info, all_anchors_list) # get the best iou boxes with the gt
            #rpn_labels = rpn_labels.astype(np.int32)
            pred_cls_score, pred_bbox_offsets = fpn_rpn_reshape(
                pred_cls_score_list, pred_bbox_offsets_list) # reshape cls and offsets
            # rpn loss
            valid_masks = rpn_labels >= 0
            objectness_loss = softmax_loss(
                pred_cls_score[valid_masks],
                rpn_labels[valid_masks],num_classes=2)

            pos_masks = rpn_labels > 0
            localization_loss = smooth_l1_loss(
                pred_bbox_offsets[pos_masks],
                rpn_bbox_targets[pos_masks],
                config.rpn_smooth_l1_beta)
            normalizer = 1 / valid_masks.sum().item()
            loss_rpn_cls = objectness_loss.sum() * normalizer
            loss_rpn_loc = localization_loss.sum() * normalizer
            loss_dict = {}
            loss_dict['loss_rpn_cls'] = loss_rpn_cls
            loss_dict['loss_rpn_loc'] = loss_rpn_loc
            return rpn_rois, loss_dict
        else:
            return rpn_rois

