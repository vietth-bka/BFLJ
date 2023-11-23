import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config_bfj import config
from det_oprs.margin_loss import ArcFace

def softmax_loss(score, label, ignore_label=-1, num_classes=2):
    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]
    score -= max_score
    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], num_classes, device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    return loss

def smooth_l1_loss(pred, target, beta: float):
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)
    return loss.sum(axis=1)

def focal_loss(inputs, targets, alpha=-1, gamma=2):
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_pred = (1 - inputs) ** gamma * torch.log(inputs)
    neg_pred = inputs ** gamma * torch.log(1 - inputs)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)

def emd_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels, num_classes=3)
    loss = objectness_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def emd_loss_focal(p_b0, p_s0, p_b1, p_s1, targets, labels):
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels,
            config.focal_loss_alpha, config.focal_loss_gamma)
    fg_masks = (labels > 0).flatten()
    localization_loss = smooth_l1_loss(
            pred_delta[fg_masks],
            targets[fg_masks],
            config.smooth_l1_beta)
    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def embedding_loss(tag0, tag1, cen0, cen1, topk):
    topk2 = topk * topk
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    num = len(tag0)

    # pull body_face
    tag = tag0.unsqueeze(1) - tag1.unsqueeze(2)
    # print('tag_bf:', tag.shape)
    tag = torch.pow(tag, 2) / (num * topk2 + 1e-4)
    pull_bf = tag.sum()

    # pull body_body
    tag_bb = tag0.unsqueeze(1) - tag0.unsqueeze(2)
    cen_bb = cen0[:, :, :2].unsqueeze(1) - cen0[:, :, :2].unsqueeze(2)
    # print('cen_bb:', cen_bb.shape)
    tag_bb = torch.pow(tag_bb, 2) / (num * topk2 + 1e-4)
    cen_bb = torch.sqrt(torch.pow(cen_bb, 2).sum(-1)) / (cen0[:, :, 2].unsqueeze(-1) + 1e-7)
    cen_bb = torch.exp(cen_bb)
    tag_bb = tag_bb * cen_bb.unsqueeze(-1)
    pull_bb = tag_bb.sum()

    # pull face_face
    tag_ff = tag1.unsqueeze(1) - tag1.unsqueeze(2)
    cen_ff = cen1[:, :, :2].unsqueeze(1) - cen1[:, :, :2].unsqueeze(2)
    tag_ff = torch.pow(tag_ff, 2) / (num * topk2 + 1e-4)
    cen_ff = torch.sqrt(torch.pow(cen_ff, 2).sum(-1)) / (cen1[:, :, 2].unsqueeze(-1) + 1e-7)
    cen_ff = torch.exp(cen_ff)
    tag_ff = tag_ff * cen_ff.unsqueeze(-1)
    pull_ff = tag_ff.sum()

    pull = pull_bb * 1.5 + pull_bf + pull_ff * 1.5

    dist1 = tag0.view(-1, tag0.shape[-1])
    dist2 = tag1.view(-1, tag1.shape[-1])
    # push body_face
    push_bf = dist1.unsqueeze(0) - dist2.unsqueeze(1)
    push_bf = torch.pow(push_bf, 2).sum(-1)
    for i in range(int(len(push_bf) / topk)):
        push_bf[i*topk:i*topk+topk, i*topk:i*topk+topk] = torch.tensor(0.0).to(device=dist1.device)
    push_bf = 2 - push_bf
    push_bf = nn.functional.relu(push_bf, inplace=True)
    push_bf = push_bf / ((num - 1) * num * topk2 + 1e-4)
    push_bf = push_bf.sum()

    ################## 
    # my code's here, no needs to reshape tag
    minus = tag0[None,:,None] - tag1[:,None,:,None] #eg [16x3x32]-[16x3x32] -> [16x16x3x3x32]
    minus = torch.pow(minus, 2).sum(-1)
    # print('minus:', minus.shape)
    # for i in range(len(minus)):
    #     minus[i, i] *= torch.tensor(0.0).to(device=minus.device)
    inds = (1-torch.eye(minus.shape[0])).bool().to(device=minus.device)
    minus = torch.where(inds[...,None,None], minus, torch.tensor(0.0).to(device=minus.device))
    minus = 2 - minus
    minus = nn.functional.relu(minus, inplace=True)
    minus = minus / ((num - 1) * num * topk2 + 1e-4)
    minus = minus.sum()
    # print('Check:', minus - push_bf, minus, push_bf)
    ##################

    # push body_body
    push_bb = dist1.unsqueeze(0) - dist1.unsqueeze(1)
    push_bb = torch.pow(push_bb, 2).sum(-1)
    for i in range(int(len(push_bb) / topk)):
        push_bb[i*topk:i*topk+topk, i*topk:i*topk+topk] = torch.tensor(0.0).to(device=dist1.device)
    push_bb = 2 - push_bb
    push_bb = nn.functional.relu(push_bb, inplace=True)
    push_bb = push_bb / ((num - 1) * num * topk2 + 1e-4)
    push_bb = push_bb.sum()

    # push face_face
    push_ff = dist2.unsqueeze(0) - dist2.unsqueeze(1)
    push_ff = torch.pow(push_ff, 2).sum(-1)
    for i in range(int(len(push_ff) / topk)):
        push_ff[i*topk:i*topk+topk, i*topk:i*topk+topk] = torch.tensor(0.0).to(device=dist1.device)
    push_ff = 2 - push_ff
    push_ff = nn.functional.relu(push_ff, inplace=True)
    push_ff = push_ff / ((num - 1) * num * topk2 + 1e-4)
    push_ff = push_ff.sum()

    push = push_bb * 1.5 + push_bf + push_ff * 1.5
    # push = push_bf

    return pull, push

def embedding_loss2(tag0, tag1, cen0, cen1, topk):
    topk2 = topk * topk
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    #normalize embs
    norm0 = torch.norm(tag0, dim=-1).unsqueeze(-1)
    ntag0 = tag0/norm0
    norm1 = torch.norm(tag1, dim=-1).unsqueeze(-1)
    ntag1 = tag1/norm1
    num = len(tag0)

    # pull body_face
    tag = (ntag0.unsqueeze(1) * ntag1.unsqueeze(2)).sum(-1)
    tag = 1 - tag.clamp(-1,1)
    pull_bf = tag.sum() / (num * topk2 + 1e-4)
    # print('pull_bf:', pull_bf)

    # pull body_body
    tag_bb = (ntag0.unsqueeze(1) * ntag0.unsqueeze(2)).sum(-1)
    tag_bb = 1 - tag_bb.clamp(-1, 1)
    # print('tag_bb:', tag_bb.shape, tag_bb.sum(-1))
    cen_bb = cen0[:, :, :2].unsqueeze(1) - cen0[:, :, :2].unsqueeze(2)
    tag_bb = tag_bb / (num * topk2 + 1e-4)
    cen_bb = torch.sqrt(torch.pow(cen_bb, 2).sum(-1)) / (cen0[:, :, 2].unsqueeze(-1) + 1e-7)
    cen_bb = torch.exp(cen_bb)
    assert cen_bb.shape[0]*cen_bb.shape[1] == tag_bb.shape[0]*tag_bb.shape[1]
    tag_bb = tag_bb * cen_bb
    pull_bb = tag_bb.sum()

    # pull face_face
    tag_ff = (ntag1.unsqueeze(1) * ntag1.unsqueeze(2)).sum(-1)
    tag_ff = 1 - tag_ff.clamp(-1,1)
    cen_ff = cen1[:, :, :2].unsqueeze(1) - cen1[:, :, :2].unsqueeze(2)
    tag_ff = tag_ff / (num * topk2 + 1e-4)
    cen_ff = torch.sqrt(torch.pow(cen_ff, 2).sum(-1)) / (cen1[:, :, 2].unsqueeze(-1) + 1e-7)
    cen_ff = torch.exp(cen_ff)
    tag_ff = tag_ff * cen_ff
    pull_ff = tag_ff.sum()

    pull = pull_bb * 1.5 + pull_bf + pull_ff * 1.5
    # pull = pull_bb + pull_bf + pull_ff
    # pull *= 1

    dist1 = ntag0.view(-1, tag0.shape[-1])   #16x3x32 -> 48x32
    dist2 = ntag1.view(-1, tag1.shape[-1])
    # push body_body
    push_bb = (dist1.unsqueeze(0) * dist1.unsqueeze(1)).sum(-1)
    # push_bb = push_bb.clamp(-1,1)    #48x48    
    for i in range(int(len(push_bb) / topk)):
        push_bb[i*topk:i*topk+topk, i*topk:i*topk+topk] = torch.tensor(0.0).to(device=dist1.device)    
    push_bb = nn.functional.relu(push_bb, inplace=True)
    push_bb = push_bb / ((num - 1) * num * topk2 + 1e-4)
    push_bb = push_bb.sum()

    #push face_face
    push_ff = (dist2.unsqueeze(0) * dist2.unsqueeze(1)).sum(-1)
    # push_ff = push_ff.clamp(-1,1)
    for i in range(int(len(push_ff) / topk)):
        push_ff[i*topk:i*topk+topk, i*topk:i*topk+topk] = torch.tensor(0.0).to(device=dist1.device)
    push_ff = nn.functional.relu(push_ff, inplace=True)
    push_ff = push_ff / ((num - 1) * num * topk2 + 1e-4)
    push_ff = push_ff.sum()

    #push body_face
    push_bf = (dist1.unsqueeze(0) * dist2.unsqueeze(1)).sum(-1)
    # push_bf = push_bf.clamp(-1,1)
    for i in range(int(len(push_bf) / topk)):
        push_bf[i*topk:i*topk+topk, i*topk:i*topk+topk] = torch.tensor(0.0).to(device=dist1.device)
    push_bf = nn.functional.relu(push_bf, inplace=True)
    push_bf = push_bf / ((num - 1) * num * topk2 + 1e-4)
    push_bf = push_bf.sum()

    push = push_bb * 1.5 + push_bf + push_ff * 1.5

    return pull*10, push*10

def embedding_loss_cse(tag0, tag1, cen0, cen1, topk, margin, scale):
    topk2 = topk * topk
    if tag0.ndim > 3:
        tag0 = tag0.squeeze()
    if tag1.ndim > 3:
        tag1 = tag1.squeeze()

    #normalize embs
    ntag0 = F.normalize(tag0, dim=-1)    
    ntag1 = F.normalize(tag1, dim=-1)
    num = len(tag0)
    
    # pull body_body
    tag_bb = (ntag0.unsqueeze(1) * ntag0.unsqueeze(2)).sum(-1)
    tag_bb = 1 - tag_bb.clamp(-1, 1)
    cen_bb = cen0[:, :, :2].unsqueeze(1) - cen0[:, :, :2].unsqueeze(2)
    tag_bb = tag_bb / (num * topk2 + 1e-4)
    cen_bb = torch.sqrt(torch.pow(cen_bb, 2).sum(-1)) / (cen0[:, :, 2].unsqueeze(-1) + 1e-7)
    cen_bb = torch.exp(cen_bb)
    assert cen_bb.shape[0]*cen_bb.shape[1] == tag_bb.shape[0]*tag_bb.shape[1], f'Size error:{cen_bb.shape}, {tag_bb.shape}'
    #Size error:torch.Size([1, 3, 3]), torch.Size([3, 32])
    tag_bb = tag_bb * cen_bb
    pull_bb = tag_bb.sum()

    # pull face_face
    tag_ff = (ntag1.unsqueeze(1) * ntag1.unsqueeze(2)).sum(-1)
    tag_ff = 1 - tag_ff.clamp(-1,1)
    cen_ff = cen1[:, :, :2].unsqueeze(1) - cen1[:, :, :2].unsqueeze(2)
    tag_ff = tag_ff / (num * topk2 + 1e-4)
    cen_ff = torch.sqrt(torch.pow(cen_ff, 2).sum(-1)) / (cen1[:, :, 2].unsqueeze(-1) + 1e-7)
    cen_ff = torch.exp(cen_ff)
    tag_ff = tag_ff * cen_ff
    pull_ff = tag_ff.sum()

    pull = pull_bb * 1.5 + pull_ff * 1.5

    assert ntag0.shape[1] == 3
    assert ntag0.shape[-1] == ntag1.shape[-1], 'Matrices size not consistent'
    # margin_loss = ArcFace(m=0.5, s=64.0)
    margin_loss = ArcFace(m=margin, s=scale)

    prototype_ntag0 = ntag0.mean(1)  #24 x 32
    prototype_ntag0 = F.normalize(prototype_ntag0, dim=-1)
    prototype_ntag1 = ntag1.mean(1)  #24 x 32
    prototype_ntag1 = F.normalize(prototype_ntag1, dim=-1)
    target = torch.arange(0, ntag0.shape[0], dtype=torch.int64).to(tag0.device)
    target = target[:, None].repeat(1,ntag0.shape[1]).flatten()

    # cosine_matrix_b = ntag0.reshape(-1,ntag0.shape[-1]).matmul(prototype_ntag1.t())
    # cosine_matrix_b = cosine_matrix_b.clamp(-1,1)
    # cse_loss_b = margin_loss(cosine_matrix_b, target)
    cse_loss_b = margin_loss(tag0, prototype_ntag1, target)

    # cosine_matrix_f = ntag1.reshape(-1,ntag0.shape[-1]).matmul(prototype_ntag0.t())
    # cosine_matrix_f = cosine_matrix_f.clamp(-1,1)
    # cse_loss_f = margin_loss(cosine_matrix_f, target)
    cse_loss_f = margin_loss(tag1, prototype_ntag0, target)

    cse = (cse_loss_b+cse_loss_f) * 0.5
    return pull*10, cse


# TODO maybe push this to nn?
def angular_loss(input, target):
    input_length = torch.sqrt((input[: ,1] ** 2 + input[:, 0] ** 2)) + 1e-7
    input = input / input_length.unsqueeze(1)
    target_length = torch.sqrt((target[: ,1] ** 2 + target[:, 0] ** 2)) + 1e-7
    target = target / target_length.unsqueeze(1)
    Cross_product = input[:, 1] * target[:, 0] - input[:, 0] * target[:, 1]
    return torch.abs(Cross_product).sum()

if __name__ == "__main__":
    a = torch.Tensor([[1,2],[2,3]])
    b = torch.Tensor([[2,3],[3,4]])
    print(vector_loss(a,b))
