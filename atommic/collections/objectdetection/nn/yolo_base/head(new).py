import torch
import torch.nn.functional as F
from torch import nn

from atommic.collections.objectdetection.nn.yolo_base import box_ops


class Head(nn.Module):
    def __init__(self, predictor, anchors, strides, 
                 match_thresh, giou_ratio, loss_weights, 
                 score_thresh, nms_thresh, detections):
        super().__init__()
        self.predictor = predictor
        self.register_buffer("anchors", torch.Tensor(anchors))
        self.strides = strides
        
        self.match_thresh = match_thresh
        self.giou_ratio = giou_ratio
        self.loss_weights = loss_weights
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections = detections
        
        self.merge = False
        self.eval_with_loss = False
        #self.min_size = 2
        
    def forward(self, features): #, targets, image_shapes=None, scale_factors=None, max_size=None):

        preds = self.predictor(features)
        return preds
        # if self.training:
        #     losses = self.compute_loss(preds, targets)
        #     return losses
        # else:
        #     losses = {}
        #     if self.eval_with_loss:
        #         losses = self.compute_loss(preds, targets)
        #
        #     results = self.inference(preds, image_shapes, scale_factors, max_size)
        #     return results, losses
        
    # def compute_loss(self, preds, targets):
    #     dtype = preds[0].dtype
    #     image_ids = torch.cat([torch.full_like(tgt["labels"], i)
    #                                for i, tgt in enumerate(targets)])
    #     gt_labels = torch.cat([tgt["labels"] for tgt in targets])
    #     gt_boxes = torch.cat([tgt["boxes"] for tgt in targets])
    #     gt_boxes = box_ops.xyxy2cxcywh(gt_boxes)
    #
    #     losses = {
    #         "loss_box": gt_boxes.new_tensor(0),
    #         "loss_obj": gt_boxes.new_tensor(0),
    #         "loss_cls": gt_boxes.new_tensor(0)}
    #     for pred, stride, wh in zip(preds, self.strides, self.anchors):
    #         anchor_id, gt_id = box_ops.size_matched_idx(wh, gt_boxes[:, 2:], self.match_thresh)
    #
    #         gt_object = torch.zeros_like(pred[..., 4])
    #         if len(anchor_id) > 0:
    #             gt_box_xy = gt_boxes[:, :2][gt_id]
    #             ids, grid_xy = box_ops.assign_targets_to_proposals(gt_box_xy / stride, pred.shape[1:3])
    #             anchor_id, gt_id = anchor_id[ids], gt_id[ids]
    #             image_id = image_ids[gt_id]
    #
    #             pred_level = pred[image_id, grid_xy[:, 1], grid_xy[:, 0], anchor_id]
    #
    #             xy = 2 * torch.sigmoid(pred_level[:, :2]) - 0.5 + grid_xy
    #             wh = 4 * torch.sigmoid(pred_level[:, 2:4]) ** 2 * wh[anchor_id] / stride
    #             box_grid = torch.cat((xy, wh), dim=1)
    #             giou = box_ops.box_giou(box_grid, gt_boxes[gt_id] / stride).to(dtype)
    #             losses["loss_box"] += (1 - giou).mean()
    #
    #             gt_object[image_id, grid_xy[:, 1], grid_xy[:, 0], anchor_id] = \
    #             self.giou_ratio * giou.detach().clamp(0) + (1 - self.giou_ratio)
    #
    #             #pos = 1 - 0.5 * self.eps
    #             #neg = 1 - pos
    #             gt_label = torch.zeros_like(pred_level[..., 5:])
    #             gt_label[range(len(gt_id)), gt_labels[gt_id]] = 1
    #             losses["loss_cls"] += F.binary_cross_entropy_with_logits(pred_level[..., 5:], gt_label)
    #         losses["loss_obj"] += F.binary_cross_entropy_with_logits(pred[..., 4], gt_object)
    #
    #     losses = {k: v * self.loss_weights[k] for k, v in losses.items()}
    #     return losses
    

    