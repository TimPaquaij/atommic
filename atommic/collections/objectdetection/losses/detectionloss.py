import numpy as np
import torch
from torch import Tensor
from atommic.core.classes.loss import Loss
from typing import Any, List, Tuple, Union,Dict
from atommic.collections.objectdetection.nn.yolo_base import box_ops
from atommic.collections.segmentation.losses.cross_entropy import BinaryCrossEntropy_with_logits_Loss
import torch.nn.functional as F

class BBoxLoss(Loss):
    def __init__(
            self,
            strides: List = (None),
            anchors: np.ndarray = [None],
            match_thresh: int = 2,

    ):

        super().__init__()
        self.strides = strides
        self.register_buffer("anchors", torch.Tensor(anchors))
        self.match_thresh = match_thresh
        #self.binary_cross_entropy = BinaryCrossEntropy_with_logits_Loss()

    def forward(self, targets: Dict, preds: torch.Tensor) -> Tuple[Union[Tensor, Any], Tensor]:  # noqa: MC0001
        dtype = preds[0].dtype
        image_ids = torch.cat([torch.full_like(tgt["labels"], i)
                               for i, tgt in enumerate(targets)])
        gt_labels = torch.cat([tgt["labels"] for tgt in targets])
        gt_boxes = torch.cat([tgt["boxes"] for tgt in targets])
        gt_boxes = box_ops.xyxy2cxcywh(gt_boxes)
        loss = {"loss_box": gt_boxes.new_tensor(0)}
        self.anchors = self.anchors.to(gt_boxes)
        for pred, stride, wh in zip(preds, self.strides, self.anchors):
            anchor_id, gt_id = box_ops.size_matched_idx(wh, gt_boxes[:, 2:], self.match_thresh)
            if len(anchor_id) > 0:
                gt_box_xy = gt_boxes[:, :2][gt_id]
                ids, grid_xy = box_ops.assign_targets_to_proposals(gt_box_xy / stride, pred.shape[1:3])
                anchor_id, gt_id = anchor_id[ids], gt_id[ids]
                image_id = image_ids[gt_id]

                pred_level = pred[image_id, grid_xy[:, 1], grid_xy[:, 0], anchor_id]
                xy = 2 * torch.sigmoid(pred_level[:, :2]) - 0.5 + grid_xy
                wh = 4 * torch.sigmoid(pred_level[:, 2:4]) ** 2 * wh[anchor_id] / stride
                box_grid = torch.cat((xy, wh), dim=1)
                giou = box_ops.box_giou(box_grid, gt_boxes[gt_id] / stride).to(dtype)
                loss["loss_box"] += (1 - giou).mean()

        return loss["loss_box"]

class ClassLoss(Loss):
    def __init__(
            self,
            strides: List = (None),
            anchors: np.ndarray = [None],
            giou_ratio: int=1,
            match_thresh: int=2,

    ):

        super().__init__()
        self.strides = strides
        self.register_buffer("anchors", torch.Tensor(anchors))
        self.match_thresh = match_thresh
        self.giou_ratio = giou_ratio
        #self.binary_cross_entropy = BinaryCrossEntropy_with_logits_Loss()
    def forward(self, targets: Dict, preds: torch.Tensor) -> Tuple[Union[Tensor, Any], Tensor]:  # noqa: MC0001
        dtype = preds[0].dtype
        image_ids = torch.cat([torch.full_like(tgt["labels"], i)
                               for i, tgt in enumerate(targets)])
        gt_labels = torch.cat([tgt["labels"] for tgt in targets])
        gt_boxes = torch.cat([tgt["boxes"] for tgt in targets])
        gt_boxes = box_ops.xyxy2cxcywh(gt_boxes)
        loss = {
            "loss_cls": gt_boxes.new_tensor(0),}
        self.anchors =self.anchors.to(gt_boxes)
        for pred, stride, wh in zip(preds, self.strides, self.anchors):
            anchor_id, gt_id = box_ops.size_matched_idx(wh, gt_boxes[:,2:], self.match_thresh)
            if len(anchor_id) > 0:
                gt_box_xy = gt_boxes[:, :2][gt_id]
                ids, grid_xy = box_ops.assign_targets_to_proposals(gt_box_xy / stride, pred.shape[1:3])
                anchor_id, gt_id = anchor_id[ids], gt_id[ids]
                image_id = image_ids[gt_id]

                pred_level = pred[image_id, grid_xy[:, 1], grid_xy[:, 0], anchor_id]
                # pos = 1 - 0.5 * self.eps
                # neg = 1 - pos
                gt_label = torch.zeros_like(pred_level[..., 5:])
                gt_label[range(len(gt_id)), gt_labels[gt_id]] = 1
                loss["loss_cls"] += F.binary_cross_entropy_with_logits(pred_level[..., 5:], gt_label)
        return loss["loss_cls"]

class ObjectLoss(Loss):
    def __init__(
            self,
            strides: List = (None),
            anchors: np.ndarray = [None],
            giou_ratio: int=1,
            match_thresh: int=2,

    ):

        super().__init__()
        self.strides = strides
        self.register_buffer("anchors", torch.Tensor(anchors))
        self.match_thresh = match_thresh
        self.giou_ratio = giou_ratio
        #self.binary_cross_entropy = BinaryCrossEntropy_with_logits_Loss()
    def forward(self, targets: Dict, preds: torch.Tensor) -> Tuple[Union[Tensor, Any], Tensor]:  # noqa: MC0001
        dtype = preds[0].dtype
        image_ids = torch.cat([torch.full_like(tgt["labels"], i)
                               for i, tgt in enumerate(targets)])
        gt_labels = torch.cat([tgt["labels"] for tgt in targets])
        gt_boxes = torch.cat([tgt["boxes"] for tgt in targets])
        gt_boxes = box_ops.xyxy2cxcywh(gt_boxes)
        losses = {"loss_obj": gt_boxes.new_tensor(0)}
        self.anchors =self.anchors.to(gt_boxes)
        for pred, stride, wh in zip(preds, self.strides, self.anchors):
            anchor_id, gt_id = box_ops.size_matched_idx(wh, gt_boxes[:,2:], self.match_thresh)
            gt_object = torch.zeros_like(pred[..., 4])
            if len(anchor_id) > 0:
                gt_box_xy = gt_boxes[:, :2][gt_id]
                ids, grid_xy = box_ops.assign_targets_to_proposals(gt_box_xy / stride, pred.shape[1:3])
                anchor_id, gt_id = anchor_id[ids], gt_id[ids]
                image_id = image_ids[gt_id]

                pred_level = pred[image_id, grid_xy[:, 1], grid_xy[:, 0], anchor_id]
                xy = 2 * torch.sigmoid(pred_level[:, :2]) - 0.5 + grid_xy
                wh = 4 * torch.sigmoid(pred_level[:, 2:4]) ** 2 * wh[anchor_id] / stride
                box_grid = torch.cat((xy, wh), dim=1)
                giou = box_ops.box_giou(box_grid, gt_boxes[gt_id] / stride).to(dtype)

                gt_object[image_id, grid_xy[:, 1], grid_xy[:, 0], anchor_id] = \
                    self.giou_ratio * giou.detach().clamp(0) + (1 - self.giou_ratio)
                # pos = 1 - 0.5 * self.eps
                # neg = 1 - pos
            losses["loss_obj"] += F.binary_cross_entropy_with_logits(pred[..., 4], gt_object)
        return losses["loss_obj"]