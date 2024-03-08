import numpy as np
import torch
from torch import Tensor
from atommic.core.classes.loss import Loss
from typing import Any, Callable, List, Optional, Tuple, Union,Dict
from atommic.collections.objectdetection.nn.yolo_base import box_ops
from atommic.collections.segmentation.losses.cross_entropy import BinaryCrossEntropy_with_logits_Loss
from torch import nn
from atommic.collections.objectdetection.parts.transform import Transformer

class DetectionLoss(Loss):
    def __init__(
            self,
            strides: List = (8,16,32),
            anchors: np.ndarray = [[[6.1, 8.1], [20.6, 12.6], [11.2, 23.7]],[[36.2, 26.8], [25.9, 57.2], [57.8, 47.9]],[[122.1, 78.3], [73.7, 143.8], [236.1, 213.1]]],
            loss_weights: Dict = {"loss_box": 0.05, "loss_obj": 1.0, "loss_cls": 0.5},
            giou_ratio: int=1,
            match_thresh: int=4,
            img_sizes: List = (256,208),

    ):

        super().__init__()
        self.strides = strides
        self.register_buffer("anchors", torch.Tensor(anchors))
        self.match_thresh = match_thresh
        self.giou_ratio = giou_ratio
        self.loss_weights = loss_weights
        self.binary_cross_entropy = BinaryCrossEntropy_with_logits_Loss()
    def forward(self, targets: Dict, preds: torch.Tensor) -> Tuple[Union[Tensor, Any], Tensor]:  # noqa: MC0001

        dtype = preds[0].dtype
        gt_labels = torch.cat([tgt['labels'].squeeze(0) for tgt in targets])
        gt_boxes = torch.cat([tgt["boxes"].squeeze(0) for tgt in targets])
        image_ids = torch.cat([torch.full_like(tgt['labels'].squeeze(0),i)  for i,tgt in enumerate(targets)])

        # if not image_ids.numel() ==0:
        #     gt_labels =gt_labels.squeeze(0)
        #     image_ids = image_ids.squeeze(0)
        gt_boxes = box_ops.xyxy2cxcywh(gt_boxes)
        losses = {
            "loss_box": gt_boxes.new_tensor(0),
            "loss_obj": gt_boxes.new_tensor(0),
            "loss_cls": gt_boxes.new_tensor(0),}
        self.anchors = self.anchors.to(gt_boxes)
        self.match_thresh = self.match_thresh
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
                giou = box_ops.box_giou(box_grid, gt_boxes[gt_id]/ stride).to(dtype)
                losses["loss_box"] += (1 - giou).mean()


                gt_object[image_id, grid_xy[:, 1], grid_xy[:, 0], anchor_id] = \
                    self.giou_ratio * giou.detach().clamp(0) + (1 - self.giou_ratio)

                # pos = 1 - 0.5 * self.eps
                # neg = 1 - pos
                gt_label = torch.zeros_like(pred_level[..., 5:])
                gt_label[range(len(gt_id)), gt_labels[gt_id]] = 1
                losses["loss_cls"] += self.binary_cross_entropy(pred_level[..., 5:], gt_label).to(dtype)
            losses["loss_obj"] += self.binary_cross_entropy(pred[..., 4], gt_object).to(dtype)

        losses = {k: v * self.loss_weights[k] for k, v in losses.items()}
        loss = sum([x for x in losses.values()])/len(losses.values())
        return loss