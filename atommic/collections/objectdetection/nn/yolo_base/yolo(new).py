import math

from torch import nn
import torch
from atommic.collections.objectdetection.nn.yolo_base import box_ops
from atommic.collections.objectdetection.nn.yolo_base.head import Head
from atommic.collections.objectdetection.nn.yolo_base.backbone import darknet_pan_backbone
from atommic.collections.objectdetection.parts.transform import Transformer


class YOLOv5(nn.Module):
    def __init__(self, num_classes, model_size=(0.33,0.5),
                 match_thresh=4, giou_ratio=1, img_sizes=(360,416),
                 score_thresh=0.1, nms_thresh=0.6, detections=100):
        super().__init__()
        # original
        anchors1 = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]
        # [320, 416]
        anchors = [
            [[6.1, 8.1], [20.6, 12.6], [11.2, 23.7]],
            [[36.2, 26.8], [25.9, 57.2], [57.8, 47.9]],
            [[122.1, 78.3], [73.7, 143.8], [236.1, 213.1]],
        ]
        loss_weights = {"loss_box": 0.05, "loss_obj": 1.0, "loss_cls": 0.5}
        
        self.backbone = darknet_pan_backbone(
            depth_multiple=model_size[0], width_multiple=model_size[1]) # 7.5M parameters
        self.register_buffer("anchors", torch.Tensor(anchors))
        in_channels_list = self.backbone.body.out_channels_list
        self.strides = ( 8,16,32)
        num_anchors = [len(s) for s in anchors]
        predictor = Predictor(in_channels_list, num_anchors, num_classes, self.strides)
        self.score_thresh =score_thresh
        self.head = Head(
            predictor, self.anchors, self.strides,
            match_thresh, giou_ratio, loss_weights, 
            self.score_thresh, nms_thresh, detections)
        if isinstance(img_sizes, int):
            img_sizes = (img_sizes, img_sizes)
        self.transformer = Transformer(
            min_size=img_sizes[0], max_size=img_sizes[1], stride=max(self.strides))
    
    def forward(self, images):
        images,targets, scale_factors, image_shapes = self.transformer(images,None)
        features = self.backbone(images)
        loss_results = self.head(features)
        inference_results = self.inference_object_detection(loss_results,image_shapes,scale_factors)

        return loss_results, inference_results

    def inference_object_detection(self, preds, image_shapes=None, scale_factors=None, max_size=None):
        ids, ps, boxes = [], [], []
        for pred, stride, wh in zip(preds, self.strides, self.anchors):  # 3.54s
            pred = torch.sigmoid(pred)
            n, y, x, a = torch.where(pred[..., 4] > self.score_thresh)
            p = pred[n, y, x, a]

            xy = torch.stack((x, y), dim=1)
            xy = (2 * p[:, :2] - 0.5 + xy) * stride
            wh = 4 * p[:, 2:4] ** 2 * wh[a]
            box = torch.cat((xy, wh), dim=1)

            ids.append(n)
            ps.append(p)
            boxes.append(box)

        ids = torch.cat(ids)
        ps = torch.cat(ps)
        boxes = torch.cat(boxes)

        boxes = box_ops.cxcywh2xyxy(boxes)
        logits = ps[:, [4]] * ps[:, 5:]
        indices, labels = torch.where(logits > self.score_thresh)  # 4.94s
        ids, boxes, scores = ids[indices], boxes[indices], logits[indices, labels]

        results = []
        for i, im_s in enumerate(image_shapes):  # 20.97s
            keep = torch.where(ids == i)[0]  # 3.11s
            box, label, score = boxes[keep], labels[keep], scores[keep]
            # ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1] # 0.27s
            # keep = torch.where((ws >= self.min_size) & (hs >= self.min_size))[0] # 3.33s
            # boxes, objectness, logits = boxes[keep], objectness[keep], logits[keep] # 0.36s

            if len(box) > 0:
                box[:, 0].clamp_(0, im_s[1])  # 0.39s
                box[:, 1].clamp_(0, im_s[0])  # ~
                box[:, 2].clamp_(0, im_s[1])  # ~
                box[:, 3].clamp_(0, im_s[0])  # ~

                keep = box_ops.batched_nms(box, score, label, self.nms_thresh, max_size)  # 4.43s
                keep = keep[:self.detections]

                nms_box, nms_label = box[keep], label[keep]
                if self.merge:  # slightly increase AP, decrease speed ~14%
                    mask = nms_label[:, None] == label[None]
                    iou = (box_ops.box_iou(nms_box, box) * mask) > self.nms_thresh  # 1.84s
                    weights = iou * score[None]  # 0.14s
                    nms_box = torch.mm(weights, box) / weights.sum(1, keepdim=True)  # 0.55s

                box, label, score = nms_box / scale_factors[i], nms_label, score[keep]  # 0.30s
            results.append(dict(boxes=box, labels=label, scores=score))  # boxes format: (xmin, ymin, xmax, ymax)

        return results
        # if self.training:
        #     losses = self.head(features, targets)
        #     return losses
        # else:
        #     max_size = max(images.shape[2:])
        #     results, losses = self.head(features, targets, image_shapes, scale_factors, max_size)
        #     return results, losses
        
    def fuse(self):
        # fusing conv and bn layers
        for m in self.modules():
            if hasattr(m, "fused"):
                m.fuse()


class Predictor(nn.Module):
    def __init__(self, in_channels_list, num_anchors, num_classes, strides):
        super().__init__()
        self.num_outputs = num_classes + 5
        self.mlp = nn.ModuleList()
        
        for in_channels, n in zip(in_channels_list, num_anchors):
            out_channels = n * self.num_outputs
            self.mlp.append(nn.Conv2d(in_channels, out_channels, 1))
            
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        for m, n, s in zip(self.mlp, num_anchors, strides):
            b = m.bias.detach().view(n, -1)
            b[:, 4] += math.log(8 / (416 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (num_classes - 0.99))
            m.bias = nn.Parameter(b.view(-1))
            
    def forward(self, x):
        N = x[0].shape[0]
        L = self.num_outputs
        preds = []
        for i in range(len(x)):
            h, w = x[i].shape[-2:]
            pred = self.mlp[i](x[i])
            pred = pred.permute(0, 2, 3, 1).reshape(N, h, w, -1, L)
            preds.append(pred)
        return preds
    
    