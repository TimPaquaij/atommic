import math

from torch import nn
from _collections import OrderedDict
import torch
from atommic.collections.objectdetection.nn.yolo_base import box_ops
from atommic.collections.objectdetection.nn.yolo_base.head import Head
from atommic.collections.objectdetection.nn.yolo_base.backbone import darknet_pan_backbone
from atommic.collections.objectdetection.parts.transform import Transformer
import numpy as np

class YOLOv5(nn.Module):
    def __init__(self, num_classes,anchors, strides,model_size=(0.33, 0.5), img_sizes=[None],
                 score_thresh=0.3, nms_thresh=0.3, detections=4,backbone_ckpt =None):
        super().__init__()
        
        self.backbone = darknet_pan_backbone(
            depth_multiple=model_size[0], width_multiple=model_size[1]) # 7.5M parameters
        if backbone_ckpt:
            ckpt = torch.load(backbone_ckpt)
            ckpt_1 = OrderedDict({".".join(name.split('.')[1:]):value for name, value in ckpt.items() if name.startswith('backbone')})
            self.backbone.load_state_dict(ckpt_1)
            for param in self.backbone.parameters():
                param.requires_grad = False
        in_channels_list = self.backbone.body.out_channels_list
        num_anchors = [len(s) for s in anchors]
        predictor = Predictor(in_channels_list, num_anchors, num_classes, strides)
        
        self.head = Head(
            predictor, anchors, strides,
            score_thresh, nms_thresh, detections)
        
        if isinstance(img_sizes, int):
            img_sizes = (img_sizes, img_sizes)
        self.transformer = Transformer(
            min_size=img_sizes[0], max_size=img_sizes[1], stride=max(strides))
    
    def forward(self, images,target):
        images, target, scale_factors, image_shapes = self.transformer(images, target)
        features = self.backbone(images)
        
        # if self.training:
        #     losses = self.head(features, targets)
        #     return losses
        max_size = max(images.shape[2:])
        preds, results = self.head(features, image_shapes, scale_factors, max_size)
        return preds,results,target
        
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
    
    