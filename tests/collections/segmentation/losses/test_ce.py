# coding=utf-8
__author__ = "Tim Paquaij"

import pytest
import torch
from atommic.collections.segmentation.losses.focal import FocalLoss
from atommic.collections.segmentation.losses.cross_entropy import CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss

class TestCELosses:

    def test_focal_loss(self):
        y = torch.randn((2,5,10,10))
        y = torch.softmax(y,dim=1)
        y = torch.argmax(y,dim=1,keepdim=True)
        x = torch.randn((2,5,10,10))
        focal_loss = FocalLoss(to_onehot_y=True,include_background=True)
        result = focal_loss(y, x)
        assert result.shape[0] ==2
    
    def test_CCE_loss(self):
        y = torch.randn((2,5,10,10))
        y = torch.softmax(y,dim=1)
        y = torch.argmax(y,dim=1,keepdim=True)
        x = torch.randn((2,5,10,10))
        ce_loss = CategoricalCrossEntropyLoss(to_onehot_y=True,include_background=True)
        result = ce_loss(y, x)
        assert result.shape[0] ==2

    def test_BCE_loss(self):
        y = torch.randn((2,2,10,10))
        y = torch.softmax(y,dim=1)
        y = torch.argmax(y,dim=1,keepdim=False)
        x = torch.randn((2,10,10))
        ce_loss = BinaryCrossEntropyLoss()
        result = ce_loss(y, x)
        assert result.shape[0] ==2

        
    
