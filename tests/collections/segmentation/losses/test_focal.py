# coding=utf-8
__author__ = "Tim Paquaij"

import pytest
import torch
from atommic.collections.segmentation.losses.focal import FocalLoss


class TestFocalLosses:

    def test_focal_loss(self):
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        y = torch.tensor([[7, 8], [9, 10], [11, 12]])
        focal_loss = FocalLoss()
        result = focal_loss(x, y)
        print(result)

        
    
