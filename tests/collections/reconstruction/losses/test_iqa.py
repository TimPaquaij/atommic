# coding=utf-8
__author__ = "Tim Paquaij"

import pytest
import torch
from atommic.collections.reconstruction.losses import VSILoss, SSIMLoss, HaarPSILoss


class TestIQLosses:

    def test_ssim_loss(self):
        y = (torch.rand(1,256,256, requires_grad=False) * 2) - 1
        x = (torch.rand(1,256,256, requires_grad=True) * 2) - 1
        SSIM_loss = SSIMLoss()
        result = SSIM_loss(x, y)
        result.backward()

    def test_vsi_loss(self):
        y = (torch.rand(1,256,256, requires_grad=False) * 2)
        x = (torch.rand(1,256,256, requires_grad=True) * 2)
        #Only supports image with pixel values above 0 

        VSI_loss = VSILoss()
        result = VSI_loss(x, y,max(y.max(), x.max()))
        result.backward()
    def test_haarpsi_loss(self):
        y = (torch.rand(1,256,256, requires_grad=False) * 2)
        x = (torch.rand(1,256,256, requires_grad=True) * 2)
        #Only supports image with pixel values above 0
        HaarPSI_loss = HaarPSILoss()
        result = HaarPSI_loss(x, y,max(y.max(), x.max()))
        result.backward()
        
    
