# coding=utf-8
__author__ = "Tim Paquaij"

import pytest
import torch
from atommic.collections.segmentation.losses.dice import Dice
from atommic.collections.segmentation.losses.gendice import GeneralizedDiceLoss

class TestDiceLosses:

    def test_dice_loss(self):
        y = torch.randn((2,5,10,10))
        y = torch.softmax(y,dim=1)
        y = torch.argmax(y,dim=1,keepdim=True)
        x = torch.randn((2,5,10,10))
        dice_loss = Dice(include_background=True,to_onehot_y=True)
        _,result = dice_loss(y, x)
        assert result.shape[0] ==2

    def test_gendice_loss(self):
        y = torch.randn((2,5,10,10))
        y = torch.softmax(y,dim=1)
        y = torch.argmax(y,dim=1,keepdim=True)
        x = torch.randn((2,5,10,10))
        gendice_loss = GeneralizedDiceLoss(include_background=True,to_onehot_y=True)
        _,result = gendice_loss(y, x)
        assert result.shape[0] ==2

        
    
