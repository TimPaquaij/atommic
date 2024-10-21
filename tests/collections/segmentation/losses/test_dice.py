# coding=utf-8
__author__ = "Tim Paquaij"

import pytest
import torch
from atommic.collections.segmentation.losses.dice import Dice, GeneralizedDiceLoss

class TestDiceLosses:

    def test_dice_loss(self):
        y = torch.randn((2,5,10,10))
        y = torch.softmax(y,dim=1)
        y = torch.argmax(y,dim=1,keepdim=True)
        x = torch.randn((2,5,10,10))
        x = torch.softmax(x,dim=1)
        dice_loss = Dice(include_background=True,to_onehot_y=True,batch=False,reduction='mean_channel',softmax=False)
        _,result = dice_loss(y, x)
        assert result.shape[0] ==2
        dice_loss = Dice(include_background=True,to_onehot_y=True,batch=False,reduction='none',softmax=False)
        _,result = dice_loss(y, x)
        assert result.shape[0] ==2 and result.shape[1] ==5

    def test_gendice_loss(self):
        y = torch.randn((2,5,10,10))
        y = torch.softmax(y,dim=1)
        y = torch.argmax(y,dim=1,keepdim=True)
        x = torch.randn((2,5,10,10))
        x = torch.softmax(x,dim=1)
        gendice_loss = GeneralizedDiceLoss(include_background=True,to_onehot_y=True,batch=False,reduction='mean_channel',softmax=False)
        _,result = gendice_loss(y, x)
        assert result.shape[0] ==2
        gendice_loss = GeneralizedDiceLoss(include_background=True,to_onehot_y=True,batch=False,reduction='none',softmax=False)
        _,result = gendice_loss(y, x)
        assert result.shape[0] ==2 and result.shape[1] ==5

        
    
