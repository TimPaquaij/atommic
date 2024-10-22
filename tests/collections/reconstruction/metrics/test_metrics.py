# coding=utf-8
__author__ = "Tim Paquaij"

import pytest
import numpy as np
from atommic.collections.reconstruction.metrics import mse, nmse, psnr, ssim, vsi3d, haarpsi3d


class TestRecMetrics:
    np.random.seed(1)
    def test_mse_score(self):
        y = (np.random.rand(256,256,256,) * 2) - 1
        x = (np.random.rand(256,256,256) * 2) - 1
        mse_score = mse(x,y)

    def test_nmse_score(self):
        y = (np.random.rand(256,256,256) * 2) - 1
        x = (np.random.rand(256,256,256) * 2) - 1
        mse_score = nmse(x,y)
    def test_psnr_score(self):
        y = (np.random.rand(256,256,256) * 2)-1
        x = (np.random.rand(256,256,256) * 2)-1
        psnr_score = psnr(x,y)
    def test_ssim_score(self):
        y = (np.random.rand(256,256,256) * 2)-1
        x = (np.random.rand(256,256,256) * 2)-1
        ssim_score = ssim(x,y)

    def test_vsi_score(self):
        y = (np.random.rand(256,256,256) * 2)
        x = (np.random.rand(256,256,256) * 2)
        vsi_score = vsi3d(x,y)
    def test_haarpsi_score(self):
        y = (np.random.rand(256,256,256) * 2)
        x = (np.random.rand(256,256,256) * 2)
        haarpsi_score = haarpsi3d(x,y)
       
        
    
