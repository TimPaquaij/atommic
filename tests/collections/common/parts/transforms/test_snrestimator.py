# coding=utf-8

# Generated by CodiumAI

import pytest
import torch

from atommic.collections.common.parts.transforms import SNREstimator


class TestSNREstimator:
    # Tests that the SNREstimator class works correctly with default parameters
    def test_default_parameters(self):
        estimator = SNREstimator(patch_size=[0, 0, 0, 0], multicoil=False)
        data = torch.randn(1, 256, 256, 2)
        result = estimator(data)
        assert result == 0

    # Tests that the SNREstimator class works correctly with multicoil data
    def test_multicoil_data(self):
        estimator = SNREstimator(patch_size=[0, 0, 0, 0], multicoil=True)
        data = torch.randn(1, 32, 256, 256, 2)
        result = estimator(data)
        assert result == 0

    # Tests that the SNREstimator class works correctly with apply_ifft=False
    def test_apply_ifft_false(self):
        estimator = SNREstimator(patch_size=[0, 0, 0, 0], apply_ifft=False)
        data = torch.randn(1, 32, 256, 256, 2)
        result = estimator(data)
        assert result == 0

    # Tests that the SNREstimator class works correctly with fft_centered=True
    def test_fft_centered_false(self):
        estimator = SNREstimator(patch_size=[0, 0, 0, 0], fft_centered=False)
        data = torch.randn(1, 32, 256, 256, 2)
        result = estimator(data)
        assert result == 0

    # Tests that the SNREstimator class works correctly with fft_normalization='backward'
    def test_fft_normalization_backward(self):
        estimator = SNREstimator(patch_size=[0, 0, 0, 0], fft_normalization="backward")
        data = torch.randn(1, 32, 256, 256, 2)
        result = estimator(data)
        assert result == 0
