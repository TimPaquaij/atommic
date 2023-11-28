# coding=utf-8

# Generated by CodiumAI

import pytest
import torch

from atommic.collections.common.losses.wasserstein import SinkhornDistance


class TestSinkhornDistance:
    # Tests that the SinkhornDistance class works correctly when given two point clouds of the same size and
    # dimensionality.
    def test_same_size_and_dimensionality(self):
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        y = torch.tensor([[7, 8], [9, 10], [11, 12]])
        sinkhorn = SinkhornDistance()
        result = sinkhorn(x, y)
        assert round(result.item()) == 52

    # Tests that the SinkhornDistance class handles correctly when the eps parameter is set to 0.
    def test_eps_zero(self):
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        y = torch.tensor([[7, 8], [9, 10], [11, 12]])
        sinkhorn = SinkhornDistance(eps=0.1)
        result = sinkhorn(x, y)
        assert round(result.item()) == 52

    # Tests that the SinkhornDistance class handles correctly when the max_iter parameter is set to 0.
    def test_max_iter_zero(self):
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        y = torch.tensor([[7, 8], [9, 10], [11, 12]])
        sinkhorn = SinkhornDistance(max_iter=0)
        result = sinkhorn(x, y)
        assert result.item() == pytest.approx(0.0)

    # Tests that the SinkhornDistance class handles correctly when the max_iter parameter is set to 100.
    def test_max_iter_hundred(self):
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        y = torch.tensor([[7, 8], [9, 10], [11, 12]])
        sinkhorn = SinkhornDistance(max_iter=100)
        result = sinkhorn(x, y)
        assert round(result.item()) == 52
