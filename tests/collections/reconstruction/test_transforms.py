# coding=utf-8
__author__ = "Dimitris Karkalousos"

import numpy as np
import pytest
import torch

from atommic.collections.common.parts.utils import center_crop, center_crop_to_smallest, complex_center_crop, to_tensor


@pytest.mark.parametrize("x", [(np.zeros([1, 320, 320]).astype(np.complex64))])
def test_to_tensor(x):
    """
    Test if the to_tensor function works as expected.

    Args:
        x: The input array.

    Returns:
        None
    """
    x = to_tensor(x)
    if x.dim() != 4:
        raise AssertionError
    if x.shape[-1] != 2:
        raise AssertionError


@pytest.mark.parametrize("x, crop_size", [(torch.zeros([320, 320]).type(torch.complex64), (160, 160))])
def test_center_crop(x, crop_size):
    """
    Test if the center_crop function works as expected.

    Args:
        x: The input array.
        crop_size: The size of the crop.

    Returns:
        None
    """
    x = center_crop(x, crop_size)
    if x.shape != crop_size:
        raise AssertionError


@pytest.mark.parametrize("x, crop_size", [(torch.zeros([320, 320, 2]).type(torch.complex64), (160, 160))])
def test_complex_center_crop(x, crop_size):
    """
    Test if the center_crop function works as expected.

    Args:
        x: The input array.
        crop_size: The size of the crop.

    Returns:
        None
    """
    x = complex_center_crop(x, crop_size)
    if x.shape[:-1] != crop_size:
        raise AssertionError


@pytest.mark.parametrize(
    "x, y", [(torch.zeros([1, 320, 320]).type(torch.complex64), torch.zeros([1, 160, 160]).type(torch.complex64))]
)
def test_center_crop_to_smallest(x, y):
    """
    Test if the center_crop_to_smallest function works as expected.

    Args:
        x: The input array.
        y: The input array.

    Returns:
        None
    """
    x, y = center_crop_to_smallest(x, y)
    if x.shape != y.shape:
        raise AssertionError
