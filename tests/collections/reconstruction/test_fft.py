# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

from typing import List

import numpy as np
import pytest
import torch

from atommic.collections.common.parts.fft import fft2, fftshift, ifft2, ifftshift, roll
from atommic.collections.common.parts.utils import complex_abs
from tests.collections.reconstruction.mri_data.conftest import create_input


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_fft2(shape: List):
    """Test centered 2D Fast Fourier Transform."""
    shape = shape + [2]
    x = create_input(shape)
    out_torch = fft2(x, centered=True, normalization="ortho", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = torch.view_as_complex(x).numpy()
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.fft2(input_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_non_centered_fft2(shape: List):
    """Test non-centered 2D Fast Fourier Transform."""
    shape = shape + [2]
    x = create_input(shape)
    out_torch = fft2(x, centered=False, normalization="ortho", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = torch.view_as_complex(x).numpy()
    out_numpy = np.fft.fft2(input_numpy, norm="ortho")

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_fft2_backward_normalization(shape: List):
    """Test centered 2D Fast Fourier Transform with backward normalization."""
    shape = shape + [2]
    x = create_input(shape)
    out_torch = fft2(x, centered=True, normalization="backward", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = torch.view_as_complex(x).numpy()
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.fft2(input_numpy, norm="backward")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_fft2_forward_normalization(shape: List):
    """Test centered 2D Fast Fourier Transform with forward normalization."""
    shape = shape + [2]
    x = create_input(shape)
    out_torch = fft2(x, centered=True, normalization="forward", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = torch.view_as_complex(x).numpy()
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.fft2(input_numpy, norm="forward")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_ifft2(shape: List):
    """Test centered 2D Inverse Fast Fourier Transform."""
    shape = shape + [2]
    x = create_input(shape)
    out_torch = ifft2(x, centered=True, normalization="ortho", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = torch.view_as_complex(x).numpy()
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.ifft2(input_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_non_centered_ifft2(shape: List):
    """Test non-centered 2D Inverse Fast Fourier Transform."""
    shape = shape + [2]
    x = create_input(shape)
    out_torch = ifft2(x, centered=False, normalization="ortho", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = torch.view_as_complex(x).numpy()
    out_numpy = np.fft.ifft2(input_numpy, norm="ortho")

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_ifft2_backward_normalization(shape: List):
    """Test centered 2D Inverse Fast Fourier Transform with backward normalization."""
    shape = shape + [2]
    x = create_input(shape)
    out_torch = ifft2(x, centered=True, normalization="backward", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = torch.view_as_complex(x).numpy()
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.ifft2(input_numpy, norm="backward")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_ifft2_forward_normalization(shape: List):
    """Test centered 2D Inverse Fast Fourier Transform with forward normalization."""
    shape = shape + [2]
    x = create_input(shape)
    out_torch = ifft2(x, centered=True, normalization="forward", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = torch.view_as_complex(x).numpy()
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.ifft2(input_numpy, norm="forward")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shift, dim", [(0, 0), (1, 0), (-1, 0), (100, 0), ([1, 2], [1, 2])])
@pytest.mark.parametrize("shape", [[5, 6, 2], [3, 4, 5]])
def test_roll(shift: List, dim: List, shape: List):
    """Test roll."""
    x = np.arange(np.product(shape)).reshape(shape)
    if isinstance(shift, int) and isinstance(dim, int):
        torch_shift = [shift]
        torch_dim = [dim]
    else:
        torch_shift = shift
        torch_dim = dim
    out_torch = roll(torch.from_numpy(x), torch_shift, torch_dim).numpy()
    out_numpy = np.roll(x, shift, dim)

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[5, 3], [2, 4, 6]])
def test_fftshift(shape: List):
    """Test fftshift."""
    x = np.arange(np.product(shape)).reshape(shape)
    out_torch = fftshift(torch.from_numpy(x)).numpy()
    out_numpy = np.fft.fftshift(x)

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[5, 3], [2, 4, 5], [2, 7, 5]])
def test_ifftshift(shape: List):
    """Test ifftshift."""
    x = np.arange(np.product(shape)).reshape(shape)
    out_torch = ifftshift(torch.from_numpy(x)).numpy()
    out_numpy = np.fft.ifftshift(x)

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError
