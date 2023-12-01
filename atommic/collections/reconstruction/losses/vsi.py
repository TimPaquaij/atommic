# coding=utf-8
__author__ = "Tim Paquaij"

import torch
# Parts of the code have been taken from https://github.com/photosynthesis-team/piq

import torch.nn as nn
from piq.vsi import vsi
import functools
class VSILoss(nn.Module):
    r"""Creates a criterion that measures Visual Saliency-induced Index error between
    each element in the input and target.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        c1: coefficient to calculate saliency component of VSI
        c2: coefficient to calculate gradient component of VSI
        c3: coefficient to calculate color component of VSI
        alpha: power for gradient component of VSI
        beta: power for color component of VSI
        omega_0: coefficient to get log Gabor filter at SDSP
        sigma_f: coefficient to get log Gabor filter at SDSP
        sigma_d: coefficient to get SDSP
        sigma_c: coefficient to get SDSP

    Examples:

        >>> loss = VSILoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        L. Zhang, Y. Shen and H. Li, "VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment,"
        IEEE Transactions on Image Processing, vol. 23, no. 10, pp. 4270-4281, Oct. 2014, doi: 10.1109/TIP.2014.2346028
        https://ieeexplore.ieee.org/document/6873260
    """

    def __init__(self, reduction: str = 'mean', c1: float = 1.27, c2: float = 386., c3: float = 130.,
                 alpha: float = 0.4, beta: float = 0.02, data_range:[int, float] = 1.,
                 omega_0: float = 0.021, sigma_f: float = 1.34, sigma_d: float = 145., sigma_c: float = 0.001) -> None:
        super().__init__()
        self.reduction = reduction
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.alpha = alpha
        self.beta = beta
        self.omega_0 = omega_0
        self.sigma_f = sigma_f
        self.sigma_d = sigma_d
        self.sigma_c = sigma_c
        self.data_range = data_range



    def forward(self, x: torch.Tensor, y: torch.Tensor,data_range: torch.Tensor):
        r"""Computation of VSI as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of VSI loss to be minimized in [0, 1] range.

        Note:
            Both inputs are supposed to have RGB channels order in accordance with the original approach.
            Nevertheless, the method supports greyscale images, which they are converted to RGB by copying the grey
            channel 3 times.
        """
        data_range = data_range[:, None, None, None]
        self.vsi = functools.partial(
            vsi, c1=self.c1, c2=self.c2, c3=self.c3, alpha=self.alpha, beta=self.beta, omega_0=self.omega_0,
            sigma_f=self.sigma_f, sigma_d=self.sigma_d, sigma_c=self.sigma_c, data_range=data_range, reduction=self.reduction)
        return 1. -self.vsi(x=x, y=y)