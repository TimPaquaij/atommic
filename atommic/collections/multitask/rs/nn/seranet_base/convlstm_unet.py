# coding=utf-8
__author__ = "Dimitris Karkalousos"

import math
from typing import List, Tuple

import torch

from atommic.collections.multitask.rs.nn.seranet_base.convlstm import ConvLSTM
from atommic.collections.reconstruction.nn.unet_base.unet_block import Unet


class ConvLSTMNormUnet(torch.nn.Module):
    """Normalized U-Net with additional Convolutional LSTM input layer model.

    This is the same as a regular U-Net, but with normalization applied to the input before the U-Net. This keeps the
    values more numerically stable during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        padding_size: int = 15,
        normalize: bool = True,
        norm_groups: int = 2,
    ):
        """Inits :class:`ConvLSTMNormUnet`.

        Parameters
        ----------
        chans : int
            Number of output channels of the first convolution layer.
        num_pools : int
            Number of down-sampling and up-sampling layers.
        in_chans : int
            Number of channels in the input to the U-Net model. Default is ``2``.
        out_chans : int
            Number of channels in the output to the U-Net model. Default is ``2``.
        drop_prob : float
            Dropout probability. Default is ``0.0``.
        padding_size: int
            Size of the padding. Default is ``15``.
        normalize: bool
            Whether to normalize the input. Default is ``True``.
        norm_groups: int
            Number of groups to use for group normalization. Default is ``2``.
        """
        super().__init__()
        self.convlstm = ConvLSTM(in_chans, chans, kernel_size=3, num_layers=1)
        self.unet = Unet(
            in_chans=chans, out_chans=out_chans, chans=chans, num_pool_layers=num_pools, drop_prob=drop_prob
        )
        self.padding_size = padding_size
        self.normalize = normalize
        self.norm_groups = norm_groups

    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        """Convert the last dimension of the input to complex."""
        b, c, h, w, two = x.shape
        if two != 2:
            raise AssertionError
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        """Convert the last dimension of the input to complex."""
        b, c2, h, w = x.shape
        if c2 % 2 != 0:
            raise AssertionError
        c = torch.div(c2, 2, rounding_mode="trunc")
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the input."""
        # group norm
        b, c, h, w = x.shape

        x = x.reshape(b, self.norm_groups, -1)

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        x = (x - mean) / std

        x = x.reshape(b, c, h, w)

        return x, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Unnormalize the input."""
        b, c, h, w = x.shape
        input_data = x.reshape(b, self.norm_groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        """Pad the input with zeros to make it square."""
        _, _, h, w = x.shape
        w_mult = ((w - 1) | self.padding_size) + 1
        h_mult = ((h - 1) | self.padding_size) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = torch.nn.functional.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        """Unpad the input."""
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`ConvLSTMNormUnet`."""
        iscomplex = False
        if x.shape[-1] == 2:
            x = self.complex_to_chan_dim(x)
            iscomplex = True

        mean = 1.0
        std = 1.0

        if self.normalize:
            x, mean, std = self.norm(x)

        x, pad_sizes = self.pad(x)

        x, _ = self.convlstm(x.unsqueeze(0))
        x = x[0]
        if x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.shape[1] == 1:
            x = x.squeeze(1)
        else:
            raise AssertionError
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)

        if self.normalize:
            x = self.unnorm(x, mean, std)

        if iscomplex:
            x = self.chan_complex_to_last_dim(x)

        return x
