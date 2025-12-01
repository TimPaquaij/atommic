# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import Any, Optional, Tuple, Union

import torch

from atommic.collections.common.parts.fft import fft1, ifft1
from atommic.collections.reconstruction_ecg.nn.rim_base import conv_layers, rim_utils, rnn_cells


class RIMBlock(torch.nn.Module):
    """RIMBlock is a block of Recurrent Inference Machines (RIMs) as presented in [Lonning19]_.

    References
    ----------
    .. [Lonning19] Lonning19 K, Putzky P, Sonke JJ, Reneman L, Caan MW, Welling M. Recurrent inference machines for
        reconstructing heterogeneous MRI data. Medical image analysis. 2019 Apr 1;53:64-78.

    """

    def __init__(
        self,
        input_channels=None,
        recurrent_layer=None,
        conv_filters=None,
        conv_kernels=None,
        conv_dilations=None,
        conv_bias=None,
        conv_act=None,
        recurrent_filters=None,
        recurrent_kernels=None,
        recurrent_dilations=None,
        recurrent_bias=None,
        depth: int = 2,
        time_steps: int = 8,
        conv_dim: int = 2,
        update_in_frequency: bool = False,
        hexad_inform: bool = False,
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
        samplebase: Optional[float] = None,
    ):
        """Inits :class:`RIMBlock`.

        Parameters
        ----------
        recurrent_layer : torch.nn.Module
            Type of the recurrent layer. It can be ``GRU``, ``MGU``, ``IndRNN``. Check ``rnn_cells`` for more details.
        conv_filters : list of int
            Number of filters in the convolutional layers.
        conv_kernels : list of int
            Kernel size of the convolutional layers.
        conv_dilations : list of int
            Dilation of the convolutional layers.
        conv_bias : list of bool
            Bias of the convolutional layers.
        recurrent_filters : list of int
            Number of filters in the recurrent layers.
        recurrent_kernels : list of int
            Kernel size of the recurrent layers.
        recurrent_dilations : list of int
            Dilation of the recurrent layers.
        recurrent_bias : list of bool
            Bias of the recurrent layers.
        depth : int
            Number of sequence of convolutional and recurrent layers. Default is ``2``.
        time_steps : int
            Number of recurrent time steps. Default is ``8``.
        conv_dim : int
            Dimension of the convolutional layers. Default is ``2``.
        no_dc : bool
            If ``True`` the DC component is not used. Default is ``True``.
        fft_centered : bool
            If ``True`` the FFT is centered. Default is ``False``.
        fft_normalization : str
            Normalization of the FFT. Default is ``"backward"``.
        spatial_dims : tuple of int
            Spatial dimensions of the input. Default is ``None``.
        coil_dim : int
            Coil dimension of the input. Default is ``1``.
        dimensionality : int
            Dimensionality of the input. Default is ``2``.
        consecutive_slices : int
            Number of consecutive slices. Default is ``1``.
        coil_combination_method : str
            Coil combination method. Default is ``"SENSE"``.
        """
        super().__init__()
        self.update_in_frequency = update_in_frequency
        self.hexad_inform = hexad_inform
        if input_channels:
            self.input_size = input_channels
            if self.update_in_frequency:
                self.input_size = self.input_size * 2
        else:
            self.input_size = depth * 2
        self.time_steps = time_steps
        self.conv_dim = conv_dim
        self.layers = torch.nn.ModuleList()
        for (
            (conv_features, conv_k_size, conv_dilation, l_conv_bias, nonlinear),
            (rnn_features, rnn_k_size, rnn_dilation, rnn_bias, rnn_type),
        ) in zip(
            zip(conv_filters, conv_kernels, conv_dilations, conv_bias, ["relu", "relu", "relu", None]),
            zip(
                recurrent_filters,
                recurrent_kernels,
                recurrent_dilations,
                recurrent_bias,
                [recurrent_layer, recurrent_layer, recurrent_layer, None],
            ),
        ):
            conv_layer = None

            if conv_features != 0:
                conv_layer = conv_layers.ConvNonlinear(
                    self.input_size,
                    conv_features,
                    conv_dim=conv_dim,
                    kernel_size=conv_k_size,
                    dilation=conv_dilation,
                    bias=l_conv_bias,
                    nonlinear=nonlinear,
                )
                self.input_size = conv_features

            if rnn_features != 0 and rnn_type is not None:
                if rnn_type.upper() == "GRU":
                    rnn_type = rnn_cells.ConvGRUCell
                elif rnn_type.upper() == "MGU":
                    rnn_type = rnn_cells.ConvMGUCell
                elif rnn_type.upper() == "INDRNN":
                    rnn_type = rnn_cells.IndRNNCell
                else:
                    raise ValueError("Please specify a proper recurrent layer type.")

                rnn_layer = rnn_type(
                    self.input_size,
                    rnn_features,
                    conv_dim=conv_dim,
                    kernel_size=rnn_k_size,
                    dilation=rnn_dilation,
                    bias=rnn_bias,
                )

                self.input_size = rnn_features

                self.layers.append(conv_layers.ConvRNNStack(conv_layer, rnn_layer))
        self.final_layer = torch.nn.Sequential(conv_layer)
        if self.update_in_frequency:
            self.lowcut = lowcut
            self.highcut = highcut
            self.samplebase = samplebase

        self.recurrent_filters = recurrent_filters

    def forward(
        self,
        prediction: torch.Tensor,
        mask: torch.Tensor,
        measured_ecg: torch.Tensor,
        hx: torch.Tensor = None,
        sigma: float = 1.0,
        keep_prediction: bool = False,
    ) -> Tuple[Any, Union[list, torch.Tensor, None]]:
        """Forward pass of :class:`RIMBlock`.

        Parameters
        ----------
        y : torch.Tensor
            Predicted k-space. Shape: ``[batch, coils, height, width, 2]``.
        masked_kspace : torch.Tensor
            Subsampled k-space. Shape: ``[batch, coils, height, width, 2]``.
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape: ``[batch, coils, height, width, 2]``.
        mask : torch.Tensor
            Subsampling mask. Shape: ``[batch, coils, height, width, 2]``.
        prediction : torch.Tensor, optional
            Initial (zero-filled) prediction. Shape: ``[batch, coils, height, width, 2]``.
        hx : torch.Tensor, optional
            Initial prediction for the hidden state. Shape: ``[batch, coils, height, width, 2]``.
        sigma : float, optional
            Noise level. Default is ``1.0``.
        keep_prediction : bool, optional
            Whether to keep the prediction. Default is ``False``.

        Returns
        -------
        Tuple[Any, Union[list, torch.Tensor, None]]
            Reconstructed image and hidden states.
        """
        if self.conv_dim == 2 and not self.update_in_frequency:
            end = slice(1, -1)
            mask = mask.unsqueeze(-1)  # [batch, leads, time, 1] 2D conv
            measured_ecg = measured_ecg.unsqueeze(-1)  # [batch, leads, time, 1] 2D conv
            if prediction.dim() == 3:
                prediction = prediction.unsqueeze(-1)  # [batch, leads, time, 1] 2D conv
        elif self.conv_dim == 1:
            end = slice(2, None)
        else:
            end = slice(1, None)
        if hx is None or (not isinstance(hx, list) and hx.dim() < 3):
            hx = [
                prediction.new_zeros((prediction.size(0), f, *prediction.size()[end]))
                for f in self.recurrent_filters
                if f != 0
            ]
        predictions = []
        for _ in range(self.time_steps):
            log_likelihood_gradient_prediction = rim_utils.log_likelihood_gradient_ecg(
                prediction,
                measured_ecg,
                mask,
                sigma,
                self.update_in_frequency,
                self.hexad_inform,
            ).contiguous()
            if self.conv_dim == 1 and self.update_in_frequency:
                B, F, L, S = log_likelihood_gradient_prediction.shape
                log_likelihood_gradient_prediction = log_likelihood_gradient_prediction.reshape(B, F * L, S)

            for h, convrnn in enumerate(self.layers):
                hx[h] = convrnn(log_likelihood_gradient_prediction, hx[h])
                log_likelihood_gradient_prediction = hx[h]

            log_likelihood_gradient_prediction = self.final_layer(log_likelihood_gradient_prediction)
            if self.conv_dim == 1 and self.update_in_frequency:
                log_likelihood_gradient_prediction = log_likelihood_gradient_prediction.reshape(B, 2, L, S)

            if self.update_in_frequency:
                if prediction.dim() == 3:
                    prediction_freq = fft1(prediction, time_dim=-1)  # Only happens first time in loop
                else:
                    prediction_freq = prediction
                    freq_map = torch.fft.fftshift(
                        torch.fft.fftfreq(log_likelihood_gradient_prediction.shape[-1], d=(1 / self.samplebase))
                    ).to(prediction.device)
                    freq_mask = (freq_map.abs() >= self.lowcut) & (freq_map.abs() <= self.highcut)
                    log_likelihood_gradient_prediction = log_likelihood_gradient_prediction * freq_mask
                prediction = prediction_freq + log_likelihood_gradient_prediction.permute(0, 2, 3, 1)
            else:
                if self.conv_dim == 1:
                    prediction = prediction + log_likelihood_gradient_prediction

                else:
                    prediction = prediction + log_likelihood_gradient_prediction.permute(0, 2, 3, 1)

            if self.conv_dim == 2 and not self.update_in_frequency:
                predictions.append(prediction.squeeze(-1))
            else:
                predictions.append(prediction)

        return predictions, hx
