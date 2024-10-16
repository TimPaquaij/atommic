# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import Any, List, Optional, Tuple, Union

import torch

from atommic.collections.quantitative.nn.base import SignalForwardModel
from atommic.collections.quantitative.nn.qrim_base.utils import analytical_log_likelihood_gradient
from atommic.collections.reconstruction.nn.rim_base import conv_layers, rnn_cells


class qRIMBlock(torch.nn.Module):
    """qRIMBlock extends a block of Recurrent Inference Machines (RIMs) as presented in [Zhang2022]_.

    References
    ----------
    .. [Zhang2022] Zhang C, Karkalousos D, Bazin PL, Coolen BF, Vrenken H, Sonke JJ, Forstmann BU, Poot DH, Caan MW.
        A unified model for reconstruction and R2* mapping of accelerated 7T data using the quantitative recurrent
        inference machine. NeuroImage. 2022 Dec 1;264:119680.
    """

    def __init__(
        self,
        recurrent_layer=None,
        conv_filters=None,
        conv_kernels=None,
        conv_dilations=None,
        conv_bias=None,
        recurrent_filters=None,
        recurrent_kernels=None,
        recurrent_dilations=None,
        recurrent_bias=None,
        depth: int = 2,
        time_steps: int = 8,
        conv_dim: int = 2,
        linear_forward_model=None,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        coil_combination_method: str = "SENSE",
        dimensionality: int = 2,  # pylint: disable=unused-argument
    ):
        """Inits :class:`qRIMBlock`.

        Parameters
        ----------
        recurrent_layer : torch.nn.Module, optional
            Recurrent layer. Default is ``None``.
        conv_filters : int, optional
            Number of filters in the convolutional layers. Default is ``None``.
        conv_kernels : int, optional
            Kernel size of the convolutional layers. Default is ``None``.
        conv_dilations : int, optional
            Dilation of the convolutional layers. Default is ``None``.
        conv_bias : bool, optional
            Bias of the convolutional layers. Default is ``None``.
        recurrent_filters : int, optional
            Number of filters in the recurrent layers. Default is ``None``.
        recurrent_kernels : int, optional
            Kernel size of the recurrent layers. Default is ``None``.
        recurrent_dilations : int, optional
            Dilation of the recurrent layers. Default is ``None``.
        recurrent_bias : bool, optional
            Bias of the recurrent layers. Default is ``None``.
        depth : int, optional
            Number of RIM layers. Default is ``2``.
        time_steps : int, optional
            Number of time steps. Default is ``8``.
        conv_dim : int, optional
            Dimension of the convolutional layers. Default is ``2``.
        linear_forward_model : SignalForwardModel, optional
            Linear forward model. Default is ``None``.
        fft_centered : bool, optional
            Whether to center the FFT. Default is ``False``.
        fft_normalization : str, optional
            Normalization of the FFT. Default is ``"backward"``.
        spatial_dims : tuple, optional
            Spatial dimensions of the input. Default is ``None``.
        coil_dim : int, optional
            Coils dimension of the input. Default is ``1``.
        coil_combination_method : str, optional
            Method to combine the coils. Default is ``"SENSE"``.
        dimensionality : int, optional
            Dimensionality of the input. Default is ``2``.
        """
        super().__init__()

        self.linear_forward_model = (
            SignalForwardModel(sequence="MEGRE") if linear_forward_model is None else linear_forward_model
        )

        self.input_size = depth * 4
        self.time_steps = time_steps

        self.layers = torch.nn.ModuleList()
        for (
            (conv_features, conv_k_size, conv_dilation, l_conv_bias, nonlinear),
            (rnn_features, rnn_k_size, rnn_dilation, rnn_bias, rnn_type),
        ) in zip(
            zip(conv_filters, conv_kernels, conv_dilations, conv_bias, ["relu", "relu", None]),
            zip(
                recurrent_filters,
                recurrent_kernels,
                recurrent_dilations,
                recurrent_bias,
                [recurrent_layer, recurrent_layer, None],
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

        self.recurrent_filters = recurrent_filters

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim
        self.coil_combination_method = coil_combination_method

    def forward(
        self,
        prediction: torch.Tensor,
        masked_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        sampling_mask: torch.Tensor,
        TEs: List,
        hx: torch.Tensor = None,
    ) -> Tuple[Any, Union[list, torch.Tensor, None]]:
        """Forward pass of :class:`qRIMBlock`.

        Parameters
        ----------
        prediction : torch.Tensor
            Initial prediction of the quantitative maps.
        masked_kspace : torch.Tensor
            Subsampled k-space of shape [batch_size, n_coils, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2].
        sampling_mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        TEs : List
            List of echo times.
        hx : torch.Tensor, optional
            Initial hidden state. If None, it will be initialized with zeros. Default is ``None``.

        Returns
        -------
        Tuple[Any, Union[list, torch.Tensor, None]]
            Tuple containing the prediction and the hidden state. The prediction is a list of the predicted
            quantitative maps of shape [batch_size, n_echoes, n_coils, n_x, n_y] and the hidden state is a list
            of the hidden states of the recurrent layers.
        """
        batch_size = masked_kspace.shape[0]

        if hx is None:
            hx = [
                prediction.new_zeros((prediction.size(0), f, *prediction.size()[2:])).to(masked_kspace)
                for f in self.recurrent_filters
                if f != 0
            ]

        predictions = []
        for _ in range(self.time_steps):
            grad_prediction = torch.zeros_like(prediction)
            for idx in range(batch_size):
                grad_prediction[idx] = (
                    analytical_log_likelihood_gradient(
                        self.linear_forward_model,
                        prediction[idx],
                        TEs,
                        sensitivity_maps[idx],
                        masked_kspace[idx],
                        sampling_mask[idx],
                        fft_centered=self.fft_centered,
                        fft_normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                        coil_dim=self.coil_dim,
                        coil_combination_method=self.coil_combination_method,
                    ).contiguous()
                    / 100
                )
            grad_prediction = torch.cat([grad_prediction, prediction], dim=self.coil_dim - 1).to(masked_kspace)
            for h, convrnn in enumerate(self.layers):
                hx[h] = convrnn(grad_prediction, hx[h])
                grad_prediction = hx[h]
            grad_prediction = self.final_layer(grad_prediction)
            prediction = prediction + grad_prediction
            prediction_tmp = prediction[:, 0, :, :]
            prediction_tmp[prediction_tmp < 0] = 0
            prediction[:, 0, :, :] = prediction_tmp
            predictions.append(prediction)

        return predictions, hx
