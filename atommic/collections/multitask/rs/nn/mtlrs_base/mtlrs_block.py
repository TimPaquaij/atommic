# coding=utf-8
__author__ = "Dimitris Karkalousos"

import math
from typing import Dict, List, Optional, Tuple, Union

import torch

from atommic.collections.common.parts.utils import rnn_weights_init
from atommic.collections.reconstruction.nn.rim_base import rim_block
from atommic.collections.reconstruction.nn.rim_base.conv_layers import ConvNonlinear
from atommic.collections.reconstruction.nn.unet_base.unet_block import Unet
from atommic.collections.segmentation.nn.attentionunet_base.attentionunet_block import AttentionUnet
from atommic.collections.segmentation.nn.lambdaunet_base.lambdaunet_block import LambdaUNet
from atommic.collections.segmentation.nn.vnet_base.vnet_block import VNet
from atommic.collections.segmentation.nn.unet3d_base.unet3d_block import UNet3D
__all__ = ["MTLRSBlock"]


class MTLRSBlock(torch.nn.Module):
    """Implementation of a Multi-Task Learning for MRI Reconstruction and Segmentation (MTLRS) block, as presented in
    [Karkalousos2023]_.

    References
    ----------
    .. [Karkalousos2023] Karkalousos, D., Išgum, I., Marquering, H., Caan, M. W. A., (2023). MultiTask Learning for
        accelerated-MRI Reconstruction and Segmentation of Brain Lesions in Multiple Sclerosis. In Proceedings of
        Machine Learning Research (Vol. 078).
    """

    def __init__(
        self,
        reconstruction_module_params: Dict,
        segmentation_module_params: Dict,
        input_channels: int,
        magnitude_input: bool = True,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        dimensionality: int = 2,
        consecutive_slices: int = 1,
        coil_combination_method: str = "SENSE",
        normalize_segmentation_output: bool = True,
    ):
        """Inits :class:`MTLRSBlock`.

        Parameters
        ----------
        reconstruction_module_params : Dict
            Parameters for the reconstruction module.
        segmentation_module_params : Dict
            Parameters for the segmentation module.
        input_channels : int
            Number of input channels.
        magnitude_input : bool
            Whether the input is magnitude or complex. Default is ``True``.
        fft_centered : bool
            Whether the FFT is centered. Default is ``False``.
        fft_normalization : str
            Normalization of the FFT. Default is ``"backward"``.
        spatial_dims : Tuple[int, int]
            Spatial dimensions of the input. Default is ``None``.
        coil_dim : int
            Coil dimension of the input. Default is ``1``.
        dimensionality : int
            Dimensionality of the input. Default is ``2``.
        consecutive_slices : int
            Number of consecutive slices to be used. Default is ``1``.
        coil_combination_method : str
            Coil combination method. Default is ``"SENSE"``.
        normalize_segmentation_output : bool
            Whether to normalize the segmentation output. Default is ``True`` .
        """
        super().__init__()

        # General parameters
        self.input_channels = input_channels
        self.magnitude_input = magnitude_input
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims
        self.coil_dim = coil_dim
        self.dimensionality = dimensionality
        if self.dimensionality != 2:
            raise NotImplementedError(f"Currently only 2D is supported for segmentation, got {self.dimensionality}D.")
        self.consecutive_slices = consecutive_slices
        self.coil_combination_method = coil_combination_method

        # Reconstruction module parameters
        self.reconstruction_module_params = reconstruction_module_params
        self.reconstruction_module_recurrent_filters = self.reconstruction_module_params["recurrent_filters"]
        self.reconstruction_module_time_steps = 8 * math.ceil(self.reconstruction_module_params["time_steps"] / 8)
        self.no_dc = self.reconstruction_module_params["no_dc"]
        self.keep_prediction = self.reconstruction_module_params["keep_prediction"]
        self.reconstruction_module_dimensionality = self.reconstruction_module_params["dimensionality"]
        reconstruction_module_consecutive_slices = (
            self.consecutive_slices if self.reconstruction_module_dimensionality == 3 else 1
        )
        self.reconstruction_module = torch.nn.ModuleList(
            [
                rim_block.RIMBlock(
                    recurrent_layer=self.reconstruction_module_params["recurrent_layer"],
                    conv_filters=self.reconstruction_module_params["conv_filters"],
                    conv_kernels=self.reconstruction_module_params["conv_kernels"],
                    conv_dilations=self.reconstruction_module_params["conv_dilations"],
                    conv_bias=self.reconstruction_module_params["conv_bias"],
                    conv_dropout = self.reconstruction_module_params["conv_dropout"],
                    conv_activations = self.reconstruction_module_params["conv_activations"],
                    recurrent_filters=self.reconstruction_module_recurrent_filters,
                    recurrent_kernels=self.reconstruction_module_params["recurrent_kernels"],
                    recurrent_dilations=self.reconstruction_module_params["recurrent_dilations"],
                    recurrent_bias=self.reconstruction_module_params["recurrent_bias"],
                    recurrent_dropout=self.reconstruction_module_params["recurrent_dropout"],
                    depth=self.reconstruction_module_params["depth"],
                    time_steps=self.reconstruction_module_time_steps,
                    conv_dim=self.reconstruction_module_params["conv_dim"],
                    no_dc=self.no_dc,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim-1,
                    dimensionality=self.reconstruction_module_dimensionality,
                    consecutive_slices=reconstruction_module_consecutive_slices,
                    coil_combination_method=self.coil_combination_method,
                )
                for _ in range(self.reconstruction_module_params["num_cascades"])
            ]
        )
        # Keep estimation through the cascades if keep_prediction is True or re-estimate it if False.
        self.reconstruction_module_keep_prediction = self.reconstruction_module_params["keep_prediction"]
        # initialize weights if not using pretrained cirim
        if not self.reconstruction_module_params["pretrained"]:
            std_init_range = 1 / self.reconstruction_module_recurrent_filters[0] ** 0.5
            self.reconstruction_module.apply(lambda module: rnn_weights_init(module, std_init_range))
        self.dc_weight = torch.nn.Parameter(torch.ones(1))
        self.accumulate_predictions = self.reconstruction_module_params["accumulate_predictions"]

        # Segmentation module parameters
        self.segmentation_module_params = segmentation_module_params
        segmentation_module = self.segmentation_module_params["segmentation_module"]
        self.segmentation_module_output_channels = self.segmentation_module_params["output_channels"]
        if segmentation_module.lower() == "unet":
            segmentation_module = Unet(
                in_chans=self.input_channels,
                out_chans=self.segmentation_module_output_channels,
                chans=self.segmentation_module_params["channels"],
                num_pool_layers=self.segmentation_module_params["pooling_layers"],
                drop_prob=self.segmentation_module_params["dropout"],
            )
        elif segmentation_module.lower() == "attentionunet":
            segmentation_module = AttentionUnet(
                in_chans=self.input_channels,
                out_chans=self.segmentation_module_output_channels,
                chans=self.segmentation_module_params["channels"],
                num_pool_layers=self.segmentation_module_params["pooling_layers"],
                drop_prob=self.segmentation_module_params["dropout"],
            )
        elif segmentation_module.lower() == "lambdaunet":
            segmentation_module = LambdaUNet(
                in_chans=self.input_channels,
                out_chans=self.segmentation_module_output_channels,
                chans=self.segmentation_module_params["channels"],
                num_pool_layers=self.segmentation_module_params["pooling_layers"],
                drop_prob=self.segmentation_module_params["dropout"],
                num_slices=self.consecutive_slices
            )
        # elif segmentation_module.lower() == "vnet":
        #     segmentation_module = VNet(
        #         in_chans=self.input_channels,
        #         out_chans=self.segmentation_module_output_channels,
        #         act=self.segmentation_module_params["activation"],
        #         drop_prob=self.segmentation_module_params["dropout"],
        #         bias=self.segmentation_module_params["bias"],
        #    )
        # elif segmentation_module.lower() == "convlayer":
        #     segmentation_module = torch.nn.Sequential(
        #         ConvNonlinear(
        #             self.input_channels,
        #             self.segmentation_module_output_channels,
        #             conv_dim=self.segmentation_module_params["conv_dim"],
        #             kernel_size=3,
        #             dilation=1,
        #             bias=False,
        #             nonlinear=None,  # No nonlinear activation
        #         )
        #     )
        elif segmentation_module.lower() == "unet3d":
            segmentation_module = torch.nn.Sequential(UNet3D(
                in_chans=self.input_channels,
                out_chans=self.segmentation_module_output_channels,
                chans=self.segmentation_module_params["channels"],
                num_pool_layers=self.segmentation_module_params["pooling_layers"],
                drop_prob=self.segmentation_module_params["dropout"],
            ))
        else:
            raise ValueError(f"Segmentation module {segmentation_module} not implemented.")
        self.segmentation_module = segmentation_module
        self.segmentation_3d = self.segmentation_module_params["segmentation_3d"]
        self.normalize_segmentation_output = normalize_segmentation_output

    def forward(  # noqa: MC0001
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_reconstruction_pred: torch.Tensor,
        target_reconstruction: torch.Tensor,  # pylint: disable=unused-argument
        hx: torch.Tensor = None,
        sigma: float = 1.0,
    ) -> Tuple[Union[List, torch.Tensor], torch.Tensor]:
        """Forward pass of :class:`MTLRSBlock`.

        Parameters
        ----------
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]
        init_reconstruction_pred : torch.Tensor
            Initial reconstruction prediction. Shape [batch_size, n_x, n_y, 2]
        target_reconstruction : torch.Tensor
            Target reconstruction. Shape [batch_size, n_x, n_y, 2]
        hx : torch.Tensor, optional
            Initial hidden state for the RNN. Default is ``None``.
        sigma : float, optional
            Standard deviation of the noise. Default is ``1.0``.

        Returns
        -------
        Tuple[Union[List, torch.Tensor], torch.Tensor]
            Tuple containing the predicted reconstruction and segmentation.
        """
        if self.consecutive_slices > 1 and self.reconstruction_module_dimensionality == 2:
            # Do per slice reconstruction
            pred_reconstruction_slices = []
            pred_loglike_slices = []
            temp_hx = []
            for slice_idx in range(self.consecutive_slices):
                y_slice = y[:, slice_idx, ...]
                prediction_slice = y_slice.clone()
                sensitivity_maps_slice = sensitivity_maps[:, slice_idx, ...]
                if mask.dim() == 1:
                    mask_slice = mask
                else:
                    mask_slice = mask[:, 0, ...]

                init_reconstruction_pred_slice = init_reconstruction_pred[:, slice_idx, ...]
                if not hx[0].shape:
                    hx_slice =hx
                else:
                    hx_slice = [x[:,slice_idx,...] for x in hx]

                _pred_reconstruction_slice = (
                    None
                    if init_reconstruction_pred_slice is None or init_reconstruction_pred_slice.dim() < 4
                    else init_reconstruction_pred_slice
                )
                cascades_predictions = []
                cascades_log_like= []

                for i, cascade in enumerate(self.reconstruction_module):
                    # Forward pass through the cascades
                    prediction_slice, hx_slice, log_like_slice= cascade(
                        prediction_slice,
                        y_slice,
                        sensitivity_maps_slice,
                        mask_slice,
                        _pred_reconstruction_slice if i == 0 else prediction_slice,
                        hx_slice,
                        sigma,
                        keep_prediction=False if i == 0 else self.keep_prediction,
                    )
                    time_steps_predictions = [torch.view_as_complex(pred) for pred in prediction_slice]
                    log_like_predictions = [torch.view_as_complex(l) for l in log_like_slice]
                    cascades_predictions.append(torch.stack(time_steps_predictions, dim=0))
                    cascades_log_like.append(torch.stack(log_like_predictions,dim = 0))
                    prediction_slice = prediction_slice[-1]
                pred_reconstruction_slices.append(torch.stack(cascades_predictions, dim=0))
                pred_loglike_slices.append(torch.stack(cascades_log_like, dim=0))
                temp_hx.append(torch.stack(hx_slice,dim=0))
            preds = torch.stack(pred_reconstruction_slices, dim=3)
            log_like = torch.stack(pred_loglike_slices, dim=3)
            hx = torch.stack(temp_hx,dim=2)


            cascades_predictions = [
                [
                    preds[cascade_prediction, time_step_prediction, ...]
                    for time_step_prediction in range(preds.shape[1])
                ]
                for cascade_prediction in range(preds.shape[0])
            ]
        else:
            prediction = y.clone()
            _pred_reconstruction = (
                None
                if init_reconstruction_pred is None or init_reconstruction_pred.dim() < 4
                else init_reconstruction_pred
            )
            sigma = 1.0
            cascades_predictions = []
            cascades_log_like = []
            for i, cascade in enumerate(self.reconstruction_module):
                # Forward pass through the cascades
                prediction, hx, log_like = cascade(
                    prediction,
                    y,
                    sensitivity_maps,
                    mask,
                    _pred_reconstruction,
                    hx,
                    sigma,
                    keep_prediction=False if i == 0 else self.keep_prediction,
                )
                time_steps_predictions = [torch.view_as_complex(pred) for pred in prediction]
                log_like_predictions = [torch.view_as_complex(l) for l in log_like]
                cascades_predictions.append(time_steps_predictions)
                cascades_log_like.append(log_like_predictions)
                log_like = cascades_log_like
        pred_reconstruction = cascades_predictions

        _pred_reconstruction = pred_reconstruction
        if isinstance(_pred_reconstruction, list):
            _pred_reconstruction = _pred_reconstruction[-1]
        if isinstance(_pred_reconstruction, list):
            _pred_reconstruction = _pred_reconstruction[-1]
        if _pred_reconstruction.shape[-1] != 2:
            _pred_reconstruction = torch.view_as_real(_pred_reconstruction)

        if self.consecutive_slices > 1 and _pred_reconstruction.dim() == 5 and not self.segmentation_3d:
            _pred_reconstruction = _pred_reconstruction.reshape(
                _pred_reconstruction.shape[0] * _pred_reconstruction.shape[1],
                *_pred_reconstruction.shape[2:],
            )
        if self.segmentation_3d == True:
            _pred_reconstruction.unsqueeze(2).permute(0,2,1,3,4,5)
        if _pred_reconstruction.shape[-1] == 2:
            if self.input_channels == 1:
                _pred_reconstruction = torch.view_as_complex(_pred_reconstruction).unsqueeze(1)
                if self.magnitude_input:
                    _pred_reconstruction = torch.abs(_pred_reconstruction)
            elif self.input_channels == 2:
                if self.magnitude_input:
                    raise ValueError("Magnitude input is not supported for 2-channel input.")
                _pred_reconstruction = _pred_reconstruction.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"The input channels must be either 1 or 2. Found: {self.input_channels}")
        else:
            _pred_reconstruction = _pred_reconstruction.unsqueeze(1)
        print(_pred_reconstruction.shape)
        pred_segmentation = self.segmentation_module(_pred_reconstruction)

        if self.normalize_segmentation_output:
            pred_segmentation = (pred_segmentation - pred_segmentation.min()) / (
                pred_segmentation.max() - pred_segmentation.min()
            )

        pred_segmentation = torch.abs(pred_segmentation)

        if self.consecutive_slices > 1 and not self.segmentation_3d:
            # get batch size and number of slices from y, because if the reconstruction module is used they will
            # not be saved before
            pred_segmentation = pred_segmentation.view([y.shape[0], y.shape[1], *pred_segmentation.shape[1:]])

        if self.consecutive_slices > 1 and self.segmentation_3d:
            # get batch size and number of slices from y, because if the reconstruction module is used they will
            # not be saved before
            pred_segmentation = pred_segmentation.permute(0,2,1,3,4)
            pred_segmentation = pred_segmentation.view([y.shape[0], y.shape[1], *pred_segmentation.shape[2:]])

        return pred_reconstruction, pred_segmentation, hx, log_like  # type: ignore
