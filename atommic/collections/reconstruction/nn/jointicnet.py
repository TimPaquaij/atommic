# coding=utf-8
__author__ = "Dimitris Karkalousos"


import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.nn.base import BaseSensitivityModel
from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import (
    check_stacked_complex,
    coil_combination_method,
    complex_conj,
    complex_mul,
)
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.unet_base.unet_block import NormUnet
from atommic.core.classes.common import typecheck

__all__ = ["JointICNet"]


class JointICNet(BaseMRIReconstructionModel):
    """Implementation of the Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network (Joint-ICNet),
     as presented in [Jun2021]_.

    References
    ----------
    .. [Jun2021] Jun, Yohan, et al. “Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network
        (Joint-ICNet) for Fast MRI.” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE,
        2021, pp. 5266–75. DOI.org (Crossref), https://doi.org/10.1109/CVPR46437.2021.00523.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`JointICNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.num_iter = cfg_dict.get("num_iter")

        self.kspace_model = NormUnet(
            cfg_dict.get("kspace_unet_num_filters"),
            cfg_dict.get("kspace_unet_num_pool_layers"),
            in_chans=2,
            out_chans=2,
            drop_prob=cfg_dict.get("kspace_unet_dropout_probability"),
            padding_size=cfg_dict.get("kspace_unet_padding_size"),
            normalize=cfg_dict.get("kspace_unet_normalize"),
        )

        self.image_model = NormUnet(
            cfg_dict.get("imspace_unet_num_filters"),
            cfg_dict.get("imspace_unet_num_pool_layers"),
            in_chans=2,
            out_chans=2,
            drop_prob=cfg_dict.get("imspace_unet_dropout_probability"),
            padding_size=cfg_dict.get("imspace_unet_padding_size"),
            normalize=cfg_dict.get("imspace_unet_normalize"),
        )

        self.sens_net = BaseSensitivityModel(
            cfg_dict.get("sens_unet_num_filters"),
            cfg_dict.get("sens_unet_num_pool_layers"),
            mask_center=cfg_dict.get("sens_unet_mask_center"),
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
            coil_dim=self.coil_dim,
            mask_type=cfg_dict.get("coil_sensitivity_maps_nn_mask_type", "2D"),
            drop_prob=cfg_dict.get("sens_unet_dropout_probability"),
            padding_size=cfg_dict.get("sens_unet_padding_size"),
            normalize=cfg_dict.get("sens_unet_normalize"),
        )

        self.conv_out = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)

        self.reg_param_I = torch.nn.Parameter(torch.ones(self.num_iter))
        self.reg_param_F = torch.nn.Parameter(torch.ones(self.num_iter))
        self.reg_param_C = torch.nn.Parameter(torch.ones(self.num_iter))

        self.lr_image = torch.nn.Parameter(torch.ones(self.num_iter))
        self.lr_sens = torch.nn.Parameter(torch.ones(self.num_iter))

    def update_C(
        self,
        idx: int,
        DC_sens: torch.Tensor,
        image: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""Updates the coil sensitivity maps.

        .. math::
            C = (1 - 2 * '\'lambda_{k}^{C} * ni_{k}) * C_{k}

            C = 2 * '\'lambda_{k}^{C} * ni_{k} * D_{C}(F^-1(b))

            A(x_{k}) = M * F * (C * x_{k})

            C = 2 * ni_{k} * F^-1(M.T * (M * F * (C * x_{k}) - b)) * x_{k}^*

        Parameters
        ----------
        idx : int
            The current iteration index.
        DC_sens : torch.Tensor
            The initial coil sensitivity maps. Shape [batch_size, num_coils, num_sens_maps, num_rows, num_cols].
        image : torch.Tensor
            The predicted image. Shape [batch_size, num_coils, num_rows, num_cols].
        sensitivity_maps : torch.Tensor
            The coil sensitivity maps. Shape [batch_size, num_coils, num_sens_maps, num_rows, num_cols].
        y : torch.Tensor
            The subsampled k-space data. Shape [batch_size, num_coils, num_rows, num_cols].
        mask : torch.Tensor
            The subsampled mask. Shape [batch_size, 1, num_rows, num_cols].

        Returns
        -------
        sensitivity_maps : torch.Tensor
            The updated coil sensitivity maps. Shape [batch_size, num_coils, num_sens_maps, num_rows, num_cols].
        """
        # (1 - 2 * lambda_{k}^{C} * ni_{k}) * C_{k}
        sense_term_1 = (1 - 2 * self.reg_param_C[idx] * self.lr_sens[idx]) * sensitivity_maps
        # 2 * lambda_{k}^{C} * ni_{k} * D_{C}(F^-1(b))
        sense_term_2 = 2 * self.reg_param_C[idx] * self.lr_sens[idx] * DC_sens
        # A(x_{k}) = M * F * (C * x_{k})
        sense_term_3_A = fft2(
            complex_mul(image.unsqueeze(self.coil_dim), sensitivity_maps),
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )
        sense_term_3_A = torch.where(mask == 0, torch.tensor([0.0], dtype=y.dtype).to(y.device), sense_term_3_A)
        # 2 * ni_{k} * F^-1(M.T * (M * F * (C * x_{k}) - b)) * x_{k}^*
        sense_term_3_mask = torch.where(
            mask == 1,
            torch.tensor([0.0], dtype=y.dtype).to(y.device),
            sense_term_3_A - y,
        )

        sense_term_3_backward = ifft2(
            sense_term_3_mask,
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )
        sense_term_3 = 2 * self.lr_sens[idx] * sense_term_3_backward * complex_conj(image).unsqueeze(self.coil_dim)
        sensitivity_maps = sense_term_1 + sense_term_2 - sense_term_3
        return sensitivity_maps

    def update_X(
        self,
        idx: int,
        image: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""Updates the image.

        .. math::
            x_{k} = (1 - 2 * '\'lamdba_{{k}_{I}} * mi_{k} - 2 * '\'lamdba_{{k}_{F}} * mi_{k}) * x_{k}

            x_{k} = 2 * mi_{k} * ('\'lambda_{{k}_{I}} * D_I(x_{k}) + '\'lambda_{{k}_{F}} * F^-1(D_F(f)))

            A(x{k} - b) = M * F * (C * x{k}) - b

            x_{k} = 2 * mi_{k} * A^* * (A(x{k} - b))

        Parameters
        ----------
        idx : int
            The current iteration index.
        image : torch.Tensor
            The predicted image. Shape [batch_size, num_coils, num_rows, num_cols].
        sensitivity_maps : torch.Tensor
            The coil sensitivity maps. Shape [batch_size, num_coils, num_sens_maps, num_rows, num_cols].
        y : torch.Tensor
            The subsampled k-space data. Shape [batch_size, num_coils, num_rows, num_cols].
        mask : torch.Tensor
            The subsampling mask. Shape [batch_size, 1, num_rows, num_cols].

        Returns
        -------
        image : torch.Tensor
            The updated image. Shape [batch_size, num_coils, num_rows, num_cols].
        """
        # (1 - 2 * lamdba_{k}_{I} * mi_{k} - 2 * lamdba_{k}_{F} * mi_{k}) * x_{k}
        image_term_1 = (
            1 - 2 * self.reg_param_I[idx] * self.lr_image[idx] - 2 * self.reg_param_F[idx] * self.lr_image[idx]
        ) * image
        # D_I(x_{k})
        image_term_2_DI = self.image_model(image.unsqueeze(self.coil_dim)).squeeze(self.coil_dim).contiguous()
        image_term_2_DF = ifft2(
            self.kspace_model(
                fft2(
                    image,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ).unsqueeze(self.coil_dim)
            )
            .squeeze(self.coil_dim)
            .contiguous(),
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )
        # 2 * mi_{k} * (lambda_{k}_{I} * D_I(x_{k}) + lambda_{k}_{F} * F^-1(D_F(f)))
        image_term_2 = (
            2
            * self.lr_image[idx]
            * (self.reg_param_I[idx] * image_term_2_DI + self.reg_param_F[idx] * image_term_2_DF)
        )
        # A(x{k}) - b) = M * F * (C * x{k}) - b
        image_term_3_A = fft2(
            complex_mul(image.unsqueeze(self.coil_dim), sensitivity_maps),
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )
        image_term_3_A = torch.where(mask == 0, torch.tensor([0.0], dtype=y.dtype).to(y.device), image_term_3_A) - y
        # 2 * mi_{k} * A^* * (A(x{k}) - b))
        image_term_3_Aconj = complex_mul(
            ifft2(
                image_term_3_A,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            ),
            complex_conj(sensitivity_maps),
        ).sum(self.coil_dim)
        image_term_3 = 2 * self.lr_image[idx] * image_term_3_Aconj
        image = image_term_1 + image_term_2 - image_term_3
        return image

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        initial_prediction: torch.Tensor,  # pylint: disable=unused-argument
        sigma: float = 1.0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Forward pass of :class:`JointICNet`.

        Parameters
        ----------
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]
        initial_prediction : torch.Tensor
            Initial prediction. Shape [batch_size, n_x, n_y, 2]
        sigma : float
            Noise level. Default is ``1.0``.

        Returns
        -------
        torch.Tensor
            Prediction of the final cascade. Shape [batch_size, n_x, n_y]
        """
        DC_sens = self.sens_net(y, mask, sensitivity_maps)
        sensitivity_maps = DC_sens.clone()
        image = coil_combination_method(
            ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims),
            sensitivity_maps,
            self.coil_combination_method,
            self.coil_dim,
        )
        for idx in range(self.num_iter):
            sensitivity_maps = self.update_C(idx, DC_sens, image, sensitivity_maps, y, mask)
            image = self.update_X(idx, image, sensitivity_maps, y, mask)
        return check_stacked_complex(image)
