# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import Dict, List, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import fft2
from atommic.collections.common.parts.utils import expand_op
from atommic.collections.multitask.rs.nn.base import BaseMRIReconstructionSegmentationModel
from atommic.collections.multitask.rs.nn.mtlrs_base.mtlrs_block import MTLRSBlock
from atommic.core.classes.common import typecheck
from atommic.collections.multitask.rs.nn.mtlrs_base.task_attention_module import TaskAttentionalModule
from atommic.collections.multitask.rs.nn.mtlrs_base.propagation_module import FeaturePropagationModule
from atommic.collections.multitask.rs.nn.mtlrs_base.spatially_adaptive_semantic_guidance_module import SASG

__all__ = ["MTLRS"]


class MTLRS(BaseMRIReconstructionSegmentationModel):
    """Implementation of the Multi-Task Learning for MRI Reconstruction and Segmentation (MTLRS) model, as presented
    in [Karkalousos2023]_.

    References
    ----------
    .. [Karkalousos2023] Karkalousos, D., Išgum, I., Marquering, H., Caan, M. W. A., (2023). MultiTask Learning for
        accelerated-MRI Reconstruction and Segmentation of Brain Lesions in Multiple Sclerosis. In Proceedings of
        Machine Learning Research (Vol. 078).

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`MTLRS`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object.
        trainer : Trainer, optional
            PyTorch Lightning trainer object. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.reconstruction_module_recurrent_filters = cfg_dict.get("reconstruction_module_recurrent_filters")
        self.reconstruction_module_time_steps = cfg_dict.get("reconstruction_module_time_steps")
        self.reconstruction_module_num_cascades = cfg_dict.get("reconstruction_module_num_cascades")
        self.reconstruction_module_accumulate_predictions = cfg_dict.get(
            "reconstruction_module_accumulate_predictions"
        )
        conv_dim = cfg_dict.get("reconstruction_module_conv_dim")
        reconstruction_module_params = {
            "num_cascades": self.reconstruction_module_num_cascades,
            "time_steps": self.reconstruction_module_time_steps,
            "no_dc": cfg_dict.get("reconstruction_module_no_dc"),
            "keep_prediction": cfg_dict.get("reconstruction_module_keep_prediction"),
            "dimensionality": cfg_dict.get("reconstruction_module_dimensionality"),
            "recurrent_layer": cfg_dict.get("reconstruction_module_recurrent_layer"),
            "conv_filters": cfg_dict.get("reconstruction_module_conv_filters"),
            "conv_kernels": cfg_dict.get("reconstruction_module_conv_kernels"),
            "conv_dilations": cfg_dict.get("reconstruction_module_conv_dilations"),
            "conv_bias": cfg_dict.get("reconstruction_module_conv_bias"),
            "recurrent_filters": self.reconstruction_module_recurrent_filters,
            "recurrent_kernels": cfg_dict.get("reconstruction_module_recurrent_kernels"),
            "recurrent_dilations": cfg_dict.get("reconstruction_module_recurrent_dilations"),
            "recurrent_bias": cfg_dict.get("reconstruction_module_recurrent_bias"),
            "depth": cfg_dict.get("reconstruction_module_depth"),
            "conv_dim": conv_dim,
            "pretrained": cfg_dict.get("pretrained"),
            "accumulate_predictions": self.reconstruction_module_accumulate_predictions,
        }

        self.segmentation_module_output_channels = cfg_dict.get("segmentation_module_output_channels", 2)
        segmentation_module_params = {
            "segmentation_module": cfg_dict.get("segmentation_module"),
            "output_channels": self.segmentation_module_output_channels,
            "channels": cfg_dict.get("segmentation_module_channels", 64),
            "pooling_layers": cfg_dict.get("segmentation_module_pooling_layers", 2),
            "dropout": cfg_dict.get("segmentation_module_dropout", 0.0),
            "temporal_kernel": cfg_dict.get("segmentation_module_temporal_kernel", 1),
            "activation": cfg_dict.get("segmentation_module_activation", "elu"),
            "bias": cfg_dict.get("segmentation_module_bias", False),
            "conv_dim": conv_dim,
        }

        self.coil_dim = cfg_dict.get("coil_dim", 1)
        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)

        self.rs_cascades = cfg_dict.get("joint_reconstruction_segmentation_module_cascades", 1)
        self.rs_module = torch.nn.ModuleList(
            [
                MTLRSBlock(
                    reconstruction_module_params=reconstruction_module_params,
                    segmentation_module_params=segmentation_module_params,
                    input_channels=cfg_dict.get("segmentation_module_input_channels", 2),
                    magnitude_input=cfg_dict.get("magnitude_input", False),
                    fft_centered=cfg_dict.get("fft_centered", False),
                    fft_normalization=cfg_dict.get("fft_normalization", "backward"),
                    spatial_dims=cfg_dict.get("spatial_dims", (-2, -1)),
                    coil_dim=self.coil_dim,
                    dimensionality=cfg_dict.get("dimensionality", 2),
                    consecutive_slices=self.consecutive_slices,
                    coil_combination_method=cfg_dict.get("coil_combination_method", "SENSE"),
                    normalize_segmentation_output=cfg_dict.get("normalize_segmentation_output", True),
                )
                for _ in range(self.rs_cascades)
            ]
        )

        self.task_adaption_type = cfg_dict.get("task_adaption_type", "multi_task_learning")
        self.attention_module = cfg_dict.get("attention_module",False)
        self.attention_module_kernel_size = cfg_dict.get("attention_module_kernel_size", 3)
        self.attention_module_padding = cfg_dict.get("attention_module_padding", 1)
        if self.attention_module == "SemanticGuidanceModule" and self.task_adaption_type == "multi_task_learning_logit":
            print("Logits can not be used with semantic guidance module, logits will be transfomed into probabilities with softmax")
            self.task_adaption_type ="multi_task_learning_softmax"
        if self.attention_module == "TaskAttentionModule":
            self.attention_module_block = torch.nn.ModuleList(
            [TaskAttentionalModule(in_channels=cfg_dict.get("reconstruction_module_conv_filters")[0]) for _ in range(self.rs_cascades)
            ]
        )
        if self.attention_module == "PropagationModule":
            self.attention_module_block = torch.nn.ModuleList(
            [FeaturePropagationModule(num_tasks=2,per_task_channels=cfg_dict.get("reconstruction_module_conv_filters")[0])for _ in range(self.rs_cascades)
            ]
        )
        if self.attention_module == "SemanticGuidanceModule":
            self.attention_module_block = torch.nn.ModuleList(
            [SASG(channels_rec=cfg_dict.get("reconstruction_module_conv_filters")[0],channels_seg=self.segmentation_module_output_channels-1,kernel_size=self.attention_module_kernel_size,padding=self.attention_module_padding)for _ in range(self.rs_cascades)
            ]
        )

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_reconstruction_pred: torch.Tensor,
        target_reconstruction: torch.Tensor,
        hx: torch.Tensor = None,
        sigma: float = 1.0,
    ) -> Tuple[Union[List, torch.Tensor], torch.Tensor]:
        """Forward pass of :class:`MTLRS`.

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
        pred_reconstructions = []
        for c,cascade in enumerate(self.rs_module):
            pred_reconstruction, pred_segmentation, hx = cascade(
                y=y,
                sensitivity_maps=sensitivity_maps,
                mask=mask,
                init_reconstruction_pred=init_reconstruction_pred,
                target_reconstruction=target_reconstruction,
                hx=hx,
                sigma=sigma,
            )
            pred_reconstructions.append(pred_reconstruction)
            init_reconstruction_pred = pred_reconstruction[-1][-1]

            if self.task_adaption_type == "multi_task_learning":
                hidden_states = [
                    torch.cat(
                        [torch.abs(init_reconstruction_pred.unsqueeze(self.coil_dim) * pred_segmentation)]
                        * (f // self.segmentation_module_output_channels),
                        dim=self.coil_dim,
                    )
                    for f in self.reconstruction_module_recurrent_filters
                    if f != 0
                ]

                if self.consecutive_slices > 1:
                    hx = [x.unsqueeze(1) for x in hx]
            
            if self.task_adaption_type == "multi_task_learning_softmax":
                if self.consecutive_slices > 1:
                    pred_segmentation_soft = torch.softmax(pred_segmentation, dim=2)
                else:
                    pred_segmentation_soft = torch.softmax(pred_segmentation, dim=1)
                if self.attention_module == "SemanticGuidanceModule":
                    hidden_states = [pred_segmentation_soft[:,1:] for _ in self.reconstruction_module_recurrent_filters]
                else:
                    hidden_states = [
                        torch.cat(
                            [torch.abs(init_reconstruction_pred.unsqueeze(
                                self.coil_dim)) * torch.sum(pred_segmentation_soft[..., 1:, :, :],dim=1,keepdim=True)]
                            * f ,
                            dim=self.coil_dim,
                        )
                        for f in self.reconstruction_module_recurrent_filters
                        if f != 0
                    ]

                # Check if the concatenated hidden states are the same size as the hidden state of the RNN
                if hidden_states[0].shape[self.coil_dim] != hx[0].shape[self.coil_dim] and not self.attention_module =="SemanticGuidanceModule":
                    prev_hidden_states = hidden_states
                    hidden_states = []
                    for hs in prev_hidden_states:
                        new_hidden_state = hs
                        for _ in range(hx[0].shape[1] - prev_hidden_states[0].shape[1]):
                            new_hidden_state = torch.cat(
                                [new_hidden_state, torch.zeros_like(hx[0][:, 0, :, :]).unsqueeze(self.coil_dim)],
                                dim=self.coil_dim,
                            )
                        hidden_states.append(new_hidden_state)
                if self.attention_module:
                    hx = [self.attention_module_block[c](hx[i],hidden_states[i]) for i in range(len(hx))]
                else:
                    hx = [hx[i] + hidden_states[i] for i in range(len(hx))]

            init_reconstruction_pred = torch.view_as_real(init_reconstruction_pred)

        return pred_reconstructions, pred_segmentation

    def process_reconstruction_loss(  # noqa: MC0001
        self,
        target: torch.Tensor,
        prediction: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        attrs: Union[Dict, torch.Tensor],
        r: Union[int, torch.Tensor],
        loss_func: torch.nn.Module,
    ) -> torch.Tensor:
        """Processes the reconstruction loss for the CIRIM model. It differs from the base class in that it can handle
        multiple cascades and time steps.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
        mask : torch.Tensor
            Mask of shape [batch_size, n_x, n_y, 2]. It will be used if self.ssdu is True, to enforce data consistency
            on the prediction.
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.
        loss_func : torch.nn.Module
            Loss function. Must be one of {torch.nn.L1Loss(), torch.nn.MSELoss(),
            atommic.collections.reconstruction.losses.ssim.SSIMLoss()}. Default is ``torch.nn.L1Loss()``.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        # If kspace reconstruction loss is used, the target needs to be transformed to k-space.
        if self.kspace_reconstruction_loss:
            # If inputs are complex, then they need to be viewed as real.
            if target.shape[-1] != 2 and torch.is_complex(target):
                target = torch.view_as_real(target)
            # If SSDU is used, then the coil-combined inputs need to be expanded to multiple coils using the
            # sensitivity maps.
            if self.ssdu:
                target = expand_op(target, sensitivity_maps, self.coil_dim)
            # Transform to k-space.
            target = fft2(target, self.fft_centered, self.fft_normalization, self.spatial_dims)
            # Ensure loss inputs are both viewed in the same way.
            target = self.__abs_output__(target / torch.max(torch.abs(target)))
        elif not self.unnormalize_loss_inputs:
            target = self.__abs_output__(target / torch.max(torch.abs(target)))

        def compute_reconstruction_loss(t, p, s):
            if self.unnormalize_loss_inputs:
                # we do the unnormalization here to avoid explicitly iterating through list of predictions, which
                # might be a list of lists.
                t, p, s = self.__unnormalize_for_loss_or_log__(t, p, s, attrs, r)

            # If kspace reconstruction loss is used, the target needs to be transformed to k-space.
            if self.kspace_reconstruction_loss:
                # If inputs are complex, then they need to be viewed as real.
                if p.shape[-1] != 2 and torch.is_complex(p):
                    p = torch.view_as_real(p)
                # If SSDU is used, then the coil-combined inputs need to be expanded to multiple coils using the
                # sensitivity maps.
                if self.ssdu:
                    p = expand_op(p, s, self.coil_dim)
                # Transform to k-space.
                p = fft2(p, self.fft_centered, self.fft_normalization, self.spatial_dims)
                # If SSDU is used, then apply the mask to the prediction to enforce data consistency.
                if self.ssdu:
                    p = p * mask
                # Ensure loss inputs are both viewed in the same way.
                p = self.__abs_output__(p / torch.max(torch.abs(p)))
            elif not self.unnormalize_loss_inputs:
                p = self.__abs_output__(p / torch.max(torch.abs(p)))

            if "ssim" in str(loss_func).lower():
                p = torch.abs(p / torch.max(torch.abs(p)))
                t = torch.abs(t / torch.max(torch.abs(t)))

                return loss_func(
                    t,
                    p,
                    data_range=torch.tensor([max(torch.max(t).item(), torch.max(p).item())]).unsqueeze(dim=0).to(t),
                )

            return loss_func(t, p)

        if self.reconstruction_module_accumulate_predictions:
            rs_cascades_weights = torch.logspace(-1, 0, steps=len(prediction)).to(target.device)
            rs_cascades_loss = []
            for rs_cascade_pred in prediction:
                cascades_weights = torch.logspace(-1, 0, steps=len(rs_cascade_pred)).to(target.device)
                cascades_loss = []
                for cascade_pred in rs_cascade_pred:
                    time_steps_weights = torch.logspace(-1, 0, steps=self.reconstruction_module_time_steps).to(target.device)
                    time_steps_loss = [
                        compute_reconstruction_loss(target, time_step_pred, sensitivity_maps)
                        for time_step_pred in cascade_pred
                    ]
                    cascade_loss = sum(x * w for x, w in zip(time_steps_loss, time_steps_weights)) / self.reconstruction_module_time_steps
                    cascades_loss.append(cascade_loss)
                rs_cascade_loss = sum(x * w for x, w in zip(cascades_loss, cascades_weights)) / len(rs_cascade_pred)
                rs_cascades_loss.append(rs_cascade_loss)
            loss = sum(x * w for x, w in zip(rs_cascades_loss, rs_cascades_weights)) / len(prediction)
        else:
            # keep the last prediction of the last cascade of the last rs cascade
            prediction = prediction[-1][-1][-1]
            loss = compute_reconstruction_loss(target, prediction, sensitivity_maps)
        return loss
