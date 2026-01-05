# coding=utf-8
__author__ = "Dimitris Karkalousos"

import math
from typing import Dict, List, Union, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import fft2
from atommic.collections.common.parts.utils import check_stacked_complex, expand_op
from atommic.collections.reconstruction_ecg.nn.base import BaseECGReconstructionModel
from atommic.collections.reconstruction_ecg.nn.rim_base.rim_block import RIMBlock
from atommic.core.classes.common import typecheck

__all__ = ["CIRIMECG"]


class CIRIMECG(BaseECGReconstructionModel):
    """Implementation of the Cascades of Independently Recurrent Inference Machines, as presented in
    [Karkalousos2022]_.

    References
    ----------
    .. [Karkalousos2022] Karkalousos D, Noteboom S, Hulst HE, Vos FM, Caan MWA. Assessment of data consistency through
        cascades of independently recurrent inference machines for fast and robust accelerated MRI reconstruction.
        Phys Med Biol. 2022 Jun 8;67(12). doi: 10.1088/1361-6560/ac6cc2. PMID: 35508147.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`CIRIM`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # make time-steps size divisible by 8 for fast fp16 training
        self.time_steps = 8 * math.ceil(cfg_dict.get("time_steps") / 8)
        self.reconstruction_module = torch.nn.ModuleList(
            [
                RIMBlock(
                    input_channels=cfg_dict.get("input_channels"),
                    recurrent_layer=cfg_dict.get("recurrent_layer"),
                    conv_filters=cfg_dict.get("conv_filters"),
                    conv_kernels=cfg_dict.get("conv_kernels"),
                    conv_dilations=cfg_dict.get("conv_dilations"),
                    conv_bias=cfg_dict.get("conv_bias"),
                    recurrent_filters=cfg_dict.get("recurrent_filters"),
                    recurrent_kernels=cfg_dict.get("recurrent_kernels"),
                    recurrent_dilations=cfg_dict.get("recurrent_dilations"),
                    recurrent_bias=cfg_dict.get("recurrent_bias"),
                    depth=cfg_dict.get("depth"),
                    no_dc=cfg_dict.get("no_dc"),
                    time_steps=self.time_steps,
                    conv_dim=cfg_dict.get("conv_dim"),
                    update_in_frequency=cfg_dict.get("update_in_frequency"),
                    hexad_inform=cfg_dict.get("hexad_inform"),
                    lowcut=cfg_dict.get("lowcut", None),
                    highcut=cfg_dict.get("highcut", None),
                    samplebase=cfg_dict.get("samplebase", None),
                )
                for _ in range(cfg_dict.get("num_cascades"))
            ]
        )

        # Keep estimation through the cascades if keep_prediction is True or re-estimate it if False.
        self.keep_prediction = cfg_dict.get("keep_prediction")

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        measured_ecg: torch.Tensor,
        mask: torch.Tensor,
        sigma: float = 1.0,
        target: Optional[torch.Tensor] = None,
    ) -> Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]:
        """Forward pass of :class:`CIRIM`.

        Parameters
        ----------
        initial_prediction : torch.Tensor
            Initial prediction. Shape [batch_size, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]
        sigma : float
            Noise level. Default is ``1.0``.

        Returns
        -------
        List of torch.Tensor
            List of the intermediate predictions for each cascade. Shape [batch_size, n_x, n_y].
        """
        prediction = measured_ecg.clone()
        hx = None
        cascades_predictions = []

        # --- synthetic branch ---
        for i, cascade in enumerate(self.reconstruction_module):
            prediction, hx = cascade.forward(
                prediction,
                mask,
                measured_ecg,
                hx,
                sigma,
                keep_prediction=False if i == 0 else self.keep_prediction,
            )
            cascades_predictions.append(prediction)
            prediction = prediction[-1]
        latent_list = []
        if target is not None:
            # Precompute masked target splits
            target_1 = target * mask
            target_2 = target * (1 - mask)

            measured_ecg_1 = target_1
            measured_ecg_2 = target_2

            hx_1, hx_2 = None, None
            latent_list = []

            for cascade in self.reconstruction_module:

                target_1, h_mask_1 = cascade.encoder_forward(
                    target=target_1,
                    mask=mask,
                    measured_ecg=measured_ecg_1,
                    hx=hx_1,
                    sigma=sigma,
                    keep_prediction=False,
                )
                hx_1 = h_mask_1[-1]

                target_2, h_mask_2 = cascade.encoder_forward(
                    target=target_2,
                    mask=mask,
                    measured_ecg=measured_ecg_2,
                    hx=hx_2,
                    sigma=sigma,
                    keep_prediction=False,
                )
                hx_2 = h_mask_2[-1]

                # combine latent states from both halves into [B, 2, ...]
                h_mlp = [torch.stack((h1, h2), dim=1) for h1, h2 in zip(h_mask_1, h_mask_2)]
                latent_list.append(h_mlp)

            labels = torch.arange(latent_list[-1][0].shape[0], device=latent_list[-1][0].device)

            return cascades_predictions, latent_list, labels

        return cascades_predictions

    def process_reconstruction_loss(  # noqa: MC0001
        self,
        target: torch.Tensor,
        prediction: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        mask: torch.Tensor,
        loss_func: torch.nn.Module,
        attrs: Dict,
        latent_features: Optional[List[List[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
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

        def compute_reconstruction_loss(t, p, m, attrs, latent_features, labels):
            if self.unnormalize_loss_inputs:
                # we do the unnormalization here to avoid explicitly iterating through list of predictions, which
                # might be a list of lists.
                t, p = self.__unnormalize_for_loss_or_log__(t, p, attrs)

            if "ssim" in str(loss_func).lower():
                return loss_func(
                    t.unsqueeze(dim=1),
                    p.unsqueeze(dim=1),
                    data_range=torch.tensor(
                        [max(torch.max(t).item(), torch.max(p).item()) - min(torch.min(t).item(), torch.min(p).item())]
                    )
                    .unsqueeze(dim=0)
                    .to(t.device),
                )

            if "mask" in str(loss_func).lower():
                return loss_func(t, p, m)

            if "supconloss" in str(loss_func).lower():
                if latent_features is None:
                    return torch.tensor(0.0, device=t.device)
                return loss_func(latent_features, labels)
            return loss_func(t, p)

        if self.accumulate_predictions:
            cascades_weights = torch.logspace(-1, 0, steps=len(prediction)).to(target.device)
            cascades_loss = []
            for idx, cascade_pred in enumerate(prediction):
                time_steps_weights = torch.logspace(-1, 0, steps=len(cascade_pred)).to(target.device)
                if latent_features is not None:
                    time_steps_loss = [
                        compute_reconstruction_loss(
                            target, time_step_pred, mask, attrs, latent_features[idx][kdx], labels
                        )
                        for kdx, time_step_pred in enumerate(cascade_pred)
                    ]
                    cascade_loss = sum(x * w for x, w in zip(time_steps_loss, time_steps_weights)) / sum(
                        time_steps_weights
                    )
                    cascades_loss.append(cascade_loss)
                else:
                    time_steps_loss = [
                        compute_reconstruction_loss(target, time_step_pred, mask, attrs, latent_features, labels)
                        for time_step_pred in cascade_pred
                    ]
                    cascade_loss = sum(x * w for x, w in zip(time_steps_loss, time_steps_weights)) / sum(
                        time_steps_weights
                    )
                    cascades_loss.append(cascade_loss)
            loss = sum(x * w for x, w in zip(cascades_loss, cascades_weights)) / sum(cascades_weights)
        else:
            # keep the last prediction of the last cascade
            prediction = prediction[-1][-1]
            latent_features = None if latent_features is None else latent_features[-1][-1]
            loss = compute_reconstruction_loss(target, prediction, mask, attrs, latent_features, labels)
        return loss
