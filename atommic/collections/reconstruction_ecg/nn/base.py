# coding=utf-8
__author__ = "Dimitris Karkalousos"

import os
import warnings
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
import h5py
import numpy as np
import torch
import json
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss, MSELoss
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from atommic.collections.common.data.subsample import create_masker
from atommic.collections.common.losses import VALID_RECONSTRUCTION_LOSSES, AggregatorLoss, SinkhornDistance
from atommic.collections.common.nn.base import BaseMRIModel, BaseSensitivityModel, DistributedMetricSum
from atommic.collections.common.parts.utils import (
    check_stacked_complex,
    coil_combination_method,
    complex_abs,
    complex_abs_sq,
    expand_op,
    is_none,
    parse_list_and_keep_last,
    unnormalize,
    unnormalize_ECG,
)
from ecgxai.utils.dataset import UniversalECGDataset
from ecgxai.utils.transforms import (
    ToTensor,
    ApplyGain,
    To12Lead,
    Resample,
    PolyFilter,
    ButterFilter,
    Masker,
    ECGNormalizer,
)
from atommic.collections.common.parts.fft import fft1, ifft1
from atommic.collections.reconstruction_ecg.losses.na import NoiseAwareLoss
from atommic.collections.reconstruction_ecg.losses.ssim import SSIMLoss
from atommic.collections.reconstruction_ecg.losses.ml1 import MaskL1Loss
from atommic.collections.reconstruction_ecg.losses.huber import MaskHuberLoss
from atommic.collections.reconstruction_ecg.losses.mse import MaskMSELoss
from atommic.collections.reconstruction_ecg.losses.contrastive import SupConLoss
from atommic.collections.reconstruction_ecg.metrics.reconstruction_metrics import mse, nmse, psnr, ssim

__all__ = ["BaseECGReconstructionModel"]


class BaseECGReconstructionModel(BaseMRIModel, ABC):
    """Base class of all MRI reconstruction models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`BaseECGReconstructionModel`.

        Parameters
        ----------
        cfg: DictConfig
            The configuration file.
        trainer: Trainer
            The PyTorch Lightning trainer.
        """
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.update_in_frequency = cfg_dict.get("update_in_frequency", False)

        self.reconstruction_losses = {}
        reconstruction_loss = cfg_dict.get("reconstruction_loss")
        reconstruction_losses_ = {}
        if reconstruction_loss is not None:
            for k, v in reconstruction_loss.items():
                if k not in VALID_RECONSTRUCTION_LOSSES:
                    raise ValueError(
                        f"Reconstruction loss {k} is not supported. Please choose one of the following: "
                        f"{VALID_RECONSTRUCTION_LOSSES}."
                    )
                if v is None or v == 0.0:
                    warnings.warn(f"The weight of reconstruction loss {k} is set to 0.0. This loss will not be used.")
                else:
                    reconstruction_losses_[k] = v
        else:
            # Default reconstruction loss is L1.
            reconstruction_losses_["l1"] = 1.0
        if sum(reconstruction_losses_.values()) != 1.0:
            warnings.warn("Sum of reconstruction losses weights is not 1.0. Adjusting weights to sum up to 1.0.")
            total_weight = sum(reconstruction_losses_.values())
            reconstruction_losses_ = {k: v / total_weight for k, v in reconstruction_losses_.items()}
        self.contrastive_loss = False
        for name in VALID_RECONSTRUCTION_LOSSES:
            if name in reconstruction_losses_:
                if name == "ssim":
                    self.reconstruction_losses[name] = SSIMLoss()
                elif name == "mse":
                    self.reconstruction_losses[name] = MSELoss()
                elif name == "wasserstein":
                    self.reconstruction_losses[name] = SinkhornDistance()
                elif name == "noise_aware":
                    self.reconstruction_losses[name] = NoiseAwareLoss()
                elif name == "l1":
                    self.reconstruction_losses[name] = L1Loss()
                elif name == "masked_l1":
                    self.reconstruction_losses[name] = MaskL1Loss(
                        weight=cfg.get("masked_weight", 2),
                        amplitude_weight=cfg.get("amplitude_weight", 2),
                        update_in_frequency=self.update_in_frequency,
                    )
                elif name == "masked_huber":
                    self.reconstruction_losses[name] = MaskHuberLoss(
                        weight=cfg.get("masked_weight", 2),
                        amplitude_weight=cfg.get("amplitude_weight", 2),
                        update_in_frequency=self.update_in_frequency,
                    )
                elif name == "masked_mse":
                    self.reconstruction_losses[name] = MaskMSELoss(
                        weight=cfg.get("masked_weight", 2),
                        amplitude_weight=cfg.get("amplitude_weight", 2),
                        update_in_frequency=self.update_in_frequency,
                    )
                elif name == "spectral_mse":
                    self.reconstruction_losses[name] = MaskMSELoss(
                        spectral=True, update_in_frequency=self.update_in_frequency
                    )
                elif name == "spectral_huber":
                    self.reconstruction_losses[name] = MaskHuberLoss(
                        spectral=True, update_in_frequency=self.update_in_frequency
                    )
                elif name == "spectral_l1":
                    self.reconstruction_losses[name] = MaskL1Loss(
                        spectral=True, update_in_frequency=self.update_in_frequency
                    )
                elif name == "contrastive_loss":
                    self.reconstruction_losses[name] = SupConLoss(temperature=0.07, contrast_mode="all")
                    self.contrastive_loss = True

        # replace losses names by 'loss_1', 'loss_2', etc. to properly iterate in the aggregator loss
        self.reconstruction_losses = {f"loss_{i+1}": v for i, v in enumerate(self.reconstruction_losses.values())}
        self.total_reconstruction_losses = len(self.reconstruction_losses)
        self.total_reconstruction_loss_weight = cfg_dict.get("total_reconstruction_loss_weight", 1.0)

        # Set normalization parameters for logging
        self.unnormalize_loss_inputs = cfg_dict.get("unnormalize_loss_inputs", False)
        self.unnormalize_log_outputs = cfg_dict.get("unnormalize_log_outputs", False)
        self.normalization_type = cfg_dict.get("normalization_type", "max")

        # Refers to cascading or iterative reconstruction methods.
        self.accumulate_predictions = cfg_dict.get("accumulate_predictions", False)

        # Initialize the module
        super().__init__(cfg=cfg, trainer=trainer)

        # Set aggregation loss
        self.total_reconstruction_loss = AggregatorLoss(
            num_inputs=self.total_reconstruction_losses, weights=list(reconstruction_losses_.values())
        )

        # Set distributed metrics
        self.MSE = DistributedMetricSum()
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()

        # Set evaluation metrics dictionaries
        self.mse_vals: Dict = defaultdict(dict)
        self.nmse_vals: Dict = defaultdict(dict)
        self.ssim_vals: Dict = defaultdict(dict)
        self.psnr_vals: Dict = defaultdict(dict)

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
        """Processes the reconstruction loss.

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
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.
        loss_func : torch.nn.Module
            Loss function. Default is ``torch.nn.L1Loss()``.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """

        def compute_reconstruction_loss(t, p, m, attrs, hx):
            if self.unnormalize_loss_inputs:
                # we do the unnormalization here to avoid explicitly iterating through list of predictions, which
                # might be a list of lists.
                t, p = self.__unnormalize_for_loss_or_log__(t, p, attrs)
            if "ssim" in str(loss_func).lower():
                p = torch.abs(p / torch.max(torch.abs(p)))
                t = torch.abs(t / torch.max(torch.abs(t)))

                return loss_func(
                    t,
                    p,
                    data_range=torch.tensor([max(torch.max(t).item(), torch.max(p).item())]).unsqueeze(dim=0).to(t),
                )
            if "masked_l1" in str(loss_func).lower():
                return loss_func(t, p, m)

            if "masked_huber" in str(loss_func).lower():
                return loss_func(t, p, m)

            if "contrastive_loss":
                return loss_func(latent_features, labels)

            return loss_func(t, p)

        return compute_reconstruction_loss(target, prediction, mask, attrs, latent_features, labels)

    def __compute_loss__(
        self,
        target: torch.Tensor,
        predictions: Union[list, torch.Tensor],
        mask: torch.Tensor,
        attrs: dict,
        latent_features: Optional[List[List[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the reconstruction loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        predictions : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        predictions_n2r : Union[list, torch.Tensor]
            Noise-to-Recon prediction(s) of shape [batch_size, n_x, n_y, 2], if Noise-to-Recon is used.
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
        ssdu_loss_mask : torch.Tensor
            SSDU loss mask of shape [batch_size, 1, n_x, n_y, 1]. It will be used if self.ssdu is True, to enforce
            data consistency on the prediction.
        attrs : Union[Dict, torch.Tensor]
            Attributes of the data with pre normalization values.
        r : Union[int, torch.Tensor]
            The selected acceleration factor.

        Returns
        -------
        loss: torch.FloatTensor
            Reconstruction loss.
        """
        weight = 1.0
        losses = {}
        if self.update_in_frequency:
            mask = mask.unsqueeze(-1)
        for name, loss_func in self.reconstruction_losses.items():
            if self.contrastive_loss:
                losses[name] = (
                    self.process_reconstruction_loss(
                        target, predictions, mask, loss_func, attrs, latent_features, labels
                    )
                    * weight
                )
            else:
                losses[name] = self.process_reconstruction_loss(target, predictions, mask, loss_func, attrs) * weight
        return self.total_reconstruction_loss(**losses) * self.total_reconstruction_loss_weight

    def __compute_and_log_metrics_and_outputs__(
        self,
        target: torch.Tensor,
        predictions: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        attrs: Union[Dict, torch.Tensor],
        fname: Union[str, torch.Tensor],
        slice_idx: Union[int, torch.Tensor],
        layout: Union[float, torch.Tensor],
    ):
        """Computes the metrics and logs the outputs.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y].
        predictions : Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]
            Prediction data of shape [batch_size, n_x, n_y, 2]. It can be a list or list of lists if iterative and/or
            cascading reconstruction methods are used.
        attrs : Union[Dict, torch.Tensor]
            Attributes of the data with pre normalization values.
        r : Union[int, torch.Tensor]
            The selected acceleration factor.
        fname : Union[str, torch.Tensor]
            File name.
        slice_idx : Union[int, torch.Tensor]
            Slice index.
        acceleration : Union[float, torch.Tensor]
            Acceleration factor.
        """
        while isinstance(predictions, list):
            predictions = predictions[-1]

        # Add dummy dimensions to target and predictions for Metrics.
        target = target.unsqueeze(1)
        predictions = predictions.unsqueeze(1)
        target = target.detach().cpu()
        predictions = predictions.detach().cpu()

        # Iterate over the batch and log the target and predictions.
        for _batch_idx_ in range(target.shape[0]):
            output_target = target[_batch_idx_]
            output_predictions = predictions[_batch_idx_]

            if self.unnormalize_log_outputs:
                # Unnormalize target and predictions with pre normalization values. This is only for logging purposes.
                # For the loss computation, the self.unnormalize_loss_inputs flag is used.
                output_target, output_predictions = self.__unnormalize_for_loss_or_log__(
                    output_target, output_predictions, attrs, _batch_idx_
                )

            # Log target and predictions, if log_image is True for this slice.
            if attrs["log_image"][_batch_idx_]:
                key = f"{fname[_batch_idx_]}-Acc={layout[_batch_idx_]}"  # type: ignore
                self.log_image(f"{key}/target", output_target[0])
                self.log_image(f"{key}/reconstruction", output_predictions[0])
                self.log_image(f"{key}/error", torch.abs(output_target[0] - output_predictions[0]))

            # Compute metrics and log them.
            output_target = output_target.numpy()
            output_predictions = output_predictions.numpy()
            self.mse_vals[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(  # type: ignore
                mse(output_target, output_predictions)
            ).view(1)
            self.nmse_vals[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(  # type: ignore
                nmse(output_target, output_predictions)
            ).view(1)

            max_value = max(np.max(output_target), np.max(output_predictions)) - min(
                np.min(output_target), np.min(output_predictions)
            )

            self.ssim_vals[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(  # type: ignore
                ssim(output_target, output_predictions, maxval=max_value)
            ).view(1)
            self.psnr_vals[fname[_batch_idx_]][str(slice_idx[_batch_idx_].item())] = torch.tensor(  # type: ignore
                psnr(output_target, output_predictions, maxval=max_value)
            ).view(1)

    def __compute_time_domain(self, target, predictions):
        if self.accumulate_predictions:
            predictions = parse_list_and_keep_last(predictions)
        predictions = ifft1(predictions, time_dim=-1)[..., 0]
        target = ifft1(target, time_dim=-1)[..., 0]
        return target, predictions

    def __unnormalize_for_loss_or_log__(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        attrs: Dict,
        batch_idx: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unnormalizes the data for computing the loss or logging.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : torch.Tensor
            Prediction data of shape [batch_size, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor or None
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2] or None.
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.
        batch_idx : int
            Batch index. Default is ``1``.

        Returns
        -------
        target : torch.Tensor
            Unnormalized target data.
        prediction : torch.Tensor
            Unnormalized prediction data.
        sensitivity_maps : torch.Tensor
            Unnormalized sensitivity maps.
        """
        min_val = attrs["target_min"]
        max_val = attrs["target_max"]
        mean_val = attrs["target_mean"]
        std_val = attrs["target_std"]
        median_val = attrs["target_median"]
        if isinstance(min_val, list):
            min_val = min_val[batch_idx]
        if isinstance(max_val, list):
            max_val = max_val[batch_idx]
        if isinstance(mean_val, list):
            mean_val = mean_val[batch_idx]
        if isinstance(std_val, list):
            std_val = std_val[batch_idx]
        if isinstance(median_val, list):
            std_val = std_val[median_val]
        target = unnormalize_ECG(
            target,
            {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val, "median": median_val},
            self.normalization_type,
        )

        min_val = attrs["prediction_min"]
        max_val = attrs["prediction_max"]
        mean_val = attrs["prediction_mean"]
        std_val = attrs["prediction_std"]
        median_val = attrs["prediction_median"]
        if isinstance(min_val, list):
            min_val = min_val[batch_idx]
        if isinstance(max_val, list):
            max_val = max_val[batch_idx]
        if isinstance(mean_val, list):
            mean_val = mean_val[batch_idx]
        if isinstance(std_val, list):
            std_val = std_val[batch_idx]
        if isinstance(median_val, list):
            std_val = std_val[median_val]

        prediction = unnormalize_ECG(
            prediction,
            {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val, "median": median_val},
            self.normalization_type,
        )

        return target, prediction

    @staticmethod
    def __process_inputs__(
        measured_ecg: Union[List, torch.Tensor],
        mask: Union[List, torch.Tensor],
        target: Union[List, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Processes lists of inputs to torch.Tensor. In the case where multiple accelerations are used, then the
        inputs are lists. This function converts the lists to torch.Tensor by randomly selecting one acceleration. If
        only one acceleration is used, then the inputs are torch.Tensor and are returned as is.

        Parameters
        ----------
        kspace : Union[List, torch.Tensor]
            Full k-space data of length n_accelerations or shape [batch_size, n_coils, n_x, n_y, 2].
        y : Union[List, torch.Tensor]
            Subsampled k-space data of length n_accelerations or shape [batch_size, n_coils, n_x, n_y, 2].
        mask : Union[List, torch.Tensor]
            Sampling mask of length n_accelerations or shape [batch_size, 1, n_x, n_y, 1].
        initial_prediction : Union[List, torch.Tensor]
            Initial prediction of length n_accelerations or shape [batch_size, n_coils, n_x, n_y, 2].
        target : Union[List, torch.Tensor]
            Target data of length n_accelerations or shape [batch_size, n_x, n_y, 2].

        Returns
        -------
        kspace : torch.Tensor
            Full k-space data of shape [batch_size, n_coils, n_x, n_y, 2].
        y : torch.Tensor
            Subsampled k-space data of shape [batch_size, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        initial_prediction : torch.Tensor
            Initial prediction of shape [batch_size, n_coils, n_x, n_y, 2].
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        r : int
            Random index used to select the acceleration.
        """
        if isinstance(measured_ecg, list):
            r = np.random.randint(len(measured_ecg))
            mask = mask[r]
            measured_ecg = measured_ecg[r]
        if isinstance(target, list):
            r = np.random.randint(len(measured_ecg))
            target = target[r]
        else:
            r = 0

        return measured_ecg, mask, target, r

    def inference_step(
        self,
        measured_ecg: Union[List[torch.Tensor], torch.Tensor],
        mask: Union[List[torch.Tensor], torch.Tensor],
        target: torch.Tensor,
        fname: Union[List[str], str],
        id: Union[List[int], int],
        layout: Union[List[str], str],
        attrs: Union[List[Dict], Dict],
    ):
        """Performs an inference step, i.e., computes the predictions of the model.

        Parameters
        ----------
        kspace : torch.Tensor
            Fully sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
        y : Union[List[torch.Tensor], torch.Tensor]
            Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
            Shape [batch_size, n_coils, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
        mask : Union[List[torch.Tensor], torch.Tensor]
            Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor. Also, if Unsupervised
            Learning methods are used, it contains their masks. Shape [batch_size, 1, n_x, n_y, 1].
        initial_prediction : Union[List, torch.Tensor]
            Initial prediction. If multiple accelerations are used, then it is a list of torch.Tensor.
            Shape [batch_size, n_x, n_y, 2].
        target : torch.Tensor
            Target data. Shape [batch_size, n_x, n_y, 2].
        fname : str
            File name.
        slice_idx : int
            Slice index.
        acceleration : float
            Acceleration factor of the sampling mask, randomly selected if multiple accelerations are used.
        attrs : Dict
            Attributes dictionary.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of processed inputs and model's predictions, with keys:
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'acceleration' : float
                    Acceleration factor of the sampling mask, randomly selected if multiple accelerations are used.
                'predictions' : Union[List[torch.Tensor], torch.Tensor]
                    Model's predictions. If accumulate predictions is True, then it is a list of torch.Tensor.
                    Shape [batch_size, n_x, n_y, 2].
                'predictions_n2r' : Union[List[torch.Tensor], torch.Tensor]
                    Model's predictions for Noise-to-Recon, if Noise-to-Recon is used. If accumulate predictions is
                    True, then it is a list of torch.Tensor. Shape [batch_size, n_x, n_y, 2].
                'target' : torch.Tensor
                    Target data. Shape [batch_size, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'loss_mask' : torch.Tensor
                    SSDU loss mask, if SSDU is used. Shape [batch_size, 1,  n_x, n_y, 1].
                'attrs' : dict
                    Attributes dictionary.
                'r' : int
                    Random index used for selected acceleration.
        """

        # Forward pass
        if self.contrastive_loss:
            predictions, h, labels = self.forward(measured_ecg, mask, target=target)
        else:
            predictions = self.forward(measured_ecg, mask)

        # Get acceleration factor from acceleration list, if multiple accelerations are used. Or if batch size > 1.
        return {
            "fname": fname,
            "id": id,
            "layout": layout,
            "predictions": predictions,
            "target": target,
            "attrs": attrs,
            "contrastive": {"latent_features": h, "labels": labels} if self.contrastive_loss else None,
        }

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Performs a training step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data with keys:
                'kspace' : List of torch.Tensor
                    Fully-sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
                'y' : Union[torch.Tensor, None]
                    Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_coils, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'mask' : Union[torch.Tensor, None]
                    Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor. Also, if
                    Unsupervised Learning methods, like Noise-to-Recon or SSDU, are used, then it is a list of
                    torch.Tensor with masks for each method. Shape [batch_size, 1, n_x, n_y, 1].
                'initial_prediction' : Union[torch.Tensor, None]
                    Initial prediction. Shape [batch_size, n_x, n_y, 2] or None.
                'target' : Union[torch.Tensor, None]
                    Target data. Shape [batch_size, n_x, n_y] or None.
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'acceleration' : float
                    Acceleration factor of the sampling mask.
                'attrs' : dict
                    Attributes dictionary.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss and log.
        """
        sample = batch
        outputs = self.inference_step(
            sample["masked_waveform"],
            sample["mask"],
            sample["waveform"],
            sample["filename"],  # type: ignore
            sample["data_idx"],  # type: ignore
            sample["layout"],
            sample["attrs"],  # type: ignore
        )
        target = outputs["target"]
        predictions = outputs["predictions"]
        if self.update_in_frequency:
            target = fft1(target, time_dim=-1)
        # Determine if contrastive loss should be applied

        if self.contrastive_loss:
            train_loss = self.__compute_loss__(
                target, predictions, sample["mask"], sample["attrs"], **outputs["contrastive"]
            )
        else:
            train_loss = self.__compute_loss__(target, predictions, sample["mask"], sample["attrs"])
        if self.update_in_frequency:
            target, predictions = self.__compute_time_domain(target, predictions)
        # Log loss for the chosen acceleration factor and the learning rate in the selected logger.
        logs = {
            f'train_loss': train_loss.item(),
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }

        # In case of Noise-to-Recon or SSDU, the target is a list.
        if isinstance(target, list):
            while isinstance(target, list):
                target = target[-1]

        # Log train loss.
        self.log(
            "reconstruction_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=target.shape[0],  # type: ignore
            sync_dist=True,
        )

        return {"loss": train_loss, "log": logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int):
        """Performs a validation step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys:
                'kspace' : List of torch.Tensor
                    Fully-sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
                'y' : Union[torch.Tensor, None]
                    Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_coils, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'mask' : Union[torch.Tensor, None]
                    Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor.. Also, if
                    Unsupervised Learning methods, like Noise-to-Recon or SSDU, are used, then it is a list of
                    torch.Tensor with masks for each method. Shape [batch_size, 1, n_x, n_y, 1].
                'initial_prediction' : Union[torch.Tensor, None]
                    Initial prediction. Shape [batch_size, n_x, n_y, 2] or None.
                'target' : Union[torch.Tensor, None]
                    Target data. Shape [batch_size, n_x, n_y] or None.
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'acceleration' : float
                    Acceleration factor of the sampling mask.
                'attrs' : dict
                    Attributes dictionary.
        batch_idx : int
            Batch index.
        """
        sample = batch
        outputs = self.inference_step(
            sample["masked_waveform"],
            sample["mask"],
            sample["waveform"],
            sample["filename"],  # type: ignore
            sample["data_idx"],  # type: ignore
            sample["layout"],
            sample["attrs"],  # type: ignore
        )
        target = outputs["target"]
        predictions = outputs["predictions"]
        if self.update_in_frequency:
            target = fft1(target, time_dim=-1)

        if self.contrastive_loss:
            val_loss = self.__compute_loss__(
                target, predictions, sample["mask"], sample["attrs"], **outputs["contrastive"]
            )
        else:
            val_loss = self.__compute_loss__(target, predictions, sample["mask"], sample["attrs"])
        self.validation_step_outputs.append({"val_loss": val_loss})
        if self.update_in_frequency:
            target, predictions = self.__compute_time_domain(target, predictions)

        # Compute metrics and log them and log outputs.
        self.__compute_and_log_metrics_and_outputs__(
            target,
            predictions,
            outputs["attrs"],
            outputs["fname"],
            outputs["id"],
            outputs["layout"],
        )

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int):
        """Performs a test step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
                'kspace' : List of torch.Tensor
                    Fully-sampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2].
                'y' : Union[torch.Tensor, None]
                    Subsampled k-space data. If multiple accelerations are used, then it is a list of torch.Tensor.
                    Shape [batch_size, n_coils, n_x, n_y, 2].
                'sensitivity_maps' : torch.Tensor
                    Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
                'mask' : Union[torch.Tensor, None]
                    Sampling mask. If multiple accelerations are used, then it is a list of torch.Tensor.. Also, if
                    Unsupervised Learning methods, like Noise-to-Recon or SSDU, are used, then it is a list of
                    torch.Tensor with masks for each method. Shape [batch_size, 1, n_x, n_y, 1].
                'initial_prediction' : Union[torch.Tensor, None]
                    Initial prediction. Shape [batch_size, n_x, n_y, 2] or None.
                'target' : Union[torch.Tensor, None]
                    Target data. Shape [batch_size, n_x, n_y] or None.
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'acceleration' : float
                    Acceleration factor of the sampling mask.
                'attrs' : dict
                    Attributes dictionary.
        batch_idx : int
            Batch index.
        """
        sample = batch
        outputs = self.inference_step(
            sample["masked_waveform"],
            sample["mask"],
            sample["waveform"],
            sample["filename"],  # type: ignore
            sample["data_idx"],  # type: ignore
            sample["layout"],
            sample["attrs"],  # type: ignore
        )

        target = outputs["target"]
        predictions = outputs["predictions"]
        if self.update_in_frequency:
            target = fft1(target, time_dim=-1)

        if self.update_in_frequency:
            target, predictions = self.__compute_time_domain(target, predictions)

        # Compute metrics and log them and log outputs.
        self.__compute_and_log_metrics_and_outputs__(
            target,
            predictions,
            outputs["attrs"],
            outputs["fname"],
            outputs["id"],
            outputs["layout"],
        )

        if self.accumulate_predictions:
            predictions = parse_list_and_keep_last(predictions)

        # If "16" or "16-mixed" fp is used, ensure complex type will be supported when saving the predictions.
        predictions = predictions.detach().cpu().numpy()
        mask = sample["mask"].detach().cpu().numpy()
        for i in range(predictions.shape[0]):
            self.test_step_outputs.append(
                [
                    sample["filename"][i],
                    sample["layout"][i],
                    mask[i],
                    predictions[i],
                ]
            )

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch to aggregate outputs."""
        self.log("val_loss", torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean(), sync_dist=True)

        # Initialize metrics.
        mse_vals = defaultdict(dict)
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)
        for k, v in self.mse_vals.items():
            mse_vals[k].update(v)
        for k, v in self.nmse_vals.items():
            nmse_vals[k].update(v)
        for k, v in self.ssim_vals.items():
            ssim_vals[k].update(v)
        for k, v in self.psnr_vals.items():
            psnr_vals[k].update(v)

        # Parse metrics and log them.
        metrics = {
            "MSE": 0,
            "NMSE": 0,
            "SSIM": 0,
            "PSNR": 0,
        }
        local_examples = 0
        for fname in mse_vals:
            local_examples += 1
            metrics["MSE"] = metrics["MSE"] + torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            metrics["NMSE"] = metrics["NMSE"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals[fname].items()])
            )
            metrics["SSIM"] = metrics["SSIM"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["PSNR"] = metrics["PSNR"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["MSE"] = self.MSE(metrics["MSE"])
        metrics["NMSE"] = self.NMSE(metrics["NMSE"])
        metrics["SSIM"] = self.SSIM(metrics["SSIM"])
        metrics["PSNR"] = self.PSNR(metrics["PSNR"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        """Called at the end of test epoch to aggregate outputs, log metrics and save predictions."""
        # Initialize metrics.
        mse_vals = defaultdict(dict)
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)

        for k, v in self.mse_vals.items():
            mse_vals[k].update(v)
        for k, v in self.nmse_vals.items():
            nmse_vals[k].update(v)
        for k, v in self.ssim_vals.items():
            ssim_vals[k].update(v)
        for k, v in self.psnr_vals.items():
            psnr_vals[k].update(v)

        # apply means across image volumes
        metrics = {
            "MSE": 0,
            "NMSE": 0,
            "SSIM": 0,
            "PSNR": 0,
        }
        local_examples = 0
        for fname in mse_vals:
            local_examples += 1
            metrics["MSE"] = metrics["MSE"] + torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            metrics["NMSE"] = metrics["NMSE"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals[fname].items()])
            )
            metrics["SSIM"] = metrics["SSIM"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["PSNR"] = metrics["PSNR"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["MSE"] = self.MSE(metrics["MSE"])
        metrics["NMSE"] = self.NMSE(metrics["NMSE"])
        metrics["SSIM"] = self.SSIM(metrics["SSIM"])
        metrics["PSNR"] = self.PSNR(metrics["PSNR"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics.items():
            self.log(f"test_metrics/{metric}", value / tot_examples, prog_bar=True, sync_dist=True)
        if "wandb" in self.logger.__module__.lower():
            out_dir = Path(os.path.join(self.logger.save_dir, "reconstructions"))
        else:
            out_dir = Path(os.path.join(self.logger.log_dir, "reconstructions"))
        out_dir.mkdir(exist_ok=True, parents=True)

        # Save predictions.
        reconstructions = defaultdict(list)
        for filename, layout, mask, reconstructions in self.test_step_outputs:
            filename = os.path.join(filename)
            file_dir = os.path.join(out_dir, filename)
            os.makedirs(os.path.split(file_dir)[0], exist_ok=True)
            np.save(file_dir, reconstructions)
            if layout == "random":
                filename = os.path.join(filename.replace(".npy", ""), "_random_mask.npy")
                file_dir = os.path.join(out_dir, filename)
                os.makedirs(os.path.split(file_dir)[0], exist_ok=True)
                np.save(file_dir, mask)

    @staticmethod
    def _setup_dataloader_from_config(cfg: DictConfig) -> DataLoader:
        """Setups the dataloader from the configuration (yaml) file.

        Parameters
        ----------
        cfg : DictConfig
            Configuration file.

        Returns
        -------
        dataloader : torch.utils.data.DataLoader
            Dataloader.
        """
        # Get mask parameters.
        mask_args = cfg.get("mask_args", None)
        mask_type = mask_args.get("type", None)
        use_seed = mask_args.get("use_seed", False)
        mask_func = None

        accelerations = mask_args.get("accelerations", [1])
        if "random" in accelerations:
            low_ratio = mask_args.get("low_ratio", 0.1)
            high_ratio = mask_args.get("high_ratio", 0.5)
            min_block = mask_args.get("min_block", 500)
            mask_func = [
                create_masker(
                    mask_type_str=mask_type,
                    accelerations=accelerations,
                    low_ratio=low_ratio,
                    high_ratio=high_ratio,
                    min_block=min_block,
                )
            ]
        else:
            mask_func = [create_masker(mask_type_str=mask_type, accelerations=accelerations)]

        dataset_format = cfg.get("dataset_format", None)
        if dataset_format == "UniversalECGDataset":
            dataloader = UniversalECGDataset

        transform = cfg.get("transforms", None)
        transforms = []
        if transform:
            for key, value in transform.items():
                if key.lower() == "applygain":
                    transforms.append(ApplyGain())
                if key.lower() == "resample":
                    transforms.append(Resample(value[0]))
                if key.lower() == "totensor":
                    transforms.append(ToTensor())
                if key.lower() == "to12lead":
                    transforms.append(To12Lead())
                if key.lower() == "butterfilter":
                    transforms.append(
                        ButterFilter(lowcut=value["lowcut"], highcut=value["highcut"], order=value["order"])
                    )
            transforms.append(Masker(mask_func, use_seed=use_seed))
        if cfg.get("normalization_type", None):
            transforms.append(
                ECGNormalizer(
                    normalization_type=cfg.get("normalization_type"),
                )
            )

        # Get dataset.
        log_figures = cfg.get("log_figures", None)
        if cfg.get("params_json", None):
            params = json.load(open(cfg.get("params_json", None), "r", encoding="utf-8"))
            df = pd.read_csv((cfg.get("dataset")), low_memory=False)[: cfg.get("dataset_number_of_examples", 10)]
            subset1 = df[df['Center'] == 'UMCU']
            subset2 = df[df['Center'] == 'CZE']

            def load_age(ds: UniversalECGDataset, row):
                return {'age': torch.tensor(row['Age_scaled'], dtype=torch.float32)}

            def load_gender(ds: UniversalECGDataset, row):
                return {'gender': torch.tensor(row['Gender'], dtype=torch.float32)}

            additional_dataset_fn = [load_age, load_gender]

            dataset1 = dataloader(
                dataset_function=cfg.get("dataset_function"),
                waveform_dir=params["umcu_data_dir"],
                dataset=subset1,
                transform=Compose(transforms),
                labels=params["labels"],
                secondary_waveform_dir=cfg.get("secondary_waveform_dir", ""),
                additional_dataset_function=additional_dataset_fn,
                log_figures=log_figures,
            )
            dataset2 = dataloader(
                dataset_function="universal",
                waveform_dir=params["cze_data_dir"],
                dataset=subset2,
                transform=Compose(transforms),
                labels=params["labels"],
                secondary_waveform_dir=cfg.get("secondary_waveform_dir", ""),
                additional_dataset_function=additional_dataset_fn,
                log_figures=log_figures,
            )

            dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
        else:
            dataset = dataloader(
                dataset_function=cfg.get("dataset_function"),
                waveform_dir=cfg.get("waveform_dir"),
                dataset=pd.read_csv((cfg.get("dataset")))[: cfg.get("dataset_number_of_examples", 10)],
                transform=Compose(transforms),
                labels=cfg.get("labels", None),
                secondary_waveform_dir=cfg.get("secondary_waveform_dir", ""),
                additional_dataset_function=cfg.get("additional_dataset_function", None),
                log_figures=log_figures,
            )
        if cfg.shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.get("batch_size", 1),
            sampler=sampler,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
