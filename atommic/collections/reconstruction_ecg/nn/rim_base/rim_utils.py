# coding=utf-8
__author__ = "Dimitris Karkalousos"

import math
import torch
import torch.nn.functional as F
from atommic.collections.common.parts.fft import fft1, ifft1
from typing import Optional


def log_likelihood_gradient_ecg(
    prediction: torch.Tensor,  # [batch, leads, time] or [batch, 1, leads, time] for 2D conv
    measured_ecg: torch.Tensor,  # [batch, leads, time] or [batch, 1, leads, time] for 2D conv
    mask: torch.Tensor,  # [batch, leads, time] or [batch, 1, leads, time] for 2D conv
    sigma: float,
    update_in_frequency: Optional[bool] = False,
    hexad_inform: Optional[bool] = False,
) -> torch.Tensor:
    """
    Compute the gradient of the log-likelihood for ECG lead reconstruction.
    Missing leads are excluded using the mask.
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.item()
    sigma = max(sigma, 1.0)

    # Observed gradient (data consistency)
    if update_in_frequency and prediction.dim() == 4:
        prediction = ifft1(prediction, time_dim=-1)  # freg to time
        measured_ecg = fft1(measured_ecg, time_dim=-1)  # time to freq
        measured_ecg = ifft1(measured_ecg, time_dim=-1)  # freq to time
        mask_gradients = (prediction - measured_ecg) * mask.unsqueeze(-1) / sigma
        if prediction.shape[1] == 12:
            pred_gradients = torch.zeros_like(prediction)
            pred_gradients[..., 2, :, :] = prediction[..., 2, :, :] - (prediction[..., 1, :, :] - prediction[..., 0, :, :])
            pred_gradients[..., 3, :, :] = prediction[..., 3, :, :] - (
                -(prediction[..., 0, :, :] + prediction[..., 1, :, :]) / 2
            )
            pred_gradients[..., 4, :, :] = prediction[..., 4, :, :] - (
                prediction[..., 0, :, :] - (prediction[..., 1, :, :] / 2)
            )
            pred_gradients[..., 5, :, :] = prediction[..., 5, :, :] - (
                prediction[..., 1, :, :] - (prediction[..., 0, :, :] / 2)
            )
            if hexad_inform is True:
                pred_gradients[..., 7, :, :] = (
                    prediction[..., 8, :, :] - 2 * prediction[..., 7, :, :] + prediction[..., 6, :, :]
                )
                pred_gradients[..., 8, :, :] = (
                    prediction[..., 9, :, :] - 2 * prediction[..., 8, :, :] + prediction[..., 7, :, :]
                )
                pred_gradients[..., 9, :, :] = (
                    prediction[..., 10, :, :] - 2 * prediction[..., 9, :, :] + prediction[..., 8, :, :]
                )
                pred_gradients[..., 10, :, :] = (
                    prediction[..., 11, :, :] - 2 * prediction[..., 10, :, :] + prediction[..., 9, :, :]
                )
        else:
            pred_gradients = None
    else:
        mask_gradients = (prediction - measured_ecg) * mask / sigma
        if prediction.shape[1] == 12:
            pred_gradients = torch.zeros_like(prediction)
            pred_gradients[..., 2, :] = prediction[..., 2, :] - (prediction[..., 1, :] - prediction[..., 0, :])
            pred_gradients[..., 3, :] = prediction[..., 3, :] - (-(prediction[..., 0, :] + prediction[..., 1, :]) / 2)
            pred_gradients[..., 4, :] = prediction[..., 4, :] - (prediction[..., 0, :] - (prediction[..., 1, :] / 2))
            pred_gradients[..., 5, :] = prediction[..., 5, :] - (prediction[..., 1, :] - (prediction[..., 0, :] / 2))
            if hexad_inform is True:
                pred_gradients[..., 7, :] = prediction[..., 8, :] - 2 * prediction[..., 7, :] + prediction[..., 6, :]
                pred_gradients[..., 8, :] = prediction[..., 9, :] - 2 * prediction[..., 8, :] + prediction[..., 7, :]
                pred_gradients[..., 9, :] = prediction[..., 10, :] - 2 * prediction[..., 9, :] + prediction[..., 8, :]
                pred_gradients[..., 10, :] = prediction[..., 11, :] - 2 * prediction[..., 10, :] + prediction[..., 9, :]
        else:
            pred_gradients = None
        
    if update_in_frequency:
        mask_gradients = fft1(mask_gradients, time_dim=-1)  # Time to freq
        if pred_gradients is not None:
            pred_gradients = fft1(pred_gradients, time_dim=-1)  # Time to freq
        prediction = fft1(prediction, time_dim=-1)  # Time to freq
        if pred_gradients is not None:
            gradients = torch.cat([prediction, mask_gradients, pred_gradients], dim=-1).permute(0, 3, 1, 2)
        else:
            gradients = torch.cat([prediction, mask_gradients], dim=-1).permute(0, 3, 1, 2)

    else:
        if prediction.dim() == 4:
            if pred_gradients is not None:
                gradients = torch.cat([prediction, mask_gradients, pred_gradients], dim=-1).permute(0, 3, 1, 2)
            else:
                gradients = torch.cat([prediction, mask_gradients], dim=-1).permute(0, 3, 1, 2)
        else:
            if pred_gradients is not None:
                gradients = torch.cat([prediction, mask_gradients, pred_gradients], dim=1)
            else:
                gradients = torch.cat([prediction, mask_gradients], dim=1)

    # Total gradient
    return gradients
