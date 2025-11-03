# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
import torch.nn.functional as F


def log_likelihood_gradient_ecg(
    prediction: torch.Tensor,  # [batch, leads]
    measured_ecg: torch.Tensor,  # [batch, leads]
    mask: torch.Tensor,  # [batch, leads], 1 for observed leads
    sigma: float,
) -> torch.Tensor:
    """
    Compute the gradient of the log-likelihood for ECG lead reconstruction.
    Missing leads are excluded using the mask.
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.item()
    sigma = max(sigma, 1.0)

    # Observed gradient (data consistency)
    gradients = (prediction - measured_ecg) * mask / sigma

    # Total gradient
    return torch.cat([prediction,gradients], dim=1)