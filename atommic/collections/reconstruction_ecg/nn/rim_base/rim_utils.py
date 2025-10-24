# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import Sequence

import torch


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
    # Ensure noise scalar
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.item()
    sigma = max(sigma, 1e-6)

    # Compute residual only for observed leads
    residual = mask * (prediction - measured_ecg)

    # Gradient of log-likelihood under Gaussian noise
    gradient = residual / (sigma**2)

    # Return prediction + gradient for iterative methods
    return gradient
