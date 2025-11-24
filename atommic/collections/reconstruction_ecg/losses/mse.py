# coding=utf-8
__author__ = "Tim Paquaij"

import torch
import torch.nn.functional as F

from atommic.core.classes.loss import Loss
from atommic.collections.common.parts.fft import ifft1
from typing import Optional


class MaskMSELoss(Loss):
    """
    MSE loss with masked weighting.
    Visible region (mask=1) → weight = 1
    Synthesised region (mask=0) → weight = `weight` (> 1)
    """

    def __init__(
        self,
        weight: float = 1.0,
        reduction: str = "mean",
        amplitude_weight: Optional[float] = None,
        spectral: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.reduction = reduction
        self.amplitude_weight = amplitude_weight
        self.spectral = spectral

    def forward(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        pred = pred.to(target.dtype)

        if self.spectral is True:
            if self.update_in_frequency:
                pred = ifft1(pred, time_dim=-1)[..., 0]
                target = ifft1(target, time_dim=-1)[..., 0]

            else:
                pred = torch.abs(torch.fft.rfft(pred, norm="ortho", dim=-1))
                target = torch.abs(torch.fft.rfft(target, norm="ortho", dim=-1))

        if mask is None or self.spectral is True:
            weighted_mask = torch.ones_like(target, dtype=target.dtype, device=target.device)
        else:
            mask = mask.to(target.dtype)
            weighted_mask = torch.where(
                mask == 1,
                torch.tensor(1.0, dtype=target.dtype, device=target.device),
                torch.tensor(self.weight, dtype=target.dtype, device=target.device),
            )
        if self.amplitude_weight:
            weighted_mask = torch.where(
                target.abs() <= 0.5,
                weighted_mask,
                torch.tensor(float(self.amplitude_weight), dtype=target.dtype, device=target.device),
            )

        # Standard MSE per element
        diff = pred - target
        mse = diff * diff

        # Apply mask weight
        loss = mse * weighted_mask

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss

        raise ValueError(f"Unknown reduction: {self.reduction}")
