# coding=utf-8
__author__ = "Tim Paquaij"

import torch
import torch.nn.functional as F
from typing import Optional
from atommic.core.classes.loss import Loss
from atommic.collections.common.parts.fft import ifft1


class MaskL1Loss(Loss):
    """
    L1 loss with masked weighting.
    Mask entries equal to 0 are replaced by a configurable weight > 1.
    """

    def __init__(
        self,
        weight: float = 2.0,
        reduction: str = "mean",
        amplitude_weight: Optional[float] = None,
        spectral: Optional[bool] = False,
        update_in_frequency: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.reduction = reduction
        self.amplitude_weight = amplitude_weight
        self.spectral = spectral
        self.update_in_frequency = update_in_frequency

    def forward(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        pred = pred.to(target.dtype)
        if self.spectral is True and not self.update_in_frequency:
            pred = torch.abs(torch.fft.rfft(pred, norm="ortho", dim=-1))
            target = torch.abs(torch.fft.rfft(target, norm="ortho", dim=-1))

        if self.update_in_frequency and not self.spectral:
            pred = ifft1(pred, time_dim=-1)[..., 0]
            target = ifft1(target, time_dim=-1)[..., 0]
            mask = mask[..., 0]

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
            amplitude_boost = (target.abs() - 0.5).clamp(min=0.0)
            weighted_mask = weighted_mask + float(self.amplitude_weight) * amplitude_boost
            weighted_mask = weighted_mask.clamp(max=self.amplitude_weight)

        diff = torch.abs(pred - target)
        loss = diff * weighted_mask

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss

        raise ValueError(f"Unknown reduction: {self.reduction}")
