# coding=utf-8
__author__ = "Tim Paquaij"

import torch
import torch.nn.functional as F

from atommic.core.classes.loss import Loss


class MaskL1Loss(Loss):
    """
    L1 loss with masked weighting.
    Mask entries equal to 0 are replaced by a configurable weight > 1.
    """

    def __init__(self, weight: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.weight = float(weight)
        self.reduction = reduction

    def forward(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        pred = pred.to(target.dtype)

        if mask is None:
            weighted_mask = torch.ones_like(target, dtype=target.dtype, device=target.device)
        else:
            mask = mask.to(target.dtype)
            weighted_mask = torch.where(
                mask == 1,
                torch.tensor(1.0, dtype=target.dtype, device=target.device),
                torch.tensor(self.weight, dtype=target.dtype, device=target.device),
            )

        diff = torch.abs(pred - target)
        loss = diff * weighted_mask

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss

        raise ValueError(f"Unknown reduction: {self.reduction}")
