# coding=utf-8
__author__ = "Dimitris Karkalousos"

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
        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        pred = pred.to(target.dtype)

        if mask is None:
            mask = torch.ones_like(target)
        else:
            mask = mask.to(target.dtype)

        # Replace zero entries with a constant multiplicative weight
        # The mask now becomes an element-wise weight matrix
        weighted_mask = torch.where(mask == 0, 
                                    torch.tensor(self.weight, dtype=mask.dtype, device=mask.device),
                                    mask)

        diff = torch.abs(pred - target)
        loss = diff * weighted_mask

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss

        raise ValueError(f"Unknown reduction: {self.reduction}")