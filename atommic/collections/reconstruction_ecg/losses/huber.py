__author__ = "Tim Paquaij"

import torch
import torch.nn.functional as F

from atommic.core.classes.loss import Loss


class MaskHuberLoss(Loss):
    """
    Masked Huber loss for ECG reconstruction.

    mask = 1  → visible region  → weight 1
    mask = 0  → synthesised region → weight = missing_weight

    Huber rule:
        if |x| <= delta: 0.5 * x^2
        else: delta * (|x| - 0.5 * delta)
    """

    def __init__(
        self,
        delta: float = 0.05,
        weight: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.delta = float(delta)
        self.weight = float(weight)
        self.reduction = reduction

    def forward(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor | None = None,
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

        diff = pred - target
        abs_diff = diff.abs()

        quadratic = 0.5 * diff.pow(2)
        linear = self.delta * (abs_diff - 0.5 * self.delta)

        huber = torch.where(abs_diff <= self.delta, quadratic, linear)

        loss = huber * weighted_mask
        if self.reduction == "mean":
            return loss.sum() / weighted_mask.sum()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss

        raise ValueError(f"Unknown reduction: {self.reduction}")
