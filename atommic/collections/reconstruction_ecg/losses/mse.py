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
        mask: torch.Tensor | None = None,
        complex_df: dict |None = None,
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
            p_mask = self.onset_offset_to_mask_batch(onsets=complex_df["p_onset"],offsets=complex_df["p_offset"],num_leads=pred.size(1), length=pred.size(2))
            qrs_mask = self.onset_offset_to_mask_batch(onsets=complex_df["qrs_onset"],offsets=complex_df["qrs_offset"],num_leads=pred.size(1), length=pred.size(2))
            t_mask = self.onset_offset_to_mask_batch(onsets=complex_df["t_onset"],offsets=complex_df["t_offset"],num_leads=pred.size(1), length=pred.size(2))
            weighted_mask = weighted_mask + float(self.amplitude_weight) * (p_mask + qrs_mask + t_mask)
            weighted_mask = weighted_mask.clamp(max=self.amplitude_weight + self.weight)

        # Standard MSE per element
        diff = pred - target
        mse = diff * diff

        # Apply mask weight
        loss = mse * weighted_mask

        # Reduction
        if self.reduction == "mean":
            if loss.ndim > 1:
                loss = loss.mean(tuple(range(1, loss.ndim)))
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss

        raise ValueError(f"Unknown reduction: {self.reduction}")
    

    def onset_offset_to_mask_batch(self,
        onsets: torch.Tensor,
        offsets: torch.Tensor,
        num_leads: int = 12,
        length: int = 5000,
    ) -> torch.Tensor:
        """
        Convert batched onset/offset tensors to binary masks.

        Parameters
        ----------
        onsets : torch.Tensor
            Shape [B, N], NaN or int indices
        offsets : torch.Tensor
            Shape [B, N], NaN or int indices
        num_leads : int
            Number of ECG leads
        length : int
            Number of samples

        Returns
        -------
        mask : torch.Tensor
            Shape [B, num_leads, length], dtype uint8
        """
        device = onsets.device
        B = onsets.shape[0]

        # ---- difference array per batch ----
        diff = torch.zeros(B, length + 1, device=device, dtype=torch.int32)

        valid = (~torch.isnan(onsets)) & (~torch.isnan(offsets))
        if not valid.any():
            return torch.zeros(B, num_leads, length, device=device, dtype=torch.uint8)

        on = onsets.clone()
        off = offsets.clone()

        on[~valid] = 0
        off[~valid] = 0

        on = on.long().clamp(0, length)
        off = off.long().clamp(0, length)

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(on)

        diff.index_put_(
            (batch_idx, on),
            torch.ones_like(on, dtype=diff.dtype),
            accumulate=True,
        )
        diff.index_put_(
            (batch_idx, off),
            -torch.ones_like(off, dtype=diff.dtype),
            accumulate=True,
        )

        time_mask = torch.cumsum(diff[:, :-1], dim=1) > 0   # [B, T]

        # ---- broadcast to leads ----
        mask = time_mask.unsqueeze(1).expand(B, num_leads, length)

        return mask.to(torch.uint8)
