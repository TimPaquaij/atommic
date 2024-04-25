# coding=utf-8
__author__ = "Tim Paquaij"

import torch
from torch import nn
from typing import Optional
from atommic.core.classes.loss import Loss
from atommic.collections.common.parts.utils import is_none
from torch.functional import F
from typing import Sequence
from torch import Tensor

class FocalLoss(Loss):
    """Wrapper around PyTorch's CrossEntropyLoss to support 2D and 3D inputs. Adapted to Focal Loss based on: https://arxiv.org/abs/1708.02002"""

    def __init__(
        self,
        num_samples: int = 50,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        alpha: float = 1,
        gamma: float = 2,
    ):
        """Inits :class:`CrossEntropyLoss`.

        Parameters
        ----------
        num_samples : int, optional
            Number of Monte Carlo samples, by default 50
        ignore_index : int, optional
            Index to ignore, by default -100
        reduction : str, optional
            Reduction method, by default "none"
        label_smoothing : float, optional
            Label smoothing, by default 0.0
        weight : torch.Tensor, optional
            Weight for each class, by default None
        """
        super().__init__()
        self.mc_samples = num_samples
        self.ignore_index =ignore_index
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.gamma = gamma



    def forward(self, target: torch.Tensor, _input: torch.Tensor, pred_log_var: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of :class:`CrossEntropyLoss`.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor. Shape: (batch_size, num_classes, *spatial_dims)
        _input : torch.Tensor
            Prediction tensor. Shape: (batch_size, num_classes, *spatial_dims)
        pred_log_var : torch.Tensor, optional
            Prediction log variance tensor. Shape: (batch_size, num_classes, *spatial_dims). Default is ``None``.

        Returns
        -------
        torch.Tensor
            Loss tensor. Shape: (batch_size, *spatial_dims)
        """
        # In case we do not have a batch dimension, add it
        if _input.dim() == 3:
            _input = _input.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        cross_entropy = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
        if self.mc_samples == 1 or pred_log_var is None:
            ce = cross_entropy(_input.float(), target.float())
            pt = torch.exp(-ce)
            loss = self.alpha*((1-pt)**self.gamma)*ce
            return loss.mean()

        pred_shape = [self.mc_samples, *_input.shape]
        noise = torch.randn(pred_shape, device=_input.device)
        noisy_pred = _input.unsqueeze(0) + torch.sqrt(torch.exp(pred_log_var)).unsqueeze(0) * noise
        noisy_pred = noisy_pred.view(-1, *_input.shape[1:])
        tiled_target = target.unsqueeze(0).tile((self.mc_samples,)).view(-1, *target.shape[1:])
        ce = cross_entropy(noisy_pred, tiled_target).to(target).view(self.mc_samples, -1, *_input.shape[-2:]).mean(0)
        pt = torch.exp(-ce)
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce

        return loss.mean()
