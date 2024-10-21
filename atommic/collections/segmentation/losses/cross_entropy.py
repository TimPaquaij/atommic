# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
from atommic.core.classes.loss import Loss
from torch import nn
import warnings
from atommic.collections.segmentation.losses.utils import one_hot

class CategoricalCrossEntropyLoss(Loss):
    """Wrapper around PyTorch's CrossEntropyLoss to support 2D and 3D inputs."""

    def __init__(
        self,
        num_samples: int = 50,
        ignore_index: int = -100,
        reduction: str = "none",
        label_smoothing: float = 0.0,
        weight: torch.Tensor = None,
        to_onehot_y: bool = False,
        include_background: bool = True,
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
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.cross_entropy = torch.nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

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

        n_pred_ch = _input.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                _input = _input[:, 1:]
        
        
        self.cross_entropy.weight = self.cross_entropy.weight.clone().to(_input.device) if self.cross_entropy.weight is not None else None

        if self.mc_samples == 1 or pred_log_var is None:
            return self.cross_entropy(_input.float(), target)

        pred_shape = [self.mc_samples, *_input.shape]
        noise = torch.randn(pred_shape, device=_input.device)
        noisy_pred = _input.unsqueeze(0) + torch.sqrt(torch.exp(pred_log_var)).unsqueeze(0) * noise
        noisy_pred = noisy_pred.view(-1, *_input.shape[1:])
        tiled_target = target.unsqueeze(0).tile((self.mc_samples,)).view(-1, *target.shape[1:])
        loss = self.cross_entropy(noisy_pred, tiled_target).view(self.mc_samples, -1, *_input.shape[-2:]).mean(0)
        return loss


class BinaryCrossEntropyLoss(Loss):
    """Wrapper around PyTorch's BinaryCrossEntropyLoss to support 2D and 3D inputs."""

    def __init__(
        self,
        num_samples: int = 50,
        weight: torch.Tensor = None,
        size_average:bool =True,
        reduce: bool =False,
        reduction:str = 'none',
    ):
        """Inits :class:`BinaryCrossEntropyLoss`.

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
        self.mc_samples=num_samples,
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.binary_cross_entropy = torch.nn.BCELoss(weight=self.weight, size_average=self.size_average, reduce=self.reduce, reduction=self.reduction)

    def forward(self, target: torch.Tensor, _input: torch.Tensor, pred_log_var: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of :class:`BinaryCrossEntropyLoss`.

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
        self.weight = self.weight.clone().to(_input.device) if self.weight is not None else None

        if self.mc_samples == 1 or pred_log_var is None:
            return self.binary_cross_entropy(_input.float(), target)

        pred_shape = [self.mc_samples, *_input.shape]
        noise = torch.randn(pred_shape, device=_input.device)
        noisy_pred = _input.unsqueeze(0) + torch.sqrt(torch.exp(pred_log_var)).unsqueeze(0) * noise
        noisy_pred = noisy_pred.view(-1, *_input.shape[1:])
        tiled_target = target.unsqueeze(0).tile((self.mc_samples,)).view(-1, *target.shape[1:])
        loss = self.binary_cross_entropy(noisy_pred, tiled_target).view(self.mc_samples, -1, *_input.shape[-2:]).mean(0)
        return loss