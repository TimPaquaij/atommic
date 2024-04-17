# coding=utf-8
__author__ = "Dimitris Karkalousos"

import torch
from torch import nn
from typing import Optional
from atommic.core.classes.loss import Loss
from atommic.collections.common.parts.utils import is_none

class CrossEntropyLoss(Loss):
    """Wrapper around PyTorch's CrossEntropyLoss to support 2D and 3D inputs."""

    def __init__(
        self,
        num_samples: int = 50,
        ignore_index: int = -100,
        reduction: str = "none",
        label_smoothing: float = 0.0,
        weight: torch.Tensor = None,
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
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction



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

        if not is_none(self.weight):
            self.weight = torch.tensor(self.weight).to(_input)
        else:
            self.weight =None

        cross_entropy = torch.nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
        if self.mc_samples == 1 or pred_log_var is None:
            return cross_entropy(_input.float(), target.float()).mean()

        pred_shape = [self.mc_samples, *_input.shape]
        noise = torch.randn(pred_shape, device=_input.device)
        noisy_pred = _input.unsqueeze(0) + torch.sqrt(torch.exp(pred_log_var)).unsqueeze(0) * noise
        noisy_pred = noisy_pred.view(-1, *_input.shape[1:])
        tiled_target = target.unsqueeze(0).tile((self.mc_samples,)).view(-1, *target.shape[1:])
        loss = cross_entropy(noisy_pred, tiled_target).to(target).view(self.mc_samples, -1, *_input.shape[-2:]).mean(0)
        return loss.mean()

class BinaryCrossEntropy_with_logits_Loss(Loss):
    def __init__(self, num_samples: int = 50,weight: Optional[torch.Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                     pos_weight: Optional[torch.Tensor] = None) -> None:
        """Inits :class:`BinaryCrossEntropy_with_logits_Loss`.

        Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples to be broadcasted with target.
            Must be a tensor with equal size along the class dimension to the number of classes.
            Pay close attention to PyTorch's broadcasting semantics in order to achieve the desired
            operations. For a target of size [B, C, H, W] (where B is batch size) pos_weight of
            size [B, C, H, W] will apply different pos_weights to each element of the batch or
            [C, H, W] the same pos_weights across the batch. To apply the same positive weight
            along all spacial dimensions for a 2D multi-class target [C, H, W] use: [C, 1, 1].
            Default: ``None``
        """


        super().__init__()
        self.mc_samples = num_samples
        self.weight =  weight
        self.size_average = size_average
        self.reduce =reduce
        self.reduction=reduction
        self.pos_weight=pos_weight

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
        # if _input.dim() == 4:
        #     _input = _input.reshape(_input.shape[0]*_input.shape[1],*_input.shape[2:])
        #     _input = _input.permute((1,2,0))
        # if target.dim() == 4:
        #     target = target.reshape(target.shape[0]*target.shape[1],*target.shape[2:])
        #     target = target.permute((1, 2, 0))

        cross_entropy = torch.nn.BCEWithLogitsLoss(
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )
        if self.mc_samples == 1 or pred_log_var is None:
            return cross_entropy(_input.float(), target).mean()

        pred_shape = [self.mc_samples, *_input.shape]
        noise = torch.randn(pred_shape, device=_input.device)
        noisy_pred = _input.unsqueeze(0) + torch.sqrt(torch.exp(pred_log_var)).unsqueeze(0) * noise
        noisy_pred = noisy_pred.view(-1, *_input.shape[1:])
        tiled_target = target.unsqueeze(0).tile((self.mc_samples,)).view(-1, *target.shape[1:])
        loss = cross_entropy(noisy_pred, tiled_target).to(target).view(self.mc_samples, -1, *_input.shape[-2:]).mean(0)
        return loss.mean()