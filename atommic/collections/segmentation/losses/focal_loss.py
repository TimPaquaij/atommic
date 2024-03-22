# coding=utf-8
__author__ = "Tim Paquaij"


from atommic.core.classes.loss import Loss
from typing import Optional, Sequence
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


# class FocalLoss(Loss):
#     """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
#
#     It is essentially an enhancement to cross entropy loss and is
#     useful for classification tasks when there is a large class imbalance.
#     x is expected to contain raw, unnormalized scores for each class.
#     y is expected to contain class labels.
#
#     Shape:
#         - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
#         - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
#     """
#
#     def __init__(self,
#                  alpha: Optional[Tensor] = torch.tensor([0.5,1,1,1,1]),
#                  gamma: float = 2,
#                  reduction: str = 'mean',
#                  ignore_index: int = -100):
#         """Constructor.
#
#         Args:
#             alpha (Tensor, optional): Weights for each class. Defaults to None.
#             gamma (float, optional): A constant, as described in the paper.
#                 Defaults to 0.
#             reduction (str, optional): 'mean', 'sum' or 'none'.
#                 Defaults to 'mean'.
#             ignore_index (int, optional): class label to ignore.
#                 Defaults to -100.
#         """
#         if reduction not in ('mean', 'sum', 'none'):
#             raise ValueError(
#                 'Reduction must be one of: "mean", "sum", "none".')
#
#         super().__init__()
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.register_buffer("alpha",alpha)
#
#
#
#     def __repr__(self):
#         arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
#         arg_vals = [self.__dict__[k] for k in arg_keys]
#         arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
#         arg_str = ', '.join(arg_strs)
#         return f'{type(self).__name__}({arg_str})'
#
#     def forward(self, x: Tensor, y: Tensor) -> Tensor:
#         if y.dim() == x.dim():
#             y = torch.argmax(y,dim=1)
#         if x.ndim > 2:
#             # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
#             c = x.shape[1]
#             x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
#             # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
#             y = y.view(-1)
#         self.nll_loss = nn.NLLLoss(
#             weight=self.alpha.to(y), reduction='none', ignore_index=self.ignore_index)
#         unignored_mask = y != self.ignore_index
#         y = y[unignored_mask]
#         if len(y) == 0:
#             return torch.tensor(0.)
#         x = x[unignored_mask]
#         print(x,y)
#         # compute weighted cross entropy term: -alpha * log(pt)
#         # (alpha is already part of self.nll_loss)
#         log_p = F.log_softmax(x, dim=-1)
#         ce = self.nll_loss(log_p, y)
#
#         # get true class column from each row
#         all_rows = torch.arange(len(x))
#         log_pt = log_p[all_rows, y]
#
#         # compute focal term: (1 - pt)^gamma
#         pt = log_pt.exp()
#         focal_term = (1 - pt)**self.gamma
#
#         # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
#         loss = focal_term * ce
#
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'sum':
#             loss = loss.sum()
#
#         return loss



class FocalLoss(Loss):
    """Wrapper around PyTorch's CrossEntropyLoss to support 2D and 3D inputs."""

    def __init__(
        self,
        num_samples: int = 50,
        ignore_index: int = -100,
        reduction: str = "none",
        label_smoothing: float = 0.0,
        weight: torch.Tensor = torch.tensor([0.5,1,1,1,1]),
        gamma: float = 2,
        alpha: float = 0.25,
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
        self.gamma =gamma
        self.alpha =alpha



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
        #target = torch.argmax(target,dim=1)
        cross_entropy = torch.nn.CrossEntropyLoss(
            weight=self.weight.to(_input),
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

        if self.mc_samples == 1 or pred_log_var is None:
            ce_loss = cross_entropy(_input.float(), target.float())
            pt = torch.exp(-ce_loss)
            focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


            return focal_loss

        pred_shape = [self.mc_samples, *_input.shape]
        noise = torch.randn(pred_shape, device=_input.device)
        noisy_pred = _input.unsqueeze(0) + torch.sqrt(torch.exp(pred_log_var)).unsqueeze(0) * noise
        noisy_pred = noisy_pred.view(-1, *_input.shape[1:])
        tiled_target = target.unsqueeze(0).tile((self.mc_samples,)).view(-1, *target.shape[1:])
        loss = cross_entropy(noisy_pred, tiled_target).to(target).view(self.mc_samples, -1, *_input.shape[-2:]).mean(0)
        return loss.mean()