# coding=utf-8
__author__ = "Tim Paquaij"

# Taken and adapted from: https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/dice.py

import warnings
from typing import Any, Callable, List, Optional, Tuple, Union
from atommic.collections.segmentation.losses.utils import do_metric_reduction
import numpy as np
import torch
from torch import Tensor

from atommic.collections.common.parts.utils import is_none
from atommic.collections.segmentation.losses.dice import one_hot
from atommic.core.classes.loss import Loss

class GeneralizedDiceLoss(Loss):
    """
    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """


    def __init__(
            self,
            include_background: bool = False,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = True,
            other_act: Optional[Callable] = None,
            w_type: str = "square",
            squared_pred: bool = False,
            jaccard: bool = False,
            flatten: bool = False,
            reduction: str = "mean_batch",
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = True,

    ):
        """Inits :class:`Dice`.

        Parameters
        ----------
        include_background : bool
            whether to skip Dice computation on the first channel of the predicted output. Default is ``True``.
        to_onehot_y : bool
            Whether to convert `y` into the one-hot format. Default is ``False``.
        sigmoid : bool
            Whether to add sigmoid function to the input data. Default is ``True``.
        softmax : bool
            Whether to add softmax function to the input data. Default is ``False``.
        other_act : Callable
            Use this parameter if you want to apply another type of activation layer. Default is ``None``.
        squared_pred : bool
            Whether to square the prediction before calculating Dice. Default is ``False``.
        jaccard : bool
            Whether to compute Jaccard Index as a loss. Default is ``False``.
        flatten : bool
            Whether to flatten input data. Default is ``False``.
        reduction : str
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied.
            'mean': the sum of the output will be divided by the number of elements in the output.
            'sum': the output will be summed.
            Default is ``mean``.
        smooth_nr : float
            A small constant added to the numerator to avoid `nan` when all items are 0. Default is ``1e-5``.
        smooth_dr : float
            A small constant added to the denominator to avoid `nan` when all items are 0. Default is ``1e-5``.
        batch : bool
            If True, compute Dice loss for each batch and return a tensor with shape (batch_size,).
            If False, compute Dice loss for the whole batch and return a tensor with shape (1,).
            Default is ``True``.
        """
        super().__init__()
        other_act = None if is_none(other_act) else other_act
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.flatten = flatten
        self.reduction = reduction
        self.w_type =w_type
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def w_func(self, grnd):
        if self.w_type == "simple":
            grnd = 1/grnd
            return grnd

        if self.w_type == "square":
            grnd= 1/(grnd * grnd)
            return grnd
        return torch.ones_like(grnd).to(grnd)




    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if isinstance(input, np.ndarray):
         input = torch.from_numpy(input)

        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        if self.flatten:
            if target.dim() == 4:
                segmentation_classes_dim = 1
            else:
                segmentation_classes_dim = 0
            target = target.reshape(target.shape[segmentation_classes_dim], 1, -1)
            input = input.reshape(input.shape[segmentation_classes_dim], 1, -1)

        if self.sigmoid:
            input = torch.sigmoid(input).to(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1).to(input)

        if self.other_act is not None:
            input = self.other_act(input)

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
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis
        intersection = torch.sum(target * input, reduce_axis)

        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(input, reduce_axis)

        denominator = ground_o + pred_o
        torch.autograd.set_detect_anomaly(True)
        w = self.w_func(ground_o.float())
        infs = torch.isinf(w)
        if self.batch:
            w[infs] = 0.0
            w = w + infs * torch.max(w)
        else:
            w[infs] = 0.0
            max_values = torch.max(w, dim=1)[0].unsqueeze(dim=1)
            w = w + infs * max_values

        final_reduce_dim = 0 if self.batch else 1
        numer = 2.0 * (intersection * w).sum(final_reduce_dim, keepdim=True) + self.smooth_nr
        denom = (denominator * w).sum(final_reduce_dim, keepdim=True) + self.smooth_dr
        dice_score, _ = do_metric_reduction((numer / denom), reduction=self.reduction)
        f: torch.Tensor = 1.0 - dice_score

        return dice_score, f