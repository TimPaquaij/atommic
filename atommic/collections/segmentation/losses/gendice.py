# coding=utf-8
__author__ = "Tim Paquaij"


# Taken and adapted from: https://docs.monai.io/en/stable/losses.html#generalizeddiceloss


import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from atommic.collections.common.parts.utils import is_none
from atommic.collections.segmentation.losses.utils import do_metric_reduction, one_hot
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
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = True,
        other_act: Callable | None = None,
        w_type: str = "square",
        reduction: str='mean',
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = True,
    ) -> None:
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            w_type: {``"square"``, ``"simple"``, ``"uniform"``}
                Type of function to transform ground truth volume to a weight factor. Defaults to ``"square"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, intersection over union is computed from each item in the batch.
                If True, the class-weighted intersection and union areas are first summed across the batches.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__()
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.reduction = reduction
        self.w_type = w_type

        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def w_func(self, grnd):
        if self.w_type == "simple":
            return torch.reciprocal(grnd)
        if self.w_type == "square":
            return torch.reciprocal(grnd * grnd)
        return torch.ones_like(grnd)

    def forward(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

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
        gendice_score= (numer / denom)
        gendice_score = torch.where(denominator > 0, gendice_score, torch.tensor(1.0).to(pred_o.device))
        gendice_score, _ = do_metric_reduction(gendice_score, reduction=self.reduction)
        f: torch.Tensor = 1.0 - gendice_score

        return gendice_score,f