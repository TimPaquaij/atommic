# coding=utf-8
__author__ = "Dimitris Karkalousos"

from atommic.collections.common.losses.aggregator import AggregatorLoss  # noqa: F401
from atommic.collections.common.losses.wasserstein import SinkhornDistance  # noqa: F401

VALID_RECONSTRUCTION_LOSSES = [
    "l1",
    "masked_l1",
    "masked_mse",
    "masked_huber",
    "spectral_l1",
    "spectral_mse",
    "spectral_huber",
    "mse",
    "ssim",
    "noise_aware",
    "wasserstein",
]
VALID_SEGMENTATION_LOSSES = ["cross_entropy", "dice"]
