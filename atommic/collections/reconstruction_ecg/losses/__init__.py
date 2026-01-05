# coding=utf-8
__author__ = "Dimitris Karkalousos"

from atommic.collections.reconstruction_ecg.losses.na import NoiseAwareLoss  # noqa: F401
from atommic.collections.reconstruction_ecg.losses.ssim import SSIMLoss  # noqa: F401
from atommic.collections.reconstruction_ecg.losses.ml1 import MaskL1Loss  # noqa: F401
from atommic.collections.reconstruction_ecg.losses.huber import MaskHuberLoss  # noqa: F401
from atommic.collections.reconstruction_ecg.losses.STFT import MultiResolutionSTFTLoss  # noqa: F401
