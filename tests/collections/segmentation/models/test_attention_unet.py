# coding=utf-8
__author__ = "Dimitris Karkalousos"

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from atommic.collections.common.data.subsample import Random1DMaskFunc
from atommic.collections.common.parts import utils
from atommic.collections.segmentation.nn.attentionunet import SegmentationAttentionUNet
from tests.collections.reconstruction.mri_data.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, dimensionality, segmentation_classes, trainer",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "use_reconstruction_module": False,
                "segmentation_module": "AttentionUNet",
                "segmentation_module_input_channels": 1,
                "segmentation_module_output_channels": 4,
                "segmentation_module_channels": 64,
                "segmentation_module_pooling_layers": 2,
                "segmentation_module_dropout": 0.0,
                "segmentation_loss": {"dice": 1.0},
                "dice_loss_include_background": False,
                "dice_loss_to_onehot_y": False,
                "dice_loss_sigmoid": True,
                "dice_loss_softmax": False,
                "dice_loss_other_act": None,
                "dice_loss_squared_pred": False,
                "dice_loss_jaccard": False,
                "dice_loss_reduction": "mean",
                "dice_loss_smooth_nr": 1,
                "dice_loss_smooth_dr": 1,
                "dice_loss_batch": True,
                "consecutive_slices": 5,
                "coil_combination_method": "SENSE",
                "magnitude_input": True,
                "use_sens_net": False,
                "fft_centered": False,
                "fft_normalization": "backward",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
            2,
            4,
            {
                "strategy": "ddp",
                "accelerator": "cpu",
                "num_nodes": 1,
                "max_epochs": 20,
                "precision": 32,
                "enable_checkpointing": False,
                "logger": False,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": -1,
                "max_steps": -1,
            },
        ),
        (
            [1, 13, 45, 45, 2],
            {
                "use_reconstruction_module": False,
                "segmentation_module": "AttentionUNet",
                "segmentation_module_input_channels": 2,
                "segmentation_module_output_channels": 4,
                "segmentation_module_channels": 64,
                "segmentation_module_pooling_layers": 2,
                "segmentation_module_dropout": 0.0,
                "segmentation_loss": {"dice": 1.0},
                "dice_loss_include_background": False,
                "dice_loss_to_onehot_y": False,
                "dice_loss_sigmoid": True,
                "dice_loss_softmax": False,
                "dice_loss_other_act": None,
                "dice_loss_squared_pred": False,
                "dice_loss_jaccard": False,
                "dice_loss_reduction": "mean",
                "dice_loss_smooth_nr": 1,
                "dice_loss_smooth_dr": 1,
                "dice_loss_batch": True,
                "consecutive_slices": 1,
                "coil_combination_method": "SENSE",
                "magnitude_input": False,
                "use_sens_net": False,
                "fft_centered": False,
                "fft_normalization": "backward",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
            2,
            4,
            {
                "strategy": "ddp",
                "accelerator": "cpu",
                "num_nodes": 1,
                "max_epochs": 20,
                "precision": 32,
                "enable_checkpointing": False,
                "logger": False,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": -1,
                "max_steps": -1,
            },
        ),
        (
            [1, 3, 32, 16, 2],
            {
                "use_reconstruction_module": False,
                "segmentation_module": "AttentionUNet",
                "segmentation_module_input_channels": 1,
                "segmentation_module_output_channels": 4,
                "segmentation_module_channels": 64,
                "segmentation_module_pooling_layers": 2,
                "segmentation_module_dropout": 0.0,
                "segmentation_loss": {"dice": 1.0},
                "dice_loss_include_background": False,
                "dice_loss_to_onehot_y": False,
                "dice_loss_sigmoid": True,
                "dice_loss_softmax": False,
                "dice_loss_other_act": None,
                "dice_loss_squared_pred": False,
                "dice_loss_jaccard": False,
                "dice_loss_reduction": "mean",
                "dice_loss_smooth_nr": 1,
                "dice_loss_smooth_dr": 1,
                "dice_loss_batch": True,
                "consecutive_slices": 1,
                "coil_combination_method": "SENSE",
                "magnitude_input": True,
                "use_sens_net": False,
                "fft_centered": False,
                "fft_normalization": "backward",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
            2,
            4,
            {
                "strategy": "ddp",
                "accelerator": "cpu",
                "num_nodes": 1,
                "max_epochs": 20,
                "precision": 32,
                "enable_checkpointing": False,
                "logger": False,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": -1,
                "max_steps": -1,
            },
        ),
        (
            [1, 13, 45, 45, 2],
            {
                "use_reconstruction_module": False,
                "segmentation_module": "AttentionUNet",
                "segmentation_module_input_channels": 1,
                "segmentation_module_output_channels": 4,
                "segmentation_module_channels": 64,
                "segmentation_module_pooling_layers": 2,
                "segmentation_module_dropout": 0.0,
                "segmentation_loss": {"dice": 1.0},
                "dice_loss_include_background": False,
                "dice_loss_to_onehot_y": False,
                "dice_loss_sigmoid": True,
                "dice_loss_softmax": False,
                "dice_loss_other_act": None,
                "dice_loss_squared_pred": False,
                "dice_loss_jaccard": False,
                "dice_loss_reduction": "mean",
                "dice_loss_smooth_nr": 1,
                "dice_loss_smooth_dr": 1,
                "dice_loss_batch": True,
                "consecutive_slices": 1,
                "coil_combination_method": "SENSE",
                "magnitude_input": True,
                "use_sens_net": False,
                "fft_centered": False,
                "fft_normalization": "backward",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
            2,
            4,
            {
                "strategy": "ddp",
                "accelerator": "cpu",
                "num_nodes": 1,
                "max_epochs": 20,
                "precision": 32,
                "enable_checkpointing": False,
                "logger": False,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": -1,
                "max_steps": -1,
            },
        ),
    ],
)
def test_attention_unet(shape, cfg, center_fractions, accelerations, dimensionality, segmentation_classes, trainer):
    """
    Test the Segmentation Attention UNet with different parameters.

    Parameters
    ----------
    shape : list of int
        Shape of the input data
    cfg : dict
        Dictionary with the parameters of the qRIM model
    center_fractions : list of float
        List of center fractions to test
    accelerations : list of float
        List of acceleration factors to test
    dimensionality : int
        Dimensionality of the data
    segmentation_classes : int
        Number of segmentation classes
    trainer : dict
        Dictionary with the parameters of the trainer
    """
    mask_func = Random1DMaskFunc(center_fractions, accelerations)
    x = create_input(shape)

    outputs, masks = [], []
    for i in range(x.shape[0]):
        output, mask, _ = utils.apply_mask(x[i : i + 1], mask_func, seed=123)
        outputs.append(output)
        masks.append(mask)

    output = torch.cat(outputs)
    mask = torch.cat(masks)

    coil_dim = cfg.get("coil_dim")
    consecutive_slices = cfg.get("consecutive_slices")
    if consecutive_slices > 1:
        output = torch.stack([output for _ in range(consecutive_slices)], 1)
        coil_dim += 1

    if dimensionality == 3 and shape[1] > 1:
        mask = torch.cat([mask, mask], 1)

    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    trainer = OmegaConf.create(trainer)
    trainer = OmegaConf.create(OmegaConf.to_container(trainer, resolve=True))
    trainer = pl.Trainer(**trainer)

    segmentation_attention_unet = SegmentationAttentionUNet(cfg, trainer=trainer)

    with torch.no_grad():
        pred_segmentation = segmentation_attention_unet.forward(output.sum(coil_dim))

    if consecutive_slices > 1:
        output = torch.view_as_complex(
            output.reshape(
                [output.shape[0] * output.shape[1], output.shape[2], output.shape[3], output.shape[4], output.shape[5]]
            ).sum(coil_dim - 1)
        )
        output = torch.stack([output for _ in range(segmentation_classes)], 1)
        pred_segmentation = pred_segmentation.reshape(
            pred_segmentation.shape[0] * pred_segmentation.shape[1], *pred_segmentation.shape[2:]
        )
        if pred_segmentation.shape != output.shape:
            raise AssertionError
    else:
        output = torch.view_as_complex(torch.stack([output for _ in range(segmentation_classes)], 1).sum(coil_dim + 1))
        if pred_segmentation.shape != output.shape:
            raise AssertionError
