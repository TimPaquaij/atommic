# coding=utf-8
__author__ = "Dimitris Karkalousos"

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from atommic.collections.common.data.subsample import Random1DMaskFunc
from atommic.collections.common.parts import utils
from atommic.collections.multitask.rs.nn.mtlrs import MTLRS
from tests.collections.reconstruction.mri_data.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, dimensionality, segmentation_classes, trainer",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "use_reconstruction_module": True,
                "task_adaption_type": "multi_task_learning",
                "joint_reconstruction_segmentation_module_cascades": 5,
                "reconstruction_module_recurrent_layer": "IndRNN",
                "reconstruction_module_conv_filters": [64, 64, 2],
                "reconstruction_module_conv_kernels": [5, 3, 3],
                "reconstruction_module_conv_dilations": [1, 2, 1],
                "reconstruction_module_conv_bias": [True, True, False],
                "reconstruction_module_recurrent_filters": [64, 64, 0],
                "reconstruction_module_recurrent_kernels": [1, 1, 0],
                "reconstruction_module_recurrent_dilations": [1, 1, 0],
                "reconstruction_module_recurrent_bias": [True, True, False],
                "reconstruction_module_depth": 2,
                "reconstruction_module_conv_dim": 2,
                "reconstruction_module_time_steps": 8,
                "reconstruction_module_num_cascades": 5,
                "reconstruction_module_dimensionality": 2,
                "reconstruction_module_accumulate_predictions": True,
                "reconstruction_module_no_dc": True,
                "reconstruction_module_keep_prediction": True,
                "reconstruction_loss": {"l1": 1.0},
                "segmentation_module": "UNet",
                "segmentation_module_accumulate_predictions": True,
                "segmentation_module_input_channels": 2,
                "segmentation_module_output_channels": 4,
                "segmentation_module_channels": 64,
                "segmentation_module_pooling_layers": 4,
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
                "dimensionality": 2,
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
                "use_reconstruction_module": True,
                "task_adaption_type": "multi_task_learning",
                "joint_reconstruction_segmentation_module_cascades": 5,
                "reconstruction_module_recurrent_layer": "IndRNN",
                "reconstruction_module_conv_filters": [64, 64, 2],
                "reconstruction_module_conv_kernels": [5, 3, 3],
                "reconstruction_module_conv_dilations": [1, 2, 1],
                "reconstruction_module_conv_bias": [True, True, False],
                "reconstruction_module_recurrent_filters": [64, 64, 0],
                "reconstruction_module_recurrent_kernels": [1, 1, 0],
                "reconstruction_module_recurrent_dilations": [1, 1, 0],
                "reconstruction_module_recurrent_bias": [True, True, False],
                "reconstruction_module_depth": 2,
                "reconstruction_module_conv_dim": 2,
                "reconstruction_module_time_steps": 8,
                "reconstruction_module_num_cascades": 5,
                "reconstruction_module_dimensionality": 2,
                "reconstruction_module_accumulate_predictions": True,
                "reconstruction_module_no_dc": True,
                "reconstruction_module_keep_prediction": True,
                "reconstruction_loss": {"l1": 1.0},
                "segmentation_module": "AttentionUNet",
                "segmentation_module_accumulate_predictions": True,
                "segmentation_module_input_channels": 1,
                "segmentation_module_output_channels": 4,
                "segmentation_module_channels": 64,
                "segmentation_module_pooling_layers": 4,
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
                "dimensionality": 2,
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
                "use_reconstruction_module": True,
                "task_adaption_type": "multi_task_learning",
                "joint_reconstruction_segmentation_module_cascades": 5,
                "reconstruction_module_recurrent_layer": "IndRNN",
                "reconstruction_module_conv_filters": [64, 64, 2],
                "reconstruction_module_conv_kernels": [5, 3, 3],
                "reconstruction_module_conv_dilations": [1, 2, 1],
                "reconstruction_module_conv_bias": [True, True, False],
                "reconstruction_module_recurrent_filters": [64, 64, 0],
                "reconstruction_module_recurrent_kernels": [1, 1, 0],
                "reconstruction_module_recurrent_dilations": [1, 1, 0],
                "reconstruction_module_recurrent_bias": [True, True, False],
                "reconstruction_module_depth": 2,
                "reconstruction_module_conv_dim": 2,
                "reconstruction_module_time_steps": 8,
                "reconstruction_module_num_cascades": 5,
                "reconstruction_module_dimensionality": 2,
                "reconstruction_module_accumulate_predictions": True,
                "reconstruction_module_no_dc": True,
                "reconstruction_module_keep_prediction": True,
                "reconstruction_loss": {"l1": 1.0},
                "segmentation_module": "ConvLayer",
                "segmentation_module_input_channels": 1,
                "segmentation_module_output_channels": 4,
                "segmentation_module_channels": 64,
                "segmentation_module_pooling_layers": 4,
                "segmentation_module_dropout": 0.0,
                "segmentation_loss": {"dice": 1.0},
                "segmentation_module_accumulate_predictions": True,
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
                "dimensionality": 2,
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
def test_mtlmrirs(shape, cfg, center_fractions, accelerations, dimensionality, segmentation_classes, trainer):
    """
    Test MultiTask Learning for accelerated-MRI Reconstruction & Segmentation with different parameters.

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
        x = torch.stack([x for _ in range(consecutive_slices)], 1)
        output = torch.stack([output for _ in range(consecutive_slices)], 1)

    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    trainer = OmegaConf.create(trainer)
    trainer = OmegaConf.create(OmegaConf.to_container(trainer, resolve=True))
    trainer = pl.Trainer(**trainer)

    mtlrs = MTLRS(cfg, trainer=trainer)

    with torch.no_grad():
        pred_reconstruction, pred_segmentation = mtlrs.forward(
            output,
            output,
            mask,
            output.sum(coil_dim),
            output.sum(coil_dim),
        )

    if cfg.get("reconstruction_module_accumulate_predictions"):
        print(len(pred_reconstruction))
        if len(pred_reconstruction) !=cfg.get("joint_reconstruction_segmentation_module_cascades"):
            raise AssertionError("Number of predictions are not equal to the number of cascades")
        if cfg.get("reconstruction_module_keep_prediction"):
            if len(pred_reconstruction[0]) !=cfg.get("reconstruction_module_time_steps"):
                raise AssertionError(f"Number of intermediate predictions are not equal to the number of time steps")

        
    
            
    if cfg.get("segmentation_module_accumulate_predictions"):
        if len(pred_segmentation) !=cfg.get("joint_reconstruction_segmentation_module_cascades"):
            raise AssertionError("Number of segmentations are not equal to the number of cascades")

    while isinstance(pred_reconstruction, list):
        pred_reconstruction = pred_reconstruction[-1]

    while isinstance(pred_segmentation, list):
        pred_segmentation = pred_segmentation[-1]

    if dimensionality == 3 or consecutive_slices > 1:
        x = x.reshape([x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]])
    if x.shape[-1] == 2:
        x = x[..., 0] + 1j * x[..., 1]

    if consecutive_slices > 1 or dimensionality == 3:
        x = x.sum(coil_dim - 1)  # sum over coils
        if pred_reconstruction.dim() == 4:
            pred_reconstruction = pred_reconstruction.reshape(
                pred_reconstruction.shape[0] * pred_reconstruction.shape[1], *pred_reconstruction.shape[2:]
            )
        if pred_reconstruction.shape != x.shape:
            raise AssertionError
        if output.dim() == 6:
            output = output.reshape(
                [output.shape[0] * output.shape[1], output.shape[2], output.shape[3], output.shape[4], output.shape[5]]
            )
            coil_dim -= 1
        output = torch.view_as_complex(output).sum(coil_dim)
        output = torch.stack([output for _ in range(segmentation_classes)], 1)
        if consecutive_slices > 1:
            pred_segmentation = pred_segmentation.reshape(
                pred_segmentation.shape[0] * pred_segmentation.shape[1], *pred_segmentation.shape[2:]
            )
        if pred_segmentation.shape != output.shape:
            raise AssertionError
    else:
        if pred_reconstruction.shape[1:] != x.shape[2:]:
            raise AssertionError
        output = torch.view_as_complex(torch.stack([output for _ in range(segmentation_classes)], 1).sum(coil_dim + 1))
        if pred_segmentation.shape != output.shape:
            raise AssertionError
