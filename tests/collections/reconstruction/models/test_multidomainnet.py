# coding=utf-8
__author__ = "Dimitris Karkalousos"

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from atommic.collections.common.data.subsample import Random1DMaskFunc
from atommic.collections.common.parts import utils
from atommic.collections.reconstruction.nn.multidomainnet import MultiDomainNet


def create_input(shape):
    """Create a random input tensor."""
    return torch.rand(shape).float()


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, dimensionality, trainer",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "standardization": True,
                "num_filters": 16,
                "num_pool_layers": 2,
                "dropout_probability": 0.0,
                "use_sens_net": False,
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "coil_combination_method": "SENSE",
                "reconstruction_loss": {"l1": 1.0},
            },
            [0.08],
            [4],
            2,
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
                "standardization": False,
                "num_filters": 64,
                "num_pool_layers": 4,
                "dropout_probability": 0.0,
                "use_sens_net": False,
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "coil_combination_method": "SENSE",
                "train_reconstruction_loss": "mse",
                "val_reconstruction_loss": "mse",
            },
            [0.08],
            [4],
            2,
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
def test_multidomainnet(shape, cfg, center_fractions, accelerations, dimensionality, trainer):
    """
    Test the multidomainnet model.

    Args:
        shape: shape of the input
        cfg: configuration of the model
        center_fractions: center fractions
        accelerations: accelerations
        dimensionality: 2D or 3D inputs
        trainer: trainer configuration

    Returns:
        None.
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

    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    trainer = OmegaConf.create(trainer)
    trainer = OmegaConf.create(OmegaConf.to_container(trainer, resolve=True))
    trainer = pl.Trainer(**trainer)

    kikinet = MultiDomainNet(cfg, trainer=trainer)

    with torch.no_grad():
        y = kikinet.forward(output, output, mask, output)

    if dimensionality == 3:
        x = x.reshape([x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]])

    if y.shape[1:] != x.shape[2:4]:
        raise AssertionError
