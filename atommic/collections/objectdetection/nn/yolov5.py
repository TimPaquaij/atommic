__author__ = "Tim Paquaij"

from typing import Dict, List, Tuple, Union
from atommic.collections.common.parts.utils import (
    check_stacked_complex,
    coil_combination_method,
    complex_abs,
    complex_abs_sq,
    expand_op,
    is_none,
    unnormalize,
)
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import fft2
from atommic.collections.common.parts.utils import expand_op
from atommic.collections.objectdetection.nn.base import BaseMRIObjectdetectionModel
from atommic.collections.objectdetection.nn.yolo_base.yolo import YOLOv5
from atommic.core.classes.common import typecheck
import matplotlib.pyplot as plt

__all__ = ["ObjectdetectionYolov5"]


class ObjectdetectionYolov5(BaseMRIObjectdetectionModel):
    """Implementation of the Multi-Task Learning for MRI Reconstruction and Segmentation (MTLRS) model, as presented
    in [Karkalousos2023]_.

    References
    ----------
    .. [Karkalousos2023] Karkalousos, D., Išgum, I., Marquering, H., Caan, M. W. A., (2023). MultiTask Learning for
        accelerated-MRI Reconstruction and Segmentation of Brain Lesions in Multiple Sclerosis. In Proceedings of
        Machine Learning Research (Vol. 078).

    """
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """inits :class:`BaseSegmentationNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object specifying the model's hyperparameters.
        trainer : Trainer, optional
            PyTorch Lightning trainer object, by default None.
        """
        super().__init__(cfg=cfg, trainer=trainer)
        #cfg_dict = DictConfig(cfg)


        self.object_detection_module = self.build_object_detection_module(cfg)
    def build_object_detection_module(self, cfg: DictConfig) -> torch.nn.Module:
        """Build the segmentation module.

        Parameterss
        ----------
        cfg : DictConfig
            Configuration object specifying the model's hyperparameters.

        Returns
        -------
        torch.nn.Module
            The segmentation module.
        """
        return YOLOv5(num_classes=cfg.get("num_classes", None),anchors=cfg.get("anchors"),
                                              strides=cfg.get("strides"),img_sizes=cfg.get("img_size"),backbone_ckpt=cfg.get("checkpoint", None))


    def forward(self, image: torch.Tensor,target: Union[List, torch.Tensor], **kwargs) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Forward pass of :class:`BaseSegmentationNet`.

        Parameters
        ----------
        image : torch.Tensor
            Input image. Shape [batch_size, n_x, n_y] or [batch_size, n_x, n_y, 2]
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Predicted segmentation. Shape [batch_size, n_classes, n_x, n_y]
        """
        image = image
        if image.shape[-1] != 2:
            _pred_reconstruction = torch.view_as_real(image)
        if image.shape[-1] == 2:
            image = torch.abs(torch.view_as_complex(image))

        if self.consecutive_slices > 1:
            batch, slices = image.shape[:2]
            image = image.reshape(batch * slices, *image.shape[2:])

        _obj_image = []
        for idv_pred in range(image.shape[0]):
            _obj_image.append(torch.cat((image[idv_pred].unsqueeze(0),
                                                       image[idv_pred].unsqueeze(0),
                                                       image[idv_pred].unsqueeze(0)), dim=0))

        pred_obj_detection, dict_obj_detection,target_obj_detection = self.object_detection_module(
            _obj_image, target)



        return pred_obj_detection, dict_obj_detection, target_obj_detection
