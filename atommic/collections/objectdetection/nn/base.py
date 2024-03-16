# coding=utf-8
__author__ = "Tim Paquaij"

import os
import warnings
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Union

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf,ListConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from atommic.collections.common.data.subsample import create_masker
from atommic.collections.common.losses import VALID_OBJ_DETECTION_LOSSES
from atommic.collections.common.losses.aggregator import AggregatorLoss
from atommic.collections.common.nn.base import BaseMRIModel, DistributedMetricSum
from atommic.collections.common.parts.utils import complex_abs, complex_abs_sq, is_none, unnormalize
from atommic.collections.objectdetection.data import mri_object_detection_loader
from atommic.collections.objectdetection.losses import ClassLoss,ObjectLoss,BBoxLoss
from atommic.collections.objectdetection.parts.transforms import ObjectdetectionMRIDataTransforms

__all__ = ["BaseMRIObjectdetectionModel"]


class BaseMRIObjectdetectionModel(BaseMRIModel, ABC):
    """Base class of all MRI Segmentation models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`BaseMRISegmentationModel`.

        Parameters
        ----------
        cfg: DictConfig
            The configuration file.
        trainer: Trainer
            The PyTorch Lightning trainer.
        """
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.acc = 1  # fixed acceleration factor to ensure acc is not None

        # Initialize the dimensionality of the data. It can be 2D or 2.5D -> meaning 2D with > 1 slices or 3D.
        self.dimensionality = cfg_dict.get("dimensionality", 2)
        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)

        # Set input channels.
        # self.input_channels = cfg_dict.get("segmentation_module_input_channels", 2)
        # if self.input_channels == 0:
        #     raise ValueError("Segmentation module input channels cannot be 0.")

        # Set type of data, i.e., magnitude only or complex valued.
        self.magnitude_input = cfg_dict.get("magnitude_input", True)
        # Refers to the type of the complex-valued data. It can be either "stacked" or "complex_abs" or
        # "complex_sqrt_abs".
        self.complex_valued_type = cfg_dict.get("complex_valued_type", "stacked")

        # Set normalization parameters for logging
        self.unnormalize_loss_inputs = cfg_dict.get("unnormalize_loss_inputs", False)
        self.unnormalize_log_outputs = cfg_dict.get("unnormalize_log_outputs", False)
        self.normalization_type = cfg_dict.get("normalization_type", "max")
        # self.normalize_segmentation_output = cfg_dict.get("normalize_segmentation_output", True)

        # Whether to log multiple modalities, e.g. T1, T2, and FLAIR will be stacked and logged.
        self.log_multiple_modalities = cfg_dict.get("log_multiple_modalities", False)

        # Set threshold for segmentation classes. If None, no thresholding is applied.
        # self.segmentation_classes_thresholds = cfg_dict.get("segmentation_classes_thresholds", None)
        # self.segmentation_activation = cfg_dict.get("segmentation_activation", None)

        # # Initialize loss related parameters.
        # self.segmentation_losses = {}
        # segmentation_loss = cfg_dict.get("segmentation_loss")
        # segmentation_losses_ = {}
        # if segmentation_loss is not None:
        #     for k, v in segmentation_loss.items():
        #         if k not in VALID_SEGMENTATION_LOSSES:
        #             raise ValueError(
        #                 f"Segmentation loss {k} is not supported. Please choose one of the following: "
        #                 f"{VALID_SEGMENTATION_LOSSES}."
        #             )
        #         if v is None or v == 0.0:
        #             warnings.warn(f"The weight of segmentation loss {k} is set to 0.0. This loss will not be used.")
        #         else:
        #             segmentation_losses_[k] = v
        # else:
        #     # Default segmentation loss is Dice.
        #     segmentation_losses_["dice"] = 1.0
        # if sum(segmentation_losses_.values()) != 1.0:
        #     warnings.warn("Sum of segmentation losses weights is not 1.0. Adjusting weights to sum up to 1.0.")
        #     total_weight = sum(segmentation_losses_.values())
        #     segmentation_losses_ = {k: v / total_weight for k, v in segmentation_losses_.items()}
        # for name in VALID_SEGMENTATION_LOSSES:
        #     if name in segmentation_losses_:
        #         if name == "cross_entropy":
        #             cross_entropy_loss_classes_weight = torch.tensor(
        #                 cfg_dict.get("cross_entropy_loss_classes_weight", 0.5)
        #             )
        #             self.segmentation_losses[name] = CrossEntropyLoss(
        #                 num_samples=cfg_dict.get("cross_entropy_loss_num_samples", 50),
        #                 ignore_index=cfg_dict.get("cross_entropy_loss_ignore_index", -100),
        #                 reduction=cfg_dict.get("cross_entropy_loss_reduction", "none"),
        #                 label_smoothing=cfg_dict.get("cross_entropy_loss_label_smoothing", 0.0),
        #                 weight=cross_entropy_loss_classes_weight,
        #             )
        #         elif name == "dice":
        #             self.segmentation_losses[name] = Dice(
        #                 include_background=cfg_dict.get("dice_loss_include_background", False),
        #                 to_onehot_y=cfg_dict.get("dice_loss_to_onehot_y", False),
        #                 sigmoid=cfg_dict.get("dice_loss_sigmoid", True),
        #                 softmax=cfg_dict.get("dice_loss_softmax", False),
        #                 other_act=cfg_dict.get("dice_loss_other_act", None),
        #                 squared_pred=cfg_dict.get("dice_loss_squared_pred", False),
        #                 jaccard=cfg_dict.get("dice_loss_jaccard", False),
        #                 flatten=cfg_dict.get("dice_loss_flatten", False),
        #                 reduction=cfg_dict.get("dice_loss_reduction", "mean"),
        #                 smooth_nr=cfg_dict.get("dice_loss_smooth_nr", 1e-5),
        #                 smooth_dr=cfg_dict.get("dice_loss_smooth_dr", 1e-5),
        #                 batch=cfg_dict.get("dice_loss_batch", False),
        #             )
        # self.segmentation_losses = {f"loss_{i+1}": v for i, v in enumerate(self.segmentation_losses.values())}
        # self.total_segmentation_losses = len(self.segmentation_losses)
        # self.total_segmentation_loss_weight = cfg_dict.get("total_segmentation_loss_weight", 1.0)
        #
        # # Set the metrics
        # cross_entropy_metric_num_samples = cfg_dict.get("cross_entropy_metric_num_samples", 50)
        # cross_entropy_metric_ignore_index = cfg_dict.get("cross_entropy_metric_ignore_index", -100)
        # cross_entropy_metric_reduction = cfg_dict.get("cross_entropy_metric_reduction", "none")
        # cross_entropy_metric_label_smoothing = cfg_dict.get("cross_entropy_metric_label_smoothing", 0.0)
        # cross_entropy_metric_classes_weight = cfg_dict.get("cross_entropy_metric_classes_weight", None)
        # dice_metric_include_background = cfg_dict.get("dice_metric_include_background", False)
        # dice_metric_to_onehot_y = cfg_dict.get("dice_metric_to_onehot_y", False)
        # dice_metric_sigmoid = cfg_dict.get("dice_metric_sigmoid", True)
        # dice_metric_softmax = cfg_dict.get("dice_metric_softmax", False)
        # dice_metric_other_act = cfg_dict.get("dice_metric_other_act", None)
        # dice_metric_squared_pred = cfg_dict.get("dice_metric_squared_pred", False)
        # dice_metric_jaccard = cfg_dict.get("dice_metric_jaccard", False)
        # dice_metric_flatten = cfg_dict.get("dice_metric_flatten", False)
        # dice_metric_reduction = cfg_dict.get("dice_metric_reduction", "mean")
        # dice_metric_smooth_nr = cfg_dict.get("dice_metric_smooth_nr", 1e-5)
        # dice_metric_smooth_dr = cfg_dict.get("dice_metric_smooth_dr", 1e-5)
        # dice_metric_batch = cfg_dict.get("dice_metric_batch", True)
        #
        # # Initialize the module
        # super().__init__(cfg=cfg, trainer=trainer)
        #
        # if not is_none(cross_entropy_metric_classes_weight):
        #     cross_entropy_metric_classes_weight = torch.tensor(cross_entropy_metric_classes_weight)
        #     self.cross_entropy_metric = CrossEntropyLoss(
        #         num_samples=cross_entropy_metric_num_samples,
        #         ignore_index=cross_entropy_metric_ignore_index,
        #         reduction=cross_entropy_metric_reduction,
        #         label_smoothing=cross_entropy_metric_label_smoothing,
        #         weight=cross_entropy_metric_classes_weight,
        #     )
        # else:
        #     self.cross_entropy_metric = None  # type: ignore
        # self.dice_metric = Dice(
        #     include_background=dice_metric_include_background,
        #     to_onehot_y=dice_metric_to_onehot_y,
        #     sigmoid=dice_metric_sigmoid,
        #     softmax=dice_metric_softmax,
        #     other_act=dice_metric_other_act,
        #     squared_pred=dice_metric_squared_pred,
        #     jaccard=dice_metric_jaccard,
        #     flatten=dice_metric_flatten,
        #     reduction=dice_metric_reduction,
        #     smooth_nr=dice_metric_smooth_nr,
        #     smooth_dr=dice_metric_smooth_dr,
        #     batch=dice_metric_batch,
        # )
        #
        # # Set aggregation loss
        # self.total_segmentation_loss = AggregatorLoss(
        #     num_inputs=self.total_segmentation_losses, weights=list(segmentation_losses_.values())
        # )
        self.obj_detection_losses = {}
        obj_detection_loss = cfg_dict.get("obj_detection_loss", None)
        obj_detection_losses_ = {}
        if obj_detection_loss is not None:
            for k, v in obj_detection_loss.items():
                if k not in VALID_OBJ_DETECTION_LOSSES:
                    raise ValueError(
                        f"Object Detection loss {k} is not supported. Please choose one of the following: "
                        f"{VALID_OBJ_DETECTION_LOSSES}."
                    )
                if v is None or v == 0.0:
                    warnings.warn(f"The weight of object detection loss {k} is set to 0.0. This loss will not be used.")
                else:
                    obj_detection_losses_[k] = v
        else:
            # Default object detection loss.
            obj_detection_losses_["object"] = 0.8
            obj_detection_losses_["bbox"] = 0.05
            obj_detection_losses_["class"] = 0.15
        if sum(obj_detection_losses_.values()) != 1.0:
            warnings.warn("Sum of segmentation losses weights is not 1.0. Adjusting weights to sum up to 1.0.")
            total_weight = sum(obj_detection_losses_.values())
            obj_detection_losses_ = {k: v / total_weight for k, v in obj_detection_losses_.items()}
        for name in VALID_OBJ_DETECTION_LOSSES:
            if name in obj_detection_losses_:
                if name == "object":
                    self.obj_detection_losses[name] = ObjectLoss(anchors=cfg_dict.get("anchors"),
                                                                 strides=cfg_dict.get("strides"))
                elif name == "class":
                    self.obj_detection_losses[name] = ClassLoss(anchors=cfg_dict.get("anchors"),
                                                                strides=cfg_dict.get("strides"))
                elif name == "bbox":
                    self.obj_detection_losses[name] = BBoxLoss(anchors=cfg_dict.get("anchors"),
                                                               strides=cfg_dict.get("strides"))
        self.obj_detection_losses = {f"loss_{i + 1}": v for i, v in enumerate(self.obj_detection_losses.values())}
        self.total_obj_detection_losses = len(self.obj_detection_losses)

        self.total_segmentation_loss_weight = cfg_dict.get("total_segmentation_loss_weight", 1.0)
        self.total_reconstruction_loss_weight = cfg_dict.get("total_reconstruction_loss_weight", 1.0)
        self.total_obj_detection_loss_weight = cfg_dict.get("total_obj_detection_loss_weight", 1.0)
        self.anchors = cfg_dict.get("anchors")
        self.strides = cfg_dict.get("strides")
        # Initialize the module
        super().__init__(cfg=cfg, trainer=trainer)
        self.total_obj_detection_loss = AggregatorLoss(
            num_inputs=self.total_obj_detection_losses, weights=list(obj_detection_losses_.values())
        )
        self.class_metric = ClassLoss(anchors=self.anchors,
                                                                 strides=self.strides) #TODO Change later into official metric
        self.object_metric = ObjectLoss(anchors=self.anchors,
                                                                 strides=self.strides)
        self.bbox_metric =BBoxLoss(anchors=self.anchors,
                                                                 strides=self.strides)

        # Set distributed metrics
        self.CLASS_BCE = DistributedMetricSum()
        self.OBJ_BCE = DistributedMetricSum()
        self.BBOX_BCE = DistributedMetricSum()
        self.class_bce_vals: Dict = defaultdict(dict)
        self.object_bce_vals: Dict = defaultdict(dict)
        self.bbox_bce_vals: Dict = defaultdict(dict)
        self.TotExamples = DistributedMetricSum()

    def __abs_output__(self, x: torch.Tensor) -> torch.Tensor:
        """Converts the input to absolute value."""
        if x.shape[-1] == 2 or torch.is_complex(x):
            if self.complex_valued_type == "stacked":
                if x.shape[-1] == 2:
                    x = torch.view_as_complex(x)
            elif self.complex_valued_type == "complex_abs":
                x = complex_abs(x)
            elif self.complex_valued_type == "complex_sqrt_abs":
                x = complex_abs_sq(x)
        return x

    def __unnormalize_for_loss_or_log__(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        attrs: Dict,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unnormalizes the data for computing the loss or logging.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : torch.Tensor
            Prediction data of shape [batch_size, n_x, n_y, 2].
        attrs : Dict
            Attributes of the data with pre normalization values.
        batch_idx : int
            Batch index. Default is ``0``.

        Returns
        -------
        target : torch.Tensor
            Unnormalized target data.
        prediction : torch.Tensor
            Unnormalized prediction data.
        sensitivity_maps : torch.Tensor
            Unnormalized sensitivity maps.
        """
        target = unnormalize(
            target,
            {
                "min": attrs["target_min"][batch_idx],
                "max": attrs["target_max"][batch_idx],
                "mean": attrs["target_mean"][batch_idx],
                "std": attrs["target_std"][batch_idx],
            },
            self.normalization_type,
        )
        prediction = unnormalize(
            prediction,
            {
                "min": attrs["prediction_min"][batch_idx],
                "max": attrs["prediction_max"][batch_idx],
                "mean": attrs["prediction_mean"][batch_idx],
                "std": attrs["prediction_std"][batch_idx],
            },
            self.normalization_type,
        )

        return target, prediction

    def process_obj_detection_loss(self, target: Dict, prediction: torch.Tensor, attrs: Dict,
                                   loss_func: torch.nn.Module) -> Dict:
        """Processes the segmentation loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, nr_classes, n_x, n_y].
        prediction : torch.Tensor
            Prediction of shape [batch_size, nr_classes, n_x, n_y].
        attrs : Dict
            Attributes of the data with pre normalization values.

        Returns
        -------
        Dict
            Dictionary containing the (multiple) loss values. For example, if the cross entropy loss and the dice loss
            are used, the dictionary will contain the keys ``cross_entropy_loss``, ``dice_loss``, and
            (combined) ``segmentation_loss``.
        """
        # if self.obj_detection_module_accumulate_predictions:
        #     rs_cascades_weights = torch.logspace(-1, 0, steps=len(prediction)).to(
        #         target[0]['labels'].device)
        #     rs_cascades_loss = []
        #     for pred in prediction:
        #         loss = loss_func(target, pred)
        #         rs_cascades_loss.append(loss)
        #     loss = sum(x * w for x, w in zip(rs_cascades_loss, rs_cascades_weights)) / sum(rs_cascades_weights)
        # else:
        prediction = prediction
        loss = loss_func(target, prediction)
        return loss

    def __compute_and_log_metrics_and_outputs__(
        self,
        dict_predictions: Union[list, Dict],
        loss_predictions: Union[list, torch.tensor],
        dict_target: Union[list, Dict],
        loss_target: Union[list, Dict],
        target_reconstruction: torch.Tensor,
        attrs: Dict,
        fname: str,
        slice_idx: int,
    ):
        """Computes the metrics and logs the outputs.

        Parameters
        ----------
        predictions : torch.Tensor
            Prediction data of shape [batch_size, n_x, n_y].
        target_reconstruction : torch.Tensor
            Target reconstruction data of shape [batch_size, n_x, n_y].
        target_segmentation : torch.Tensor
            Target segmentation data of shape [batch_size, segmentation_classes, n_x, n_y].
        attrs : Dict
            Attributes of the data with pre normalization values.
        fname : str
            File name.
        slice_idx : int
            Slice index.
        """

        # Ensure all inputs are both viewed in the same way.

        if self.consecutive_slices > 1:
            dict_target = [dict_target[2//self.consecutive_slices]]
            dict_predictions = [dict_predictions[2 // self.consecutive_slices]]
            target_reconstruction = target_reconstruction[:, 2//self.consecutive_slices]

        batch_size = target_reconstruction.shape[0]



        # Iterate over the batch and log the target and predictions.
        for _batch_idx_ in range(batch_size):
            dict_target = dict_target[_batch_idx_]
            dict_predictions = dict_predictions[_batch_idx_]
            output_target_reconstruction = target_reconstruction[_batch_idx_]

            # Ensure loss inputs are both viewed in the same way.
            output_target_reconstruction = self.__abs_output__(output_target_reconstruction)
            output_target_reconstruction = torch.abs(
                output_target_reconstruction / torch.max(torch.abs(output_target_reconstruction)))
            if torch.max(target_reconstruction) != 1:
                output_target_reconstruction = torch.clip(output_target_reconstruction, 0, 1)




            if self.unnormalize_log_outputs:
                # Unnormalize target and predictions with pre normalization values. This is only for logging purposes.
                # For the loss computation, the self.unnormalize_loss_inputs flag is used.
                output_target_segmentation, output_predictions = self.__unnormalize_for_loss_or_log__(
                    output_target_segmentation, output_predictions, attrs, batch_idx=_batch_idx_
                )

            # Log target and predictions, if log_image is True for this slice.
            if attrs["log_image"][_batch_idx_] and dict_target["boxes"].reshape(-1,4).numel() != 0:
                key = f"{fname[_batch_idx_]}_slice_{int(slice_idx[_batch_idx_])}"  # type: ignore


                if self.log_multiple_modalities:
                    # concatenate the reconstruction predictions for logging
                    output_target_reconstruction = torch.cat(
                        [output_target_reconstruction[i] for i in range(output_target_reconstruction.shape[0])], dim=-1
                    )

                self.log_image(f"{key}/a/input", output_target_reconstruction)
                self.log_boundary_box(f"{key}/b/object_detection/target", output_target_reconstruction,
                                      dict_target, attrs['categories'])
                self.log_boundary_box(f"{key}/c/object_detection/prediction", output_target_reconstruction,
                                      dict_predictions, attrs['categories'])



            self.class_bce_vals[fname][slice_idx] = 1-self.class_metric(
                    loss_target, loss_predictions
                )
            self.bbox_bce_vals[fname][slice_idx] = 1-self.bbox_metric(
                loss_target, loss_predictions
            )
            self.object_bce_vals[fname][slice_idx] = 1-self.object_metric(
                loss_target, loss_predictions
            )



    def inference_step(
        self, image: torch.Tensor, target: torch.Tensor, fname: str, slice_idx: int, attrs: Dict
    ) -> Dict[str, torch.Tensor]:
        """Performs an inference step, i.e., computes the predictions of the model.

        Parameters
        ----------
        image : torch.Tensor
            Input data. Shape [batch_size, n_x, n_y, 2].
        target : torch.Tensor
            Target data. Shape [batch_size, n_x, n_y, 2].
        fname : str
            File name.
        slice_idx : int
            Slice index.
        attrs : Dict
            Attributes dictionary.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of processed inputs and model's predictions, with keys:
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'predictions' : Union[List[torch.Tensor], torch.Tensor]
                    Model's predictions. Shape [batch_size, segmentation_classes, n_x, n_y, 2].
                'target' : torch.Tensor
                    Target data. Shape [batch_size, n_x, n_y, 2].
                'attrs' : dict
                    Attributes dictionary.
        """
        # Model forward pass
        target = [{key: value.squeeze(0) for key, value in tgt.items()} for tgt in target]
        pred_obj_detection, dict_obj_detection, target_obj_detection = self.forward(image,target)


        return {
            "fname": fname,
            "slice_idx": slice_idx,
            "predictions_dict": dict_obj_detection,
            "predictions_loss":pred_obj_detection,
            "target_loss": target_obj_detection,
            "target_dict":target,
            "attrs": attrs,
        }
    def __compute_loss__(
        self,
        target_obj_detection: Dict,
        predictions_obj_detection: Union[list, torch.Tensor],
        attrs: Dict,
    ) -> torch.Tensor:
        #TODO change imput description
        """Computes the reconstruction loss.

        Parameters
        ----------
        predictions_reconstruction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        predictions_reconstruction_n2r : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2], if Noise-to-Recon is used. Otherwise, None.
        target_reconstruction : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        predictions_segmentation : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, nr_classes, n_x, n_y].
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
        ssdu_loss_mask : torch.Tensor
            SSDU loss mask of shape [batch_size, 1, n_x, n_y, 1]. It will be used if self.ssdu is True, to enforce
            data consistency on the prediction.
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """

        losses = {}
        for name, loss_func in self.obj_detection_losses.items():

            losses[name] = (
                    self.process_obj_detection_loss(
                        target_obj_detection,
                        predictions_obj_detection,
                        attrs,
                        loss_func=loss_func,
                    )
            )
        loss = self.total_obj_detection_loss(**losses)


        return loss
    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Performs a training step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data with keys:
                'kspace' : List of torch.Tensor
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'y' : Union[torch.Tensor, None]
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'sensitivity_maps' : torch.Tensor
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'mask' : Union[torch.Tensor, None]
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'initial_prediction_reconstruction' : torch.Tensor
                    Initial reconstruction prediction. Shape [batch_size, n_x, n_y, 2].
                'target_reconstruction' : torch.Tensor
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'segmentation_labels' : torch.Tensor
                    Target segmentation labels. Shape [batch_size, segmentation_classes, n_x, n_y].
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'acceleration' : float
                    Acceleration factor of the sampling mask.
                'attrs' : dict
                    Attributes dictionary.

        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss and log.
        """
        (
            _,
            _,
            _,
            _,
            _,
            target_reconstruction,
            _,
            target_obj_detection,
            fname,
            slice_idx,
            _,
            attrs,
        ) = batch


        outputs = self.inference_step(
            target_reconstruction,
            target_obj_detection,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            attrs,  # type: ignore
        )

        train_loss = self.__compute_loss__(outputs["target_loss"], outputs["predictions_loss"], attrs)  # type: ignore

        tensorboard_logs = {
            "train_loss": train_loss.item(),  # type: ignore
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }

        self.log(
            "train_objectdetection_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,  # type: ignore
            sync_dist=True,
        )

        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int):
        """Performs a validation step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data with keys:
                'kspace' : List of torch.Tensor
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'y' : Union[torch.Tensor, None]
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'sensitivity_maps' : torch.Tensor
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'mask' : Union[torch.Tensor, None]
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'initial_prediction_reconstruction' : torch.Tensor
                    Initial reconstruction prediction. Shape [batch_size, n_x, n_y, 2].
                'target_reconstruction' : torch.Tensor
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'segmentation_labels' : torch.Tensor
                    Target segmentation labels. Shape [batch_size, segmentation_classes, n_x, n_y].
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'acceleration' : float
                    Acceleration factor of the sampling mask.
                'attrs' : dict
                    Attributes dictionary.

        batch_idx : int
            Batch index.
        """
        (
            _,
            _,
            _,
            _,
            _,
            target_reconstruction,
            _,
            target_obj_detection,
            fname,
            slice_idx,
            _,
            attrs,
        ) = batch


        outputs = self.inference_step(
            target_reconstruction,
            target_obj_detection,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            attrs,  # type: ignore
        )



        # print memory usage for debugging
        val_loss = self.__compute_loss__(outputs["target_loss"], outputs["predictions_loss"], attrs)  # type: ignore
        # Compute metrics and log them and log outputs.
        self.__compute_and_log_metrics_and_outputs__(
            outputs["predictions_dict"],
            outputs["predictions_loss"],
            outputs["target_dict"],
            outputs["target_loss"],
            target_reconstruction,
            attrs,  # type: ignore
            fname,  # type: ignore
            slice_idx,  # type: ignore
        )

        self.validation_step_outputs.append({"val_loss": val_loss})

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int):
        """Performs a test step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data with keys:
                'kspace' : List of torch.Tensor
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'y' : Union[torch.Tensor, None]
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'sensitivity_maps' : torch.Tensor
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'mask' : Union[torch.Tensor, None]
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'initial_prediction_reconstruction' : torch.Tensor
                    Initial reconstruction prediction. Shape [batch_size, n_x, n_y, 2].
                'target_reconstruction' : torch.Tensor
                    Placeholder to keep the same structure as the base (task) classes. Not used.
                'segmentation_labels' : torch.Tensor
                    Target segmentation labels. Shape [batch_size, segmentation_classes, n_x, n_y].
                'fname' : str
                    File name.
                'slice_idx' : int
                    Slice index.
                'acceleration' : float
                    Acceleration factor of the sampling mask.
                'attrs' : dict
                    Attributes dictionary.

        batch_idx : int
            Batch index.
        """
        (
            _,
            _,
            _,
            _,
            _,
            target_reconstruction,
            _,
            target_obj_detection,
            fname,
            slice_idx,
            _,
            attrs,
        ) = batch


        outputs = self.inference_step(
            target_reconstruction,
            target_obj_detection,
            fname,  # type: ignore
            slice_idx,  # type: ignore
            attrs,  # type: ignore
        )


        # Compute metrics and log them and log outputs.
        self.__compute_and_log_metrics_and_outputs__(
            outputs["predictions_dict"],
            outputs["predictions_loss"],
            target_reconstruction,
            outputs["target_dict"],
            outputs["target_loss"],
            attrs,  # type: ignore
            fname,  # type: ignore
            slice_idx,  # type: ignore
        )

        # Get the file name.
        fname = attrs['fname'][0]  # type: ignore
        if '.nii.gz' in fname or '.nii' in fname or '.h5' in fname:  # type: ignore
            fname = fname.split('.')[0]  # type: ignore

        self.test_step_outputs.append([fname, slice_idx, predictions.detach().cpu()])

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch to aggregate outputs.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        self.log("val_loss", torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean(), sync_dist=True)

        # Log metrics.

        class_bce_vals = defaultdict(dict)
        for k, v in self.class_bce_vals.items():
            class_bce_vals[k].update(v)

        bbox_bce_vals = defaultdict(dict)
        for k, v in self.bbox_bce_vals.items():
            bbox_bce_vals[k].update(v)

        object_bce_vals = defaultdict(dict)
        for k, v in self.object_bce_vals.items():
            object_bce_vals[k].update(v)

        metrics = {"Class": 0, "BBox": 0, "Object": 0}

        local_examples = 0
        for fname in class_bce_vals:
            local_examples += 1

            metrics["Class"] = metrics["Class"] + torch.mean(
                torch.cat([v.view(-1) for _, v in class_bce_vals[fname].items()])
            )
            metrics["BBox"] = metrics["BBox"] + torch.mean(
                torch.cat([v.view(-1) for _, v in bbox_bce_vals[fname].items()])
            )
            metrics["Object"] = metrics["Object"] + torch.mean(
                torch.cat([v.view(-1) for _, v in bbox_bce_vals[fname].items()])
            )

        # reduce across ddp via sum

        metrics["Class"] = self.CLASS_BCE(metrics["Class"])
        metrics["BBox"] = self.CLASS_BCE(metrics["BBox"])
        metrics["Object"] = self.CLASS_BCE(metrics["Object"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):  # noqa: MC0001
        """Called at the end of test epoch to aggregate outputs, log metrics and save predictions.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        # Log metrics.
        class_bce_vals = defaultdict(dict)
        for k, v in self.class_bce_vals.items():
            class_bce_vals[k].update(v)

        bbox_bce_vals = defaultdict(dict)
        for k, v in self.bbox_bce_vals.items():
            bbox_bce_vals[k].update(v)

        object_bce_vals = defaultdict(dict)
        for k, v in self.object_bce_vals.items():
            object_bce_vals[k].update(v)

        metrics = {"Class": 0, "BBox": 0, "Object": 0}

        local_examples = 0
        for fname in class_bce_vals:
            local_examples += 1

            metrics["Class"] = metrics["Class"] + torch.mean(
                torch.cat([v.view(-1) for _, v in class_bce_vals[fname].items()])
            )
            metrics["BBox"] = metrics["BBox"] + torch.mean(
                torch.cat([v.view(-1) for _, v in bbox_bce_vals[fname].items()])
            )
            metrics["Object"] = metrics["Object"] + torch.mean(
                torch.cat([v.view(-1) for _, v in bbox_bce_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["Class"] = self.CLASS_BCE(metrics["Class"])
        metrics["BBox"] = self.CLASS_BCE(metrics["BBox"])
        metrics["Object"] = self.CLASS_BCE(metrics["Object"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics.items():
            self.log(f"test_metrics/{metric}", value / tot_examples, prog_bar=True, sync_dist=True)


    @staticmethod
    def _setup_dataloader_from_config(cfg: DictConfig) -> DataLoader:
        """Setups the dataloader from the configuration (yaml) file.

        Parameters
        ----------
        cfg : DictConfig
            Configuration file.

        Returns
        -------
        dataloader : torch.utils.data.DataLoader
            Dataloader.
        """
        mask_root = cfg.get("mask_path", None)
        mask_args = cfg.get("mask_args", None)
        shift_mask = mask_args.get("shift_mask", False)
        mask_type = mask_args.get("type", None)

        mask_func = None
        mask_center_scale = 0.02

        if is_none(mask_root) and not is_none(mask_type):
            accelerations = mask_args.get("accelerations", [1])
            accelerations = list(accelerations)
            if len(accelerations) == 1:
                accelerations = accelerations * 2
            center_fractions = mask_args.get("center_fractions", [1])
            center_fractions = list(center_fractions)
            if len(center_fractions) == 1:
                center_fractions = center_fractions * 2
            mask_center_scale = mask_args.get("center_scale", 0.02)

            mask_func = [create_masker(mask_type, center_fractions, accelerations)]

        complex_data = cfg.get("complex_data", True)

        dataset_format = cfg.get("dataset_format", None)
        if any(s.lower() in (
                "skm-tea-echo1",
                "skm-tea-echo2",
                "skm-tea-echo1+echo2",
                "skm-tea-echo1+echo2-mc") for s in dataset_format):
            if any(a.lower() == "lateral" for a in dataset_format):
                dataloader = mri_object_detection_loader.SKMTEAOBJMRIDatasetlateral
            else:
                dataloader = mri_object_detection_loader.SKMTEARSMRIDataset
        else:
            dataloader = mri_object_detection_loader.ObjectMRIDataset

        dataset = dataloader(
            root=cfg.get("data_path"),
            coil_sensitivity_maps_root=cfg.get("coil_sensitivity_maps_path", None),
            mask_root=mask_root,
            noise_root=cfg.get("noise_path", None),
            initial_predictions_root=cfg.get("initial_predictions_path"),
            dataset_format=dataset_format,
            sample_rate=cfg.get("sample_rate", 1.0),
            volume_sample_rate=cfg.get("volume_sample_rate", None),
            use_dataset_cache=cfg.get("use_dataset_cache", False),
            dataset_cache_file=cfg.get("dataset_cache_file", None),
            num_cols=cfg.get("num_cols", None),
            consecutive_slices=cfg.get("consecutive_slices", 1),
            data_saved_per_slice=cfg.get("data_saved_per_slice", False),
            n2r_supervised_rate=cfg.get("n2r_supervised_rate", 0.0),
            complex_target=cfg.get("complex_target", False),
            log_images_rate=cfg.get("log_images_rate", 1.0),
            transform=ObjectdetectionMRIDataTransforms(
                complex_data=complex_data,
                dataset_format=dataset_format,
                apply_prewhitening=cfg.get("apply_prewhitening", False),
                find_patch_size=cfg.get("find_patch_size", False),
                prewhitening_scale_factor=cfg.get("prewhitening_scale_factor", 1.0),
                prewhitening_patch_start=cfg.get("prewhitening_patch_start", 10),
                prewhitening_patch_length=cfg.get("prewhitening_patch_length", 30),
                apply_gcc=cfg.get("apply_gcc", False),
                gcc_virtual_coils=cfg.get("gcc_virtual_coils", 10),
                gcc_calib_lines=cfg.get("gcc_calib_lines", 10),
                gcc_align_data=cfg.get("gcc_align_data", False),
                apply_random_motion=cfg.get("apply_random_motion", False),
                random_motion_type=cfg.get("random_motion_type", "gaussian"),
                random_motion_percentage=cfg.get("random_motion_percentage", [10, 10]),
                random_motion_angle=cfg.get("random_motion_angle", 10),
                random_motion_translation=cfg.get("random_motion_translation", 10),
                random_motion_center_percentage=cfg.get("random_motion_center_percentage", 0.02),
                random_motion_num_segments=cfg.get("random_motion_num_segments", 8),
                random_motion_random_num_segments=cfg.get("random_motion_random_num_segments", True),
                random_motion_non_uniform=cfg.get("random_motion_non_uniform", False),
                estimate_coil_sensitivity_maps=cfg.get("estimate_coil_sensitivity_maps", False),
                coil_sensitivity_maps_type=cfg.get("coil_sensitivity_maps_type", "espirit"),
                coil_sensitivity_maps_gaussian_sigma=cfg.get("coil_sensitivity_maps_gaussian_sigma", 0.0),
                coil_sensitivity_maps_espirit_threshold=cfg.get("coil_sensitivity_maps_espirit_threshold", 0.05),
                coil_sensitivity_maps_espirit_kernel_size=cfg.get("coil_sensitivity_maps_espirit_kernel_size", 6),
                coil_sensitivity_maps_espirit_crop=cfg.get("coil_sensitivity_maps_espirit_crop", 0.95),
                coil_sensitivity_maps_espirit_max_iters=cfg.get("coil_sensitivity_maps_espirit_max_iters", 30),
                coil_combination_method=cfg.get("coil_combination_method", "SENSE"),
                dimensionality=cfg.get("dimensionality", 2),
                mask_func=mask_func,
                shift_mask=shift_mask,
                mask_center_scale=mask_center_scale,
                remask=cfg.get("remask", False),
                ssdu=cfg.get("ssdu", False),
                ssdu_mask_type=cfg.get("ssdu_mask_type", "Gaussian"),
                ssdu_rho=cfg.get("ssdu_rho", 0.4),
                ssdu_acs_block_size=cfg.get("ssdu_acs_block_size", (4, 4)),
                ssdu_gaussian_std_scaling_factor=cfg.get("ssdu_gaussian_std_scaling_factor", 4.0),
                ssdu_outer_kspace_fraction=cfg.get("ssdu_outer_kspace_fraction", 0.0),
                ssdu_export_and_reuse_masks=cfg.get("ssdu_export_and_reuse_masks", False),
                n2r=cfg.get("n2r", False),
                n2r_supervised_rate=cfg.get("n2r_supervised_rate", 0.0),
                n2r_probability=cfg.get("n2r_probability", 0.5),
                n2r_std_devs=cfg.get("n2r_std_devs", (0.0, 0.0)),
                n2r_rhos=cfg.get("n2r_rhos", (0.4, 0.4)),
                n2r_use_mask=cfg.get("n2r_use_mask", True),
                unsupervised_masked_target=cfg.get("unsupervised_masked_target", False),
                crop_size=cfg.get("crop_size", None),
                kspace_crop=cfg.get("kspace_crop", False),
                crop_before_masking=cfg.get("crop_before_masking", False),
                kspace_zero_filling_size=cfg.get("kspace_zero_filling_size", None),
                normalize_inputs=cfg.get("normalize_inputs", True),
                normalization_type=cfg.get("normalization_type", "max"),
                kspace_normalization=cfg.get("kspace_normalization", False),
                fft_centered=cfg.get("fft_centered", False),
                fft_normalization=cfg.get("fft_normalization", "backward"),
                spatial_dims=cfg.get("spatial_dims", None),
                coil_dim=cfg.get("coil_dim", 1),
                consecutive_slices=cfg.get("consecutive_slices", 1),
                use_seed=cfg.get("use_seed", True),
                construct_knee_label=cfg.get('construct_knee_label', False),
                include_background_label=cfg.get('include_background_label', False),
            ),
            segmentations_root=cfg.get("segmentations_path"),
            segmentation_classes=cfg.get("segmentation_classes", 2),
            segmentation_classes_to_remove=cfg.get("segmentation_classes_to_remove", None),
            segmentation_classes_to_combine=cfg.get("segmentation_classes_to_combine", None),
            segmentation_classes_to_separate=cfg.get("segmentation_classes_to_separate", None),
            segmentation_classes_thresholds=cfg.get("segmentation_classes_thresholds", None),
            complex_data=complex_data,
            annotations_root=cfg.get('annotations_path', None)
        )
        if cfg.shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.get("batch_size", 1),
            sampler=sampler,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
