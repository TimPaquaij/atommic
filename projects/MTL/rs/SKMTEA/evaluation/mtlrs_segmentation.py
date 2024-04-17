# coding=utf-8
__author__ = "Dimitris Karkalousos"

import json
import os
from pathlib import Path
import torch
import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm
import pandas as pd
from atommic.collections.segmentation.losses.dice import one_hot
from atommic.collections.segmentation.metrics.segmentation_metrics import (
    SegmentationMetrics,
    dice_metric,
    f1_per_class_metric,
    hausdorff_distance_95_metric,
    iou_metric,
)

METRIC_FUNCS = {
    "DICE": dice_metric,
    "F1": f1_per_class_metric,
    "HD95": hausdorff_distance_95_metric,
    "IOU": iou_metric,
}


def process_segmentation_labels(segmentation_labels):
    """Process the segmentation labels to match the predictions."""
    # 0: Patellar Cartilage
    patellar_cartilage = np.zeros_like(segmentation_labels)
    patellar_cartilage[segmentation_labels == 1] = 1
    # 1: Femoral Cartilage
    femoral_cartilage = np.zeros_like(segmentation_labels)
    femoral_cartilage[segmentation_labels == 2] = 1
    # 2: Lateral Tibial Cartilage
    lateral_tibial_cartilage = np.zeros_like(segmentation_labels)
    lateral_tibial_cartilage[segmentation_labels == 3] = 1
    # 3: Medial Tibial Cartilage
    medial_tibial_cartilage = np.zeros_like(segmentation_labels)
    medial_tibial_cartilage[segmentation_labels == 4] = 1
    # 4: Lateral Meniscus
    lateral_meniscus = np.zeros_like(segmentation_labels)
    lateral_meniscus[segmentation_labels == 5] = 1
    # 5: Medial Meniscus
    medial_meniscus = np.zeros_like(segmentation_labels)
    medial_meniscus[segmentation_labels == 6] = 1
    # combine Lateral Tibial Cartilage and Medial Tibial Cartilage
    tibial_cartilage = lateral_tibial_cartilage + medial_tibial_cartilage
    # combine Lateral Meniscus and Medial Meniscus
    medial_meniscus = lateral_meniscus + medial_meniscus

    segmentation_labels = np.stack(
        [patellar_cartilage, femoral_cartilage, tibial_cartilage, medial_meniscus],
        axis=1,
    )

    # TODO: This is hardcoded on the SKM-TEA side, how to generalize this?
    # We need to crop the segmentation labels in the frequency domain to reduce the FOV.
    segmentation_labels = np.fft.fftshift(np.fft.fft2(segmentation_labels))
    segmentation_labels = segmentation_labels[:, :, 48:-48, 40:-40]
    segmentation_labels = np.fft.ifft2(np.fft.ifftshift(segmentation_labels)).real

    return segmentation_labels


def main(args):
    # if json file
    if args.targets_dir.endswith(".json"):
        with open(args.targets_dir, "r", encoding="utf-8") as f:
            targets = json.load(f)
        targets = [Path(target) for target in targets]
    else:
        targets = list(Path(args.targets_dir).iterdir())

    evaluation_type = args.evaluation_type
    dataframe = pd.DataFrame()
    for target in tqdm(targets):
        scores = SegmentationMetrics(METRIC_FUNCS)
        fname = str(target).rsplit("/", maxsplit=1)[-1]
        if ".h5" in fname:
            fname = fname.split(".h5")[0]
        elif ".nii" in fname:
            fname = fname.split(".nii")[0]

        predictions = h5py.File(Path(args.predictions_dir) / f"{fname}.h5", "r")["segmentation"][()].squeeze()
        predictions = torch.softmax(torch.from_numpy(predictions), dim=1)
        predictions = one_hot(torch.argmax(torch.abs(predictions), dim=1, keepdim=True),
                                     num_classes=predictions.shape[1]).float().numpy()

        segmentation_labels = h5py.File(Path(args.predictions_dir) / f"{fname}.h5", "r")["target_segmentation"][()].squeeze()
        #segmentation_labels = np.transpose(segmentation_labels, (0, 3, 1, 2))
        #segmentation_labels = process_segmentation_labels(segmentation_labels)
        segmentation_labels = torch.softmax(torch.from_numpy(segmentation_labels), dim=1)
        segmentation_labels = one_hot(torch.argmax(torch.abs(segmentation_labels), dim=1, keepdim=True),
                              num_classes=segmentation_labels.shape[1]).float().numpy()
        if evaluation_type == "per_slice":
            for sl in range(segmentation_labels.shape[0]):
                scores.push(segmentation_labels[sl,1:][np.newaxis], predictions[sl,1:][np.newaxis])
        elif evaluation_type == "per_volume":
            scores.push(segmentation_labels[:,1:], predictions[:,1:])

        model = args.predictions_dir.split("/")
        model = model[-4] if model[-4] != "default" else model[-5]
        print(f"{model+'_'+ str(target).rsplit('/', maxsplit=1)[-1]}: {repr(scores)}")
        new_row = {"patiend_id":str(target).rsplit('/', maxsplit=1)[-1].replace('.h5',"")}
        new_row.update(scores.means())
        dataframe= dataframe._append(new_row,ignore_index=True)
        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # if file exists dont' overwrite, but append in a new line
            with open(output_dir / "results_segmentation.txt", "a", encoding="utf-8") as f:
                f.write(f"{model+'_'+ str(target).rsplit('/', maxsplit=1)[-1]}: {repr(scores)}")
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_dir / "results_segmentation.csv")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("targets_dir", type=str)
    #parser.add_argument("segmentations_dir", type=str)
    parser.add_argument("predictions_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--dataset_format", choices=["skm-tea", "brats", "private"], default="private")
    parser.add_argument("--evaluation_type", choices=["per_slice", "per_volume"], default="per_slice")
    parser.add_argument("--fill_target_path", action="store_true")
    parser.add_argument("--fill_pred_path", action="store_true")
    args = parser.parse_args()
    print(args)
    if args.fill_target_path:
        input_dir = ""
        for root, dirs, files in os.walk(args.targets_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        # check if after dir we have "/segmentations" or "/predictions" dir
        if os.path.exists(os.path.join(input_dir, "segmentations")):
            args.targets_dir = os.path.join(input_dir, "segmentations")
        elif os.path.exists(os.path.join(input_dir, "predictions")):
            args.targets_dir = os.path.join(input_dir, "predictions")

    if args.fill_pred_path:
        input_dir = ""
        for root, dirs, files in os.walk(args.predictions_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        # check if after dir we have "/segmentations" or "/predictions" dir
        if os.path.exists(os.path.join(input_dir, "segmentations")):
            args.predictions_dir = os.path.join(input_dir, "segmentations")
        elif os.path.exists(os.path.join(input_dir, "predictions")):
            args.predictions_dir = os.path.join(input_dir, "predictions")

    main(args)
