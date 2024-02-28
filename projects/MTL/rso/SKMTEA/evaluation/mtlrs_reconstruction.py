# coding=utf-8
__author__ = "Dimitris Karkalousos"

import json
import os
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
import torch
from atommic.collections.reconstruction.metrics.reconstruction_metrics import (
    ReconstructionMetrics,
    mse,
    nmse,
    psnr,
    ssim,
haarpsi3d,
vsi3d

)
import matplotlib.pyplot as plt

METRIC_FUNCS = {"SSIM": ssim, "HaarPSI":haarpsi3d,"VSI":vsi3d}


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
        scores = ReconstructionMetrics(METRIC_FUNCS)
        #For now reconstruction made based on ksapce and maps not saved in file yet
        reconstruction = h5py.File(Path(args.reconstructions_dir) / str(target).rsplit("/", maxsplit=1)[-1], "r")[
            "reconstruction"
        ][()].squeeze()
        if'target_reconstruction' in h5py.File(Path(args.reconstructions_dir) / str(target).rsplit("/", maxsplit=1)[-1], "r").keys():
            target_scan = h5py.File(Path(args.reconstructions_dir) / str(target).rsplit("/", maxsplit=1)[-1], "r")[
                "target_reconstruction"
            ][()].squeeze()
        else:
            kspace = h5py.File(target, "r")["kspace"][()]
            maps = h5py.File(target, "r")["maps"][()]
            target_scan = np.zeros(reconstruction.shape)
            for i in range(target.shape[0]):
                kspace_fft_4 = torch.fft.ifftshift(torch.from_numpy(kspace[i, :, :, :, :]), dim=(0, 1))
                kspace_fft_5 = torch.fft.ifft2(kspace_fft_4, dim=(0, 1))
                kspace_fft_6 = torch.fft.fftshift(kspace_fft_5, dim=(0, 1))
                image_fft = kspace_fft_6 * torch.conj(torch.as_tensor(maps[i, :, :, :, :]))
                target_scan[i,...] = torch.sum(image_fft, dim=-1, keepdim=False)[:,:,0].numpy() ##Select which echo to compare
        if 'target_segmentation' in h5py.File(Path(args.reconstructions_dir) / str(target).rsplit("/", maxsplit=1)[-1], "r").keys() and args.seg ==True:
            target_seg = h5py.File(Path(args.reconstructions_dir) / str(target).rsplit("/", maxsplit=1)[-1], "r")[
                "target_segmentation"
            ][()].squeeze()
            target_seg = np.abs(target_seg.astype(np.float32))
            target_seg = np.where(target_seg > 0.5, 1, 0)
            target_seg =np.sum(target_seg,axis=1)
        # normalize per slice
        for sl in range(target_scan.shape[0]):
            if args.normalisation_method == 'complex_abs':
                target_scan[sl] = target_scan[sl]**2
                reconstruction[sl] = reconstruction[sl] **2
            if args.normalisation_method == 'complex_abs_sqrt':
                target_scan[sl] = np.sqrt(target_scan[sl] ** 2)
                reconstruction[sl] = np.sqrt(reconstruction[sl] ** 2)
            if  args.normalisation_method == 'stacked':
                target_scan[sl] = target_scan[sl]
                reconstruction[sl] = reconstruction[sl]




            target_scan[sl] = target_scan[sl] / np.max(np.abs(target_scan[sl]))
            reconstruction[sl] = reconstruction[sl] / np.max(np.abs(reconstruction[sl]))
        if args.seg==True:
            reconstruction = np.abs(reconstruction).real.astype(np.float32)*target_seg
            target_scan = np.abs(target_scan).real.astype(np.float32)*target_seg
        else:
            reconstruction = np.abs(reconstruction).real.astype(np.float32)
            target_scan = np.abs(target_scan).real.astype(np.float32)
        maxvalue = max(np.max(target_scan), np.max(reconstruction))

        if evaluation_type == "per_slice":
            for sl in range(target_scan.shape[0]):
                if args.seg ==True:
                    non_zero_coord = np.column_stack(np.where(target_scan[sl] > 0))
                    if len(non_zero_coord) != 0:
                        min_x, min_y = np.min(non_zero_coord, axis=0)
                        max_x, max_y = np.max(non_zero_coord, axis=0)
                        if (max_x-min_x) >16 and (max_y-min_y)>16:
                            reconstruction_slice = reconstruction[sl,min_x:max_x, min_y:max_y]
                            target_slice = target_scan[sl,min_x:max_x, min_y:max_y]
                        else:
                            reconstruction_slice = reconstruction[sl, min_x-10:max_x+10, min_y-10:max_y+10]
                            target_slice = target_scan[sl, min_x-10:max_x+10, min_y-10:max_y+10]
                    else:
                        reconstruction_slice = reconstruction[sl]
                        target_slice = target_scan[sl]
                else:
                    reconstruction_slice = reconstruction[sl]
                    target_slice = target_scan[sl]
                scores.push(target_slice, reconstruction_slice, maxval=maxvalue)

        elif evaluation_type == "per_volume":
            scores.push(target, reconstruction, maxval=maxvalue)

        model = args.reconstructions_dir.split("/")
        model = model[-4] if model[-4] != "default" else model[-5]
        print(f"{model+'_'+ str(target).rsplit('/', maxsplit=1)[-1]}: {repr(scores)}")
        new_row = {"patiend_id": str(target).rsplit('/', maxsplit=1)[-1].replace('.h5', "")}
        new_row.update(scores.means())
        dataframe = dataframe._append(new_row, ignore_index=True)
        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # if file exists dont' overwrite, but append in a new line
            with open(output_dir / ("results_reconstruction_" + args.normalisation_method + "_seg:"+ str(args.seg) +  "_.txt"), "a", encoding="utf-8") as f:
                f.write(f"{model+'_'+ str(target).rsplit('/', maxsplit=1)[-1]}: {repr(scores)}\n")
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_dir / ("results_reconstruction_" + args.normalisation_method + "_seg:"+ str(args.seg) + "_.csv") )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("targets_dir", type=str)
    parser.add_argument("reconstructions_dir", type=str)
    parser.add_argument("normalisation_method", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--seg", type=bool,default=False)
    parser.add_argument("--evaluation_type", choices=["per_slice", "per_volume"], default="per_slice")
    parser.add_argument("--fill_target_path", action="store_true")
    parser.add_argument("--fill_pred_path", action="store_true")
    args = parser.parse_args()

    if args.fill_target_path:
        input_dir = ""
        for root, dirs, files in os.walk(args.targets_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        # check if after dir we have "/reconstructions" or "/predictions" dir
        if os.path.exists(os.path.join(input_dir, "reconstructions")):
            args.targets_dir = os.path.join(input_dir, "reconstructions")
        elif os.path.exists(os.path.join(input_dir, "predictions")):
            args.targets_dir = os.path.join(input_dir, "predictions")

    if args.fill_pred_path:
        input_dir = ""
        for root, dirs, files in os.walk(args.reconstructions_dir, topdown=False):
            for name in dirs:
                input_dir = os.path.join(root, name)
        # check if after dir we have "/reconstructions" or "/predictions" dir
        if os.path.exists(os.path.join(input_dir, "reconstructions")):
            args.reconstructions_dir = os.path.join(input_dir, "reconstructions")
        elif os.path.exists(os.path.join(input_dir, "predictions")):
            args.reconstructions_dir = os.path.join(input_dir, "predictions")

    main(args)
