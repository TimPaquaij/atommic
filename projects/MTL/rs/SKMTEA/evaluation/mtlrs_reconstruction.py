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

METRIC_FUNCS = {"SSIM": ssim, "HaarPSI":haarpsi3d,"VSI":vsi3d, "PSNR":psnr}


def main(args):
    # if json file
    if args.targets_dir.endswith(".json"):
        with open(args.targets_dir, "r", encoding="utf-8") as f:
            targets = json.load(f)
        targets = [Path(target) for target in targets]
    else:
        targets = list(Path(args.targets_dir).iterdir())

    evaluation_type = args.evaluation_type

    dataframe_echo_1 = pd.DataFrame()
    dataframe_echo_2 = pd.DataFrame()
    dataframe = pd.DataFrame()
    for target in tqdm(targets):
        #For now reconstruction made based on ksapce and maps not saved in file yet
        reconstruction = h5py.File(Path(args.reconstructions_dir) / str(target).rsplit("/", maxsplit=1)[-1], "r")[
            "reconstruction"
        ][()].squeeze()
        if 'intermediate_reconstruction' in h5py.File(Path(args.reconstructions_dir) / str(target).rsplit("/", maxsplit=1)[-1], "r").keys() and args.inter:
            reconstruction = h5py.File(Path(args.reconstructions_dir) / str(target).rsplit("/", maxsplit=1)[-1], "r")[
                "intermediate_reconstruction"
            ][()].squeeze()[:,0]
        if "target" in h5py.File(target, "r").keys() and args.target:
            target_scan =np.transpose(h5py.File(target, "r")["target"][()].squeeze(),(0,3,1,2))
        elif 'target_reconstruction' in h5py.File(Path(args.reconstructions_dir) / str(target).rsplit("/", maxsplit=1)[-1], "r").keys():
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

        target_scan = np.abs(target_scan)
        reconstruction = np.abs(reconstruction)

        if target_scan.ndim ==4 and reconstruction.ndim ==4:
            for i in range(target_scan.shape[1]):
                scores = ReconstructionMetrics(METRIC_FUNCS)
                if evaluation_type == "per_slice":
                    for sl in range(target_scan.shape[0]):
                        reconstruction_slice = reconstruction[sl,i]
                        target_slice = target_scan[sl,i]
                        scores.push(target_slice, reconstruction_slice)

                elif evaluation_type == "per_volume":
                    scores.push(target_scan[:,i], reconstruction[:,i])

                model = args.reconstructions_dir.split("/")
                model = model[-4] if model[-4] != "default" else model[-5]
                new_row = {"id": str(target).rsplit('/', maxsplit=1)[-1].replace('.h5', "")}
                new_row.update(scores.means())
                if i ==0:
                    dataframe_echo_1 = dataframe_echo_1._append(new_row, ignore_index=True)
                else:
                    dataframe_echo_2=dataframe_echo_2._append(new_row, ignore_index=True)
        elif target_scan.ndim ==4 and reconstruction.ndim ==5:
            for i in range(reconstruction.shape[2]):
                for c in range(reconstruction.shape[1]):
                    scores = ReconstructionMetrics(METRIC_FUNCS)
                    if evaluation_type == "per_slice":
                        print(target_scan.shape, reconstruction.shape)
                        for sl in range(reconstruction.shape[0]):
                            reconstruction_slice = reconstruction[sl,c,i]
                            target_slice = target_scan[sl,i]
                            scores.push(target_slice, reconstruction_slice)

                    elif evaluation_type == "per_volume":
                        print(target_scan.shape,reconstruction.shape)
                        scores.push(target_scan[:,i], reconstruction[:,c,i])

                    model = args.reconstructions_dir.split("/")
                    model = model[-4] if model[-4] != "default" else model[-5]
                    new_row = {"id": str(target).rsplit('/', maxsplit=1)[-1].replace('.h5', ""),"Cascade":int(c+1)}
                    new_row.update(scores.means())
                    if i ==0:
                        dataframe_echo_1 = dataframe_echo_1._append(new_row, ignore_index=True)
                    else:
                        dataframe_echo_2=dataframe_echo_2._append(new_row, ignore_index=True)
        else:
            scores = ReconstructionMetrics(METRIC_FUNCS)
            if evaluation_type == "per_slice":
                print(target_scan.shape, reconstruction.shape)
                for sl in range(target_scan.shape[0]):
                    reconstruction_slice = reconstruction[sl]
                    target_slice = target_scan[sl]
                    scores.push(target_slice, reconstruction_slice)

            elif evaluation_type == "per_volume":
                scores.push(target_scan, reconstruction)

            # model = args.reconstructions_dir.split("/")
            # model = model[-4] if model[-4] != "default" else model[-5]
            new_row = {"id": str(target).rsplit('/', maxsplit=1)[-1].replace('.h5', "")}
            new_row.update(scores.means())
            dataframe = dataframe._append(new_row, ignore_index=True)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.reconstructions_dir)
    with pd.ExcelWriter(output_dir / "results_reconstruction_inter.xlsx") as writer:
        dataframe_echo_1.to_excel(writer,sheet_name="Echo 1" )
        dataframe_echo_2.to_excel(writer, sheet_name="Echo 2")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("targets_dir", type=str)
    parser.add_argument("reconstructions_dir", type=str)
    parser.add_argument("--normalisation_method", type=str, default='stacked')
    parser.add_argument("--target", type=bool, default=False)
    parser.add_argument("--inter", type=bool, default=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--evaluation_type", choices=["per_slice", "per_volume"], default="per_volume")
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
