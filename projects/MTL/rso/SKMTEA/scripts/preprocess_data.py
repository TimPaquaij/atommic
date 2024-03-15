# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch.fft
import shutil
from tqdm import tqdm
import nibabel as nib
from typing import Sequence, Union
from scipy.interpolate import RegularGridInterpolator
import pandas as pd

def categorical_to_one_hot(x, channel_dim: int = 1, background=0, num_categories=None, dtype=None):
    """Converts categorical predictions to one-hot encoded predictions.

    Args:
        x (torch.Tensor | np.ndarray): Categorical array or tensor.
        channel_dim (int, optional): Channel dimension for output tensor.
        background (int | NoneType, optional): The numerical label of the
            background category. If ``None``, assumes that the background is
            a class that should be one-hot encoded.
        num_categories (int, optional): Number of categories (excluding background).
            Defaults to the ``max(x) + 1``.
        dtype (type, optional): Data type of the output.
            Defaults to boolean (``torch.bool`` or ``np.bool``).

    Returns:
        torch.Tensor | np.ndarray: One-hot encoded predictions.
    """
    is_ndarray = isinstance(x, np.ndarray)
    if is_ndarray:
        x = torch.from_numpy(x)

    if num_categories is None:
        num_categories = torch.max(x).type(torch.long).cpu().item()
    num_categories += 1

    shape = x.shape
    out_shape = (num_categories,) + shape

    if dtype is None:
        dtype = torch.bool
    default_value = True if dtype == torch.bool else 1
    if x.dtype != torch.long:
        x = x.type(torch.long)

    out = torch.zeros(out_shape, dtype=dtype, device=x.device)
    out.scatter_(0, x.reshape((1,) + x.shape), default_value)
    if background is not None:
        out = torch.cat([out[0:background], out[background + 1 :]], dim=0)
    if channel_dim != 0:
        if channel_dim < 0:
            channel_dim = out.ndim + channel_dim
        order = (channel_dim,) + tuple(d for d in range(out.ndim) if d != channel_dim)
        out = out.permute(tuple(np.argsort(order)))
        out = out.contiguous()

    if is_ndarray:
        out = out.numpy()
    return out

def collect_mask(
    mask: np.ndarray,
    index: Sequence[Union[int, Sequence[int], int]],
    out_channel_first: bool = True,
):
    """Collect masks by index.

    Collated indices will be summed. For example, `index=(1,(3,4))` will return
    `np.stack(mask[...,1], mask[...,3]+mask[...,4])`.

    TODO: Add support for adding background.

    Args:
        mask (ndarray): A (...)xC array.
        index (Sequence[int]): The index/indices to select in mask.
            If sub-indices are collated, they will be summed.
        out_channel_first (bool, optional): Reorders dimensions of output mask to Cx(...)
    """
    if isinstance(index, int):
        index = (index,)

    if not any(isinstance(idx, Sequence) for idx in index):
        mask = mask[..., index]
    else:
        o_seg = []
        for idx in index:
            c_seg = mask[..., idx]
            if isinstance(idx, Sequence):
                c_seg = np.sum(c_seg, axis=-1)
            o_seg.append(c_seg)
        mask = np.stack(o_seg, axis=-1)

    if out_channel_first:
        last_idx = len(mask.shape) - 1
        mask = np.transpose(mask, (last_idx,) + tuple(range(0, last_idx)))

    return mask

def one_hot_to_categorical(x, channel_dim: int = 1, background=False):
    """Converts one-hot encoded predictions to categorical predictions.

    Args:
        x (torch.Tensor | np.ndarray): One-hot encoded predictions.
        channel_dim (int, optional): Channel dimension.
            Defaults to ``1`` (i.e. ``(B,C,...)``).
        background (bool, optional): If ``True``, assumes index 0 in the
            channel dimension is the background.

    Returns:
        torch.Tensor | np.ndarray: Categorical array or tensor. If ``background=False``,
        the output will be 1-indexed such that ``0`` corresponds to the background.
    """
    is_ndarray = isinstance(x, np.ndarray)
    if is_ndarray:
        x = torch.as_tensor(x)

    if background is not None and background is not False:
        out = torch.argmax(x, channel_dim)
    else:
        out = torch.argmax(x.type(torch.long), dim=channel_dim) + 1
        out = torch.where(x.sum(channel_dim) == 0, torch.tensor([0], device=x.device), out)

    if is_ndarray:
        out = out.numpy()
    return out
def main(args):
    if args.data_type == "raw":
        data_type = "files_recon_calib-24"
        seg_type = str("segmentation_masks/raw-data-track")
        annotation_type = str("annotations/v1.0.0/")
    else:
        data_type = "image_files"

    # remove "annotations/v1.0.0/" from args.annotations_path and add "files_recon_calib-24" to get the raw_data_path
    save_data_path = Path(args.target_dir) / data_type

    seg_data_path = Path(args.data_path).parent /seg_type
    raw_data_path = Path(args.data_path)
    crop_scale = args.crop_scale


    seg_save_data_path = Path(args.target_dir) /seg_type
    annotation_save_path = Path(args.target_dir) / annotation_type
    # get train.json, val.json and test.json filenames from args.annotations_path
    annotations_sets = list(Path(args.annotations_path_org).iterdir())
    print(annotations_sets)
    for annotation_set_path in annotations_sets:

        # read json file
        with open(annotation_set_path, "r", encoding="utf-8") as f:
            annotation_set = json.load(f)
        # read the "images" key and for every instance get the "file_name" key
        for image in tqdm(range(len(annotation_set["images"]))):
            rfname = f'{raw_data_path}/{annotation_set["images"][image]["file_name"]}'
            wfname = f'{save_data_path}/{annotation_set["images"][image]["file_name"]}'
            segrfname = f'{seg_data_path}/{annotation_set["images"][image]["file_name"]}'.replace('.h5','.nii.gz')
            segwfname = f'{seg_save_data_path}/{annotation_set["images"][image]["file_name"]}'.replace('.h5', '.nii.gz')



            with h5py.File(rfname, "r") as rf:
                kspace = rf['kspace'][()]
                maps = rf['maps'][()]

            segmentation = nib.load(segrfname).get_fdata()
            segmentation_one = categorical_to_one_hot(segmentation,channel_dim=-1)
            segmentation_one = collect_mask(segmentation_one, (0, 1, (2, 3), (4, 5)), out_channel_first=False)

            # Remove undersampling kspace and prepare for lateral reconstruction
            kspace = kspace[:, 48:-48, 40:-40, ...]
            kspace = torch.fft.fft(torch.fft.ifftshift(torch.as_tensor(kspace), dim=0), dim=0)
            kspace = torch.fft.fftshift(kspace, dim=0)
            kspace = kspace[int(kspace.shape[0] / 2 - (kspace.shape[0] * crop_scale[0] / 2)):int(kspace.shape[0] / 2 + (kspace.shape[0] * crop_scale[0] / 2)),
                            int(kspace.shape[1] / 2 - (kspace.shape[1] * crop_scale[1] / 2)):int(kspace.shape[1] / 2 + (kspace.shape[1] * crop_scale[1] / 2)),
                            int(kspace.shape[2] / 2 - (kspace.shape[2] * crop_scale[2] / 2)):int(kspace.shape[2] / 2 + (kspace.shape[2] * crop_scale[2] / 2)),
                            :,
                            :]

            kspace = torch.fft.ifft(torch.fft.ifftshift(kspace, dim=2), dim=2)
            kspace = torch.fft.fftshift(kspace, dim=2).numpy()
            kspace = np.transpose(kspace, (2, 0, 1, 3, 4))
            annotation_set["images"][image]['matrix_shape'] = [kspace.shape[0], kspace.shape[1], kspace.shape[2]]
            # print('Cropped and hybridized kspace',kspace.shape)
            ####Interpolate segmentation ####
            arr_shape = np.transpose(maps,(2,0,1,3,4)).shape[0:3]

            pixelsize_x_old = annotation_set["images"][image]['voxel_spacing'][0]
            pixelsize_y_old = annotation_set["images"][image]['voxel_spacing'][1]
            slice_thickness_old = annotation_set["images"][image]['voxel_spacing'][2]

            pixelsize_x_new = pixelsize_x_old * (1 / crop_scale[0])
            pixelsize_y_new = pixelsize_y_old * (1 / (416 / 512)) * (1 / crop_scale[1])
            slice_thickness_new = slice_thickness_old * (1 / (80 / 160)) * (1 / crop_scale[2])

            annotation_set["images"][image]['voxel_spacing'][0] =  slice_thickness_new
            annotation_set["images"][image]['voxel_spacing'][1] = pixelsize_x_new
            annotation_set["images"][image]['voxel_spacing'][2] = pixelsize_y_new


            x_old = np.linspace(0, (arr_shape[1] - 1) * pixelsize_x_old, arr_shape[1])
            y_old = np.linspace(0, (arr_shape[2] - 1) * pixelsize_y_old, arr_shape[2])
            z_old = np.arange(0, (arr_shape[0])) * slice_thickness_old

            x_new = kspace.shape[1]
            y_new = kspace.shape[2]
            z_new = kspace.shape[0]

            # pts is the new grid
            pts = np.indices((z_new, x_new, y_new)).transpose((1, 2, 3, 0))
            pts = pts.reshape(1, z_new * x_new * y_new, 1, 3).reshape(z_new * x_new * y_new, 3)
            pts = np.array(pts, dtype=float)
            pts[:, 1] = pts[:, 1] * pixelsize_x_new
            pts[:, 2] = pts[:, 2] * pixelsize_y_new
            pts[:, 0] = pts[:, 0] * slice_thickness_new

            ##### Interpolate Segmentation  #####
            target_shape = np.array([z_new, x_new, y_new, segmentation_one.shape[3]], int)
            new_segmentation = np.zeros(shape=target_shape, dtype=int)
            for l in range(segmentation_one.shape[-1]):
                arr = np.transpose(segmentation_one[:, :, :, l], (2, 0, 1))
                my_interpolating_object = RegularGridInterpolator((z_old, x_old, y_old), arr, method='nearest',
                                                                  bounds_error=False)
                interpolated_data = my_interpolating_object(pts)
                interpolated_data = interpolated_data.reshape(z_new, x_new, y_new)
                new_segmentation[:, :, :, l] = interpolated_data
                print('Interpolated Segmentaion Class:', l+1)
            new_segmentation = np.transpose(new_segmentation, (0, 1, 2, 3))

            print('Interpolated all Segmentation Classes', new_segmentation.shape)
            ##### Interpolate Sensitivity Maps ######
            target_shape = np.array([z_new, x_new, y_new, maps.shape[3], maps.shape[4]], int)
            new_arr = np.empty(shape=target_shape, dtype=complex)
            for k in range(maps.shape[-2]):
                map = np.transpose(maps[:, :, :, k, 0], (2, 0, 1))
                my_interpolating_object = RegularGridInterpolator((z_old, x_old, y_old), map, method='nearest',
                                                                  bounds_error=False)
                interpolated_data = my_interpolating_object(pts)
                interpolated_data = interpolated_data.reshape(z_new, x_new, y_new)
                new_arr[:, :, :, k, 0] = interpolated_data
                print('Interpolated Sensitivity map:', k + 1)
            maps = np.transpose(new_arr, (0, 1, 2, 4, 3))
            print('Interpolated all Sensitivity Coils', maps.shape)
            annotation_set["images"][image]['matrix_shape'] = [z_new,x_new,y_new]
            annotation_set["images"][image]['orientation'] = [annotation_set["images"][image]['orientation'][2],annotation_set["images"][image]['orientation'][0],annotation_set["images"][image]['orientation'][1]]

            #### Crop Sensitivity Maps #######
            maps = maps[48:-48, :, 40:-40, ...]
            maps = maps[int(maps.shape[0] / 2 - (maps.shape[0] * crop_scale[0] / 2)):int(
                maps.shape[0] / 2 + (maps.shape[0] * crop_scale[0] / 2)),
                     int(maps.shape[1] / 2 - (maps.shape[1] * crop_scale[1] / 2)):int(
                         maps.shape[1] / 2 + (maps.shape[1] * crop_scale[1] / 2)),
                     int(maps.shape[2] / 2 - (maps.shape[2] * crop_scale[2] / 2)):int(
                         maps.shape[2] / 2 + (maps.shape[2] * crop_scale[2] / 2)),
                     :,
                     :]
            maps = np.transpose(maps, (2, 0, 1, 4, 3))
            ###### Object Detection ######
            for annotation in range(len(annotation_set['annotations'])):
                if annotation_set['annotations'][annotation]['image_id'] == annotation_set["images"][image]['id']:
                    print(annotation_set['annotations'][annotation]['bbox'])
                    annotation_set['annotations'][annotation]['bbox'] = [int((80/160)*crop_scale[2]*annotation_set['annotations'][annotation]['bbox'][2]),int(crop_scale[0]*annotation_set['annotations'][annotation]['bbox'][0]),int(((512-96)/512)*annotation_set['annotations'][annotation]['bbox'][1]*crop_scale[1]),int((80/160)*crop_scale[2]*annotation_set['annotations'][annotation]['bbox'][5]),int(crop_scale[0]*annotation_set['annotations'][annotation]['bbox'][3]),int(((512-96)/512)*annotation_set['annotations'][annotation]['bbox'][4]*crop_scale[1])]
                    print(annotation_set['annotations'][annotation]['bbox'])
            with h5py.File(wfname, "w") as wf:
                wf.create_dataset('kspace',data=kspace)
                wf.create_dataset('maps',data=maps)
            nib.save(nib.Nifti1Image(new_segmentation,affine=np.eye(4),dtype= float),segwfname)

        with open(annotation_save_path/annotation_set_path.name,"w") as f:
            json.dump(annotation_set,f)










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, default=None, help="Path to the raw files.")
    parser.add_argument("annotations_path_org", type=Path, default=None, help="Path to the annotations json file.")
    parser.add_argument("target_dir", type=Path, default=None, help="Path to the location")
    parser.add_argument("--data_type", choices=["raw", "image"], default="raw", help="Type of data to split.")
    parser.add_argument("--crop_scale", type=list, default=[1,1,1], help="List with scalers for croppig")
    args = parser.parse_args()
    main(args)