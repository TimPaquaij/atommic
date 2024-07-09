import h5py

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Sequence
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import torch
from atommic.collections.reconstruction.metrics.reconstruction_metrics import ssim,psnr,haarpsi3d,vsi3d
from atommic.collections.segmentation.metrics.segmentation_metrics import asd, dice_metric, hausdorff_distance_95_metric, f1_per_class_metric,iou_metric
from atommic.collections.segmentation.losses.dice import one_hot

def get_scaled_image(
        x: Union[torch.Tensor, np.ndarray], percentile=0.99, clip=False
):
    """Scales image by intensity percentile (and optionally clips to [0, 1]).

    Args:
      x (torch.Tensor | np.ndarray): The image to process.
      percentile (float): The percentile of magnitude to scale by.
      clip (bool): If True, clip values between [0, 1]

    Returns:
      torch.Tensor | np.ndarray: The scaled image.
    """
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x = torch.as_tensor(x)

    scale_factor = torch.quantile(x, percentile)
    x = x / scale_factor
    if clip:
        x = torch.clip(x, 0, 1)

    if is_numpy:
        x = x.numpy()

    return x


def plot_images(
        images, processor=None, disable_ticks=True, titles: Sequence[str] = None,
        ylabel: str = None, xlabels: Sequence[str] = None, cmap: str = "gray",
        show_cbar: bool = False, overlay=None, opacity: float = 0.4,
        hsize=5, wsize=5, axs=None, fontsize=40,text=None,show_cbar_overlay: bool=False,ticks_overlay:bool=False,ticks:bool=False,
):
    """Plot multiple images in a single row.

    Add an overlay with the `overlay=` argument.
    Add a colorbar with `show_cbar=True`.
    """

    def get_default_values(x, default=""):
        if x is None:
            return [default] * len(images)
        return x

    titles = get_default_values(titles)
    ylabels = get_default_values(images)
    xlabels = get_default_values(xlabels)

    N = len(images)
    if axs is None:
        fig, axs = plt.subplots(1, N, figsize=(wsize * N, hsize))
    else:
        assert len(axs) >= N
        fig = axs.flatten()[0].get_figure()
    k = 0
    for ax, img, title, xlabel in zip(axs, images, titles, xlabels):
        if processor is not None:
            img = processor(img)
        if type(cmap) == list:
            if cmap[k] =='viridis' or cmap[k] =='jet' :
                if type(ticks[0])==list:
                    im = ax.imshow(img, cmap=cmap[k], vmax=ticks[k][-1], vmin=ticks[k][0])
                else:
                    im = ax.imshow(img, cmap=cmap[k],vmax=ticks[-1],vmin=ticks[0])
            else:
                im = ax.imshow(img, cmap=cmap[k])
            if type(show_cbar) == list:
                if show_cbar[k]:
                    cbaxes = inset_axes(ax, width="30%", height="5%")
                    cbar = fig.colorbar(im, cax=cbaxes, orientation='horizontal', cmap='jet', ticks=ticks[k])
                    cbar.ax.set_xticklabels(labels=ticks[k], color='yellow', fontsize=fontsize,fontweight='bold')
        else:
            im = ax.imshow(img, cmap=cmap)
        k = k + 1
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel)

    if type(overlay) == list:
        for i, ax in enumerate(axs.flatten()):
            if overlay[i] is not None:
                im = ax.imshow(overlay[i], alpha=opacity,cmap='jet',vmax=ticks_overlay[-1],vmin=ticks_overlay[0])
                if type(show_cbar_overlay) == list:
                    if show_cbar_overlay[i]:
                        cbaxes = inset_axes(ax, width="25%", height="5%",loc=1,borderpad = 1)
                        cbar = fig.colorbar(im,cax=cbaxes, orientation='horizontal',cmap='jet',ticks=ticks_overlay)
                        cbar.ax.set_xticklabels(labels= ticks_overlay,color='yellow',fontsize=fontsize,fontweight='bold')

    if type(text) == list:
        for i, ax in enumerate(axs.flatten()):
            if text[i] is not None:
               im = ax.text(.01, .99, text[i], ha='left', va='top', transform=ax.transAxes,fontsize=fontsize,color='yellow',fontweight='bold')

    if disable_ticks:
        for ax in axs.flatten():
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    return axs


# Function for transforming segmentation classes into categorical
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



def complex_abs(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor. The last dimension must be of size 2.

    Returns
    -------
    torch.Tensor
        Absolute value of complex tensor.

    Examples
    --------
    >>> from atommic.collections.common.parts.utils import complex_abs
    >>> import torch
    >>> data = torch.tensor([1+1j, 2+2j, 3+3j])
    >>> complex_abs(data)
    tensor([1.4142, 2.8284, 4.2426])
    """
    if x.shape[-1] != 2:
        if torch.is_complex(x):
            x = torch.view_as_real(x)
        else:
            raise ValueError("Tensor does not have separate complex dim.")
    return (x ** 2).sum(dim=-1)


def check_stacked_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Check if tensor is stacked complex (real & imaginary parts stacked along last dim) and convert it to a combined
    complex tensor.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to check.

    Returns
    -------
    torch.Tensor
        Tensor with stacked complex converted to combined complex.

    Examples
    --------
    >>> from atommic.collections.common.parts.utils import check_stacked_complex
    >>> import torch
    >>> data = torch.tensor([1+1j, 2+2j, 3+3j])
    >>> data.shape
    torch.Size([3])
    >>> data = torch.view_as_real(data)
    >>> data.shape
    >>> check_stacked_complex(data)
    tensor([1.+1.j, 2.+2.j, 3.+3.j])
    >>> check_stacked_complex(data).shape
    torch.Size([3])
    >>> data = torch.tensor([1+1j, 2+2j, 3+3j])
    >>> data.shape
    torch.Size([3])
    >>> check_stacked_complex(data)
    tensor([1.+1.j, 2.+2.j, 3.+3.j])
    >>> check_stacked_complex(data).shape
    torch.Size([3])
    """
    return torch.view_as_complex(x) if x.shape[-1] == 2 else x


def generate_figure(patients,echo, slice,intermidiate_form,dataframe,ticks_overlay,hsize,fontsize):
    with pd.ExcelWriter(
            "/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/results_UQ_intermediate_"+ intermidiate_form+"_soft_tissue.xlsx") as writer:
        mean_all_patients = []
        std_all_patients = []
        mean_std_all_patients = []
        mean_all_patients_2= []
        std_inter_all_patients = []

        for Patient_id in patients:
            # %%
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SENSE_NO_MTL_def/2024-05-23_12-22-34/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                prediction_mtlrs_1 = f['segmentation'][()].squeeze()
                target_mtlrs_1 = f['target_reconstruction'][()].squeeze()
                reconstruction_mtlrs_1 = f['reconstruction'][()].squeeze()
                segmentation_labels_mtlrs_1 = f['target_segmentation'][()].squeeze()
                inter_pred_mtlrs_1 = f['intermediate_' + intermidiate_form][()].squeeze()
                zero_filled_mtlrs_1 = f['zero_filled'][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SENSE_NO_MTL_def/2024-05-23_12-22-37/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_2 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_2 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_2 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SENSE_NO_MTL_def/2024-05-23_12-22-40/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_3 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_3 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_3 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SENSE_NO_MTL_def/2024-06-08_11-56-28/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_4 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_4 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_4 = f['intermediate_' + intermidiate_form][()].squeeze()
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SENSE_NO_MTL_def/2024-06-09_19-16-53/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_5 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_5 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_5 = f['intermediate_' + intermidiate_form][()].squeeze()


            mean_no_mtl = np.mean([np.abs(inter_pred_mtlrs_1[:,0]),np.abs(inter_pred_mtlrs_2[:,0]),np.abs(inter_pred_mtlrs_3[:,0]),np.abs(inter_pred_mtlrs_4[:,0]),np.abs(inter_pred_mtlrs_5[:,0])],axis=0)
            std_no_mtl = np.std([np.abs(inter_pred_mtlrs_1[:,0]),np.abs(inter_pred_mtlrs_2[:,0]),np.abs(inter_pred_mtlrs_3[:,0]),np.abs(inter_pred_mtlrs_4[:,0]),np.abs(inter_pred_mtlrs_5[:,0])],axis=0)
            inter_std_no_mtl = np.mean([np.abs(inter_pred_mtlrs_1[:,1]),np.abs(inter_pred_mtlrs_2[:,1]),np.abs(inter_pred_mtlrs_3[:,1]),np.abs(inter_pred_mtlrs_4[:,1]),np.abs(inter_pred_mtlrs_5[:,1])],axis=0)
            reconstruction_no_mtl = np.mean(
                [np.abs(reconstruction_mtlrs_1), np.abs(reconstruction_mtlrs_2), np.abs(reconstruction_mtlrs_3), np.abs(reconstruction_mtlrs_4), np.abs(reconstruction_mtlrs_5)],
                axis=0)
            inter_seg_no_mtl = one_hot(torch.argmax(torch.softmax(torch.from_numpy(
                    np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3, prediction_mtlrs_4, prediction_mtlrs_5], axis=0)),
                    dim=2), dim=2, keepdim=True),
                    num_classes=segmentation_labels_mtlrs_1.shape[1],dim=2).float().numpy()
            mean_seg_no_mtl = torch.softmax(torch.from_numpy(
                    np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3, prediction_mtlrs_4, prediction_mtlrs_5], axis=0)),
                    dim=2).float().numpy()
            entropy_seg_no_mtl = -np.sum(mean_seg_no_mtl*np.log(mean_seg_no_mtl+1e-10),axis=2)
            print("prepared JOINT results")
            # %%
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_LOGIT_SUM_DEF_2/2024-06-02_18-24-03/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_1 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_1 = f['segmentation'][()].squeeze()

                inter_pred_mtlrs_1 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_LOGIT_SUM_DEF_2/2024-06-02_18-24-32/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_2 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_2 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_2 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_LOGIT_SUM_DEF_2/2024-06-02_18-24-37/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_3 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_3 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_3 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_LOGIT_SUM_DEF_2/2024-06-08_11-55-10/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_4 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_4 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_4 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_LOGIT_SUM_DEF_2/2024-06-08_11-55-07/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_5 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_5 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_5 = f['intermediate_' + intermidiate_form][()].squeeze()
            mean_logit = np.mean(
                [np.abs(inter_pred_mtlrs_1[:, 0]), np.abs(inter_pred_mtlrs_2[:, 0]), np.abs(inter_pred_mtlrs_3[:, 0]), np.abs(inter_pred_mtlrs_4[:, 0]), np.abs(inter_pred_mtlrs_5[:, 0])],
                axis=0)
            std_logit = np.std(
                [np.abs(inter_pred_mtlrs_1[:, 0]), np.abs(inter_pred_mtlrs_2[:, 0]), np.abs(inter_pred_mtlrs_3[:, 0]), np.abs(inter_pred_mtlrs_4[:, 0]), np.abs(inter_pred_mtlrs_5[:, 0])],
                axis=0)
            inter_std_logit = np.mean(
                [np.abs(inter_pred_mtlrs_1[:, 1]), np.abs(inter_pred_mtlrs_2[:, 1]), np.abs(inter_pred_mtlrs_3[:, 1]), np.abs(inter_pred_mtlrs_4[:, 1]), np.abs(inter_pred_mtlrs_5[:, 1])],
                axis=0)
            reconstruction_logit = np.mean(
                [np.abs(reconstruction_mtlrs_1), np.abs(reconstruction_mtlrs_2), np.abs(reconstruction_mtlrs_3), np.abs(reconstruction_mtlrs_4), np.abs(reconstruction_mtlrs_5)],
                axis=0)
            inter_seg_logit = one_hot(torch.argmax(torch.softmax(torch.from_numpy(
                np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3, prediction_mtlrs_4, prediction_mtlrs_5], axis=0)),
                dim=2), dim=2, keepdim=True),
                num_classes=segmentation_labels_mtlrs_1.shape[1],dim=2).float().numpy()
            mean_seg_logit = torch.softmax(torch.from_numpy(
                np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3, prediction_mtlrs_4,
                         prediction_mtlrs_5], axis=0)),
                dim=2).float().numpy()
            entropy_seg_logit = -np.sum(mean_seg_logit * np.log(mean_seg_logit + 1e-10), axis=2)
            print("prepared logit results")
            # %%
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_SUM_def/2024-05-30_10-14-44/predictions/{Patient_id}.h5'
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_SUM_DEF_2/2024-06-05_18-24-47/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_1 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_1 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_1 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_SUM_def/2024-05-30_10-14-41/predictions/{Patient_id}.h5'
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_SUM_DEF_2/2024-06-06_05-59-24/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_2 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_2 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_2 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_SUM_def/2024-05-29_11-49-51/predictions/{Patient_id}.h5'
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_SUM_DEF_2/2024-06-05_18-24-50/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_3 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_3 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_3 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_SUM_def/2024-05-29_11-49-51/predictions/{Patient_id}.h5'
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_SUM_DEF_2/2024-06-06_08-19-34/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_4 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_4 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_4 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_SUM_DEF_2/2024-06-09_11-35-15/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_5 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_5 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_5 = f['intermediate_' + intermidiate_form][()].squeeze()

            mean_softmax = np.mean(
                [np.abs(inter_pred_mtlrs_1[:, 0]), np.abs(inter_pred_mtlrs_2[:, 0]), np.abs(inter_pred_mtlrs_3[:, 0]), np.abs(inter_pred_mtlrs_4[:, 0]), np.abs(inter_pred_mtlrs_5[:, 0])],
                axis=0)
            std_softmax = np.std(
                [np.abs(inter_pred_mtlrs_1[:, 0]), np.abs(inter_pred_mtlrs_2[:, 0]), np.abs(inter_pred_mtlrs_3[:, 0]), np.abs(inter_pred_mtlrs_4[:, 0]), np.abs(inter_pred_mtlrs_5[:, 0])],
                axis=0)
            inter_std_softmax = np.mean(
                [np.abs(inter_pred_mtlrs_1[:, 1]), np.abs(inter_pred_mtlrs_2[:, 1]), np.abs(inter_pred_mtlrs_3[:, 1]), np.abs(inter_pred_mtlrs_4[:, 1]), np.abs(inter_pred_mtlrs_5[:, 1])],
                axis=0)
            reconstruction_softmax = np.mean(
                [np.abs(reconstruction_mtlrs_1), np.abs(reconstruction_mtlrs_2), np.abs(reconstruction_mtlrs_3), np.abs(reconstruction_mtlrs_4), np.abs(reconstruction_mtlrs_5)],
                axis=0)
            inter_seg_softmax = one_hot(torch.argmax(torch.softmax(torch.from_numpy(
                np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3, prediction_mtlrs_4,prediction_mtlrs_5], axis=0)),
                dim=2), dim=2, keepdim=True),
                num_classes=segmentation_labels_mtlrs_1.shape[1],dim=2).float().numpy()
            mean_seg_softmax = torch.softmax(torch.from_numpy(
                np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3, prediction_mtlrs_4,
                         prediction_mtlrs_5], axis=0)),
                dim=2).float().numpy()
            entropy_seg_softmax = -np.sum(mean_seg_softmax * np.log(mean_seg_softmax + 1e-10), axis=2)
            print("prepared softmax results")
            # %%

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SASG_DEF_2/2024-06-03_14-35-29/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                prediction_mtlrs_1 = f['segmentation'][()].squeeze()
                reconstruction_mtlrs_1 = f['reconstruction'][()].squeeze()
                inter_pred_mtlrs_1 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SASG_DEF_2/2024-06-04_10-08-40/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_2 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_2 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_2 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SASG_DEF_2/2024-06-04_11-32-04/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_3 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_3 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_3 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SASG_DEF_2/2024-06-04_11-33-32/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_4 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_4 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_4 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SASG_DEF_2/2024-06-04_11-33-44/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_5 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_5 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_5 = f['intermediate_' + intermidiate_form][()].squeeze()

            mean_sasg = np.mean(
                [np.abs(inter_pred_mtlrs_1[:, 0]), np.abs(inter_pred_mtlrs_2[:, 0]), np.abs(inter_pred_mtlrs_3[:, 0]) ,np.abs(inter_pred_mtlrs_4[:, 0]), np.abs(inter_pred_mtlrs_5[:, 0])],
                axis=0)
            std_sasg  = np.std(
                [np.abs(inter_pred_mtlrs_1[:, 0]), np.abs(inter_pred_mtlrs_2[:, 0]), np.abs(inter_pred_mtlrs_3[:, 0]),np.abs(inter_pred_mtlrs_4[:, 0]), np.abs(inter_pred_mtlrs_5[:, 0])],
                axis=0)
            inter_std_sasg = np.mean(
                [np.abs(inter_pred_mtlrs_1[:, 1]), np.abs(inter_pred_mtlrs_2[:, 1]), np.abs(inter_pred_mtlrs_3[:, 1]),np.abs(inter_pred_mtlrs_4[:, 1]), np.abs(inter_pred_mtlrs_5[:, 1])],
                axis=0)
            reconstruction_sasg = np.mean(
                [np.abs(reconstruction_mtlrs_1), np.abs(reconstruction_mtlrs_2), np.abs(reconstruction_mtlrs_3),np.abs(reconstruction_mtlrs_4), np.abs(reconstruction_mtlrs_5)],
                axis=0)
            inter_seg_sasg = one_hot(torch.argmax(torch.softmax(torch.from_numpy(
                np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3,prediction_mtlrs_4,prediction_mtlrs_5], axis=0)),
                dim=2), dim=2, keepdim=True),
                num_classes=segmentation_labels_mtlrs_1.shape[1],dim=2).float().numpy()
            mean_seg_sasg = torch.softmax(torch.from_numpy(
                np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3, prediction_mtlrs_4,
                         prediction_mtlrs_5], axis=0)),
                dim=2).float().numpy()
            entropy_seg_sasg = -np.sum(mean_seg_sasg * np.log(mean_seg_sasg + 1e-10), axis=2)
            print("prepared sasg results")
            # %%
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_TAM_def/2024-05-29_15-20-28/predictions/{Patient_id}.h5'
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_TAM_DEF_2/2024-06-06_15-48-49/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                prediction_mtlrs_1 = f['segmentation'][()].squeeze()
                reconstruction_mtlrs_1 = f['reconstruction'][()].squeeze()
                inter_pred_mtlrs_1 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_TAM_def/2024-05-29_15-20-31/predictions/{Patient_id}.h5'
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_TAM_DEF_2/2024-06-06_15-48-52/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_2 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_2 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_2 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_TAM_def/2024-05-31_08-40-15/predictions/{Patient_id}.h5'
            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_TAM_DEF_2/2024-06-07_18-50-31/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_3 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_3 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_3 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_TAM_DEF_2/2024-06-07_21-32-30/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_4 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_4 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_4 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_SOFTMAX_TAM_DEF_2/2024-06-08_04-11-21/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_5 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_5 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_5 = f['intermediate_' + intermidiate_form][()].squeeze()
            mean_softmax_tam = np.mean(
                [np.abs(inter_pred_mtlrs_1[:, 0]), np.abs(inter_pred_mtlrs_2[:, 0]), np.abs(inter_pred_mtlrs_3[:, 0]), np.abs(inter_pred_mtlrs_4[:, 0]), np.abs(inter_pred_mtlrs_5[:, 0])],
                axis=0)
            std_softmax_tam = np.std(
                [np.abs(inter_pred_mtlrs_1[:, 0]), np.abs(inter_pred_mtlrs_2[:, 0]), np.abs(inter_pred_mtlrs_3[:, 0]), np.abs(inter_pred_mtlrs_4[:, 0]), np.abs(inter_pred_mtlrs_5[:, 0])],
                axis=0)
            inter_std_softmax_tam = np.mean(
                [np.abs(inter_pred_mtlrs_1[:, 1]), np.abs(inter_pred_mtlrs_2[:, 1]), np.abs(inter_pred_mtlrs_3[:, 1]), np.abs(inter_pred_mtlrs_4[:, 1]), np.abs(inter_pred_mtlrs_5[:, 1])],
                axis=0)
            reconstruction_softmax_tam= np.mean(
                [np.abs(reconstruction_mtlrs_1), np.abs(reconstruction_mtlrs_2), np.abs(reconstruction_mtlrs_3), np.abs(reconstruction_mtlrs_4), np.abs(reconstruction_mtlrs_5)],
                axis=0)
            inter_seg_softmax_tam = one_hot(torch.argmax(torch.softmax(torch.from_numpy(
                np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3, prediction_mtlrs_4, prediction_mtlrs_5], axis=0)),
                dim=2), dim=2, keepdim=True),
                num_classes=segmentation_labels_mtlrs_1.shape[1],dim=2).float().numpy()
            mean_seg_softmax_tam = torch.softmax(torch.from_numpy(
                np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3, prediction_mtlrs_4,
                         prediction_mtlrs_5], axis=0)),
                dim=2).float().numpy()
            entropy_seg_softmax_tam = -np.sum(mean_seg_softmax_tam * np.log(mean_seg_softmax_tam + 1e-10), axis=2)
            print("prepared softmax TAM results")
            # %%

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_LOGIT_TAM_DEF_2/2024-06-02_21-02-20/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_1 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_1 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_1 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_LOGIT_TAM_DEF_2/2024-06-02_21-02-23/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_2 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_2 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_2 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_LOGIT_TAM_DEF_2/2024-06-02_21-02-30/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_3 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_3 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_3 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_LOGIT_TAM_DEF_2/2024-06-04_14-05-48/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_4 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_4 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_4 = f['intermediate_' + intermidiate_form][()].squeeze()

            fname = f'/data/projects/utwente/recon/SKM-TEA/v1-release/predictions/MTLRS_SENSE/predictions_8x/MTLRS_SKMTEA_LOGIT_TAM_DEF_2/2024-06-05_00-49-24/predictions/{Patient_id}.h5'
            with h5py.File(fname, "r") as f:
                reconstruction_mtlrs_5 = f['reconstruction'][()].squeeze()
                prediction_mtlrs_5 = f['segmentation'][()].squeeze()
                inter_pred_mtlrs_5 = f['intermediate_' + intermidiate_form][()].squeeze()
            mean_logit_tam = np.mean(
                [np.abs(inter_pred_mtlrs_1[:, 0]), np.abs(inter_pred_mtlrs_2[:, 0]), np.abs(inter_pred_mtlrs_3[:, 0]),np.abs(inter_pred_mtlrs_4[:, 0]),np.abs(inter_pred_mtlrs_5[:, 0])],
                axis=0)
            std_logit_tam = np.std(
                [np.abs(inter_pred_mtlrs_1[:, 0]), np.abs(inter_pred_mtlrs_2[:, 0]), np.abs(inter_pred_mtlrs_3[:, 0]),np.abs(inter_pred_mtlrs_4[:, 0]),np.abs(inter_pred_mtlrs_5[:, 0])],
                axis=0)
            inter_std_logit_tam = np.mean(
                [np.abs(inter_pred_mtlrs_1[:, 1]), np.abs(inter_pred_mtlrs_2[:, 1]), np.abs(inter_pred_mtlrs_3[:, 1]),np.abs(inter_pred_mtlrs_4[:, 1]),np.abs(inter_pred_mtlrs_5[:, 1])],
                axis=0)
            reconstruction_logit_tam = np.mean(
                [np.abs(reconstruction_mtlrs_1), np.abs(reconstruction_mtlrs_2), np.abs(reconstruction_mtlrs_3),np.abs(reconstruction_mtlrs_4),np.abs(reconstruction_mtlrs_5)],
                axis=0)
            inter_seg_logit_tam = one_hot(torch.argmax(torch.softmax(torch.from_numpy(
                np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3,prediction_mtlrs_4,prediction_mtlrs_5], axis=0)),
                dim=2), dim=2, keepdim=True),
                num_classes=segmentation_labels_mtlrs_1.shape[1],dim=2).float().numpy()
            mean_seg_logit_tam = torch.softmax(torch.from_numpy(
                np.mean([prediction_mtlrs_1, prediction_mtlrs_2, prediction_mtlrs_3, prediction_mtlrs_4,
                         prediction_mtlrs_5], axis=0)),
                dim=2).float().numpy()
            entropy_seg_logit_tam = -np.sum(mean_seg_logit_tam * np.log(mean_seg_logit_tam + 1e-10), axis=2)
            print("prepared logit TAM results")
            # %%

            zero_filled_mtlrs_1 = np.abs(zero_filled_mtlrs_1)
            target_mtlrs_1 = np.abs(target_mtlrs_1)

            # %%
            # wsize = hsize / target_mtlrs_1.shape[2] * target_mtlrs_1.shape[3]
            # fig, axs = plt.subplots(6, 5, figsize=(5 * wsize, 6 * hsize))
            # plot_images([mean_no_mtl[slice, 0, echo], mean_no_mtl[slice, 1, echo],
            #              mean_no_mtl[slice, 2, echo], mean_no_mtl[slice, 3, echo], mean_no_mtl[slice, 4, echo]], axs=axs[0],
            #             overlay=[std_no_mtl[slice, 0, echo], std_no_mtl[slice, 1, echo], std_no_mtl[slice, 2, echo],
            #                      std_no_mtl[slice, 3, echo], std_no_mtl[slice, 4, echo]],
            #             show_cbar_overlay=[True, True, True, True, True], fontsize=fontsize,
            #             text=[f"JOINT \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2", "Cascade: 3", "Cascade: 4", "Cascade: 5"], ticks_overlay=ticks_overlay)
            # plot_images([mean_logit[slice, 0, echo], mean_logit[slice, 1, echo],
            #              mean_logit[slice, 2, echo], mean_logit[slice, 3, echo], mean_logit[slice, 4, echo]], axs=axs[1],
            #             overlay=[std_logit[slice, 0, echo], std_logit[slice, 1, echo], std_logit[slice, 2, echo],
            #                      std_logit[slice, 3, echo], std_logit[slice, 4, echo]],
            #             show_cbar_overlay=[True, True, True, True, True], fontsize=fontsize,
            #             text=[f"SUM LOGIT \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2", "Cascade: 3", "Cascade: 4", "Cascade: 5"], ticks_overlay=ticks_overlay)
            # plot_images(
            #     [mean_softmax[slice, 0, echo], mean_softmax[slice, 1, echo],
            #      mean_softmax[slice, 2, echo], mean_softmax[slice, 3, echo], mean_softmax[slice, 4, echo]], axs=axs[2],
            #     overlay=[std_softmax[slice, 0, echo], std_softmax[slice, 1, echo],
            #              std_softmax[slice, 2, echo], std_softmax[slice, 3, echo], std_softmax[slice, 4, echo]], show_cbar_overlay=[True, True, True, True, True],
            #     fontsize=fontsize, text=[f"SUM SOFTMAX \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2", "Cascade: 3", "Cascade: 4", "Cascade: 5"], ticks_overlay=ticks_overlay)
            # plot_images([mean_sasg[slice, 0, echo], mean_sasg[slice, 1, echo],
            #              mean_sasg[slice, 2, echo], mean_sasg[slice, 3, echo], mean_sasg[slice, 4, echo]], axs=axs[3],
            #             overlay=[std_sasg[slice, 0, echo], std_sasg[slice, 1, echo], std_sasg[slice, 2, echo],
            #                      std_sasg[slice, 3, echo], std_sasg[slice, 4, echo]],
            #             show_cbar_overlay=[True, True, True, True, True], fontsize=fontsize,
            #             text=[f"SASG \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2", "Cascade: 3", "Cascade: 4", "Cascade: 5"],ticks_overlay=ticks_overlay)
            # plot_images([mean_logit_tam[slice, 0, echo], mean_logit_tam[slice, 1, echo],
            #              mean_logit_tam[slice, 2, echo], mean_logit_tam[slice, 3, echo], mean_logit_tam[slice, 4, echo]], axs=axs[4],
            #             overlay=[std_logit_tam[slice, 0, echo], std_logit_tam[slice, 1, echo],
            #                      std_logit_tam[slice, 2, echo], std_logit_tam[slice, 3, echo],
            #                      std_logit_tam[slice, 4, echo]],
            #             show_cbar_overlay=[True, True, True, True, True], fontsize=fontsize,
            #             text=[f"TAM LOGIT \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2", "Cascade: 3", "Cascade: 4", "Cascade: 5"], ticks_overlay=ticks_overlay)
            # plot_images(
            #     [mean_softmax_tam[slice, 0, echo], mean_softmax_tam[slice, 1, echo],
            #      mean_softmax_tam[slice, 2, echo], mean_softmax_tam[slice, 3, echo], mean_softmax_tam[slice, 4, echo]], axs=axs[5],
            #     overlay=[std_softmax_tam[slice, 0, echo], std_softmax_tam[slice, 1, echo],
            #              std_softmax_tam[slice, 2, echo], std_softmax_tam[slice, 3, echo], std_softmax_tam[slice, 4, echo]],
            # show_cbar_overlay=[True, True, True, True, True],
            #     fontsize=fontsize, text=[f"TAM SOFTMAX \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2", "Cascade: 3", "Cascade: 4", "Cascade: 5"],ticks_overlay=ticks_overlay)
            # plt.tight_layout(pad=0)
            # plt.savefig(
            #     "/scratch/tmpaquaij/Figures/IP/Intermediate_" + intermidiate_form + f"_DeepEnsemble_UQ_{Patient_id}_Echo:{str(echo)}_slice:{str(slice)}.png")
            # plt.close()
            # # #
            # wsize = hsize / target_mtlrs_1.shape[2] * target_mtlrs_1.shape[3]
            # fig, axs = plt.subplots(6, 5, figsize=(5 * wsize, 6 * hsize))
            # plot_images([mean_no_mtl[slice, 0, echo], mean_no_mtl[slice, 1, echo],
            #              mean_no_mtl[slice, 2, echo], mean_no_mtl[slice, 3, echo], mean_no_mtl[slice, 4, echo]],
            #             axs=axs[0],
            #             overlay=[inter_std_no_mtl[slice, 0, echo], inter_std_no_mtl[slice, 1, echo], inter_std_no_mtl[slice, 2, echo],
            #                      inter_std_no_mtl[slice, 3, echo], inter_std_no_mtl[slice, 4, echo]],
            #             show_cbar_overlay=[True, True, True, True, True], fontsize=fontsize,
            #             text=[f"JOINT \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2",
            #                   "Cascade: 3", "Cascade: 4", "Cascade: 5"], ticks_overlay=ticks_overlay)
            # plot_images([mean_logit[slice, 0, echo], mean_logit[slice, 1, echo],
            #              mean_logit[slice, 2, echo], mean_logit[slice, 3, echo], mean_logit[slice, 4, echo]],
            #             axs=axs[1],
            #             overlay=[inter_std_logit[slice, 0, echo], inter_std_logit[slice, 1, echo], inter_std_logit[slice, 2, echo],
            #                      inter_std_logit[slice, 3, echo], inter_std_logit[slice, 4, echo]],
            #             show_cbar_overlay=[True, True, True, True, True], fontsize=fontsize,
            #             text=[f"SUM LOGIT \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2",
            #                   "Cascade: 3", "Cascade: 4", "Cascade: 5"], ticks_overlay=ticks_overlay)
            # plot_images(
            #     [mean_softmax[slice, 0, echo], mean_softmax[slice, 1, echo],
            #      mean_softmax[slice, 2, echo], mean_softmax[slice, 3, echo], mean_softmax[slice, 4, echo]], axs=axs[2],
            #     overlay=[inter_std_softmax[slice, 0, echo], inter_std_softmax[slice, 1, echo],
            #              inter_std_softmax[slice, 2, echo], inter_std_softmax[slice, 3, echo], inter_std_softmax[slice, 4, echo]],
            #     show_cbar_overlay=[True, True, True, True, True],
            #     fontsize=fontsize,
            #     text=[f"SUM SOFTMAX \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2",
            #           "Cascade: 3", "Cascade: 4", "Cascade: 5"], ticks_overlay=ticks_overlay)
            # plot_images([mean_sasg[slice, 0, echo], mean_sasg[slice, 1, echo],
            #              mean_sasg[slice, 2, echo], mean_sasg[slice, 3, echo], mean_sasg[slice, 4, echo]], axs=axs[3],
            #             overlay=[inter_std_sasg[slice, 0, echo], inter_std_sasg[slice, 1, echo], inter_std_sasg[slice, 2, echo],
            #                      inter_std_sasg[slice, 3, echo], inter_std_sasg[slice, 4, echo]],
            #             show_cbar_overlay=[True, True, True, True, True], fontsize=fontsize,
            #             text=[f"SASG \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2",
            #                   "Cascade: 3", "Cascade: 4", "Cascade: 5"], ticks_overlay=ticks_overlay)
            # plot_images([mean_logit_tam[slice, 0, echo], mean_logit_tam[slice, 1, echo],
            #              mean_logit_tam[slice, 2, echo], mean_logit_tam[slice, 3, echo],
            #              mean_logit_tam[slice, 4, echo]], axs=axs[4],
            #             overlay=[inter_std_logit_tam[slice, 0, echo], inter_std_logit_tam[slice, 1, echo],
            #                      inter_std_logit_tam[slice, 2, echo], inter_std_logit_tam[slice, 3, echo],
            #                      inter_std_logit_tam[slice, 4, echo]],
            #             show_cbar_overlay=[True, True, True, True, True], fontsize=fontsize,
            #             text=[f"TAM LOGIT \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2",
            #                   "Cascade: 3", "Cascade: 4", "Cascade: 5"], ticks_overlay=ticks_overlay)
            # plot_images(
            #     [mean_softmax_tam[slice, 0, echo], mean_softmax_tam[slice, 1, echo],
            #      mean_softmax_tam[slice, 2, echo], mean_softmax_tam[slice, 3, echo], mean_softmax_tam[slice, 4, echo]],
            #     axs=axs[5],
            #     overlay=[inter_std_softmax_tam[slice, 0, echo], inter_std_softmax_tam[slice, 1, echo],
            #              inter_std_softmax_tam[slice, 2, echo], inter_std_softmax_tam[slice, 3, echo],
            #              inter_std_softmax_tam[slice, 4, echo]],
            #     show_cbar_overlay=[True, True, True, True, True],
            #     fontsize=fontsize,
            #     text=[f"TAM SOFTMAX \nCascade: 1 \nPatient: {Patient_id} \nSlice: {slice}", "Cascade: 2",
            #           "Cascade: 3", "Cascade: 4", "Cascade: 5"], ticks_overlay=ticks_overlay)
            # plt.tight_layout(pad=0)
            # plt.savefig(
            #     "/scratch/tmpaquaij/Figures/IP/Intermediate_" + intermidiate_form + f"_IP_STD_UQ_{Patient_id}_Echo:{str(echo)}_slice:{str(slice)}.png")
            # plt.close()
            # # print(f"Saved {Patient_id} specific Deep Ensemble figure")
            wsize = hsize / target_mtlrs_1.shape[2] * target_mtlrs_1.shape[3]
            ssim_score_zf = ssim(target_mtlrs_1[slice, echo], zero_filled_mtlrs_1[slice, echo])
            psnr_score_zf = psnr(target_mtlrs_1[slice, echo], zero_filled_mtlrs_1[slice, echo])
            haarpsi_score_zf = haarpsi3d(target_mtlrs_1[slice, echo], zero_filled_mtlrs_1[slice, echo])
            ssim_score_no_mtl = ssim(target_mtlrs_1[slice, echo], reconstruction_no_mtl[slice, echo])
            psnr_score_no_mtl= psnr(target_mtlrs_1[slice, echo], reconstruction_no_mtl[slice, echo])
            haarpsi_score_no_mtl = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_no_mtl[slice, echo])
            ssim_score_logit = ssim(target_mtlrs_1[slice, echo], reconstruction_logit[slice, echo])
            psnr_score_logit = psnr(target_mtlrs_1[slice, echo], reconstruction_logit[slice, echo])
            haarpsi_score_logit = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_logit[slice, echo])
            ssim_score_softmax = ssim(target_mtlrs_1[slice, echo], reconstruction_softmax[slice, echo])
            psnr_score_softmax = psnr(target_mtlrs_1[slice, echo], reconstruction_softmax[slice, echo])
            haarpsi_score_softmax = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_softmax[slice, echo])
            ssim_score_sasg = ssim(target_mtlrs_1[slice, echo], reconstruction_sasg[slice, echo])
            psnr_score_sasg = psnr(target_mtlrs_1[slice, echo], reconstruction_sasg[slice, echo])
            haarpsi_score_sasg = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_sasg[slice, echo])
            ssim_score_logit_tam = ssim(target_mtlrs_1[slice, echo], reconstruction_logit_tam[slice, echo])
            psnr_score_logit_tam = psnr(target_mtlrs_1[slice, echo], reconstruction_logit_tam[slice, echo])
            haarpsi_score_logit_tam = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_logit_tam[slice, echo])
            ssim_score_softmax_tam = ssim(target_mtlrs_1[slice, echo], reconstruction_softmax_tam[slice, echo])
            psnr_score_softmax_tam = psnr(target_mtlrs_1[slice, echo], reconstruction_softmax_tam[slice, echo])
            haarpsi_score_softmax_tam = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_softmax_tam[slice, echo])
            fig, axs = plt.subplots(2, 4, figsize=(4 * wsize, 2 * hsize))
            plot_images([target_mtlrs_1[slice, echo],zero_filled_mtlrs_1[slice, echo],
                         reconstruction_no_mtl[slice, echo], reconstruction_logit[slice, echo]], axs=axs[0], fontsize=fontsize,
                        text=[f'Target \nPatient: {Patient_id} \nSlice: {slice}',
                              f'Zero-Filled \nSSIM: {round(ssim_score_zf, 3)} \nPSNR: {round(psnr_score_zf, 3)} \nHaarPSI: {round(haarpsi_score_zf, 3)}',
                              f'JOINT \nSSIM: {round(ssim_score_no_mtl, 3)} \nPSNR: {round(psnr_score_no_mtl, 3)} \nHaarPSI: {round(haarpsi_score_no_mtl, 3)}',
                              f'SUM LOGIT \nSSIM: {round(ssim_score_logit, 3)} \nPSNR: {round(psnr_score_logit, 3)} \nHaarPSI: {round(haarpsi_score_logit, 3)}',
                              ])
            plot_images([reconstruction_softmax[slice, echo], reconstruction_sasg[slice, echo],
                         reconstruction_logit_tam[slice, echo], reconstruction_softmax_tam[slice, echo]],
                        axs=axs[1], fontsize=fontsize,
                        text=[
                            f'SUM SOFTMAX \nSSIM: {round(ssim_score_softmax, 3)} \nPSNR: {round(psnr_score_softmax, 3)} \nHaarPSI: {round(haarpsi_score_softmax, 3)}',
                            f'SASG \nSSIM: {round(ssim_score_sasg, 3)} \nPSNR: {round(psnr_score_sasg, 3)} \nHaarPSI: {round(haarpsi_score_sasg, 3)}',
                            f'TAM LOGIT \nSSIM: {round(ssim_score_logit_tam, 3)} \nPSNR: {round(psnr_score_logit_tam, 3)} \nHaarPSI: {round(haarpsi_score_logit_tam, 3)}',
                            f'TAM SOFTMAX \nSSIM: {round(ssim_score_softmax_tam, 3)} \nPSNR: {round(psnr_score_softmax_tam, 3)} \nHaarPSI: {round(haarpsi_score_softmax_tam, 3)}',
                            ])
            plt.tight_layout(pad=0)
            plt.savefig(
                f"/scratch/tmpaquaij/Figures/IP/Reconstruction_{Patient_id}_Echo:{str(echo)}_slice:{str(slice)}.png")
            plt.close()
            # print(f"Saved {Patient_id} specific Deep Ensemble figure")
            # fig, axs = plt.subplots(2, 3, figsize=(3 * wsize, 2 * hsize))
            # plot_images([mean_no_mtl[slice,1, echo], mean_logit[slice,1, echo],
            #              mean_softmax[slice,1, echo]], axs=axs[0],overlay=[std_no_mtl[slice,1, echo], std_logit[slice,1, echo],
            #              std_softmax[slice,1, echo]],show_cbar_overlay=[True,True,True],ticks_overlay=ticks_overlay,
            #             fontsize=fontsize,
            #             text=[
            #                   f'JOINT \nPatient: {Patient_id} \nSlice: {slice}',
            #                   f'SUM LOGIT',
            #                   'SUM SOFTMAX'])
            # plot_images([mean_sasg[slice,1, echo], mean_logit_tam[slice,1, echo],
            #              mean_softmax_tam[slice,1, echo]],
            #             overlay=[std_sasg[slice,1, echo], std_logit_tam[slice,1, echo],
            #              std_softmax_tam[slice,1, echo]],show_cbar_overlay=[True,True,True],
            #             axs=axs[1], fontsize=fontsize,ticks_overlay=ticks_overlay,
            #             text=[
            #                 f'SASG ',
            #                 f'TAM LOGIT ',
            #                 f'TAM SOFTMAX ',
            #             ])
            # plt.tight_layout(pad=0)
            # plt.savefig(
            #     f"/scratch/tmpaquaij/Figures/IP/UQ_cascade_2_{Patient_id}_Echo:{str(echo)}_slice:{str(slice)}.png")
            # plt.close()
            #
            #
            # # %%
            # ssim_score = ssim(target_mtlrs_1[slice, echo], reconstruction_no_mtl[slice, echo])
            # psnr_score = psnr(target_mtlrs_1[slice, echo], reconstruction_no_mtl[slice, echo])
            # haarpsi_score = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_no_mtl[slice, echo])
            # vsi_score = vsi3d(target_mtlrs_1[slice, echo], reconstruction_no_mtl[slice, echo])
            # fig, axs = plt.subplots(6, 8, figsize=(8 * wsize, 6 * hsize))
            # plot_images([zero_filled_mtlrs_1[slice, echo], mean_no_mtl[slice, 0, echo], mean_no_mtl[slice, 1, echo],
            #              mean_no_mtl[slice, 2, echo], mean_no_mtl[slice, 3, echo], mean_no_mtl[slice, 4, echo],
            #              reconstruction_no_mtl[slice, echo], target_mtlrs_1[slice, echo]],
            #             titles=["Zero Filled", intermidiate_form + " C1", intermidiate_form + " C2",
            #                     intermidiate_form + " C3", intermidiate_form + " C4", intermidiate_form + " C5",
            #                     "Final mean prediction", "Target"], axs=axs[0],
            #             overlay=[None, inter_std_no_mtl[slice, 0, echo], inter_std_no_mtl[slice, 1, echo],
            #                      inter_std_no_mtl[slice, 2, echo], inter_std_no_mtl[slice, 3, echo],
            #                      inter_std_no_mtl[slice, 4, echo], None, None],
            #             show_cbar_overlay=[False, True, True, True, True, True, False, False], fontsize=fontsize,
            #             text=[None, None, None, None, None, None,
            #                   f'SSIM: {round(ssim_score, 3)} \nPSNR: {round(psnr_score, 3)} \nHaarPSI: {round(haarpsi_score, 3)} \nVSI: {round(vsi_score, 3)}',
            #                   f'Patient: {Patient_id}'], ticks_overlay=ticks_overlay)
            # axs[0, 0].set_ylabel("JOINT", fontsize=fontsize)
            # ssim_score = ssim(target_mtlrs_1[slice, echo], reconstruction_logit[slice, echo])
            # psnr_score = psnr(target_mtlrs_1[slice, echo], reconstruction_logit[slice, echo])
            # haarpsi_score = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_logit[slice, echo])
            # vsi_score = vsi3d(target_mtlrs_1[slice, echo], reconstruction_logit[slice, echo])
            # plot_images([zero_filled_mtlrs_1[slice, echo], mean_logit[slice, 0, echo], mean_logit[slice, 1, echo],
            #              mean_logit[slice, 2, echo], mean_logit[slice, 3, echo], mean_logit[slice, 4, echo],
            #              reconstruction_logit[slice, echo], target_mtlrs_1[slice, echo]], axs=axs[1],
            #             overlay=[None, inter_std_logit[slice, 0, echo], inter_std_logit[slice, 1, echo],
            #                      inter_std_logit[slice, 2, echo], inter_std_logit[slice, 3, echo],
            #                      inter_std_logit[slice, 4, echo], None, None],
            #             show_cbar_overlay=[False, True, True, True, True, True, False, False], fontsize=fontsize,
            #             text=[None, None, None, None, None, None,
            #                   f'SSIM: {round(ssim_score, 3)} \nPSNR: {round(psnr_score, 3)} \nHaarPSI: {round(haarpsi_score, 3)} \nVSI: {round(vsi_score, 3)}',
            #                   f'Patient: {Patient_id}\nSlice: {slice}'], ticks_overlay=ticks_overlay)
            # axs[1, 0].set_ylabel("SUM LOGIT", fontsize=fontsize)
            # ssim_score = ssim(target_mtlrs_1[slice, echo], reconstruction_softmax[slice, echo])
            # psnr_score = psnr(target_mtlrs_1[slice, echo], reconstruction_softmax[slice, echo])
            # haarpsi_score = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_softmax[slice, echo])
            # vsi_score = vsi3d(target_mtlrs_1[slice, echo], reconstruction_softmax[slice, echo])
            # plot_images(
            #     [zero_filled_mtlrs_1[slice, echo], mean_softmax[slice, 0, echo], mean_softmax[slice, 1, echo],
            #      mean_softmax[slice, 2, echo], mean_softmax[slice, 3, echo], mean_softmax[slice, 4, echo],
            #      reconstruction_softmax[slice, echo], target_mtlrs_1[slice, echo]], axs=axs[2],
            #     overlay=[None, inter_std_softmax[slice, 0, echo], inter_std_softmax[slice, 1, echo],
            #              inter_std_softmax[slice, 2, echo], inter_std_softmax[slice, 3, echo],
            #              inter_std_softmax[slice, 4, echo], None, None],
            #     show_cbar_overlay=[False, True, True, True, True, True, False, False], fontsize=fontsize,
            #     text=[None, None, None, None, None, None,
            #           f'SSIM: {round(ssim_score, 3)} \nPSNR: {round(psnr_score, 3)} \nHaarPSI: {round(haarpsi_score, 3)} \nVSI: {round(vsi_score, 3)}',
            #           f'Patient: {Patient_id}\nSlice: {slice}'], ticks_overlay=ticks_overlay)
            # axs[2, 0].set_ylabel("SUM SOFTMAX", fontsize=fontsize)
            # ssim_score = ssim(target_mtlrs_1[slice, echo], reconstruction_sasg[slice, echo])
            # psnr_score = psnr(target_mtlrs_1[slice, echo], reconstruction_sasg[slice, echo])
            # haarpsi_score = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_sasg[slice, echo])
            # vsi_score = vsi3d(target_mtlrs_1[slice, echo], reconstruction_sasg[slice, echo])
            # plot_images([zero_filled_mtlrs_1[slice, echo], mean_sasg[slice, 0, echo], mean_sasg[slice, 1, echo],
            #              mean_sasg[slice, 2, echo], mean_sasg[slice, 3, echo], mean_sasg[slice, 4, echo],
            #              reconstruction_sasg[slice, echo], target_mtlrs_1[slice, echo]], axs=axs[3],
            #             overlay=[None, inter_std_sasg[slice, 0, echo], inter_std_sasg[slice, 1, echo],
            #                      inter_std_sasg[slice, 2, echo], inter_std_sasg[slice, 3, echo],
            #                      inter_std_sasg[slice, 4, echo], None, None],
            #             show_cbar_overlay=[False, True, True, True, True, True, False, False], fontsize=fontsize,
            #             text=[None, None, None, None, None, None,
            #                   f'SSIM: {round(ssim_score, 3)} \nPSNR: {round(psnr_score, 3)} \nHaarPSI: {round(haarpsi_score, 3)} \nVSI: {round(vsi_score, 3)}',
            #                   f'Patient: {Patient_id}\nSlice: {slice}'], ticks_overlay=ticks_overlay)
            # axs[3, 0].set_ylabel("SASG", fontsize=fontsize)
            # ssim_score = ssim(target_mtlrs_1[slice, echo], reconstruction_logit_tam[slice, echo])
            # psnr_score = psnr(target_mtlrs_1[slice, echo], reconstruction_logit_tam[slice, echo])
            # haarpsi_score = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_logit_tam[slice, echo])
            # vsi_score = vsi3d(target_mtlrs_1[slice, echo], reconstruction_logit_tam[slice, echo])
            # plot_images([zero_filled_mtlrs_1[slice, echo], mean_logit_tam[slice, 0, echo], mean_logit_tam[slice, 1, echo],
            #              mean_logit_tam[slice, 2, echo], mean_logit_tam[slice, 3, echo], mean_logit_tam[slice, 4, echo],
            #              reconstruction_logit_tam[slice, echo], target_mtlrs_1[slice, echo]], axs=axs[4],
            #             overlay=[None, inter_std_logit_tam[slice, 0, echo], inter_std_logit_tam[slice, 1, echo],
            #                      inter_std_logit_tam[slice, 2, echo], inter_std_logit_tam[slice, 3, echo],
            #                      inter_std_logit_tam[slice, 4, echo], None, None],
            #             show_cbar_overlay=[False, True, True, True, True, True, False, False], fontsize=fontsize,
            #             text=[None, None, None, None, None, None,
            #                   f'SSIM: {round(ssim_score, 3)} \nPSNR: {round(psnr_score, 3)} \nHaarPSI: {round(haarpsi_score, 3)} \nVSI: {round(vsi_score, 3)}',
            #                   f'Patient: {Patient_id}'], ticks_overlay=ticks_overlay)
            # axs[4, 0].set_ylabel("LOGIT TAM", fontsize=fontsize)
            # ssim_score = ssim(target_mtlrs_1[slice, echo], reconstruction_softmax_tam[slice, echo])
            # psnr_score = psnr(target_mtlrs_1[slice, echo], reconstruction_softmax_tam[slice, echo])
            # haarpsi_score = haarpsi3d(target_mtlrs_1[slice, echo], reconstruction_softmax_tam[slice, echo])
            # vsi_score = vsi3d(target_mtlrs_1[slice, echo], reconstruction_softmax_tam[slice, echo])
            # plot_images(
            #     [zero_filled_mtlrs_1[slice, echo], mean_softmax_tam[slice, 0, echo], mean_softmax_tam[slice, 1, echo],
            #      mean_softmax_tam[slice, 2, echo], mean_softmax_tam[slice, 3, echo], mean_softmax_tam[slice, 4, echo],
            #      reconstruction_softmax_tam[slice, echo], target_mtlrs_1[slice, echo]], axs=axs[5],
            #     overlay=[None, inter_std_softmax_tam[slice, 0, echo], inter_std_softmax_tam[slice, 1, echo],
            #              inter_std_softmax_tam[slice, 2, echo], inter_std_softmax_tam[slice, 3, echo],
            #              inter_std_softmax_tam[slice, 4, echo], None, None],
            #     show_cbar_overlay=[False, True, True, True, True, True, False, False], fontsize=fontsize,
            #     text=[None, None, None, None, None, None,
            #           f'SSIM: {round(ssim_score, 3)} \nPSNR: {round(psnr_score, 3)} \nHaarPSI: {round(haarpsi_score, 3)} \nVSI: {round(vsi_score, 3)}',
            #           f'Patient: {Patient_id}\nSlice: {slice}'], ticks_overlay=ticks_overlay)
            # axs[5, 0].set_ylabel("SOFTMAX TAM", fontsize=fontsize)
            # plt.tight_layout(pad=0)
            # plt.savefig(
            #     "/scratch/tmpaquaij/Figures/IP/Intermediate_" + intermidiate_form + f"_timestep_UQ_{Patient_id}_Echo:{str(echo)}_slice:{str(slice)}.png")
            # plt.close()
            # print(f"Saved {Patient_id} specific ITS figure")

            # %%
            # fig, axs = plt.subplots(6, 2, figsize=(2 * wsize, 6 * hsize))
            # plot_images([mean_no_mtl[slice, 4, echo],mean_no_mtl[slice, 4, echo]],
            #             titles=["Deep Ensemble", "IP"], axs=axs[0],
            #             overlay=[std_no_mtl[slice, 4, echo], inter_std_no_mtl[slice, 4, echo]],
            #             show_cbar_overlay=[True,True], fontsize=fontsize, ticks_overlay=ticks_overlay)
            # axs[0, 0].set_ylabel("JOINT", fontsize=fontsize)
            # plot_images([mean_logit[slice, 4, echo],mean_logit[slice, 4, echo]], axs=axs[1],
            #             overlay=[std_logit[slice, 4, echo],inter_std_logit[slice, 4, echo]],
            #             show_cbar_overlay=[True, True], fontsize=fontsize, ticks_overlay=ticks_overlay)
            # axs[1, 0].set_ylabel("SUM LOGIT", fontsize=fontsize)
            # plot_images(
            #     [mean_softmax[slice, 4, echo],mean_softmax[slice, 4, echo]], axs=axs[2],
            #     overlay=[std_softmax[slice, 4, echo],inter_std_softmax[slice, 4, echo]],
            #     show_cbar_overlay=[True,True],fontsize=fontsize,ticks_overlay=ticks_overlay)
            # axs[2, 0].set_ylabel("SUM SOFTMAX", fontsize=fontsize)
            # plot_images([mean_sasg[slice, 4, echo],mean_sasg[slice, 4, echo]], axs=axs[3],
            #             overlay=[std_sasg[slice, 4, echo], inter_std_sasg[slice, 4, echo]],
            #             show_cbar_overlay=[True, True], fontsize=fontsize, ticks_overlay=ticks_overlay)
            # axs[3, 0].set_ylabel("SASG", fontsize=fontsize)
            # plot_images(
            #     [mean_logit_tam[slice, 4, echo],mean_logit_tam[slice, 4, echo]], axs=axs[4],
            #     overlay=[std_logit_tam[slice, 4, echo],inter_std_logit_tam[slice, 4, echo]],
            #     show_cbar_overlay=[True,True], fontsize=fontsize, ticks_overlay=ticks_overlay)
            # axs[4, 0].set_ylabel("LOGIT TAM", fontsize=fontsize)
            # plot_images(
            #     [mean_softmax_tam[slice, 4, echo],mean_softmax_tam[slice, 4, echo]], axs=axs[5],
            #     overlay=[std_softmax_tam[slice, 4, echo],inter_std_softmax_tam[slice, 4, echo]],
            #     show_cbar_overlay=[True, True], fontsize=fontsize,ticks_overlay=ticks_overlay)
            # axs[5, 0].set_ylabel("SOFTMAX TAM", fontsize=fontsize)
            # plt.tight_layout(pad=0)
            # plt.savefig(
            #     "/scratch/tmpaquaij/Figures/IP/Intermediate_" + intermidiate_form + f"_UQ_comparison_{Patient_id}_Echo:{str(echo)}_slice:{str(slice)}.png")
            # plt.close()
            # print(f"Saved {Patient_id} specific comparison figure")






            # %%
            # ticks = [0, 1, 2, 3, 4]
            # wsize = hsize / target_mtlrs_1.shape[2] * target_mtlrs_1.shape[3]
            # dice_score = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                          inter_seg_no_mtl[slice,-1, 1:][np.newaxis])
            # assd_score = float(
            #         asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_no_mtl[slice,-1, 1:][np.newaxis]))
            # f1_score = float(f1_per_class_metric(segmentation_labels_mtlrs_1[slice, 1:, ][np.newaxis],
            #                                      inter_seg_no_mtl[slice,-1, 1:][np.newaxis]))
            # HD95_score = float(hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                                                 inter_seg_no_mtl[slice,-1, 1:][np.newaxis]))
            # IOU_score = float(
            #     iou_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_no_mtl[slice,-1, 1:][np.newaxis]))
            #
            # inter_seg_no_mtl_plot = torch.argmax(torch.from_numpy(inter_seg_no_mtl), dim=2)
            # segmentation_labels_mtlrs_1_plot = torch.argmax(torch.from_numpy(segmentation_labels_mtlrs_1), dim=1)
            #
            # fig, axs = plt.subplots(6, 7, figsize=(7 * wsize, 6 * hsize))
            # plot_images([entropy_seg_no_mtl[slice, 0], entropy_seg_no_mtl[slice, 1], entropy_seg_no_mtl[slice, 2],
            #              entropy_seg_no_mtl[slice, 3], entropy_seg_no_mtl[slice, 4],inter_seg_no_mtl_plot[slice, 4],
            #              segmentation_labels_mtlrs_1_plot[slice]],show_cbar=[True,True,True,True,True,False,False],
            #             titles=["Entropy C1", "Entropy C2", "Entropy C3", "Entropy C4",
            #                     "Entropy C5","Predicted Segmentation", "Target"], axs=axs[0], fontsize=fontsize,
            #             cmap=['jet', 'jet', 'jet', 'jet', 'jet', 'viridis', 'viridis'], ticks=[[0,1],[0,1],[0,1],[0,1],[0,1],ticks,ticks],
            #             text=[None, None, None, None,None,
            #                   f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm \nF1: {round(f1_score, 3)} \nHD95: {round(HD95_score, 3)} \nIoU: {round(IOU_score, 3)}',
            #                   f'Patient: {Patient_id}\nSlice: {slice}'])
            # axs[0][0].set_ylabel("JOINT", fontsize=fontsize)
            # dice_score = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                          inter_seg_logit[slice,-1, 1:][np.newaxis])
            #
            # assd_score = float(
            #     asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_logit[slice,-1, 1:][np.newaxis]))
            # f1_score = float(f1_per_class_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                                      inter_seg_logit[slice,-1, 1:][np.newaxis]))
            # HD95_score = float(hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                                                 inter_seg_logit[slice,-1, 1:][np.newaxis]))
            # IOU_score = float(
            #     iou_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_logit[slice,-1, 1:][np.newaxis]))
            #
            # inter_seg_logit_plot = torch.argmax(torch.from_numpy(inter_seg_logit), dim=2)
            # plot_images([entropy_seg_logit[slice, 0], entropy_seg_logit[slice, 1], entropy_seg_logit[slice, 2],
            #              entropy_seg_logit[slice, 3],entropy_seg_logit[slice, 4], inter_seg_logit_plot[slice, 4],
            #              segmentation_labels_mtlrs_1_plot[slice]],
            #             axs=axs[1], fontsize=fontsize,show_cbar=[True,True,True,True,True,False,False],
            #             cmap=['jet', 'jet', 'jet', 'jet', 'jet', 'viridis', 'viridis'], ticks=[[0,1],[0,1],[0,1],[0,1],[0,1],ticks,ticks],
            #             text=[None, None, None, None,None,
            #                   f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm \nF1: {round(f1_score, 3)} \nHD95: {round(HD95_score, 3)} \nIoU: {round(IOU_score, 3)}',
            #                   f'Patient: {Patient_id}\nSlice: {slice}'])
            # axs[1, 0].set_ylabel("SUM LOGIT", fontsize=fontsize)
            # dice_score = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                          inter_seg_softmax[slice,-1, 1:][np.newaxis])
            #
            # assd_score = float(
            #     asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_softmax[slice,-1, 1:][np.newaxis]))
            # f1_score = float(f1_per_class_metric(segmentation_labels_mtlrs_1[slice, 1:, ][np.newaxis],
            #                                      inter_seg_softmax[slice,-1, 1:][np.newaxis]))
            # HD95_score = float(hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                                                 inter_seg_softmax[slice,-1, 1:][np.newaxis]))
            # IOU_score = float(iou_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                              inter_seg_softmax[slice,-1, 1:][np.newaxis]))
            #
            # inter_seg_softmax_plot = torch.argmax(torch.from_numpy(inter_seg_softmax), dim=2)
            # plot_images([entropy_seg_softmax[slice, 0], entropy_seg_softmax[slice, 1],
            #              entropy_seg_softmax[slice, 2], entropy_seg_softmax[slice, 3],entropy_seg_softmax[slice, 4],
            #              inter_seg_softmax_plot[slice, 4], segmentation_labels_mtlrs_1_plot[slice]],
            #             axs=axs[2],show_cbar=[True,True,True,True,True,False,False],
            #             fontsize=fontsize, cmap=['jet', 'jet', 'jet', 'jet', 'jet', 'viridis', 'viridis'], ticks=[[0,1],[0,1],[0,1],[0,1],[0,1],ticks,ticks],
            #             text=[None, None, None, None,None,
            #                                f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm \nF1: {round(f1_score, 3)} \nHD95: {round(HD95_score, 3)} \nIoU: {round(IOU_score, 3)}',
            #                                f'Patient: {Patient_id}\nSlice: {slice}'])
            # axs[2, 0].set_ylabel("SUM SOFTMAX", fontsize=fontsize)
            # dice_score = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_sasg[slice,-1, 1:][np.newaxis])
            #
            # assd_score = float(
            #     asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_sasg[slice,-1, 1:][np.newaxis]))
            # f1_score = float(f1_per_class_metric(segmentation_labels_mtlrs_1[slice, 1:, ][np.newaxis], inter_seg_sasg[slice,-1, 1:][np.newaxis]))
            # HD95_score = float(
            #     hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_sasg[slice,-1, 1:][np.newaxis]))
            # IOU_score = float(iou_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_sasg[slice,-1, 1:][np.newaxis]))
            #
            # inter_seg_sasg_plot = torch.argmax(torch.from_numpy(inter_seg_sasg), dim=2)
            # plot_images([entropy_seg_sasg[slice, 0], entropy_seg_sasg[slice, 1], entropy_seg_sasg[slice, 2],
            #              entropy_seg_sasg[slice, 3], entropy_seg_sasg[slice, 4], inter_seg_sasg_plot[slice, 4],
            #              segmentation_labels_mtlrs_1_plot[slice]],
            #             axs=axs[3], fontsize=fontsize,show_cbar=[True,True,True,True,True,False,False],
            #             cmap=['jet', 'jet', 'jet', 'jet', 'jet', 'viridis', 'viridis'], ticks=[[0,1],[0,1],[0,1],[0,1],[0,1],ticks,ticks],
            #             text=[None, None, None, None,None,
            #                   f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm \nF1: {round(f1_score, 3)} \nHD95: {round(HD95_score, 3)} \nIoU: {round(IOU_score, 3)}',
            #                   f'Patient: {Patient_id}\nSlice: {slice}'])
            # axs[3, 0].set_ylabel("SASG", fontsize=fontsize)
            # dice_score = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                          inter_seg_logit_tam[slice,-1, 1:][np.newaxis])
            # assd_score = float(
            #     asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_logit_tam[slice,-1, 1:][np.newaxis]))
            # f1_score = float(f1_per_class_metric(segmentation_labels_mtlrs_1[slice, 1:, ][np.newaxis],
            #                                      inter_seg_logit_tam[slice,-1, 1:][np.newaxis]))
            # HD95_score = float(hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                                                 inter_seg_logit_tam[slice,-1, 1:][np.newaxis]))
            # IOU_score = float(iou_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                              inter_seg_logit_tam[slice,-1, 1:][np.newaxis]))
            #
            # inter_seg_logit_tam_plot = torch.argmax(torch.from_numpy(inter_seg_logit_tam), dim=2)
            # plot_images(
            #     [entropy_seg_logit_tam[slice, 0], entropy_seg_logit_tam[slice, 1], entropy_seg_logit_tam[slice, 2],
            #      entropy_seg_logit_tam[slice, 3], entropy_seg_logit_tam[slice, 4], inter_seg_logit_tam_plot[slice, 4],
            #      segmentation_labels_mtlrs_1_plot[slice]],
            #     axs=axs[4], fontsize=fontsize,show_cbar=[True,True,True,True,True,False,False],
            #     cmap=['jet', 'jet', 'jet', 'jet', 'jet', 'viridis', 'viridis'], ticks=[[0,1],[0,1],[0,1],[0,1],[0,1],ticks,ticks],
            #     text=[None,None, None, None, None,
            #           f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm \nF1: {round(f1_score, 3)} \nHD95: {round(HD95_score, 3)} \nIoU: {round(IOU_score, 3)}',
            #           f'Patient: {Patient_id}\nSlice: {slice}'])
            #
            # axs[4, 0].set_ylabel("LOGIT TAM", fontsize=fontsize)
            # dice_score = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                          inter_seg_softmax_tam[slice,-1, 1:][np.newaxis])
            # assd_score = float(
            #     asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_softmax_tam[slice,-1, 1:][np.newaxis]))
            # f1_score = float(f1_per_class_metric(segmentation_labels_mtlrs_1[slice, 1:, ][np.newaxis],
            #                                      inter_seg_softmax_tam[slice,-1, 1:][np.newaxis]))
            # HD95_score = float(hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                                                 inter_seg_softmax_tam[slice,-1, 1:][np.newaxis]))
            # IOU_score = float(iou_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
            #                              inter_seg_softmax_tam[slice,-1, 1:][np.newaxis]))
            # inter_seg_softmax_tam_plot = torch.argmax(torch.from_numpy(inter_seg_softmax_tam), dim=2)
            # plot_images([entropy_seg_softmax_tam[slice, 0], entropy_seg_softmax_tam[slice, 1],
            #              entropy_seg_softmax_tam[slice, 2], entropy_seg_softmax_tam[slice, 3],entropy_seg_softmax_tam[slice, 4],
            #              inter_seg_softmax_tam_plot[slice, 4], segmentation_labels_mtlrs_1_plot[slice]],
            #             axs=axs[5],show_cbar=[True,True,True,True,True,False,False],
            #             fontsize=fontsize, cmap=['jet', 'jet', 'jet', 'jet', 'jet', 'viridis', 'viridis'], ticks=[[0,1],[0,1],[0,1],[0,1],[0,1],ticks,ticks],
            #              text=[None, None, None, None,None,
            #                                f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm \nF1: {round(f1_score, 3)} \nHD95: {round(HD95_score, 3)} \nIoU: {round(IOU_score, 3)}',
            #                                f'Patient: {Patient_id}\nSlice: {slice}'])
            # axs[5, 0].set_ylabel("SOFTMAX TAM", fontsize=fontsize)
            # plt.tight_layout(pad=0)
            # plt.savefig(
            #     f"/scratch/tmpaquaij/Figures/IP/Intermediate_segmentation_inter_UQ_{Patient_id}_slice:{str(slice)}.png")
            #
            # plt.close()

            ticks = [0, 1, 2, 3, 4]
            wsize = hsize / target_mtlrs_1.shape[2] * target_mtlrs_1.shape[3]
            dice_score_no_mtl = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                     inter_seg_no_mtl[slice, -1, 1:][np.newaxis])
            assd_score_no_mtl = float(
                asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_no_mtl[slice, -1, 1:][np.newaxis]))
            HD95_score_no_mtl = float(hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                                            inter_seg_no_mtl[slice, -1, 1:][np.newaxis]))
            dice_score_logit = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                     inter_seg_logit[slice, -1, 1:][np.newaxis])

            assd_score_logit= float(
                asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_logit[slice, -1, 1:][np.newaxis]))

            HD95_score_logit = float(hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                                            inter_seg_logit[slice, -1, 1:][np.newaxis]))
            dice_score_softmax = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                     inter_seg_softmax[slice, -1, 1:][np.newaxis])

            assd_score_softmax = float(
                asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_softmax[slice, -1, 1:][np.newaxis]))

            HD95_score_softmax = float(hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                                            inter_seg_softmax[slice, -1, 1:][np.newaxis]))
            dice_score_sasg = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                     inter_seg_sasg[slice, -1, 1:][np.newaxis])

            assd_score_sasg = float(
                asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_sasg[slice, -1, 1:][np.newaxis]))

            HD95_score_sasg = float(
                hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                             inter_seg_sasg[slice, -1, 1:][np.newaxis]))
            dice_score_logit_tam = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                     inter_seg_logit_tam[slice, -1, 1:][np.newaxis])
            assd_score_logit_tam = float(
                asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis], inter_seg_logit_tam[slice, -1, 1:][np.newaxis]))

            HD95_score_logit_tam = float(hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                                            inter_seg_logit_tam[slice, -1, 1:][np.newaxis]))
            dice_score_softmax_tam = dice_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                     inter_seg_softmax_tam[slice, -1, 1:][np.newaxis])
            assd_score_softmax_tam = float(
                asd(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                    inter_seg_softmax_tam[slice, -1, 1:][np.newaxis]))

            HD95_score_softmax_tam = float(hausdorff_distance_95_metric(segmentation_labels_mtlrs_1[slice, 1:][np.newaxis],
                                                            inter_seg_softmax_tam[slice, -1, 1:][np.newaxis]))

            inter_seg_no_mtl_plot = torch.argmax(torch.from_numpy(inter_seg_no_mtl), dim=2)
            inter_seg_logit_plot = torch.argmax(torch.from_numpy(inter_seg_logit), dim=2)
            inter_seg_softmax_plot = torch.argmax(torch.from_numpy(inter_seg_softmax), dim=2)
            inter_seg_sasg_plot = torch.argmax(torch.from_numpy(inter_seg_sasg), dim=2)
            inter_seg_softmax_tam_plot = torch.argmax(torch.from_numpy(inter_seg_softmax_tam), dim=2)
            inter_seg_logit_tam_plot = torch.argmax(torch.from_numpy(inter_seg_logit_tam), dim=2)
            segmentation_labels_mtlrs_1_plot = torch.argmax(torch.from_numpy(segmentation_labels_mtlrs_1), dim=1)

            fig, axs = plt.subplots(2, 4, figsize=(4 * wsize, 2 * hsize))
            plot_images([target_mtlrs_1[slice, echo], zero_filled_mtlrs_1[slice, echo],
                         reconstruction_no_mtl[slice, echo], reconstruction_logit[slice, echo]], axs=axs[0],
                        fontsize=fontsize,overlay=[segmentation_labels_mtlrs_1_plot[slice],None,inter_seg_no_mtl_plot[slice,4],inter_seg_logit_plot[slice,4]],ticks_overlay=[0,1,2,3,4],
                        text=[f'Target \nPatient: {Patient_id} \nSlice: {slice}',
                              f'Zero-Filled ',
                              f'JOINT \nDICE: {round(dice_score_no_mtl, 3)} \nHD95: {round(HD95_score_no_mtl, 3)} \nASSD: {round(assd_score_no_mtl, 3)}',
                              f'SUM LOGIT \nDICE: {round(dice_score_logit, 3)} \nHD95: {round(HD95_score_logit, 3)} \nASSD: {round(assd_score_logit, 3)}',
                              ])
            plot_images([reconstruction_softmax[slice, echo], reconstruction_sasg[slice, echo],
                         reconstruction_logit_tam[slice, echo], reconstruction_softmax_tam[slice, echo]],
                        axs=axs[1], fontsize=fontsize,overlay=[inter_seg_softmax_plot[slice,4],inter_seg_sasg_plot[slice,4],inter_seg_logit_tam_plot[slice,4],inter_seg_softmax_tam_plot[slice,4]],ticks_overlay=[0,1,2,3,4],
                        text=[
                            f'SUM SOFTMAX \nDICE: {round(dice_score_softmax, 3)} \nHD95: {round(HD95_score_softmax, 3)} \nASSD: {round(assd_score_softmax, 3)}',
                            f'SASG \nDICE: {round(dice_score_sasg, 3)} \nHD95: {round(HD95_score_sasg, 3)} \nASSD: {round(assd_score_sasg, 3)}',
                            f'TAM LOGIT \nDICE: {round(dice_score_logit_tam, 3)} \nHD95: {round(HD95_score_logit_tam, 3)} \nASSD: {round(assd_score_logit_tam, 3)}',
                            f'TAM SOFTMAX \nDICE: {round(dice_score_softmax_tam, 3)} \nHD95: {round(HD95_score_softmax_tam, 3)} \nASSD: {round(assd_score_softmax_tam, 3)}',
                        ])
            plt.tight_layout(pad=0)
            plt.savefig(
                f"/scratch/tmpaquaij/Figures/IP/Segmentation_{Patient_id}_Echo:{str(echo)}_slice:{str(slice)}.png")
            plt.close()
            print(f"Saved {Patient_id} specific Deep Ensemble figure")

            # fig, axs = plt.subplots(6, 3, figsize=(3 * wsize, 6 * hsize))
            # plot_images([entropy_seg_no_mtl[slice, 4], inter_seg_no_mtl_plot[slice, 4],
            #              segmentation_labels_mtlrs_1_plot[slice]],
            #             show_cbar=[True, False, False],
            #             titles=["Predicted Entropy", "Predicted Segmentation", "Target"], axs=axs[0], fontsize=fontsize,
            #             cmap=[ 'jet', 'viridis', 'viridis'],
            #             ticks=[ [0, 1], ticks, ticks],
            #             text=[ None,
            #                   f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm  \nHD95: {round(HD95_score, 3)}',
            #                   f'Patient: {Patient_id}\nSlice: {slice}'])
            # axs[0][0].set_ylabel("JOINT", fontsize=fontsize)
            #
            #
            #
            # inter_seg_logit_plot = torch.argmax(torch.from_numpy(inter_seg_logit), dim=2)
            # plot_images([entropy_seg_logit[slice, 4], inter_seg_logit_plot[slice, 4],
            #              segmentation_labels_mtlrs_1_plot[slice]],
            #             axs=axs[1], fontsize=fontsize, show_cbar=[True, False, False],
            #             cmap=['jet', 'viridis', 'viridis'],
            #             ticks=[ [0, 1], ticks, ticks],
            #             text=[None,
            #                   f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm  \nHD95: {round(HD95_score, 3)} ',
            #                   f'Patient: {Patient_id}\nSlice: {slice}'])
            # axs[1, 0].set_ylabel("SUM LOGIT", fontsize=fontsize)
            #
            #
            # inter_seg_softmax_plot = torch.argmax(torch.from_numpy(inter_seg_softmax), dim=2)
            # plot_images([ entropy_seg_softmax[slice, 4],
            #              inter_seg_softmax_plot[slice, 4], segmentation_labels_mtlrs_1_plot[slice]],
            #             axs=axs[2], show_cbar=[ True, False, False],
            #             fontsize=fontsize, cmap=['jet', 'viridis', 'viridis'],
            #             ticks=[ [0, 1], ticks, ticks],
            #             text=[None,
            #                   f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm  \nHD95: {round(HD95_score, 3)} ',
            #                   f'Patient: {Patient_id}\nSlice: {slice}'])
            # axs[2, 0].set_ylabel("SUM SOFTMAX", fontsize=fontsize)
            #
            #
            #
            #
            # plot_images([entropy_seg_sasg[slice, 4], inter_seg_sasg_plot[slice, 4],
            #              segmentation_labels_mtlrs_1_plot[slice]],
            #             axs=axs[3], fontsize=fontsize, show_cbar=[True, False, False],
            #             cmap=[ 'jet', 'viridis', 'viridis'],
            #             ticks=[ [0, 1], ticks, ticks],
            #             text=[ None,
            #                   f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm  \nHD95: {round(HD95_score, 3)} ',
            #                   f'Patient: {Patient_id}\nSlice: {slice}'])
            # axs[3, 0].set_ylabel("SASG", fontsize=fontsize)
            #
            #
            # inter_seg_logit_tam_plot = torch.argmax(torch.from_numpy(inter_seg_logit_tam), dim=2)
            # plot_images(
            #     [entropy_seg_logit_tam[slice, 4], inter_seg_logit_tam_plot[slice, 4],
            #      segmentation_labels_mtlrs_1_plot[slice]],
            #     axs=axs[4], fontsize=fontsize, show_cbar=[True, False, False],
            #     cmap=[ 'jet', 'viridis', 'viridis'],
            #     ticks=[ [0, 1], ticks, ticks],
            #     text=[ None,
            #           f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm  \nHD95: {round(HD95_score, 3)} ',
            #           f'Patient: {Patient_id}\nSlice: {slice}'])
            #
            # axs[4, 0].set_ylabel("LOGIT TAM", fontsize=fontsize)
            #
            # inter_seg_softmax_tam_plot = torch.argmax(torch.from_numpy(inter_seg_softmax_tam), dim=2)
            # plot_images([entropy_seg_softmax_tam[slice, 4],
            #              inter_seg_softmax_tam_plot[slice, 4], segmentation_labels_mtlrs_1_plot[slice]],
            #             axs=axs[5], show_cbar=[ True, False, False],
            #             fontsize=fontsize, cmap=['jet', 'viridis', 'viridis'],
            #             ticks=[ [0, 1], ticks, ticks],
            #             text=[None,
            #                   f'DICE: {round(dice_score, 3)} \nASSD: {round(assd_score, 3)} mm  \nHD95: {round(HD95_score, 3)} ',
            #                   f'Patient: {Patient_id}\nSlice: {slice}'])
            # axs[5, 0].set_ylabel("SOFTMAX TAM", fontsize=fontsize)
            # plt.tight_layout(pad=0)
            # plt.savefig(
            #     f"/scratch/tmpaquaij/Figures/IP/Intermediate_segmentation_inter_UQ_{Patient_id}_slice:{str(slice)}.png")
            #
            # plt.close()



            # mean_std_cascade = []
            # for cascade in range(5):
            #     std_no_mtl_soft = std_no_mtl * np.expand_dims(np.sum(segmentation_labels_mtlrs_1[:, 1:], axis=1),
            #                                                   axis=[1, 2])
            #     mean_std_no_mtl_soft = round(np.mean(std_no_mtl_soft[:, cascade, echo]), 5)
            #     std_logit_soft = std_logit * np.expand_dims(np.sum(segmentation_labels_mtlrs_1[:, 1:], axis=1), axis=[1, 2])
            #     mean_std_logit_soft = round(np.mean(std_logit_soft[:, cascade, echo]), 5)
            #     std_softmax_soft = std_softmax * np.expand_dims(np.sum(segmentation_labels_mtlrs_1[:, 1:], axis=1),
            #                                                             axis=[1, 2])
            #     mean_std_softmax_soft = round(np.mean(std_softmax_soft[:, cascade, echo]), 5)
            #     std_sasg_soft = std_sasg * np.expand_dims(np.sum(segmentation_labels_mtlrs_1[:, 1:], axis=1), axis=[1, 2])
            #     mean_std_sasg_soft = round(np.mean(std_sasg_soft[:, cascade, echo]), 5)
            #     std_logit_tam_soft = std_logit_tam * np.expand_dims(np.sum(segmentation_labels_mtlrs_1[:, 1:], axis=1),
            #                                                         axis=[1, 2])
            #     mean_std_logit_tam_soft = round(np.mean(std_logit_tam_soft[:, cascade, echo]), 5)
            #     std_softmax_tam_soft = std_softmax_tam * np.expand_dims(np.sum(segmentation_labels_mtlrs_1[:, 1:], axis=1),
            #                                                             axis=[1, 2])
            #     mean_std_softmax_tam_soft = round(np.mean(std_softmax_tam_soft[:, cascade, echo]), 5)
            #     dataframe[f'Cascade: {cascade}'] = [mean_std_no_mtl_soft, mean_std_logit_soft, mean_std_softmax_soft,
            #                                         mean_std_sasg_soft, mean_std_logit_tam_soft, mean_std_softmax_tam_soft]
            #     mean_std_cascade.append([mean_std_no_mtl_soft,mean_std_logit_soft,mean_std_softmax_soft,mean_std_sasg_soft,mean_std_logit_tam_soft,mean_std_softmax_tam_soft])
            # mean_std_all_patients.append(mean_std_cascade)


            # dataframe.to_excel(writer, sheet_name=f'{Patient_id}')
            # print(f'finished patient: {Patient_id}')
        #     mean_all_patients.append([mean_no_mtl[slice,1,echo],mean_logit[slice,1,echo],mean_softmax[slice,1,echo],mean_sasg[slice,1,echo],mean_logit_tam[slice,1,echo],mean_softmax_tam[slice,1,echo]])
        #     std_all_patients.append([std_no_mtl[slice,1,echo], std_logit[slice,1,echo], std_softmax[slice,1,echo], std_sasg[slice,1,echo], std_logit_tam[slice,1,echo], std_softmax_tam[slice,1,echo]])
        #     mean_all_patients_2.append(
        #         [mean_no_mtl[slice, 2, echo], mean_logit[slice, 2, echo], mean_softmax[slice, 2, echo],
        #          mean_sasg[slice, 2, echo], mean_logit_tam[slice, 2, echo], mean_softmax_tam[slice, 2, echo]])
        #     std_inter_all_patients.append([inter_std_no_mtl[slice,2,echo], inter_std_logit[slice,2,echo], inter_std_softmax[slice,2,echo], inter_std_sasg[slice,2,echo], inter_std_logit_tam[slice,2,echo], inter_std_softmax_tam[slice,2,echo]])
        #
        # mean_std_all_patients = np.mean(mean_std_all_patients, axis=0).T
        # dataframe_mean = pd.DataFrame(index=dataframe.index, columns=dataframe.columns, data=mean_std_all_patients)
        # dataframe_mean.to_excel(writer, sheet_name=f'Mean UQ')
        # dataframe.to_excel(writer, sheet_name=f'{Patient_id}')
        # wsize = hsize / target_mtlrs_1.shape[2] * target_mtlrs_1.shape[3]
        # fig, axs = plt.subplots(6, len(patients), figsize=(len(patients) * wsize, 6 * hsize))
        # for i in range(len(patients)):
        #     plot_images(images=[mean_all_patients[i][0],mean_all_patients[i][1],mean_all_patients[i][2],mean_all_patients[i][3],mean_all_patients[i][4],mean_all_patients[i][5]],
        #                 axs=axs[:,i],
        #                 overlay=[std_all_patients[i][0],std_all_patients[i][1],std_all_patients[i][2],std_all_patients[i][3],std_all_patients[i][4],std_all_patients[i][5]],
        #                 show_cbar_overlay=[True, True, True, True, True, True], fontsize=fontsize,
        #                 ticks_overlay=ticks_overlay, opacity=0.3,text=[f'JOINT \nPatient: {Patient_id} \nSlice: {slice}',"SUM LOGIT","SUM SOFTMAX","SASG","TAM LOGIT","TAM SOFTMAX"] )
        #
        #
        # plt.tight_layout(pad=0)
        # plt.savefig(
        #     "/scratch/tmpaquaij/Figures/IP/Intermediate_" + intermidiate_form + f"_DeepEnsemble_UQ_Echo:{str(echo)}_slice:{str(slice)}_Cascade_2.png")
        # plt.close()
        # wsize = hsize / target_mtlrs_1.shape[2] * target_mtlrs_1.shape[3]
        # fig, axs = plt.subplots(6, len(patients), figsize=(len(patients) * wsize, 6 * hsize))
        # for i in range(len(patients)):
        #     plot_images(images=[mean_all_patients_2[i][0], mean_all_patients_2[i][1], mean_all_patients_2[i][2],
        #                         mean_all_patients_2[i][3], mean_all_patients_2[i][4], mean_all_patients_2[i][5]],
        #                 axs=axs[:, i],
        #                 overlay=[std_inter_all_patients[i][0], std_inter_all_patients[i][1], std_inter_all_patients[i][2],
        #                          std_inter_all_patients[i][3], std_inter_all_patients[i][4], std_inter_all_patients[i][5]],
        #                 show_cbar_overlay=[True, True, True, True, True, True], fontsize=fontsize,
        #                 ticks_overlay=ticks_overlay, opacity=0.3,
        #                 text=[f'JOINT \nPatient: {patients[i]} \nSlice: {slice}', "SUM LOGIT", "SUM SOFTMAX", "SASG",
        #                       "TAM LOGIT", "TAM SOFTMAX"])
        #
        # plt.tight_layout(pad=0)
        # plt.savefig(
        #     "/scratch/tmpaquaij/Figures/IP/Intermediate_" + intermidiate_form + f"_Inter_UQ_Echo:{str(echo)}_slice:{str(slice)}_Cascade_3.png")
        # plt.close()
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=list,default = ["MTR_052","MTR_066","MTR_120","MTR_196","MTR_227"],choices=["MTR_052","MTR_066","MTR_120","MTR_196","MTR_227"])
    parser.add_argument("--echo", type=int,default = 0, choices=[0,1])
    parser.add_argument("--slice", type=int, default=34,choices =["interger between 0 and 79"])
    parser.add_argument("--intermidiate_form", type=str, default="reconstruction",choices=["reconstruction","loglike"])
    parser.add_argument("--ticks_overlay", type=list, default=[0.00, 0.04])
    parser.add_argument("--hsize", type=float, default=5)
    parser.add_argument("--fontsize", type=int, default=15)

    args = parser.parse_args()
    index = ['JOINT', "LOGIT SUM", "SOFTMAX SUM", "SASG", "LOGIT TAM", "SOFTMAX TAM"]
    dataframe = pd.DataFrame(index=index)
    generate_figure(patients=args.patients,echo=args.echo,slice=args.slice,dataframe=dataframe,intermidiate_form=args.intermidiate_form,ticks_overlay=args.ticks_overlay,hsize=args.hsize,fontsize=args.fontsize)