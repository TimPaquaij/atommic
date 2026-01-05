# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import neurokit2 as nk
from tslearn.metrics import dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def mse(x: np.ndarray, y: np.ndarray, maxval: np.ndarray = None) -> float:  # pylint: disable=unused-argument
    """Computes Mean Squared Error (MSE).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.
    maxval : np.ndarray
        Maximum value of the images. If None, it is computed from the images. If the images are normalized, maxval
        should be 1.

    Returns
    -------
    float
        Mean Squared Error.

    Examples
    --------
    >>> from atommic.collections.reconstruction.metrics.reconstruction_metrics import mse
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = np.random.rand(3, 100, 100)
    >>> mse(datax, datay)
    0.17035991151556373
    """
    return np.mean((x - y) ** 2)


def nmse(x: np.ndarray, y: np.ndarray, maxval: np.ndarray = None) -> float:  # pylint: disable=unused-argument
    """Computes Normalized Mean Squared Error (NMSE).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.
    maxval : np.ndarray
        Maximum value of the images. If None, it is computed from the images. If the images are normalized, maxval
        should be 1.

    Returns
    -------
    float
        Normalized Mean Squared Error.

    Examples
    --------
    >>> from atommic.collections.reconstruction.metrics.reconstruction_metrics import nmse
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = np.random.rand(3, 100, 100)
    >>> nmse(datax, datay)
    0.5001060028222054
    """
    return np.linalg.norm(x - y) ** 2 / np.linalg.norm(x) ** 2


def psnr(x: np.ndarray, y: np.ndarray, maxval: np.ndarray = None) -> float:
    """Computes Peak Signal to Noise Ratio (PSNR).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.
    maxval : np.ndarray
        Maximum value of the images. If None, it is computed from the images. If the images are normalized, maxval
        should be 1.

    Returns
    -------
    float
        Peak Signal to Noise Ratio.

    Examples
    --------
    >>> from atommic.collections.reconstruction.metrics.reconstruction_metrics import psnr
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = np.random.rand(3, 100, 100)
    >>> psnr(datax, datay)
    7.6700572264458

    .. note::
        x and y must be normalized to the same range, e.g. [0, 1].

        The PSNR is computed using the scikit-image implementation of the PSNR metric.
        Source: https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio
    """
    maxval = max(np.max(x) - np.min(x), np.max(y) - np.min(y)) if maxval is None else maxval
    return peak_signal_noise_ratio(x, y, data_range=maxval)


def ssim(x: np.ndarray, y: np.ndarray, maxval: np.ndarray = None) -> float:
    """Computes Structural Similarity Index Measure (SSIM).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.
    maxval : np.ndarray
        Maximum value of the images. If None, it is computed from the images. If the images are normalized, maxval
        should be 1.

    Returns
    -------
    float
        Structural Similarity Index Measure.

    Examples
    --------
    >>> from atommic.collections.reconstruction.metrics.reconstruction_metrics import ssim
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = datax * 0.5
    >>> ssim(datax, datay)
    0.01833040155119426

    .. note::
        x and y must be normalized to the same range, e.g. [0, 1].

        The SSIM is computed using the scikit-image implementation of the SSIM metric.
        Source: https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity
    """
    if x.ndim == 2:
        x = x[np.newaxis, :, :]
    if y.ndim == 2:
        y = y[np.newaxis, :, :]
    if x.ndim != 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if x.ndim != y.ndim:
        raise ValueError("Ground truth dimensions does not match prediction dimensions.")

    maxval = max(np.max(x) - np.min(x), np.max(y) - np.min(y)) if maxval is None else maxval
    maxval = max(maxval, 1)
    ssim_score = sum(
        structural_similarity(x[slice_num], y[slice_num], data_range=maxval) for slice_num in range(x.shape[0])
    )
    return ssim_score / x.shape[0]


def rpeak_timing_error_ms(x: np.ndarray, y: np.ndarray, fs: int = 500):
    """
    Compute R-peak timing error (ms) between ground truth and reconstructed ECG.

    Parameters
    ----------
    ecg_gt : np.ndarray
        Ground truth ECG, shape [1, 12, T]
    ecg_rec : np.ndarray
        Reconstructed ECG, shape [1, 12, T]
    fs : int
        Sampling frequency in Hz

    Returns
    -------
    mean_error_ms : float
        Mean absolute R-peak timing error across leads (ms)
    per_lead_error_ms : np.ndarray
        Array of shape [12] with per-lead timing errors (ms)
    """

    assert x.shape == y.shape, "GT and reconstructed ECG must have same shape"

    num_leads = x.shape[1]
    per_lead_error_ms = []

    for lead in range(num_leads):
        signal_gt = y[0, lead]
        signal_rec = x[0, lead]

        # Detect R-peaks
        _, rpeaks_gt = nk.ecg_peaks(signal_gt, sampling_rate=fs)
        _, rpeaks_rec = nk.ecg_peaks(signal_rec, sampling_rate=fs)

        r_gt = rpeaks_gt["ECG_R_Peaks"]
        r_rec = rpeaks_rec["ECG_R_Peaks"]

        # Handle missing detections
        if len(r_gt) == 0 or len(r_rec) == 0:
            per_lead_error_ms.append(np.nan)
            continue

        # Nearest-neighbor matching
        errors = []
        for r in r_gt:
            idx = np.argmin(np.abs(r_rec - r))
            error_ms = (r_rec[idx] - r) / fs * 1000.0
            errors.append(abs(error_ms))

        per_lead_error_ms.append(np.mean(errors))

    per_lead_error_ms = np.array(per_lead_error_ms)

    # Mean across leads (ignore NaNs)
    mean_error_ms = np.nanmean(per_lead_error_ms)

    return mean_error_ms


def dtw_distance_ecg(ecg_gt, ecg_rec):
    """
    Robust DTW for ECG signals [1, 12, T].
    Forces correct dtype and shape for tslearn.
    """

    assert ecg_gt.shape == ecg_rec.shape
    assert ecg_gt.ndim == 3 and ecg_gt.shape[0] == 1

    per_lead_dtw = []

    for lead in range(ecg_gt.shape[1]):
        # FORCE clean 1D float64 arrays
        gt = np.asarray(ecg_gt[0, lead], dtype=np.float64).reshape(-1)
        rec = np.asarray(ecg_rec[0, lead], dtype=np.float64).reshape(-1)

        # Optional normalization (recommended)
        gt = (gt - gt.mean()) / (gt.std() + 1e-8)
        rec = (rec - rec.mean()) / (rec.std() + 1e-8)

        d = dtw(gt, rec)
        per_lead_dtw.append(d)

    per_lead_dtw = np.array(per_lead_dtw)
    return per_lead_dtw.mean()


def dtw_distance_ecg_fast(ecg_gt, ecg_rec):
    """
    Robust DTW metric for ECG.
    Works safely inside PyTorch Lightning validation.
    """

    assert ecg_gt.shape == ecg_rec.shape
    assert ecg_gt.ndim == 3 and ecg_gt.shape[0] == 1

    per_lead_dtw = []

    for lead in range(ecg_gt.shape[1]):
        # CRITICAL: force scalar sequence
        gt = np.asarray(ecg_gt[0, lead], dtype=np.float64).squeeze()
        rec = np.asarray(ecg_rec[0, lead], dtype=np.float64).squeeze()

        # Optional but recommended normalization
        gt = (np.array(gt) - np.mean(gt)) / (np.std(gt) + 1e-8)
        rec = (np.array(rec) - np.mean(rec)) / (np.std(rec) + 1e-8)

        # FINAL GUARANTEE: flatten to 1D of scalars
        gt = gt.tolist()
        rec = rec.tolist()

        d, _ = fastdtw(gt, rec)
        per_lead_dtw.append(d)

    per_lead_dtw = np.array(per_lead_dtw)
    return per_lead_dtw.mean()


def ncc_nan_safe_vectorized(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Zero-lag normalized cross-correlation (NCC) per lead.
    NaN-safe. Assumes signals are already aligned.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays of shape [1, 12, T]

    Returns
    -------
    np.ndarray
        Array of shape [12] with NCC per lead.
    """

    eps = 1e-8

    mean_x = np.nanmean(x, axis=2, keepdims=True)
    mean_y = np.nanmean(y, axis=2, keepdims=True)

    std_x = np.nanstd(x, axis=2, keepdims=True, ddof=0)
    std_y = np.nanstd(y, axis=2, keepdims=True, ddof=0)

    std_x = np.maximum(std_x, eps)
    std_y = np.maximum(std_y, eps)

    zx = (x - mean_x) / std_x
    zy = (y - mean_y) / std_y

    valid = np.isfinite(zx) & np.isfinite(zy)
    n_valid = np.sum(valid, axis=2)

    prod = zx * zy
    num = np.nansum(prod, axis=2)

    # squeeze batch dimension
    num = num[0]  # [12]
    n_valid = n_valid[0]  # [12]

    ncc = np.full(12, np.nan)
    good = n_valid > 1
    ncc[good] = num[good] / n_valid[good]

    return np.nanmean(ncc)


METRIC_FUNCS = {
    "MSE": mse,
    "NMSE": nmse,
    "PSNR": psnr,
    "SSIM": ssim,
    "RPEMS": rpeak_timing_error_ms,
    "DTW": dtw_distance_ecg_fast,
    "NCCc": ncc_nan_safe_vectorized,
}


class ReconstructionMetrics:
    r"""Maintains running statistics for a given collection of reconstruction metrics.

    Examples
    --------
    >>> from atommic.collections.reconstruction.metrics.reconstruction_metrics import ReconstructionMetrics
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = np.random.rand(3, 100, 100)
    >>> metrics = ReconstructionMetrics(METRIC_FUNCS, 'output', 'method')
    >>> metrics.push(datax, datay)
    >>> metrics.means()
    {'MSE': 0.17035991151556373, 'NMSE': 0.5001060028222054, 'PSNR': 7.6700572264458, 'SSIM': 0.01833040155119426}
    >>> metrics.__repr__()
    'MSE = 0.1704 +/- 0.01072 NMSE = 0.5001 +/- 0.01636 PSNR = 7.67 +/- 0.319 SSIM = 0.01833 +/- 0.03527\n'
    """

    def __init__(self, metric_funcs):
        """Inits :class:`ReconstructionMetrics`.

        Parameters
        ----------
        metric_funcs : dict
            A dict where the keys are metric names and the values are Python functions for evaluating that metric.
        """
        self.metrics_scores = {metric: Statistics() for metric in metric_funcs}

    def push(self, x, y, maxval=None):
        """Pushes a new batch of metrics to the running statistics.

        Parameters
        ----------
        x : np.ndarray
            Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
            images, the first dimension should be 1.
        y : np.ndarray
            Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
            images, the first dimension should be 1.
        maxval : np.ndarray
            Maximum value of the images. If None, it is computed from the images. If the images are normalized, maxval
            should be 1. Default is ``None``.

        Returns
        -------
        dict
            A dict where the keys are metric names and the values are the computed metric scores.
        """
        for metric, func in METRIC_FUNCS.items():
            self.metrics_scores[metric].push(func(x, y, maxval=maxval))

    def means(self):
        """Mean of the means of each metric."""
        return {metric: stat.mean() for metric, stat in self.metrics_scores.items()}

    def stddevs(self):
        """Standard deviation of the means of each metric."""
        return {metric: stat.stddev() for metric, stat in self.metrics_scores.items()}

    def __repr__(self):
        """Representation of the metrics."""
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))

        res = " ".join(f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}" for name in metric_names) + "\n"

        return res
