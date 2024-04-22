""" Utilities for tests """
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

from skimage._shared.utils import (
    _supported_float_type,
    check_shape_equality,
    warn,
)
from skimage.util.dtype import dtype_range

from src.pyvsnr import vsnr2d


def measure_vsnr(
    img, filters, maxit=20, algo='auto', beta=10.0, nblocks="auto", norm=True
):
    """ Measure the execution time of a vsnr2d algorithm """
    t0 = time.perf_counter()

    img_corr = vsnr2d(img, filters, maxit=maxit, algo=algo, beta=beta, norm=norm)

    process_time = round(time.perf_counter() - t0, 3)

    if algo == 'cuda':
        print("\033[92m" + f"CUDA: {process_time} sec" + "\033[0m")
    else:
        print("\033[94m" + f"{algo}: {process_time} sec" + "\033[0m")

    return img_corr


def measure_vsnr_cuda(img, filters, nite=20, beta=10.0, nblocks="auto", norm=True):
    """ Measure the execution time of the vsnr2d_cuda algorithm """
    img_corr_cuda = measure_vsnr(
        img, filters, algo='cuda', maxit=nite, beta=beta, nblocks=nblocks, norm=norm
    )
    return img_corr_cuda


def measure_vsnr_cupy(img, filters, maxit=20, beta=10.0, norm=True):
    """ Measure the execution time of the vsnr2d_cupy algorithm """
    img_corr_cupy = measure_vsnr(
        img, filters, algo='cupy', maxit=maxit, beta=beta, norm=norm
    )
    return img_corr_cupy


def measure_vsnr_numpy(img, filters, maxit=20, beta=10.0, norm=True):
    """ Measure the execution time of the vsnr2d_numpy algorithm """
    img_corr_numpy = measure_vsnr(
        img, filters, algo='numpy', maxit=maxit, beta=beta, norm=norm
    )
    return img_corr_numpy


def print_max_diff(img_corr_py, img_corr_cuda, xp):
    """ Print the maximum difference between the CUDA and Python implementations """
    difference = np.abs(img_corr_cuda - img_corr_py).max()

    print(
        f"Greatest difference between CUDA and {xp.__name__}:"
        f" {format(difference)}"
    )
    assert np.allclose(
        img_corr_cuda, img_corr_py, atol=1e-2 # usually 1e-4 but if image is in 255 range it is 1e-2
    ), "CUDA and Python implementations are not equal"


def mean_squared_error(image0, image1):
    """
    Compute the mean-squared error between two images.

    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.
    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)


def _as_floats(image0, image1):
    """ Promote im1, im2 to nearest appropriate floating point precision. """
    float_type = _supported_float_type((image0.dtype, image1.dtype))
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1


def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.

    Returns
    -------
    psnr : float
        The PSNR metric.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """
    if isinstance(image_test, cp.ndarray):
        image_test = cp.asnumpy(image_test)

    check_shape_equality(image_true, image_test)

    if data_range is None:
        if image_true.dtype != image_test.dtype:
            warn(
                "Inputs have mismatched dtype.  Setting data_range based on "
                "image_true."
            )
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = np.min(image_true), np.max(image_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "image_true has intensity values outside the range expected "
                "for its data type. Please manually specify the data_range."
            )
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    image_true, image_test = _as_floats(image_true, image_test)

    err = mean_squared_error(image_true, image_test)

    if err == 0:
        return np.inf
    
    return 10 * np.log10((data_range**2) / err)


def print_psnr(img, noisy_img, img_corr_py, img_corr_cuda):
    """ Print the PSNR of the noisy image, the CUDA corrected image and the Python """

    psnr_noisy = peak_signal_noise_ratio(img, noisy_img)
    psnr_corrected_cuda = peak_signal_noise_ratio(img, img_corr_cuda)
    psnr_corrected_py = peak_signal_noise_ratio(img, img_corr_py)

    print(
        "\033[95m"
        + f"PSNR noisy image: {np.round(psnr_noisy, 2)}dB"
        + "\033[0m"
    )
    print(
        "\033[92m"
        + f"PSNR CUDA corrected image: {np.round(psnr_corrected_cuda, 2)}dB"
        + "\033[0m"
    )
    print(
        "\033[94m"
        + f"PSNR Python corrected image: {np.round(psnr_corrected_py, 2)}dB"
        + "\033[0m"
    )


def plot_results(
        img, noisy_img, img_corr_py, img_corr_cuda, xp, save_plots=False, title="fig.png", vmin=0, vmax=1):
    """ Plot the original image, the noisy image, the Python corrected image and """

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(noisy_img, vmin=vmin, vmax=vmax)
    plt.title("Noisy image")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(img_corr_py, vmin=vmin, vmax=vmax)
    plt.title(f"Cleaned {xp.__name__} image")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(img_corr_cuda, vmin=vmin, vmax=vmax)
    plt.title("Cleaned CUDA image")
    plt.axis("off")

    if save_plots:
        plt.savefig(
            os.path.join(os.path.dirname(__file__), "images", title),
            bbox_inches="tight",
        )

    plt.show()
