import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

from skimage._shared.utils import _supported_float_type, check_shape_equality, warn
from skimage.util.dtype import dtype_range
from pyvsnr.vsnr2d import create_gabor
from pyvsnr import vsnr2d, vsnr2d_cuda

def measure_vsnr_cuda(img, filters, nite=20, beta=10.):
    t0 = time.perf_counter()
    img_corr_cuda = vsnr2d_cuda(img, filters, nite=nite, beta=beta, nblocks='auto')
    process_time = round(time.perf_counter() - t0,3)
    print("\033[92m" + f"CUDA: {process_time} sec" + "\033[0m")

    return img_corr_cuda

def measure_vsnr_py(img, filters, nite=20, beta=10., xp=np):
    t0 = time.perf_counter()
    img_corr_py = vsnr2d(img, filters, nite=nite, beta=beta, xp=xp)
    process_time = round(time.perf_counter() - t0,3)
    print("\033[94m" + f"{xp.__name__}: {process_time} sec" + "\033[0m")

    return img_corr_py

def measure_vsnr_cupy(img, filters, nite=20, beta=10.):
    img_corr_cupy = measure_vsnr_py(img, filters, nite=nite, beta=beta, xp=cp)
    return img_corr_cupy

def measure_vsnr_numpy(img, filters, nite=20, beta=10.):
    img_corr_numpy = measure_vsnr_py(img, filters, nite=nite, beta=beta, xp=np)
    return img_corr_numpy

def print_max_diff(img_corr_py, img_corr_cuda, xp):
    difference = xp.abs(xp.asarray(img_corr_cuda) - xp.asarray(img_corr_py)).max()
    print(f"Greatest difference between CUDA and {xp.__name__}: {format(difference)}")
    assert xp.allclose(img_corr_cuda, img_corr_py, atol=1e-4), "CUDA and Python implementations are not equal"

def vmax_encoding(arr):
    """ Basic function to estimate vmax value related to array type encoding"""
    if arr.max() <= 1.:
        return 1.
    elif arr.max() <= 255:
        return 255
    elif arr.max() <= 65535:
        return 65535
    else:
        raise IOError("maximum value over 65535 !!!")
    
def pad_centered(arr, shape_ref, value=0):
    """
    Return a centered ND-array with a surrounding padding 'value'

    Parameters
    ----------
    arr: numpy.ndarray()
        Array to handle
    shape_ref:
        Final array shape to reach
    value: float, optional
        Value used for the padding

    Returns
    -------
    arr_pad: numpy.ndarray()
        The resulting padded array


    .. plot::

        from common_ex.core.ex_array import ex_pad_centered
        ex_pad_centered(make_plots=True)
    """
    assert (len(shape_ref) == len(arr.shape))

    dim = len(shape_ref)
    arr_pad = arr.copy()
    for k in range(dim):
        # gap between shape_ref and shape_max to pad
        gap = shape_ref[k] - arr.shape[k]
        gap2 = gap // 2

        # swap axes to work on axis=0
        arr_pad = np.swapaxes(arr_pad, 0, k)

        # padding
        if gap >= 0:
            width = (gap2, gap - gap2)
            if dim > 1:
                width = (width,) + (dim - 1) * ((0, 0),)
            arr_pad = np.pad(arr_pad, width, constant_values=value)
        # cutting
        else:
            arr_pad = arr_pad[-gap2:-gap2 + shape_ref[k], ...]

        # return to original axis
        arr_pad = np.swapaxes(arr_pad, 0, k)

    return arr_pad

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

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.

    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)

def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
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

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_psnr`` to
        ``skimage.metrics.peak_signal_noise_ratio``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """
    if type(image_test) == cp.ndarray:
        image_test = cp.asnumpy(image_test)
        
    check_shape_equality(image_true, image_test)

    if data_range is None:
        if image_true.dtype != image_test.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "image_true.")
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = np.min(image_true), np.max(image_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "image_true has intensity values outside the range expected "
                "for its data type. Please manually specify the data_range.")
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    image_true, image_test = _as_floats(image_true, image_test)

    err = mean_squared_error(image_true, image_test)
    return 10 * np.log10((data_range ** 2) / err)

def print_psnr(img, noisy_img, img_corr_py, img_corr_cuda):
    if type(img_corr_py) == cp.ndarray:
        img_corr_py = img_corr_py.get()

    psnr_noisy = peak_signal_noise_ratio(img, noisy_img)
    psnr_corrected_cuda = peak_signal_noise_ratio(img, img_corr_cuda)
    psnr_corrected_py = peak_signal_noise_ratio(img, img_corr_py)

    print("\033[95m" + f"PSNR noisy image: {np.round(psnr_noisy, 2)}dB" + "\033[0m")
    print("\033[92m" + f"PSNR CUDA corrected image: {np.round(psnr_corrected_cuda, 2)}dB" + "\033[0m")
    print("\033[94m" + f"PSNR Python corrected image: {np.round(psnr_corrected_py, 2)}dB" + "\033[0m")

def plot_results(img, noisy_img, img_corr_py, img_corr_cuda, xp, save_plots=False, title="fig.png"):
    if type(img_corr_py) == cp.ndarray:
        img_corr_py = cp.asnumpy(img_corr_py)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(img, vmin=0, vmax=1)
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(noisy_img, vmin=0, vmax=1)
    plt.title('Noisy image')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(img_corr_py, vmin=0, vmax=1)
    plt.title(f'Cleaned {xp.__name__} image')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(img_corr_cuda, vmin=0, vmax=1)
    plt.title('Cleaned CUDA image')
    plt.axis('off')

    if save_plots:
        plt.savefig(os.path.join(os.path.dirname(__file__), 'images', title), bbox_inches='tight')

    plt.show()

def stripes_addition(img_base, amplitude):
    np.random.seed(0) # DEBUG
    # norm_img = np.linalg.norm(img_base) # DEBUG
    # test=np.zeros_like(img_base) # DEBUG

    noisy_img = img_base.copy()
    vmax=vmax_encoding(noisy_img)

    noisy_img = noisy_img / vmax

    for i in range(img_base.shape[0]):
        noise = amplitude * (np.random.random() - 0.5)
        noisy_img[i] += noise

        # test[i] = noise # DEBUG

    # print("norme de l'image : ", norm_img) # DEBUG
    # print("norme du bruit ajouté : ", np.linalg.norm(test)) # DEBUG
    # print(f"noise norm : {round(np.linalg.norm(test) / norm_img * 100, 3)} %") # DEBUG

    noisy_img = np.clip(noisy_img, 0, 1)

    return noisy_img

def curtains_addition(img_ref, seed=0, amplitude=0.2, sigma=(3,40), angle=0, threshold=0.999):
    """
    Add curtains effects in a image

    Parameters
    ----------
    img_ref: numpy.ndarray((m, n))
        Original image
    seed: float, optional
        Seed associated to randomized noise (for stripes definition)
    amplitude: float, optional
        Positive relative amplitude associated to the strides.
    sigma: tuple of 2 floats, optional
        Pixel sizes of the spots (curtains) in x and y directions respectively
    theta: float, optional
        Spot orientation (angle (in clockwise) [in degrees])
    threshold: float, optional
        Parameters to select more or less curtains positions. The higher the
        threshold is, the less positions there are

    Returns
    -------
    img: numpy.ndarray((m, n))
        Image with curtains
    """
    assert (amplitude >= 0.)

    np.random.seed(seed)

    n0,n1 = img_ref.shape
    sigmax, sigmay = sigma

    # relative to absolute noise amplitude conversion
    vmax = vmax_encoding(img_ref)
    amplitude *= vmax

    # curtains definition (from gabor filter) and location
    psi = create_gabor(n0, n1, 0.2, sigmax, sigmay, angle=angle, phase=0, lambda_=0., xp=np).reshape(n0, n1)

    psi *= 0.01 / psi.max()  # renormalization
    psi = pad_centered(psi, img_ref.shape, value=0)

    position = np.random.random(img_ref.shape)
    position = (position > threshold).astype(float)
    noise = np.fft.irfft2(np.fft.rfft2(position) * np.fft.rfft2(psi))

    noise *= amplitude / noise.max()

    # if dark_curtains:
    #     noise *= -1.

    img = img_ref + noise

    # if renormalization:
    #     img = np.clip(img, 0, vmax)

    return img