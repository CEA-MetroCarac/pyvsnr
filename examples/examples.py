"""
Example of filtering methods like destriping (curtains and stripes correction)
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread

from vsnr import VSNR, GPU_ENV


def ex_camera(defects_type='stripes'):
    """
    Example of stripes or curtains removal with 'camera' image
    """
    assert (defects_type in ['stripes', 'curtains'])

    img_ref = imread(os.path.join("data", "camera.tif"))
    img = imread(os.path.join("data", f"camera_{defects_type}.tif"))

    # vsnr object creation
    vsnr = VSNR(img_ref.shape)
    if defects_type == 'stripes':
        vsnr.add_filter(alpha=5e-2, name='dirac_h')
    else:
        vsnr.add_filter(alpha=5e-2, name='gabor', sigma=(3, 40))

    # image processing
    t0 = time.process_time()
    vsnr.initialize()
    img_corr = vsnr.eval(img, maxit=100)
    print("CPU/GPU running time :", time.process_time() - t0)

    # image renormalization
    img_corr = np.clip(img_corr, 0., 1.)

    # plotting
    fig0 = plt.figure(figsize=(14, 4))
    fig0.sfn = f"ex_camera_{defects_type}"
    plt.subplot(131)
    plt.title("Reference")
    plt.imshow(img_ref)
    plt.subplot(132)
    plt.title("Reference + noise")
    plt.imshow(img)
    plt.subplot(133)
    plt.title("Corrected")
    plt.imshow(img_corr)
    plt.tight_layout()

    fig1 = vsnr.plot_cvg_criteria()
    fig1.sfn = f"ex_camera_{defects_type}_cvg"


def ex_fib_sem():
    """
    Example of vsnr application on a real FIB-SEM image
    """
    img = imread(os.path.join("data", "fib_sem_sample.tif"))

    # image renormalization into [0, 1]
    vmax = img.max()
    img = img / vmax

    # vsnr object creation
    vsnr = VSNR(img.shape)
    vsnr.add_filter(alpha=1e-2, name='gabor', sigma=(1, 30), theta=358)
    # vsnr.add_filter(alpha=1e-1, name='gabor', sigma=(5, 30), theta=358)

    # image processing
    t0 = time.process_time()
    vsnr.initialize()
    img_corr = vsnr.eval(img, maxit=100)
    print("CPU/GPU running time :", time.process_time() - t0)

    # image renormalization to recover original format
    img_corr = (np.clip(img_corr, 0., 1.) * vmax).astype(np.uint8)

    # plotting
    fig0 = plt.figure(figsize=(12, 6))
    fig0.sfn = "ex_fib_sem"
    plt.subplot(121)
    plt.title("Original")
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.title("Corrected")
    plt.imshow(img_corr, cmap='gray')
    plt.tight_layout()

    fig1 = vsnr.plot_cvg_criteria()
    fig1.sfn = "ex_fib_sem_cvg"


def ex_vsnr_perf_evaluation():
    """
    Compare CPU and GPU running time on different image sizes
    """
    img = imread(os.path.join("data", "fib_sem_sample.tif"))

    # image renormalization into [0, 1]
    img = img / 255.

    # image sizes to explore
    sizes = [256, 512, 1024, 2048]

    # numerical CPU/GPU libraries
    numerical_libs = ['numpy']
    if GPU_ENV:
        numerical_libs += ['cupy']

    # image processing and plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.sfn = "ex_vsnr_perf_evaluation"
    for numerical_lib in numerical_libs:
        tcpus_1 = []
        tcpus_2 = []
        for size in sizes:
            img_test = img[:size, :size]

            vsnr = VSNR(img_test.shape)
            vsnr.add_filter(alpha=1e-2, name='gabor', sigma=(1, 30), theta=358)

            # first run
            t1 = time.process_time()
            vsnr.initialize(is_gpu=(numerical_lib == 'cupy'))
            vsnr.eval(img_test, maxit=20, cvg_threshold=0)
            tcpu_1 = time.process_time() - t1
            tcpus_1.append(tcpu_1)

            # second run
            t2 = time.process_time()
            vsnr.initialize(is_gpu=(numerical_lib == 'cupy'))
            vsnr.eval(img_test, maxit=20, cvg_threshold=0)
            tcpu_2 = time.process_time() - t2
            tcpus_2.append(tcpu_2)

        ax1.plot(sizes, tcpus_1, label=numerical_lib)
        ax2.plot(sizes, tcpus_2, label=numerical_lib)
    ax1.legend()
    ax1.set_title("First run")
    ax1.set_xlabel("Image size [px]")
    ax1.set_ylabel("Running time [s]")
    ax2.legend()
    ax2.set_title("Second run")
    ax2.set_xlabel("Image size [px]")
    ax2.set_ylabel("Running time [s]")


if __name__ == '__main__':
    # pylint: disable=I0011,C0103

    ex_camera(defects_type='stripes')
    ex_camera(defects_type='curtains')
    # ex_fib_sem()
    # ex_vsnr_perf_evaluation()

    plt.show()
