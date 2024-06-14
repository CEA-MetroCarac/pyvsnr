"""Pytest file to verify that the python code is equivalent to the cuda code"""
import pathlib

import cupy as cp
import numpy as np
from skimage import data


from pyvsnr import vsnr2d

from pyvsnr.vsnr2d import vsnr_admm as vsnr_admm_batch

from src.pyvsnr.utils import stripes_addition

DIRNAME = pathlib.Path(__file__).parent
PRECOMPILED_PATH = DIRNAME.parent / "src" / "pyvsnr" / "precompiled"

xp = cp

def test_vsnr2d_batch_single_img():
    """ Test the batch processing """
    img = data.camera().astype(np.float32)
    img = xp.asarray(img)


    filters = [{"name": "Dirac", "noise_level": 10}]

    gu0_individually = vsnr2d(img, filters, algo="cuda")
    gu0_batch = vsnr2d(img, filters, algo="cupy")

    xp.testing.assert_allclose(gu0_individually, gu0_batch, atol=1e-2)

def test_vsnr2d_batch_random_imgs():
    """ Test the batch processing """
    gu0 = data.camera()

    if xp == cp:
        # convert numpy array to cupy array
        gu0 = cp.asarray(gu0)

    # Create a batch of 10 different images by adding an amount of random noise to each image
    gu0_batch = xp.stack([gu0 + xp.random.normal(scale=10, size=gu0.shape) for _ in range(10)])

    filters = [{"name": "Dirac", "noise_level": 10}]

    gu0_individually = vsnr2d(gu0_batch, filters, algo="cuda")
    gu0_batch = vsnr2d(gu0_batch, filters, algo="cupy")

    for i in range(10):
        xp.testing.assert_allclose(gu0_individually[i], gu0_batch[i], atol=1e-2) # 100.01 vs 100.00
