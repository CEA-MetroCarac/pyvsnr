""" Pytest file to verify that the python code is equivalent to the cuda code """
import pathlib

import cupy as cp
import numpy as np
from skimage import data

from src.pyvsnr import vsnr2d
from src.pyvsnr.vsnr2d_batch import vsnr2d_batch

from src.pyvsnr.vsnr2d import create_filters
from src.pyvsnr.vsnr2d_batch import create_filters_batch

from src.pyvsnr.utils import stripes_addition

DIRNAME = pathlib.Path(__file__).parent
PRECOMPILED_PATH = DIRNAME.parent / "src" / "pyvsnr" / "precompiled"

xp = cp


def test_create_filters_batch():
    """ Test the create_filters functions """

    u0 = np.loadtxt(DIRNAME / "TEST_VSNR_ADMM/camera.txt", dtype=np.float32)
    gpsi_kernel = np.loadtxt(
        DIRNAME / "TEST_CREATE_FILTERS/out.txt", dtype=np.float32
    )
    u0 = u0.reshape(512, 512)

    # CREATE_FILTERS
    gu0 = u0.copy()
    gu0_batch = np.repeat(gu0[np.newaxis, :], 10, axis=0)
    gpsi1 = create_filters(
        [{"name": "Dirac", "noise_level": 10}], gu0, 512, 512, xp
    )
    gpsi = create_filters_batch(
        [{"name": "Dirac", "noise_level": 10}], gu0_batch, 512, 512, xp
    )

    for i in range(10):
        assert np.allclose(gpsi[i], gpsi1)


def test_batch():
    """ Test the batch processing """
    gu0 = data.camera()

    if xp == cp:
        # convert numpy array to cupy array
        gu0 = cp.asarray(gu0)

    gu0_batch = xp.repeat(gu0[np.newaxis, :], 10, axis=0)
    filters = [{"name": "Dirac", "noise_level": 10}]
    print(gu0.shape, gu0_batch.shape)

    gu0 = vsnr2d(gu0, filters)
    gu0_batch = vsnr2d_batch(gu0_batch, filters)  

    for i in range(10):
        assert xp.allclose(gu0_batch[i], gu0)