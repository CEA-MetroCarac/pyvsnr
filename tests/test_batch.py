""" Pytest file to verify that the python code is equivalent to the cuda code """
import pathlib

import cupy as cp
import numpy as np
from skimage import data

from src.pyvsnr import vsnr2d
from src.pyvsnr.vsnr2d_batch import vsnr2d_batch

from src.pyvsnr.vsnr2d import create_filters, vsnr_admm
from src.pyvsnr.vsnr2d_batch import create_filters_batch
from src.pyvsnr.vsnr2d_batch import vsnr_admm as vsnr_admm_batch

from src.pyvsnr.utils import stripes_addition

DIRNAME = pathlib.Path(__file__).parent
PRECOMPILED_PATH = DIRNAME.parent / "src" / "pyvsnr" / "precompiled"

xp = cp


def test_create_filters_batch():
    """ Test the create_filters functions """

    u0 = data.camera()
    u0 = u0.reshape(512, 512)

    # CREATE_FILTERS
    gu0 = xp.asarray(u0.copy())
    gu0_batch = xp.repeat(gu0[xp.newaxis, :], 10, axis=0)
    gpsi1 = create_filters(
        [{"name": "Dirac", "noise_level": 10}], gu0, 512, 512, xp
    )
    gpsi = create_filters_batch(
        [{"name": "Dirac", "noise_level": 10}], gu0_batch, 512, 512, xp
    )

    for i in range(10):
        assert xp.allclose(gpsi[i], gpsi1)
        
def test_admm_batch():
    """ Test the vsnr_admm_batch function """

    u0 = data.camera()
    u0 = u0.reshape(512, 512).ravel()

    # CREATE_FILTERS
    gu0 = xp.asarray(u0.copy())
    gu0_batch = xp.stack([gu0 + xp.random.normal(scale=10, size=gu0.shape) for _ in range(10)])

    filters = [{"name": "Dirac", "noise_level": 10}]

    # VSNR_ADMM
    nit = 10
    beta = 1.0
    cvg_threshold = 0.01
    
    gu0_individually = [vsnr_admm(gu0, create_filters(filters, gu0, 512, 512, xp), 512, 512, nit, beta, xp, cvg_threshold=cvg_threshold)[0] for gu0 in gu0_batch]
    gu0_batch = vsnr_admm_batch(gu0_batch, create_filters_batch(filters, gu0_batch, 512, 512, xp), 512, 512, nit, beta, xp, cvg_threshold=cvg_threshold)[0]

    for i in range(10):
        assert xp.allclose(gu0_batch[i], gu0_individually[i], atol=1e-5)

def test_vsnr2d_batch_same_img():
    """ Test the batch processing """
    gu0 = data.camera()

    if xp == cp:
        # convert numpy array to cupy array
        gu0 = cp.asarray(gu0)

    gu0_batch = xp.repeat(gu0[xp.newaxis, :], 10, axis=0)
    filters = [{"name": "Dirac", "noise_level": 10}]
    print(gu0.shape, gu0_batch.shape)

    gu0 = vsnr2d(gu0, filters)
    gu0_batch = vsnr2d_batch(gu0_batch, filters)  

    for i in range(10):
        assert xp.allclose(gu0_batch[i], gu0)

def test_vsnr2d_batch_random_imgs():
    """ Test the batch processing """
    gu0 = data.camera()

    if xp == cp:
        # convert numpy array to cupy array
        gu0 = cp.asarray(gu0)

    # Create a batch of 10 different images by adding an amount of random noise to each image
    gu0_batch = xp.stack([gu0 + xp.random.normal(scale=10, size=gu0.shape) for _ in range(10)])

    filters = [{"name": "Dirac", "noise_level": 10}]

    gu0_individually = [vsnr2d(gu0, filters) for gu0 in gu0_batch]
    gu0_batch = vsnr2d_batch(gu0_batch, filters)  

    for i in range(10):
        #print(np.max(np.abs(gu0_individually[i] - gu0_batch[i])))
        assert xp.allclose(gu0_individually[i], gu0_batch[i], atol=1e-5) # This test pass only with atol 1e-5

def test_vsnr2d_batch_single_img():
    """ Test the batch processing """
    img = data.camera()
    img = xp.asarray(img)


    filters = [{"name": "Dirac", "noise_level": 10}]

    gu0_individually = vsnr2d(img, filters)
    gu0_batch = vsnr2d_batch(img, filters)  

    assert xp.allclose(gu0_individually, gu0_batch, atol=1e-5) # This test pass only with atol 1e-5