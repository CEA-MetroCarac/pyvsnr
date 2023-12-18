import os
import numpy as np
import pathlib
from ctypes import POINTER, c_int, c_float, CDLL

# precompiled path = file path/../precompiled
PRECOMPILED_PATH = pathlib.Path(__file__).parent / 'precompiled'

def get_dll():
    try:
        if os.name == 'nt':
            os.add_dll_directory(str(PRECOMPILED_PATH))
            return CDLL(
                str(PRECOMPILED_PATH / "libvsnr2d.dll"),
                winmode=0,
            )
        else:
            # nvcc -lcufft -lcublas --compiler-options '-fPIC'
            # -o precompiled/libvsnr2d.so --shared vsnr2d.cu
            return CDLL(str(PRECOMPILED_PATH / "libvsnr2d.so"))
    except OSError as err:
        raise OSError('Problem loading the compiled library from '
                      f'{PRECOMPILED_PATH}, please try recompiling '
                      '(see readme)') from err
   
def get_nblocks():
    """ Get the number of maximum threads per block library"""
    dll = get_dll()
    return dll.getMaxBlocks()

def get_vsnr2d():
    """ Load the 'cuda' function from the dedicated .dll library"""
    dll = get_dll()
    func = dll.VSNR_2D_FIJI_GPU
    func.argtypes = [POINTER(c_float), c_int, POINTER(c_float),
                     c_int, c_int, c_int,
                     c_float, POINTER(c_float), c_int, c_float]
    return func

def vsnr2d_cuda(img, filters, nite=20, beta=10., nblocks='auto'):
    r"""
    Calculate the corrected image using the 2D-VSNR algorithm in libvsnr2d.dll

    .. note:
    To ease code comparison with the original onde, most of the variable names
    have been kept as nearly as possible during the code transcription.
    Accordingly, PEP8 formatting compatibility is not always respected.

    Parameters
    ----------
    img: numpy.ndarray((n0, n1))
        The image to process
    filters: list of dicts
        Dictionaries that contains filters definition.
        Example For a 'Dirac' filter:
        - filter={'name':'Dirac', 'noise_level':10}
        Example For a 'Gabor' filter:
        - filter={'name':'Gabor', 'noise_level':5, 'sigma':(3, 40), 'theta':45}
        For further informations, see :
        https://www.math.univ-toulouse.fr/~weiss/Codes/VSNR/Documentation_VSNR_V2_Fiji.pdf
    nite: int, optional
        Number of iterations in the denoising processing
    beta: float, optional
        Beta parameters
    nblocks: 'auto' or int, optional
        Number of threads per block to work with

    Returns
    -------
    img_corr: numpy.ndarray((n0, n1))
        The corrected image
    """
    length = len(filters)
    n0, n1 = img.shape

    # psis definition from filters
    psis = []
    for filt in filters:
        name = filt['name']
        noise_level = filt['noise_level']
        if name == 'Dirac':
            psis += [0, noise_level]
        elif name == 'Gabor':
            sigma = filt['sigma']
            theta = filt['theta']
            psis += [1, noise_level, sigma[0], sigma[1], theta]
        else:
            raise IOError(f"filter name '{name}' should be 'Dirac' or 'Gabor'")

    # flattened arrays and corresponding pointers definition
    psis = np.asarray(psis).flatten()
    u0 = img.flatten()
    u = np.zeros_like(u0)

    psis_ = (c_float * len(psis))(*psis)
    u0_ = (c_float * len(u0))(*u0)
    u_ = (c_float * len(u))(*u)

    # 'auto' nblocks definition
    nblocks_max = get_nblocks()
    if nblocks == 'auto':
        nblocks = nblocks_max
    else:
        nblocks = max(nblocks_max, nblocks)

    # calculation
    vmax = u0.max()
    vsnr_func = get_vsnr2d()
    vsnr_func(psis_, length, u0_, n0, n1, nite, beta, u_, nblocks, vmax)

    # reshaping
    img_corr = np.array(u_).reshape(n0, n1).astype(float)

    img_corr = np.clip(img_corr, 0, 1)
    
    return img_corr
