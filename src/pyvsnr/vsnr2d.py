"""
This module contains the 2D-VSNR algorithm implementation.
its a port of the original CUDA implementation to python using cupy.
It contains several improvements in terms of performance and memory usage.
"""
import numpy as np
from .vsnr2d_cuda import get_dll, vsnr2d_cuda

# determine the algo to use for auto mode
# 1. cupy; 2. cuda; 3. numpy
CUDA_AVAILABLE = False
CUPY_AVAILABLE = False
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    try:
        get_dll()
        CUDA_AVAILABLE = True
    except:
        pass

def compute_phi(fphi1, fphi2, beta, xp):
    """Compute the value of fphi based on the values of fphi1, fphi2, and beta."""
    # Compute the squares of the real and imaginary parts of fphi1 and fphi2
    fphi1_squared_real = xp.square(xp.abs(fphi1))
    fphi2_squared_real = xp.square(xp.abs(fphi2))

    # Compute the value of fphi
    fphi = 1 + beta * (fphi1_squared_real + fphi2_squared_real)

    # Create a complex CuPy array with the computed values of fphi
    fphi_complex = xp.zeros_like(fphi, dtype=xp.complex64)
    fphi_complex.real = fphi

    return fphi_complex


def update_psi(fpsitemp, fsum, alpha, xp):
    """Update the value of fsum based on the values of fpsitemp, fsum, and alpha."""
    # // fsum += |fpsitemp|^2 / alpha_i;
    fsum = xp.add(fsum, xp.divide(fpsitemp, alpha))
    return fsum


def create_dirac(n0, n1, val, xp):
    """Create a Dirac filter."""
    psi = xp.zeros((n0, n1), dtype=xp.float32)
    psi[0, 0] = val
    return psi


def create_gabor(n0, n1, level, sigmax, sigmay, angle, phase, lambda_, xp):
    """Create a Gabor filter."""
    psi = xp.zeros((n0, n1), dtype=xp.float32)

    theta = xp.radians(angle)
    offset_x = (n1 / 2.0) + 1.0
    offset_y = (n0 / 2.0) + 1.0
    phase = xp.radians(phase)
    nn = xp.pi / xp.sqrt(sigmax * sigmay)

    i = xp.arange(n1)
    j = xp.arange(n0)
    x, y = xp.meshgrid(i, j)

    x = offset_x - x
    y = offset_y - y
    x_theta = x * xp.cos(theta) + y * xp.sin(theta)
    y_theta = y * xp.cos(theta) - x * xp.sin(theta)

    val = xp.exp(
        -0.5 * ((x_theta / sigmax) ** 2 + (y_theta / sigmay) ** 2)
    ) * xp.cos((x_theta * lambda_ / sigmax) + phase)
    psi = level * val / nn

    return psi


def update_y(d1u0, d2u0, tmp1, tmp2, lambda1, lambda2, beta, xp):
    """
    Update the values of y1 and y2 based on the values of d1u0, d2u0, tmp1,
    tmp2, lambda1, lambda2, and beta.
    """
    t1 = d1u0 - (tmp1 + (lambda1 / beta))
    t2 = d2u0 - (tmp2 + (lambda2 / beta))
    ng = (
        xp.sqrt(xp.add(xp.square(t1), xp.square(t2))) + xp.finfo(float).eps
    )  # Adding a small epsilon to avoid division by zero
    mask = ng > 1.0 / beta

    coef = 1.0 - (1.0 / (beta * ng))
    y1 = d1u0 - mask * (t1 * coef)  # cp inf * 0 = 0
    y2 = d2u0 - mask * (t2 * coef)  # while np inf * 0 = nan

    return y1, y2


def setd1(n0, n1, xp):
    """Set the values of d1."""
    d1 = xp.zeros((n0, n1), dtype=xp.complex64)
    d1[0, 0] = 1
    d1[0, n1 - 1] = -1

    return d1


def setd2(n0, n1, xp):
    """Set the values of d2."""
    d2 = xp.zeros((n0, n1), dtype=xp.complex64)
    d2[0, 0] = 1
    d2[n0 - 1, 0] = -1

    return d2


def compute_vsnr(filters, u0, n0, n1, nit, beta, vmax, xp, cvg_threshold):
    """Calculate the corrected image using the 2D-VSNR algorithm"""
    gu = xp.zeros_like(u0, dtype=xp.float32)

    gu0 = u0.copy()
    gpsi = xp.zeros_like(u0, dtype=xp.float32)

    gu0 = xp.divide(gu0, vmax[:, None, None])

    # 2. Prepares filters
    gpsi = create_filters_batch(filters, gu0, n0, n1, xp)

    # 3. Denoises the image
    gu, cvg_criteria = vsnr_admm(
        gu0, gpsi, n0, n1, nit, beta, xp, cvg_threshold=cvg_threshold
    )

    # 4. Copies the result to u
    gu = xp.multiply(gu, vmax[:, None, None])
    u = xp.copy(gu)

    return u, cvg_criteria

def create_filters_batch(filters, gu0, n0, n1, xp):
    """
    Create filters based on filters parameters stored in the filters array.
    The calculated filters are stored in the gpsi array.
    """
    batch_size, _, _ = gu0.shape
    gpsi = xp.zeros((batch_size, n0, n1), dtype=xp.complex64)

    for i in range(batch_size):
        gpsi[i] = create_filters(filters, gu0[i], n0, n1, xp)

    return gpsi

def create_filters(filters, gu0, n0, n1, xp):
    """
    Create filters based on filters parameters stored in the filters array.
    The calculated filters are stored in the gpsi array.
    """
    # Computes the l2 norm of u0 on GPU
    fsum = 0
    norm = xp.linalg.norm(gu0)

    # Computes fd1
    fd1 = setd1(n0, n1, xp)  # // d1[0] = 1; d1[n1-1] = -1;
    fd1 = xp.fft.fft2(fd1)
    fd1 = xp.abs(fd1)  # // fd1 = |fd1|;

    # Computes fd2
    fd2 = setd2(n0, n1, xp)  # // d2[0] = 1; d2[(n0-1)*n1] = -1;
    fd2 = xp.fft.fft2(fd2)
    fd2 = xp.abs(fd2)

    # Computes PSI = sum_{i=1}^m |PSI_i|^2/alpha_i, where alpha_i is defined in the paper.

    for filt in filters:
        if filt["name"] == "Dirac":
            psitemp = create_dirac(n0, n1, 1, xp)
            eta = filt["noise_level"]
        elif filt["name"] == "Gabor":
            sigma = filt["sigma"]
            theta = filt["theta"]
            psitemp = create_gabor(n0, n1, 1, sigma[0], sigma[1], theta, 0, 0, xp)
            eta = filt["noise_level"]

        psitemp = xp.fft.fft2(psitemp)

        psitemp = xp.square(xp.abs(psitemp))  # // psitemp = |psitemp|^2;)

        ftmp = xp.multiply(xp.abs(fd1), xp.abs(psitemp))
        imax = xp.unravel_index(xp.argmax(xp.abs(ftmp)), ftmp.shape)  # find the index of the maximum element
        max1 = ftmp[imax]

        ftmp = xp.multiply(xp.abs(fd2), xp.abs(psitemp))
        imax = xp.unravel_index(xp.argmax(xp.abs(ftmp)), ftmp.shape)
        max2 = ftmp[imax]
        nmax = max(max1, max2)
        alpha = xp.sqrt(n0*n1) * (n0*n1)**2 * nmax / (norm * eta)

        fsum = update_psi(psitemp, fsum, alpha, xp)  # fsum += |psitemp|^2 / alpha_i;

    fsum = xp.sqrt(fsum)  # // fsum = sqrtf(fsum);
    gpsi = xp.fft.ifft2(fsum, norm="forward")

    return gpsi


def vsnr_admm(u0, psi, n0, n1, nit, beta, xp, cvg_threshold=0):
    """Denoise the image u0 using the VSNR algorithm."""
    batch_size = u0.shape[0]

    lambda1 = xp.zeros((batch_size, n0, n1), dtype=xp.float32)
    lambda2 = xp.zeros((batch_size, n0, n1), dtype=xp.float32)
    y1 = xp.zeros((batch_size, n0, n1), dtype=xp.float32)
    y2 = xp.zeros((batch_size, n0, n1), dtype=xp.float32)

    fu0 = xp.fft.fft2(u0)
    fpsi = xp.fft.fft2(psi)

    # Computes fd1
    fd1 = setd1(n0, n1, xp)
    fd1 = xp.fft.fft2(fd1)

    # Computes fd2
    fd2 = setd2(n0, n1, xp)
    fd2 = xp.fft.fft2(fd2)

    # Computes d1u0
    ftmp1 = xp.multiply(fd1, fu0)
    d1u0 = xp.real(xp.fft.ifft2(ftmp1, norm="forward"))
    d1u0 = xp.divide(d1u0, n0*n1)

    # Computes d2u0
    ftmp2 = xp.multiply(fd2, fu0)
    d2u0 = xp.real(xp.fft.ifft2(ftmp2, norm="forward"))
    d2u0 = xp.divide(d2u0, n0*n1)

    # Computes fphi1 and fphi2
    fphi1 = xp.multiply(fpsi, fd1)
    fphi2 = xp.multiply(fpsi, fd2)

    # Computes fphi
    fphi = compute_phi(fphi1, fphi2, beta, xp)

    cvg_criterion = 1000.0
    cvg_criteria = []
    i = 0
    fx_old = xp.zeros((batch_size, n0, n1), dtype=xp.float32)
    while i < nit and cvg_criterion > cvg_threshold:
        # First step, x update : (I+beta ATA)x = AT (-lambda+beta*ATy)
        ftmp1 = xp.subtract(xp.multiply(y1, beta), lambda1)
        ftmp2 = xp.subtract(xp.multiply(y2, beta), lambda2)
        ftmp1 = xp.fft.fft2(ftmp1)
        ftmp2 = xp.fft.fft2(ftmp2)

        # Computes w = conj(u) * v
        ftmp1 = xp.multiply(xp.conj(fphi1), ftmp1)
        ftmp2 = xp.multiply(xp.conj(fphi2), ftmp2)

        # fx = (tmp1 + tmp2) / fphi;
        fx = xp.divide(xp.add(ftmp1, ftmp2), fphi)

        # Second step y update : y = prox_{f1/beta}(Ax+lambda/beta)
        ftmp1 = xp.multiply(fphi1, fx)
        ftmp2 = xp.multiply(fphi2, fx)
        ftmp1 = xp.fft.ifft2(ftmp1, norm="forward").real
        ftmp2 = xp.fft.ifft2(ftmp2, norm="forward").real
        ftmp1 = xp.divide(ftmp1, n0*n1)
        ftmp2 = xp.divide(ftmp2, n0*n1)

        y1, y2 = update_y(d1u0, d2u0, ftmp1, ftmp2, lambda1, lambda2, beta, xp)

        # Third step lambda update : lambda = lambda + beta * (Ax - y)
        lambda1 = lambda1 + xp.multiply(beta, xp.subtract(ftmp1, y1))
        lambda2 = lambda2 + xp.multiply(beta, xp.subtract(ftmp2, y2))

        if i != 0:
            cvg_criterion = float(
                xp.max(xp.abs(fx - fx_old)) / xp.max(xp.abs(fx))
            )
            cvg_criteria.append(cvg_criterion)
        fx_old = fx
        i += 1

    # Last but not the least : u = u0 - (psi * x)
    ftmp1 = xp.multiply(fx, fpsi)
    u = xp.fft.ifft2(ftmp1, norm="forward").real
    u = xp.divide(u, n0*n1)
    u = xp.subtract(u0, u)

    return u, cvg_criteria

def vsnr2d_py(
    imgs,
    filters,
    maxit=20,
    xp=np,
    beta=10.0,
    norm=True,
    cvg_threshold=0,
    return_cvg=False,
):
    # If imgs is a 2D array, add an extra dimension to make it a 3D array with one image
    if len(imgs.shape) == 2:
        imgs = imgs[np.newaxis, :, :]

    if hasattr(imgs, 'get'):
        imgs = imgs.get()
    imgs = xp.asarray(imgs)

    batch_size, n0, n1 = imgs.shape
    dtype = imgs.dtype

    vmin, vmax = imgs.min(axis=(1,2)), imgs.max(axis=(1,2))

    if norm:
        imgs = (imgs - vmin[:, None, None]) / (vmax[:, None, None] - vmin[:, None, None])
        vmax_norm = xp.ones(batch_size)
    else:
        vmax_norm = vmax

    # u0 = imgs.reshape(batch_size, -1)
    # u = xp.zeros_like(u0)

    # calculation
    imgs_corr, cvg_criteria = compute_vsnr(filters, imgs, n0, n1, maxit, beta, vmax_norm, xp, cvg_threshold)

    if norm:
        imgs_corr = xp.clip(imgs_corr, 0, 1)
        imgs_corr = (imgs_corr - imgs_corr.min(axis=(1,2))[:, None, None]) / (
            imgs_corr.max(axis=(1,2))[:, None, None] - imgs_corr.min(axis=(1,2))[:, None, None]
        )
        imgs_corr = vmin[:, None, None] + imgs_corr * (vmax[:, None, None] - vmin[:, None, None])
    elif xp.issubdtype(dtype, xp.integer):
        # If dtype is integer, clip at its maximum value to avoid overflow
        imgs_corr = xp.clip(imgs_corr, 0, xp.iinfo(dtype).max)

    # cast to original dtype
    imgs_corr = imgs_corr.astype(dtype)

    # Handle cupy to numpy conversion if needed
    try:
        if xp == cp:
            imgs_corr = imgs_corr.get()
    except:
        pass

    # Remove the extra dimension from the output if the original input was a 2D array
    imgs_corr = xp.squeeze(imgs_corr)

    if return_cvg:
        return imgs_corr, cvg_criteria

    return imgs_corr

def vsnr2d(
    imgs,
    filters,
    maxit=20,
    algo='auto',
    beta=10.0,
    norm=True,
    cvg_threshold=0,
    return_cvg=False,
    verbose=False,
):
    r"""
    Calculate the corrected image using the 2D-VSNR algorithm

    Notes
    -----
    To ease code comparison with the original onde, most of the variable names
    have been kept as nearly as possible during the code transcription.
    Accordingly, PEP8 formatting compatibility may not be always respected.

    Parameters
    ----------
    imgs: numpy.ndarray((batch_size, n0, n1))
        The images to process
    filters: list of dicts
        Dictionaries that contains filters definition.
        Example For a 'Dirac' filter:
        - filter={'name':'Dirac', 'noise_level':10}
        Example For a 'Gabor' filter:
        - filter={'name':'Gabor', 'noise_level':100, 'sigma':(1000,0.1), 'theta':45}
        For further informations, see :
        https://www.math.univ-toulouse.fr/~weiss/Codes/VSNR/Documentation_VSNR_V2_Fiji.pdf

    maxit: int, optional
        Number of iterations in the denoising processing
    algo: str, optional
        The algorithm to use for the computation. Can be 'cupy', 'cuda', or 'numpy'.
        If 'auto', the best available algorithm will be used.
    beta: float
        The regularization parameter in the VSNR model. Controls the trade-off between the data
        fidelity term and the regularization term. A higher beta value gives more weight to the
        regularization term, which encourages sparser solutions but may result in more denoising
        (and potentially more distortion). A lower beta value gives more weight to the data fidelity
        term, which encourages solutions that are closer to the original image but may result in
        less denoising.
    norm: bool, optional
        If True, the image is normalized before processing and the output
        image is renormalized to the original range
    return_cvg: bool, optional
        If True, the function returns the convergence criterion for each iteration
    verbose: bool, optional
        If True, print the used algorithm

    Returns
    -------
    imgs_corr: numpy.ndarray((batch_size, n0, n1))
        The corrected images
    """

    if algo == 'auto':
        if CUPY_AVAILABLE:
            algo = 'cupy'
        elif CUDA_AVAILABLE:
            algo = 'cuda'
        else:
            algo = 'numpy'

    if verbose:
        print(f"Using {algo} algorithm")

    if algo == 'cupy':
        return vsnr2d_py(
            imgs,
            filters,
            maxit=maxit,
            xp=cp,
            beta=beta,
            norm=norm,
            cvg_threshold=cvg_threshold,
            return_cvg=return_cvg,
        )

    elif algo == 'cuda':
        return vsnr2d_cuda(
            imgs,
            filters,
            nite=maxit,
            beta=beta,
            nblocks="auto",
            norm=norm,
        )

    elif algo == 'numpy':
        return vsnr2d_py(
            imgs,
            filters,
            maxit=maxit,
            xp=np,
            beta=beta,
            norm=norm,
            cvg_threshold=cvg_threshold,
            return_cvg=return_cvg,
        )

    else:
        raise ValueError("algo must be 'cupy', 'numpy', or 'cuda'.")
