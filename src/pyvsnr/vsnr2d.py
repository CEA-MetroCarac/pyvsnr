"""
This module contains the 2D-VSNR algorithm implementation.
its a port of the original CUDA implementation to python using cupy.
It contains several improvements in terms of performance and memory usage.
"""
import numpy as np

def compute_phi(fphi1, fphi2, beta, xp):
    """ Compute the value of fphi based on the values of fphi1, fphi2, and beta. """
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
    """ Update the value of fsum based on the values of fpsitemp, fsum, and alpha. """
    # // fsum += |fpsitemp|^2 / alpha_i;
    fsum = xp.add(fsum, xp.divide(fpsitemp, alpha))
    return fsum


def create_dirac(n, val, xp):
    """ Create a Dirac filter. """
    psi = xp.zeros((n,), dtype=xp.float32)
    psi[0] = val
    return psi


def create_gabor(n0, n1, level, sigmax, sigmay, angle, phase, lambda_, xp):
    """ Create a Gabor filter. """
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

    return psi.ravel()


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


def setd1(n, n1, xp):
    """ Set the values of d1. """
    d1 = xp.zeros((n,), dtype=xp.complex64)
    d1[0] = 1
    d1[n1 - 1] = -1

    return d1


def setd2(n, n1, xp):
    """ Set the values of d2. """
    d2 = xp.zeros((n,), dtype=xp.complex64)
    d2[0] = 1
    d2[n - n1] = -1

    return d2


def compute_vsnr(filters, u0, n0, n1, nit, beta, vmax, xp, cvg_threshold):
    """ Calculate the corrected image using the 2D-VSNR algorithm """
    n = n0 * n1
    gu = xp.zeros((n,), dtype=xp.float32)

    gu0 = u0.copy()
    gpsi = xp.zeros((n,), dtype=xp.float32)

    gu0 = xp.divide(gu0, vmax)

    # 2. Prepares filters
    gpsi = create_filters(filters, gu0, n0, n1, xp)

    # 3. Denoises the image
    gu, cvg_criterias = vsnr_admm(gu0, gpsi, n0, n1, nit, beta, xp, cvg_threshold=cvg_threshold)

    # 4. Copies the result to u
    gu = xp.multiply(gu, vmax)
    u = xp.copy(gu)

    return u, cvg_criterias


def create_filters(filters, gu0, n0, n1, xp):
    """
    Create filters based on filters parameters stored in the filters array.
    The calculated filters are stored in the gpsi array.
    """
    # Computes the l2 norm of u0 on GPU
    fsum = 0
    norm = xp.linalg.norm(gu0)

    # Computes d1 and fd1
    n = n0 * n1
    # d1=xp.array([-3-10j,-2+2j],dtype=xp.complex64) #test
    d1 = setd1(n, n1, xp)  # // d1[0] = 1; d1[n1-1] = -1;
    fd1 = xp.fft.fft2(d1.reshape(n0, n1)).flatten()
    fd1 = xp.abs(fd1)  # // fd1 = |fd1|;

    # Computes d2 and fd2
    # d2=xp.array([5-1j,-4+2j],dtype=xp.complex64) #test
    d2 = setd2(n, n1, xp)  # // d2[0] = 1; d2[(n0-1)*n1] = -1;
    fd2 = xp.fft.fft2(d2.reshape(n0, n1)).flatten()
    fd2 = xp.abs(fd2)

    # Computes PSI = sum_{i=1}^m |PSI_i|^2/alpha_i, where alpha_i is defined in the paper.

    for filt in filters:
        if filt["name"] == "Dirac":
            psitemp = create_dirac(n, 1, xp)
            eta = filt["noise_level"]
        elif filt["name"] == "Gabor":
            sigma = filt["sigma"]
            theta = filt["theta"]
            psitemp = create_gabor(n0, n1, 1, sigma[0], sigma[1], theta, 0, 0, xp)
            eta = filt["noise_level"]

        fpsitemp = xp.fft.fft2(psitemp.reshape(n0, n1)).flatten()

        fpsitemp = xp.square(xp.abs(fpsitemp))  # // fpsitemp = |fpsitemp|^2;)

        ftmp = xp.multiply(xp.abs(fd1), xp.abs(fpsitemp))
        imax = xp.argmax(xp.abs(ftmp))  # find the index of the maximum element
        max1 = ftmp[imax]

        ftmp = xp.multiply(xp.abs(fd2), xp.abs(fpsitemp))
        imax = xp.argmax(xp.abs(ftmp))
        max2 = ftmp[imax]
        nmax = max(max1, max2)
        alpha = np.sqrt(n) * n**2 * nmax / (norm * eta)

        fsum = update_psi(fpsitemp, fsum, alpha, xp)  # fsum += |fpsitemp|^2 / alpha_i;

    fsum = xp.sqrt(fsum)  # // fsum = sqrtf(fsum);
    gpsi = xp.fft.ifft2(fsum.reshape(n0, n1), norm="forward").flatten()

    return gpsi


def vsnr_admm(u0, psi, n0, n1, nit, beta, xp, cvg_threshold=0):
    """ Denoise the image u0 using the VSNR algorithm. """
    n = n0 * n1
    # m=n0*(n1//2+1)

    lambda1 = xp.zeros((n,), dtype=xp.float32)
    lambda2 = xp.zeros((n,), dtype=xp.float32)
    y1 = xp.zeros((n,), dtype=xp.float32)
    y2 = xp.zeros((n,), dtype=xp.float32)
    tmp1 = xp.zeros((n,), dtype=xp.float32)
    tmp2 = xp.zeros((n,), dtype=xp.float32)

    fu0 = xp.fft.fft2(u0.reshape(n0, n1)).flatten()
    fpsi = xp.fft.fft2(psi.reshape(n0, n1)).flatten()

    # // Computes d1 and fd1
    d1 = setd1(n, n1, xp)  # // d1[0] = 1; d1[n1-1] = -1;
    fd1 = xp.fft.fft2(d1.reshape(n0, n1)).flatten()

    # // Computes d2 and fd2
    d2 = setd2(n, n1, xp)  # // d2[0] = 1; d2[(n0-1)*n1] = -1;
    fd2 = xp.fft.fft2(d2.reshape(n0, n1)).flatten()

    # // Computes d1u0
    ftmp1 = xp.multiply(fd1, fu0)
    d1u0 = xp.real(
        xp.fft.ifft2(ftmp1.reshape(n0, n1), norm="forward").flatten()
    )
    d1u0 = xp.divide(d1u0, n)  # d1u0=normalize_array(d1u0)

    # // Computes d2u0
    ftmp2 = xp.multiply(fd2, fu0)
    d2u0 = xp.real(
        xp.fft.ifft2(ftmp2.reshape(n0, n1), norm="forward").flatten()
    )
    d2u0 = xp.divide(d2u0, n)

    # // Computes fphi1 and fphi2
    fphi1 = xp.multiply(fpsi, fd1)
    fphi2 = xp.multiply(fpsi, fd2)

    # // Computes fphi
    fphi = compute_phi(fphi1, fphi2, beta, xp)

    cvg_criteria=1000.
    cvg_criterias=[]
    i=0
    fx_old=0
    while i<nit and cvg_criteria>cvg_threshold:
        #     // -------------------------------------------------------------
        #     // First step, x update : (I+beta ATA)x = AT (-lambda+beta*ATy)
        #     // -------------------------------------------------------------
        tmp1 = xp.subtract(xp.multiply(y1, beta), lambda1)
        tmp2 = xp.subtract(xp.multiply(y2, beta), lambda2)
        ftmp1 = xp.fft.fft2(tmp1.reshape(n0, n1)).flatten()
        ftmp2 = xp.fft.fft2(tmp2.reshape(n0, n1)).flatten()

        # Computes w = conj(u) * v
        ftmp1 = xp.multiply(xp.conj(fphi1), ftmp1)
        ftmp2 = xp.multiply(xp.conj(fphi2), ftmp2)

        # fx = (tmp1 + tmp2) / fphi;
        fx = xp.divide(xp.add(ftmp1, ftmp2), fphi)

        #     // --------------------------------------------------------
        #     // Second step y update : y = prox_{f1/beta}(Ax+lambda/beta)
        #     // --------------------------------------------------------

        ftmp1 = xp.multiply(fphi1, fx)
        ftmp2 = xp.multiply(fphi2, fx)
        tmp1 = xp.fft.ifft2(ftmp1.reshape(n0, n1), norm="forward").flatten().real
        tmp2 = xp.fft.ifft2(ftmp2.reshape(n0, n1), norm="forward").flatten().real
        tmp1 = xp.divide(tmp1, n)  # normalize_array(tmp1)
        tmp2 = xp.divide(tmp2, n)  # normalize_array(tmp2)

        y1, y2 = update_y(d1u0, d2u0, tmp1, tmp2, lambda1, lambda2, beta, xp)

        #     // --------------------------
        #     // Third step lambda update
        #     // --------------------------
        lambda1 = lambda1 + xp.multiply(beta, xp.subtract(tmp1, y1))
        lambda2 = lambda2 + xp.multiply(beta, xp.subtract(tmp2, y2))

        if i!=0:
            cvg_criteria = float(xp.max(xp.abs(fx - fx_old))/xp.max(xp.abs(fx)))
            cvg_criterias.append(cvg_criteria)
        fx_old=fx
        i+=1
    
    # // Last but not the least : u = u0 - (psi * x)
    ftmp1 = xp.multiply(fx, fpsi)
    u = xp.fft.ifft2(ftmp1.reshape(n0, n1), norm="forward").flatten().real
    u = xp.divide(u, n)
    u = xp.subtract(u0, u)

    return u, cvg_criterias


def vsnr2d(img, filters, maxit=20, xp=np, beta=10.0, norm=True, cvg_threshold=0, return_cvg=False):
    r"""
    Calculate the corrected image using the 2D-VSNR algorithm in libvsnr2d.dll

    Notes
    -----
    To ease code comparison with the original onde, most of the variable names
    have been kept as nearly as possible during the code transcription.
    Accordingly, PEP8 formatting compatibility may not be always respected.

    Parameters
    ----------
    img: numpy.ndarray((n0, n1))
        The image to process
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
        If True, the function returns the convergence criteria for each iteration

    Returns
    -------
    img_corr: numpy.ndarray((n0, n1))
        The corrected image
    """
    img = xp.asarray(img)
    n0, n1 = img.shape
    dtype = img.dtype

    vmin, vmax = img.min(), img.max()

    if norm:
        img = (img - vmin) / (vmax - vmin)

    u0 = img.flatten()
    u = xp.zeros_like(u0)

    # calculation
    u, cvg_criterias = compute_vsnr(filters, u0, n0, n1, maxit, beta, u0.max(), xp, cvg_threshold=cvg_threshold)

    # reshaping
    img_corr = xp.array(u).reshape(n0, n1)

    if norm:
        img_corr = xp.clip(img_corr, 0, 1)
        img_corr = (img_corr - img_corr.min()) / (img_corr.max() - img_corr.min())
        img_corr = vmin + img_corr * (vmax - vmin)

    # cast to original dtype
    img_corr = img_corr.astype(dtype)

    if return_cvg:
        return img_corr, cvg_criterias
    
    return img_corr
