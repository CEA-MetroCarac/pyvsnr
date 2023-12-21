""" Pytest file to verify that the python code is equivalent to the cuda code """
import pathlib
import os
from ctypes import POINTER, c_int, c_float, CDLL

import cupy as cp
import numpy as np
from skimage import exposure, data

from src.pyvsnr.vsnr2d import (
    vsnr_admm,
    create_filters,
    setd1,
    setd2,
    compute_phi,
    update_psi,
    update_y,
    create_dirac,
    create_gabor,
)

from src.pyvsnr import vsnr2d, vsnr2d_cuda

from tests.utils import stripes_addition

DIRNAME = pathlib.Path(__file__).parent
PRECOMPILED_PATH = DIRNAME.parent / "src" / "pyvsnr" / "precompiled"

xp = cp

cuda_code = """
#include <math.h>
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
extern "C" {


#define PI (3.141592653589793)
#define SQ(a) ((a)*(a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

//struct CuC {float x, y;};
typedef cufftComplex CuC;

__global__ void normalize(float* u, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step)
        u[i] = u[i] / (float)n;
}

__global__ void product_carray(CuC* u1, CuC* u2, CuC* out, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step) {
        out[i].x = (u1[i].x * u2[i].x) - (u1[i].y * u2[i].y);
        out[i].y = (u1[i].y * u2[i].x) + (u1[i].x * u2[i].y);
    }
}

__global__ void setd1(float* d1, int n, int n1)
{
    int i     = blockIdx.x * blockDim.x + threadIdx.x ;
    int step  = blockDim.x * gridDim.x;
    int id[2] = {0, n1-1};

    for ( ; i < n ; i += step) {
        if      (i == id[0]) d1[i] =  1;
        else if (i == id[1]) d1[i] = -1;
        else                 d1[i] =  0;
    }
}


__global__ void setd2(float* d2, int n, int n1)
{
    int i     = blockIdx.x * blockDim.x + threadIdx.x;
    int step  = blockDim.x * gridDim.x;
    int id[2] = {0, n-n1};

    for ( ; i < n ; i += step) {
        if      (i == id[0]) d2[i] =  1;
        else if (i == id[1]) d2[i] = -1;
        else                 d2[i] =  0;
    }
}

__global__ void compute_norm(CuC* fpsi, int m)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < m ; i += step) {
        fpsi[i].x = sqrtf(SQ(fpsi[i].x) + SQ(fpsi[i].y));
        fpsi[i].y = 0.0;
    }
}

__global__ void create_dirac(float* psi, float val, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step) {
        if (i == 0) psi[0] = val;
        else        psi[i] = 0.0;
    }
}

__global__ void create_gabor(float* psi, int n0, int n1, float level, float sigmax, float sigmay, float angle, float phase, float lambda)
{
    int n = n0*n1;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    float theta = angle * PI / 180.0;
    float offset_x = (n1 / 2.0) + 1.0;
    float offset_y = (n0 / 2.0) + 1.0;
    float x, y, x_theta, y_theta, val, nn;
    int i, j;

    phase = phase * PI / 180.0;
    nn    = PI / sqrtf(sigmax*sigmay);

    for ( ; k < n ; k += step) {
        i = k % n1;
        j = k / n1;
        x = offset_x - i;
        y = offset_y - j;
        x_theta = (x * cos(theta)) + (y * sin(theta));
        y_theta = (y * cos(theta)) - (x * sin(theta));
        val = exp(-0.5*(SQ(x_theta/sigmax)+SQ(y_theta/sigmay)))*cos((x_theta*lambda/sigmax)+phase);
        psi[k] = level * val / nn;
    }
}

__global__ void compute_squared_norm(CuC* fpsi, int m)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < m ; i += step) {
        fpsi[i].x = SQ(fpsi[i].x) + SQ(fpsi[i].y);
        fpsi[i].y = 0.0;
    }
}

__global__ void compute_product(CuC* fpsi, CuC* fd, float* ftmp, int m)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < m ; i += step)
        ftmp[i] = fpsi[i].x * fd[i].x;
}

__global__ void update_psi(CuC* fpsitemp, CuC* fsum, float alpha, int m)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < m ; i += step) {
        fsum[i].x += fpsitemp[i].x / alpha;
    }
}

__global__ void compute_sqrtf(CuC* fsum, int m)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < m ; i += step) {
        fsum[i].x = sqrtf(fsum[i].x);
        fsum[i].y = 0.0;
    }
}

__global__ void update_y(float* d1u0, float* d2u0, float* tmp1, float* tmp2, float* lambda1, float* lambda2, float* y1, float* y2, float beta, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    float ng, t1, t2;

    for ( ; i < n ; i += step) {
        t1 = d1u0[i] - (tmp1[i] + (lambda1[i] / beta));
        t2 = d2u0[i] - (tmp2[i] + (lambda2[i] / beta));
        ng = sqrtf((t1 * t1) + (t2 * t2));

        if (ng > 1.0 / beta) {
            y1[i] = d1u0[i] - t1 * (1.0 - (1.0 / (beta * ng)));
            y2[i] = d2u0[i] - t2 * (1.0 - (1.0 / (beta * ng)));
        } else {
            y1[i] = d1u0[i];
            y2[i] = d2u0[i];
        }
    }
}

__global__ void compute_phi(CuC* fphi1, CuC* fphi2, CuC* fphi, float beta, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step) {
        fphi[i].x = 1 + beta*(fphi1[i].x*fphi1[i].x + fphi1[i].y*fphi1[i].y + fphi2[i].x*fphi2[i].x + fphi2[i].y*fphi2[i].y);
        fphi[i].y = 0;
    }
}

__global__ void betay_m_lambda(float* lambda1, float* lambda2, float* y1, float* y2, float* tmp1, float* tmp2, float beta, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step) {
        tmp1[i] = (beta * y1[i]) - lambda1[i];
        tmp2[i] = (beta * y2[i]) - lambda2[i];
    }
}

void CREATE_FILTERS(float* psis, float* gu0, int length, float* gpsi, int n0, int n1, int dimGrid, int dimBlock)
{
    int i = 0;
    int n = n0*n1;
    int m = n0*(n1/2+1);
    cublasHandle_t handle;

    float eta, alpha, max1, max2, mmax, norm;
    float *psitemp, *ftmp;
    CuC *fpsitemp, *fsum, *fd1, *fd2;
    float *d1, *d2;
    cufftHandle planR2C, planC2R;
    int imax;

    cudaMalloc((void**)&psitemp,  n*sizeof(float));
    cudaMalloc((void**)&fpsitemp, m*sizeof(CuC));
    cudaMalloc((void**)&ftmp, 	  m*sizeof(float));
    cudaMalloc((void**)&fsum, 	  m*sizeof(CuC));
    cudaMalloc((void**)&d1, 	  n*sizeof(float));
    cudaMalloc((void**)&fd1,	  m*sizeof(CuC));
    cudaMalloc((void**)&d2, 	  n*sizeof(float));
    cudaMalloc((void**)&fd2,	  m*sizeof(CuC));

    cudaMemset(fsum, 0, m*sizeof(CuC));

    cublasCreate(&handle);

    cufftPlan2d(&planR2C, n0, n1, CUFFT_R2C);
    cufftPlan2d(&planC2R, n0, n1, CUFFT_C2R);

    // Computes the l2 norm of u0 on GPU
    cublasSnrm2(handle, n, gu0, 1, &norm);

    // Computes d1 and fd1
    setd1<<<dimGrid,dimBlock>>>(d1, n, n1); // d1[0] = 1; d1[n1-1] = -1;
    cufftExecR2C(planR2C, d1, fd1);
    compute_norm<<<dimGrid,dimBlock>>>(fd1, m); // fd1 = |fd1|;
    cudaFree(d1);

    // Computes d2 and fd2
    setd2<<<dimGrid,dimBlock>>>(d2, n, n1); // d2[0] = 1; d2[(n0-1)*n1] = -1;
    cufftExecR2C(planR2C, d2, fd2);
    compute_norm<<<dimGrid,dimBlock>>>(fd2, m); // fd2 = |fd2|;
    cudaFree(d2);

    // Computes PSI = sum_{i=1}^m |PSI_i|^2/alpha_i, where alpha_i is defined in the paper.
    while (i < length) {

        if (psis[i] == 0) {
            create_dirac<<<dimGrid,dimBlock>>>(psitemp, 1, n);
            eta = psis[i+1];
            i += 2;
        } else if (psis[i] == 1) {
            // 1 : amplitude, 2 : gammaX, 3 : gammaY, 4 : angle, 5 : phase_psi, 6 :frequency
            create_gabor<<<dimGrid,dimBlock>>>(psitemp, n0, n1, 1, psis[i+2], psis[i+3], psis[i+4], 0, 0);
            eta = psis[i+1];
            i += 5;
        }

        cufftExecR2C(planR2C, psitemp, fpsitemp);

        compute_squared_norm<<<dimGrid,dimBlock>>>(fpsitemp, m); // fpsitemp = |fpsitemp|^2;

        compute_product<<<dimGrid,dimBlock>>>(fpsitemp, fd1, ftmp, m); // ftmp = |fd1|*|fpsitemp|;
        cublasIsamax(handle, m, ftmp, 1, &imax);
        cudaMemcpy(&max1, &ftmp[imax-1], sizeof(float), cudaMemcpyDeviceToHost); // max1 = ftmp[imax];

        compute_product<<<dimGrid,dimBlock>>>(fpsitemp, fd2, ftmp, m); // ftmp = |fd2|*|fpsitemp|;
        cublasIsamax(handle, m, ftmp, 1, &imax);
        cudaMemcpy(&max2, &ftmp[imax-1], sizeof(float), cudaMemcpyDeviceToHost); // max2 = ftmp[imax];

        mmax = MAX(max1, max2);

        alpha = sqrtf((float)n) * SQ((float)n) * mmax / (norm * eta); // eta = noise level

        update_psi<<<dimGrid,dimBlock>>>(fpsitemp, fsum, alpha, m); // fsum += |fpsitemp|^2 / alpha_i;

    }

    compute_sqrtf<<<dimGrid,dimBlock>>>(fsum, m); // fsum = sqrtf(fsum);

    cufftExecC2R(planC2R, fsum, gpsi);

    cudaFree(psitemp);
    cudaFree(fpsitemp);
    cudaFree(ftmp);
    cudaFree(fsum);
    cudaFree(fd1);
    cudaFree(fd2);
    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
    cublasDestroy(handle);
}

}
"""

mod = cp.RawModule(
    code=cuda_code,
    backend="nvcc",
    options=(
        "-lcublas",
        "-lcufft",
    ),
)


def get_dll():
    """ Load the dedicated .dll library """
    try:
        if os.name == "nt":
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
        raise OSError(
            "Problem loading the compiled library from "
            f"{PRECOMPILED_PATH}, please try recompiling "
            "(see readme)"
        ) from err


def get_vsnr2d():
    """ Load the 'cuda' function from the dedicated .dll library """
    dll = get_dll()
    func = dll.VSNR_2D_FIJI_GPU
    func.argtypes = [
        POINTER(c_float),
        c_int,
        POINTER(c_float),
        c_int,
        c_int,
        c_int,
        c_float,
        POINTER(c_float),
        c_int,
        c_float,
    ]
    return func


def get_nblocks():
    """ Get the number of maximum threads per block library """
    dll = get_dll()
    return dll.getMaxBlocks()

def cuda_algo():
    """ Test the cuda algorithm """
    # image loading and intensity rescaling
    img0 = data.camera()

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img0, cmap='gray')
    # plt.title('original')

    per2, per98 = xp.percentile(img0, (2, 98))
    img0 = exposure.rescale_intensity(img0, in_range=(per2, per98))

    filter_0 = {"name": "Dirac", "noise_level": 10}
    vsnr_kwargs = {
        "algo": "cupy",
        "maxit": 5,
        "cvg_threshold": 1e-4,
        "nbfilters": 1,
        "filter_0": filter_0,
    }

    filters = [filter_0]
    img_corr = vsnr2d_cuda(img0, filters, nite=vsnr_kwargs["maxit"])

    return img_corr


def test_multiply():
    """ Test the multiply functions """
    multiply_kernel = mod.get_function("product_carray")

    a_kernel = (
        cp.random.rand(
            10000,
        )
        + 1j
        * cp.random.rand(
            10000,
        )
    ).astype(cp.complex64)
    b_kernel = (
        cp.random.rand(
            10000,
        )
        + 1j
        * cp.random.rand(
            10000,
        )
    ).astype(cp.complex64)
    res_kernel = cp.zeros_like(a_kernel)

    a = a_kernel.copy()
    b = b_kernel.copy()
    res = res_kernel.copy()

    m = 100 * 100
    blocksize = 256
    gridsize = (m + blocksize - 1) // blocksize

    multiply_kernel(
        (gridsize, 1), (blocksize, 1), (a_kernel, b_kernel, res_kernel, m)
    )
    res = cp.multiply(a, b)

    assert cp.allclose(res, res_kernel, atol=1e-7)


def test_setd1():
    """ Test the setd1 functions """
    # setd1_kernel = cp.RawKernel(cuda_code, 'setd1')
    setd1_kernel = mod.get_function("setd1")
    n1 = 100
    d1_kernel = cp.random.rand(100, 100, dtype=xp.float32)  # setd1 from CUDA
    d1_kernel = d1_kernel.flatten()
    d1 = d1_kernel.copy()
    setd1_kernel((10, 10), (32, 32), (d1_kernel, 100 * 100, n1))
    d1 = setd1(n1**2, n1, xp)
    assert xp.allclose(d1, d1_kernel)


def test_setd2():
    """ Test the setd2 functions """
    # setd2_kernel = cp.RawKernel(cuda_code, 'setd2')
    setd2_kernel = mod.get_function("setd2")
    n1 = 100
    d2_kernel = cp.random.rand(100, 100, dtype=xp.float32)  # setd2 from CUDA
    d2_kernel = d2_kernel.flatten()
    d2 = d2_kernel.copy()
    setd2_kernel((10, 10), (32, 32), (d2_kernel, 100 * 100, n1))
    d2 = setd2(n1**2, n1, xp)
    xp.testing.assert_allclose(d2, d2_kernel)


def test_compute_norm():
    """ Test the compute_norm functions """
    # compute_norm_kernel = cp.RawKernel(cuda_code, 'compute_norm')
    compute_norm_kernel = mod.get_function("compute_norm")
    fpsi_kernel = xp.random.rand(100, 100, dtype=xp.float64).astype(
        xp.complex64
    )
    fpsi_kernel = fpsi_kernel.flatten()
    fpsi = fpsi_kernel.copy()
    m = 100
    compute_norm_kernel((10, 10), (32, 32), (fpsi_kernel, m))
    fpsi = xp.absolute(fpsi)
    xp.testing.assert_allclose(fpsi, fpsi_kernel)


def test_compute_product():
    """ Test the compute_product functions """
    # compute_product_kernel = cp.RawKernel(cuda_code, 'compute_product')
    compute_product_kernel = mod.get_function("compute_product")
    fpsi_kernel = xp.random.rand(100, 100, dtype=xp.float32).astype(
        xp.complex64
    )
    fpsi_kernel = fpsi_kernel.flatten()
    fpsi = fpsi_kernel.copy()
    fd_kernel = xp.random.rand(100, 100, dtype=xp.float32).astype(xp.complex64)
    fd_kernel = fd_kernel.flatten()
    fd = fd_kernel.copy()
    ftmp_kernel = xp.random.rand(100, 100, dtype=xp.float32)
    ftmp_kernel = ftmp_kernel.flatten()
    ftmp = ftmp_kernel.copy()
    m = 100 * 100
    compute_product_kernel(
        (256, 1), (40, 1), (fpsi_kernel, fd_kernel, ftmp_kernel, m)
    )
    ftmp = xp.multiply(fpsi, fd)
    xp.testing.assert_allclose(ftmp, ftmp_kernel)

def test_compute_sqrtf():
    """ Test the compute_sqrtf functions """
    # compute_sqrtf_kernel = cp.RawKernel(cuda_code, 'compute_sqrtf')
    compute_sqrtf_kernel = mod.get_function("compute_sqrtf")
    fsum_kernel = xp.random.rand(100, 100, dtype=xp.float32).astype(
        xp.complex64
    )
    fsum_kernel = fsum_kernel.flatten()
    fsum = fsum_kernel.copy()
    m = 100 * 100
    compute_sqrtf_kernel((256, 1), (40, 1), (fsum_kernel, m))
    fsum = xp.sqrt(fsum)
    xp.testing.assert_allclose(fsum, fsum_kernel)


def test_compute_phi():
    """ Test the compute_phi functions """
    # compute_phi_kernel = cp.RawKernel(cuda_code, 'compute_phi')
    compute_phi_kernel = mod.get_function("compute_phi")
    fphi_kernel = xp.random.rand(100, 100, dtype=xp.float32).astype(
        xp.complex64
    )
    fphi_kernel = fphi_kernel.flatten()
    fphi = fphi_kernel.copy()
    fphi1_kernel = xp.random.rand(100, 100, dtype=xp.float32).astype(
        xp.complex64
    )
    fphi1_kernel = fphi1_kernel.flatten()
    fphi1 = fphi1_kernel.copy()
    fphi2_kernel = xp.random.rand(100, 100, dtype=xp.float32).astype(
        xp.complex64
    )
    fphi2_kernel = fphi2_kernel.flatten()
    fphi2 = fphi2_kernel.copy()
    beta = xp.float32(1.5)
    n = 100 * 100

    compute_phi_kernel(
        (256, 1), (40, 1), (fphi1_kernel, fphi2_kernel, fphi_kernel, beta, n)
    )
    fphi = compute_phi(fphi1, fphi2, beta, xp)
    xp.testing.assert_allclose(
        fphi, fphi_kernel, atol=1e-7
    )  # ?? fonctionnait sans préciser la tolerance avec assert np.allclose

def test_update_psi():
    """ Test the update_psi functions """
    # update_psi_kernel = cp.RawKernel(cuda_code, 'update_psi')
    update_psi_kernel = mod.get_function("update_psi")
    fpsitemp_kernel = xp.random.rand(100, 100, dtype=xp.float32).astype(
        xp.complex64
    )
    fpsitemp_kernel = fpsitemp_kernel.flatten()
    fpsitemp = fpsitemp_kernel.copy()
    fsum_kernel = xp.random.rand(100, 100, dtype=xp.float32).astype(
        xp.complex64
    )
    fsum_kernel = fsum_kernel.flatten()
    fsum = fsum_kernel.copy()
    alpha = xp.float32(2.3)
    m = 100 * 100
    update_psi_kernel(
        (256, 1), (40, 1), (fpsitemp_kernel, fsum_kernel, alpha, m)
    )
    fsum = update_psi(fpsitemp, fsum, 2.3, xp)
    xp.testing.assert_allclose(
        fsum, fsum_kernel, atol=1e-7
    )  # ?? fonctionnait sans préciser la tolerance avec assert np.allclose


def test_update_y():
    """ Test the update_y functions """
    # update_y_kernel = cp.RawKernel(cuda_code, 'update_y')
    update_y_kernel = mod.get_function("update_y")
    d1u0_kernel = xp.zeros((10000,), dtype=xp.float32)
    d1u0 = d1u0_kernel.copy()
    d2u0_kernel = xp.zeros((10000,), dtype=xp.float32)
    d2u0 = d2u0_kernel.copy()
    tmp1_kernel = xp.zeros((10000,), dtype=xp.float32)
    tmp1 = tmp1_kernel.copy()
    tmp2_kernel = xp.zeros((10000,), dtype=xp.float32)
    tmp2 = tmp2_kernel.copy()
    lambda1_kernel = xp.zeros((10000,), dtype=xp.float32)
    lambda1 = lambda1_kernel.copy()
    lambda2_kernel = xp.zeros((10000,), dtype=xp.float32)
    lambda2 = lambda2_kernel.copy()
    y1_kernel = xp.zeros((10000,), dtype=xp.float32)
    y1 = y1_kernel.copy()
    y2_kernel = xp.zeros((10000,), dtype=xp.float32)
    y2 = y2_kernel.copy()
    beta = xp.float32(1.5)
    n1 = 100 * 100

    update_y_kernel(
        (256, 1),
        (40, 1),
        (
            d1u0_kernel,
            d2u0_kernel,
            tmp1_kernel,
            tmp2_kernel,
            lambda1_kernel,
            lambda2_kernel,
            y1_kernel,
            y2_kernel,
            beta,
            n1,
        ),
    )
    update_y(d1u0, d2u0, tmp1, tmp2, lambda1, lambda2, beta, xp)
    xp.testing.assert_allclose(
        y1, y1_kernel, atol=1e-7
    )  # ?? fonctionnait sans préciser la tolerance avec assert np.allclose
    xp.testing.assert_allclose(y2, y2_kernel, atol=1e-7)


def test_create_dirac():
    """ Test the create_dirac functions """
    # create_dirac_kernel = cp.RawKernel(cuda_code, 'create_dirac')
    create_dirac_kernel = mod.get_function("create_dirac")
    psi_kernel = xp.zeros((10000,), dtype=xp.float32)
    psi = psi_kernel.copy()
    val = xp.float32(1.5)
    n = 100 * 100
    create_dirac_kernel((10, 10), (32, 32), (psi_kernel, val, n))
    psi = create_dirac(n, val, xp)
    xp.testing.assert_allclose(psi, psi_kernel)


def test_create_gabor():
    """ Test the create_gabor functions """
    # create_gabor_kernel = cp.RawKernel(cuda_code, 'create_gabor')
    create_gabor_kernel = mod.get_function("create_gabor")
    psi_kernel = xp.zeros((10000,), dtype=xp.float32)
    psi = psi_kernel.copy()
    n0 = 100
    n1 = 100
    level = xp.float32(5.0)
    sigmax = xp.float32(3.0)
    sigmay = xp.float32(40.0)
    angle = xp.float32(8.0)
    phase = xp.float32(0.0)
    lambda_ = xp.float32(0.0)
    create_gabor_kernel(
        (10, 10),
        (32, 32),
        (psi_kernel, n0, n1, level, sigmax, sigmay, angle, phase, lambda_),
    )
    psi = create_gabor(100, 100, 5.0, 3.0, 40.0, 8.0, 0.0, 0.0, xp)
    assert xp.allclose(psi, psi_kernel, atol=1e-4)

def test_create_filters():
    """ Test the create_filters functions """

    u0 = np.loadtxt(DIRNAME / "TEST_VSNR_ADMM/camera.txt", dtype=np.float32)
    gpsi_kernel = np.loadtxt(
        DIRNAME / "TEST_CREATE_FILTERS/out.txt", dtype=np.float32
    )
    u0 = u0.reshape(512, 512)

    # CREATE_FILTERS
    gu0 = u0.copy()
    gpsi = np.zeros((512 * 512,), dtype=np.float32)
    gpsi = create_filters(
        [{"name": "Dirac", "noise_level": 10}], gu0, 512, 512, xp
    )

    assert np.allclose(gpsi, gpsi_kernel)


def test_vsnr_admm():
    """ Test the vsnr_admm functions """
    gu0 = data.camera().flatten()

    if xp == cp:
        # convert numpy array to cupy array
        gu0 = cp.asarray(gu0)

    # write to a file camera.txt
    with open(
        DIRNAME / "TEST_VSNR_ADMM/camera.txt", "w", encoding="utf-8"
    ) as f:
        for item in gu0:
            f.write("%s\n" % item)

    n0, n1 = 512, 512
    nit = 2
    beta = 0.1
    gpsi = xp.loadtxt(
        DIRNAME / "TEST_CREATE_FILTERS/out.txt", dtype=xp.float32
    )
    gu = xp.zeros((512, 512), dtype=xp.float32)

    # gu0=gu0.reshape(64,64)
    # gpsi=gpsi.reshape(64,64)
    # gu=gu.reshape(64,64)

    gu, cvg_dummy = vsnr_admm(gu0, gpsi, n0, n1, nit, beta, xp)

    gu_kernel = xp.loadtxt(
        DIRNAME / "TEST_VSNR_ADMM/out.txt", dtype=xp.float32
    )

    gu = gu.reshape(512, 512)
    gu_kernel = gu_kernel.reshape(512, 512)

    xp.testing.assert_allclose(gu, gu_kernel, atol=1e-3)


def test_cuda_equals_cupy_numpy():
    """ Test if the cuda code is equivalent to the cupy code """
    img = data.camera()

    maxit = 20
    filters = [{"name": "Dirac", "noise_level": 0.4}]

    img = stripes_addition(img, 0.2)

    img_corr_py = vsnr2d(img, filters, maxit=maxit)
    img_corr_cuda = vsnr2d_cuda(img, filters, nite=maxit)

    xp.testing.assert_allclose(img_corr_cuda, img_corr_py, atol=1e-3)

def test_data_min_max_preserved():
    """ Test if the min and max of the original image are preserved """
    img = data.camera()

    maxit = 20
    filters = [{"name": "Dirac", "noise_level": 0.4}]

    img = stripes_addition(img, 0.2)

    img_corr = vsnr2d(img, filters, maxit=maxit)
    img_corr_cuda = vsnr2d_cuda(img, filters, nite=maxit)

    assert img_corr.min() == img_corr_cuda.min() == img.min()
    assert img_corr.max() == img_corr_cuda.max() == img.max()

def test_original_img_preserved():
    """ Test if the original image is preserved """
    img = data.camera()
    maxit=20
    filters = [{"name": "Dirac", "noise_level": 0.4}]

    img = stripes_addition(img, 0.2)
    img_copy = img.copy()

    vsnr2d(img, filters, maxit=maxit)
    vsnr2d_cuda(img, filters, nite=maxit)

    assert xp.array_equal(img, img_copy)