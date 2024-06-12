""" Pytest file to verify that the python code is equivalent to the cuda code """
import pathlib
import os
from ctypes import POINTER, c_int, c_float, CDLL

import cupy as cp
import numpy as np
from skimage import exposure, data

from src.pyvsnr import vsnr2d, vsnr2d_cuda
from src.pyvsnr.utils import stripes_addition

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

def test_cuda_equals_py():
    """ Test if the cuda code is equivalent to the cupy code """
    img = data.camera()

    maxit = 20
    filters = [{"name": "Dirac", "noise_level": 0.4}]

    img = stripes_addition(img, 0.2)

    img_corr_py = vsnr2d(img, filters, maxit=maxit, norm=False)
    img_corr_cuda = vsnr2d_cuda(img, filters, nite=maxit, norm=False)

    xp.testing.assert_allclose(img_corr_cuda, img_corr_py, atol=1e-5)

    img_corr_py_norm = vsnr2d(img, filters, maxit=maxit, norm=True)
    img_corr_cuda_norm = vsnr2d_cuda(img, filters, nite=maxit, norm=True)

    xp.testing.assert_allclose(img_corr_cuda_norm, img_corr_py_norm, atol=1e-5)

def test_data_min_max_preserved():
    """ Test if the min and max of the original image are preserved """
    img = data.camera()

    maxit = 20
    # filters = [{"name": "Dirac", "noise_level": 0.4}]
    filters = [{'name': 'Gabor', 'noise_level': 200, 'sigma': [2, 80], 'theta': 10}]

    img = stripes_addition(img, 0.2)

    img_corr = vsnr2d(img, filters, maxit=maxit, norm=False)
    img_corr_cuda = vsnr2d_cuda(img, filters, nite=maxit, norm=False)

    assert np.allclose(img_corr.min(), img_corr_cuda.min(), atol=1e-5)
    assert np.allclose(img_corr.max(), img_corr_cuda.max(), atol=1e-5)

    img_corr_norm = vsnr2d(img, filters, maxit=maxit, norm=True)
    img_corr_cuda_norm = vsnr2d_cuda(img, filters, nite=maxit, norm=True)

    assert np.allclose(img_corr_norm.min(), img_corr_cuda_norm.min(), atol=1e-5)
    assert np.allclose(img_corr_norm.max(), img_corr_cuda_norm.max(), atol=1e-5)

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