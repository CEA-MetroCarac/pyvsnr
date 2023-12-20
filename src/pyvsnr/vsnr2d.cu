

// ---------------------------------------------------- //
//                                                      //
//             VSNR 2D CUDA DYNAMIC LIBRARY             //
//                                                      //
// ---------------------------------------------------- //
// Original Algorithm :                                 //
//   Pierre WEISS, Jerome FEHRENBACH                    //
// Developers :                                         //
//   Pierre WEISS, Mogan GAUTHIER, Jean EYMERIE         //
// ---------------------------------------------------- //

#ifdef __linux
#define _export_ extern "C"
#elif _WIN32
#define _export_ extern "C" __declspec(dllexport)
#endif


#include <math.h>
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define PI (3.141592653589793)

#define SQ(a) ((a)*(a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef cufftComplex CuC; // struct { float x, y }
typedef cufftReal    CuR; // float


// DEBUG
// -------------------------------------------------------------------------


// Disp lastCudaError in file
void __dispLastCudaError(FILE* file, const char* string)
{
    // -
    fprintf(file,"~ %s: %s\n", string, cudaGetErrorString(cudaGetLastError()));
}

// Disp a string relative to err from a cufft function in file
void __dispCufftError(FILE* file, const char* string, int err)
{
    switch (err) {
        case 0 :
            fprintf(file, "# %s: CUFFT_SUCCESS\n", string);
            break;
        case 1 :
            fprintf(file, "# %s: CUFFT_INVALID_PLAN\n", string);
            break;
        case 2 :
            fprintf(file, "# %s: CUFFT_ALLOC_FAILED\n", string);
            break;
        case 3 :
            fprintf(file, "# %s: CUFFT_INVALID_TYPE\n", string);
            break;
        case 4 :
            fprintf(file, "# %s: CUFFT_INVALID_VALUE\n", string);
            break;
        case 5 :
            fprintf(file, "# %s: CUFFT_INTERNAL_ERROR\n", string);
            break;
        case 6 :
            fprintf(file, "# %s: CUFFT_EXEC_FAILED\n", string);
            break;
        case 7 :
            fprintf(file, "# %s: CUFFT_SETUP_FAILED\n", string);
            break;
        case 8 :
            fprintf(file, "# %s: CUFFT_INVALID_SIZE\n", string);
            break;
        case 9 :
            fprintf(file, "# %s: CUFFT_UNALIGNED_DATA\n", string);
            break;
        case 10 :
            fprintf(file, "# %s: CUFFT_INCOMPLETE_PARAMETER_LIST\n", string);
            break;
        case 11 :
            fprintf(file, "# %s: CUFFT_INVALID_DEVICE\n", string);
            break;
        case 12 :
            fprintf(file, "# %s: CUFFT_PARSE_ERROR\n", string);
            break;
        case 13 :
            fprintf(file, "# %s: CUFFT_NO_WORKSPACE\n", string);
            break;
        case 14 :
            fprintf(file, "# %s: CUFFT_NOT_IMPLEMENTED\n", string);
            break;
        case 15 :
            fprintf(file, "# %s: CUFFT_LICENSE_ERROR\n", string);
            break;
        case 16 :
            fprintf(file, "# %s: CUFFT_NOT_SUPPORTED\n", string);
            break;
        default :
            fprintf(file, "# %s: UNKNOW_ERROR\n", string);
    }
}

// Displays a complex array as a vector
void disp_array2(FILE* file, float* u, int n)
{
    float* copy_u = (float*)malloc(n*sizeof(float));
    cudaMemcpy(copy_u, u, n*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0 ; i < n ; ++i)
        fprintf(file,"%1.4f    ", copy_u[i]);

    fprintf(file,"\n\n");  
    free(copy_u);
}

// Displays a complex array as a vector
void disp_carray2(FILE* file, CuC* u, int n)
{
    float2* copy_u = (float2*)malloc(n*sizeof(float2));
    cudaMemcpy(copy_u, u, n*sizeof(float2), cudaMemcpyDeviceToHost);

    for (int i = 0 ; i < n ; ++i)
        fprintf(file,"%1.4f + i %1.4f     ",copy_u[i].x,copy_u[i].y);

    fprintf(file,"\n\n");  
    free(copy_u);
}


// -------------------------------------------------------------------------


// Computes out = u1.*u2
__global__ void product_carray(CuC* u1, CuC* u2, CuC* out, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step) {
        out[i].x = (u1[i].x * u2[i].x) - (u1[i].y * u2[i].y);
        out[i].y = (u1[i].y * u2[i].x) + (u1[i].x * u2[i].y);
    }
}

// Normalize an array
__global__ void normalize(CuR* u, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step)
        u[i] = u[i] / (float)n;
}

// u = u*val;
__global__ void multiply(CuR* u, int n, float val)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step)
        u[i] = u[i] * val;
}

// u = u/val;
__global__ void divide(CuR* u, int n, float val)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step)
        u[i] = u[i] / val;
}

// adds two vectors w = u + v
__global__ void add(CuR* u, CuR* v, CuR* w, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step)
        w[i] = u[i] + v[i];
}

// substracts two vectors w = u - v
__global__ void substract(CuR* u, CuR* v, CuR* w, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step)
        w[i] = u[i] - v[i];
}

// Sets finite difference 1
__global__ void setd1(CuR* d1, int n, int n1)
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

// Sets finite difference 2
__global__ void setd2(CuR* d2, int n, int n1)
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

// Compute Phi
__global__ void compute_phi(CuC* fphi1, CuC* fphi2, CuC* fphi, float beta, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step) {
        fphi[i].x = 1 + beta*(fphi1[i].x*fphi1[i].x + fphi1[i].y*fphi1[i].y + fphi2[i].x*fphi2[i].x + fphi2[i].y*fphi2[i].y);
        fphi[i].y = 0;
    }
}

// Computes tmpi = -lambdai + beta * yi
__global__ void betay_m_lambda(CuR* lambda1, CuR* lambda2, CuR* y1, CuR* y2, CuR* tmp1, CuR* tmp2, float beta, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step) {
        tmp1[i] = (beta * y1[i]) - lambda1[i];
        tmp2[i] = (beta * y2[i]) - lambda2[i];
    }
}

// Computes w = conj(u) * vcr
__global__ void conju_x_v(CuC* u, CuC* v, CuC* w, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    float a1, a2, b1, b2;

    for ( ; i < n ; i += step) {
        a1 = u[i].x;
        b1 = u[i].y;
        a2 = v[i].x;
        b2 = v[i].y;
        w[i].x = (a1 * a2) + (b1 * b2);
        w[i].y = (b2 * a1) - (b1 * a2);
    }
}

// fx = (tmp1 + tmp2) / fphi;
__global__ void update_fx(CuC* ftmp1, CuC* ftmp2, CuC* fphi, CuC* fx, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step) {
        fx[i].x = (ftmp1[i].x + ftmp2[i].x) / fphi[i].x;
        fx[i].y = (ftmp1[i].y + ftmp2[i].y) / fphi[i].x;
    }
}

__global__ void update_y(CuR* d1u0, CuR* d2u0, CuR* tmp1, CuR* tmp2, CuR* lambda1, CuR* lambda2, CuR* y1, CuR* y2, float beta, int n)
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

// -
__global__ void update_lambda(CuR* lambda, CuR* tmp, CuR* y, float beta, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x ;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step)
        lambda[i] = lambda[i] + (beta * (tmp[i] - y[i]));
}

// Main function
void VSNR_ADMM_GPU(float* u0, float* psi, int n0, int n1, int nit, float beta, float* u, int dimGrid, int dimBlock)
{
    cufftHandle planR2C, planC2R;
    CuC *fpsi, *fu0;
    CuC *fd1, *fd2, *fphi1, *fphi2, *fphi, *ftmp1, *ftmp2, *fx;
    CuR *d1, *d2, *d1u0, *d2u0, *tmp1, *tmp2, *y1, *y2, *lambda1, *lambda2;

    int n = n0*n1;
    int m = n0*(n1/2+1);

    cudaMalloc((void**)&fpsi, m*sizeof(CuC));
    cudaMalloc((void**)&fu0,  m*sizeof(CuC));
    cudaMalloc((void**)&d1u0, n*sizeof(CuR));
    cudaMalloc((void**)&d2u0, n*sizeof(CuR));

    // Allocation for the main loop
    cudaMalloc((void**)&tmp1,  n*sizeof(CuR));
    cudaMalloc((void**)&tmp2,  n*sizeof(CuR));
    cudaMalloc((void**)&ftmp1, m*sizeof(CuC));
    cudaMalloc((void**)&ftmp2, m*sizeof(CuC));

    cufftPlan2d(&planR2C, n0, n1, CUFFT_R2C);
    cufftPlan2d(&planC2R, n0, n1, CUFFT_C2R);

    cufftExecR2C(planR2C, u0,  fu0); // fu0 = fftn(u0);
    cufftExecR2C(planR2C, psi, fpsi); // fpsi = fftn(psi);

    // Computes d1 and fd1
    cudaMalloc((void**)&d1,  n*sizeof(CuR));
    cudaMalloc((void**)&fd1, m*sizeof(CuC));
    setd1<<<dimGrid,dimBlock>>>(d1, n, n1);  // d1[0] = 1; d1[n1-1] = -1;
    cufftExecR2C(planR2C, d1, fd1); // fd1 = fftn(d1);
    cudaFree(d1);

    // Computes d2 and fd2
    cudaMalloc((void**)&d2,  n*sizeof(CuR));
    cudaMalloc((void**)&fd2, m*sizeof(CuC));
    setd2<<<dimGrid,dimBlock>>>(d2, n, n1);  // d2[0] = 1; d2[n-n1] = -1;
    cufftExecR2C(planR2C, d2, fd2); // fd2 = fftn(d2)
    cudaFree(d2);

    // Computes d1u0
    product_carray<<<dimGrid,dimBlock>>>(fd1, fu0, ftmp1, m);
    cufftExecC2R(planC2R, ftmp1, d1u0);  // d1u0 = ifftn(fd1.*fu0);
    normalize<<<dimGrid,dimBlock>>>(d1u0, n);

    // Computes d2u0
    product_carray<<<dimGrid,dimBlock>>>(fd2, fu0, ftmp2, m);
    cufftExecC2R(planC2R, ftmp2, d2u0);  // d2u0 = ifftn(fd2.*fu0);
    normalize<<<dimGrid,dimBlock>>>(d2u0, n);

    cudaFree(fu0); // This is unused until the end

    // Computes fphi1 and fphi2
    cudaMalloc((void**)&fphi1, m*sizeof(CuC));
    cudaMalloc((void**)&fphi2, m*sizeof(CuC));
    product_carray<<<dimGrid,dimBlock>>>(fd1, fpsi, fphi1, m); // fphi1 = fpsi.*fd1;
    product_carray<<<dimGrid,dimBlock>>>(fd2, fpsi, fphi2, m); // fphi2 = fpsi.*fd2;

    cudaFree(fd1);
    cudaFree(fd2);

    // Computes fphi
    cudaMalloc((void**)&fphi, m*sizeof(CuC));
    compute_phi<<<dimGrid,dimBlock>>>(fphi1, fphi2, fphi, beta, m);

    // Initialization
    cudaMalloc((void**)&y1, 	 n*sizeof(CuR));
    cudaMalloc((void**)&y2, 	 n*sizeof(CuR));
    cudaMalloc((void**)&lambda1, n*sizeof(CuR));
    cudaMalloc((void**)&lambda2, n*sizeof(CuR));
    cudaMalloc((void**)&fx, 	 m*sizeof(CuC));

    cudaMemset(y1, 		0, n*sizeof(CuR));
    cudaMemset(y2, 		0, n*sizeof(CuR));
    cudaMemset(lambda1, 0, n*sizeof(CuR));
    cudaMemset(lambda2, 0, n*sizeof(CuR));

    // Main algorithm
    for (int k = 0 ; k < nit ; ++k) {

        // -------------------------------------------------------------
        // First step, x update : (I+beta ATA)x = AT (-lambda+beta*ATy)
        // -------------------------------------------------------------
        // ftmp1 = conj(fphi1).*(fftn(-lambda1+beta*y1));
        // ftmp2 = conj(fphi2).*(fftn(-lambda2+beta*y2));
        betay_m_lambda<<<dimGrid,dimBlock>>>(lambda1, lambda2, y1, y2, tmp1, tmp2, beta, n);
        cufftExecR2C(planR2C, tmp1, ftmp1);
        cufftExecR2C(planR2C, tmp2, ftmp2);
        conju_x_v<<<dimGrid,dimBlock>>>(fphi1, ftmp1, ftmp1, m);
        conju_x_v<<<dimGrid,dimBlock>>>(fphi2, ftmp2, ftmp2, m);
        update_fx<<<dimGrid,dimBlock>>>(ftmp1, ftmp2, fphi, fx, m);

        // --------------------------------------------------------
        // Second step y update : y = prox_{f1/beta}(Ax+lambda/beta)
        // --------------------------------------------------------
        product_carray<<<dimGrid,dimBlock>>>(fphi1, fx, ftmp1, m);
        product_carray<<<dimGrid,dimBlock>>>(fphi2, fx, ftmp2, m);
        cufftExecC2R(planC2R, ftmp1, tmp1); // tmp1 = Ax1
        cufftExecC2R(planC2R, ftmp2, tmp2); // tmp2 = Ax2
        normalize<<<dimGrid,dimBlock>>>(tmp1, n);
        normalize<<<dimGrid,dimBlock>>>(tmp2, n);
        update_y<<<dimGrid,dimBlock>>>(d1u0, d2u0, tmp1, tmp2, lambda1, lambda2, y1, y2, beta, n);

        // --------------------------
        // Third step lambda update
        // --------------------------
        update_lambda<<<dimGrid,dimBlock>>>(lambda1, tmp1, y1, beta, n);
        update_lambda<<<dimGrid,dimBlock>>>(lambda2, tmp2, y2, beta, n);

    }

    // Last but not the least : u = u0 - (psi * x)
    product_carray<<<dimGrid,dimBlock>>>(fx, fpsi, ftmp1, m);
    cufftExecC2R(planC2R, ftmp1, u);
    normalize<<<dimGrid,dimBlock>>>(u, n);
    substract<<<dimGrid,dimBlock>>>(u0, u, u, n);

    // Free memory
    cudaFree(fpsi);
    cudaFree(fphi);
    cudaFree(fphi1);
    cudaFree(fphi2);
    cudaFree(ftmp1);
    cudaFree(ftmp2);
    cudaFree(fx);

    cudaFree(d1u0);
    cudaFree(d2u0);
    cudaFree(y1);
    cudaFree(y2);
    cudaFree(lambda1);
    cudaFree(lambda2);
    cudaFree(tmp1);
    cudaFree(tmp2);

    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
}

// Sets Gabor
__global__ void create_gabor(CuR* psi, int n0, int n1, float level, float sigmax, float sigmay, float angle, float phase, float lambda)
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

// Sets dirac
__global__ void create_dirac(CuR* psi, float val, int n)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < n ; i += step) {
        if (i == 0) psi[i] = val;
        else        psi[i] = 0.0;
    }
}

// Sets Psi = |Psi|^2
__global__ void compute_squared_norm(CuC* fpsi, int m)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < m ; i += step) {
        fpsi[i].x = SQ(fpsi[i].x) + SQ(fpsi[i].y);
        fpsi[i].y = 0.0;
    }
}

// Sets Psi = sqrtf(|Psi|^2)
__global__ void compute_norm(CuC* fpsi, int m)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < m ; i += step) {
        fpsi[i].x = sqrtf(SQ(fpsi[i].x) + SQ(fpsi[i].y));
        fpsi[i].y = 0.0;
    }
}

// Sets fsum = sqrtf(fsum)
__global__ void compute_sqrtf(CuC* fsum, int m)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < m ; i += step) {
        fsum[i].x = sqrtf(fsum[i].x);
        fsum[i].y = 0.0;
    }
}

// Sets ftmp = fpsi * fd
__global__ void compute_product(CuC* fpsi, CuC* fd, float* ftmp, int m)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for ( ; i < m ; i += step)
        ftmp[i] = fpsi[i].x * fd[i].x;
}

// Sets fsum += fpsitemp / alpha
__global__ void update_psi(CuC* fpsitemp, CuC* fsum, float alpha, int m)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for ( ; i < m ; i += step) {
        fsum[i].x += fpsitemp[i].x / alpha;
        // printf("%g",fsum);
    }
}

// This function creates the filters from a Java list of filters
void CREATE_FILTERS(float* psis, float* gu0, int length, float* gpsi, int n0, int n1, int dimGrid, int dimBlock)
{
    int i = 0;
    int n = n0*n1;
    int m = n0*(n1/2+1);
    cublasHandle_t handle;

    float eta, alpha, max1, max2, mmax, norm;
    float *psitemp, *ftmp;
    CuC *fpsitemp, *fsum, *fd1, *fd2;
    CuR *d1, *d2;
    cufftHandle planR2C, planC2R;
    int imax;

    cudaMalloc((void**)&psitemp,  n*sizeof(float));
    cudaMalloc((void**)&fpsitemp, m*sizeof(CuC));
    cudaMalloc((void**)&ftmp, 	  m*sizeof(float));
    cudaMalloc((void**)&fsum, 	  m*sizeof(CuC));
    cudaMalloc((void**)&d1, 	  n*sizeof(CuR));
    cudaMalloc((void**)&fd1,	  m*sizeof(CuC));
    cudaMalloc((void**)&d2, 	  n*sizeof(CuR));
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

        alpha = sqrtf((float)n) * SQ((float)n) * mmax / (norm * eta);
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

// -
_export_ int getMaxGrid()
{
    struct cudaDeviceProp properties;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&properties, device);
    return properties.maxGridSize[1];
}

// -
_export_ int getMaxBlocks()
{
    struct cudaDeviceProp properties;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&properties, device);
    return properties.maxThreadsDim[0];
}

// -
_export_ void VSNR_2D_FIJI_GPU(float* psis, int length, float* u0, int n0, int n1, int nit, float beta, float* u, int nBlocks, float max)
{
    int n = n0*n1;
    float *gu, *gu0, *gpsi;

    int dimBlock = MIN(nBlocks, getMaxBlocks());
    dimBlock = MAX(dimBlock, 1);
    int dimGrid = MIN(n/dimBlock, getMaxGrid());
    dimGrid = MAX(dimGrid, 1);

    // 1. Alloc memory
    cudaMalloc((void**)&gu,   n*sizeof(float));
    cudaMalloc((void**)&gpsi, n*sizeof(float));
    cudaMalloc((void**)&gu0,  n*sizeof(float));

    cudaMemcpy(gu0, u0, n*sizeof(float), cudaMemcpyHostToDevice);
    divide<<<dimGrid, dimBlock>>>(gu0, n, max);

    // 2. Prepares filters
    CREATE_FILTERS(psis, gu0, length, gpsi, n0, n1, dimGrid, dimBlock);

    // 3. Denoises the image
    VSNR_ADMM_GPU(gu0, gpsi, n0, n1, nit, beta, gu, dimGrid, dimBlock);

    // 4. Copies the result to u
    multiply<<<dimGrid, dimBlock>>>(gu, n, max);
    cudaMemcpy(u, gu, n*sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Frees memory
    cudaFree(gu);
    cudaFree(gu0);
    cudaFree(gpsi);
}

