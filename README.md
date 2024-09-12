<!--
TODO Pyvsnr 2.2.0:
    TODO NumPy 2.0.0 support when CuPy (v14 will) and pyfftw support it
        TODO int to float32 still needed for CuPy ?
    TODO optionally take file path as input, load and process concurrently ?
        TODO Load Directly onto GPU for CuPy using nvimagecodec. https://github.com/NVIDIA/nvImageCodec/issues/5 fixed ?
        TODO auto set batch size based on user input MAX_GPU_MEM and image size (https://docs.cupy.dev/en/stable/user_guide/memory.html#limiting-gpu-memory-usage)
        TODO Use threading to overlap datatransfer with vsnr2d calls to speed up process
    TODO Speed up using less kernels thanks to User Kernel, currently 45 CUDA kernels     
-->
# pyvsnr

![](https://raw.githubusercontent.com/CEA-MetroCarac/pyvsnr/main/tests/images/fib_sem_corr.png)

## Description

`pyvsnr` is a Python library for computing the Variational Signal-to-Noise Ratio (VSNR) in 2D images. It provides both CPU (NumPy) and GPU (CuPy or CUDA) implementations for efficient computation.

## Installation

### From PyPI (recommended)

```bash
pip install pyvsnr
```
### From GitHub (latest version)

```bash
pip install git+https://github.com/CEA-MetroCarac/pyvsnr
```

In case of problem during CUDA execution (typically OSError or 'access memory error'),
it may be necessary to **recompile** the shared library from source ([see below](#shared-library-re-compilation)).

## Requirements

For **CPU** execution, the vsnr algorithm requires only the
[NumPy](https://numpy.org/) package, with **matplotlib** and **scikit-image**
packages for examples and tests running.

For **GPU** execution, a working CUDA installation is necessary wether you want to use the CUDA version directly or the cupy one. We recommend using the [CuPy](https://cupy.dev) library, which was 10x faster in the tests. Please ensure that you install the correct version of CuPy that corresponds to your CUDA version (for example, if you're using CUDA 12.x, you should install `cupy-cuda12x`). See the [installation instructions](https://docs.cupy.dev/en/stable/install.html) for more details.

- numpy
- pyfftw
- cupy (**optional** but strongly recommended, allows for GPU computation)
- matplotlib, scikit-image (**optional**, for examples and tests execution only)

<!-- The Jupyter notebook requires all the above packages, plus jupyter. It is only used for examples and tests. -->

## Usage

`pyvsnr` allows you to process either a single image (2D array) or a batch of images (3D array). Here are some basic examples:

### Single Image

```python
import numpy as np
from pyvsnr import vsnr2d

img = np.random.random((100, 100))  # Input image
filters = [{'name':'Dirac', 'noise_level':0.35}]  # List of filters

img_corr_py = vsnr2d(img, filters) # output is a 2D array (100, 100)
```

### Batch of Images

```python
import numpy as np
from pyvsnr import vsnr2d

imgs = np.random.random((10, 100, 100))  # Batch of 10 images
filters = [{'name':'Dirac', 'noise_level':0.35}]  # List of filters

imgs_corr_py = vsnr2d(imgs, filters) # Output is a 3D array (10, 100, 100)
```

By default, the `vsnr2d` function uses auto detection to determine whether to use the CPU or GPU implementation. If CuPy is installed, it will use it, otherwise it will try to use the CUDA implementation. If neither are available, it will fall back to the NumPy CPU implementation.

If you want to choose the algorithm to use, you can do so using the `algo` argument. The possible values are `'auto'`, `'cupy'`, `'cuda'`, and `'numpy'`. The default value is `'auto'`.

Please note that the CUDA implementation doesn't support batch processing, you can still pass 3D arrays but it'll process them one at a time.

```python
# Compute VSNR using CUDA
from pyvsnr import vsnr2d
img_corr_cuda = vsnr2d(img, filters, algo='cuda') # OK
imgs_corr_cuda = vsnr2d(imgs, filters, algo='cuda') # Will process images one at a time
```

The `cvg_threshold` parameter is a stopping criterion based on the relative-change of the denoised image between successive iterations.
The default value is 0. It stops the algorithm when relative-change between iterations falls below this threshold. This means that the result has nearly stabilized, indicating that further iterations are unlikely to significantly improve the denoising result. Adjusting this parameter impact the speed and the quality of the results. Here is the formula used to compute the convergence criterion at iteration k:

$$ CV_{k} = \frac{\max|\Delta u_k - \Delta u_{k-1}|}{\max|\Delta u_k|} $$

Where $\Delta u_{k}$ is the estimated noise of the image at iteration k, such as

$$ u_k = u_0 - \Delta u_k $$

Use `return_cvg=True` to visualize the algorithm's convergence. You can also set the maximum number of iterations with the `maxit` parameter, the default value is 20.

NOTE: For now the `return_cvg` parameter is only available for Cupy and NumPy implementation.

```python
n = 30   # Maximum number of iterations
img_corr, cvg = vsnr2d(img, filters, maxit=n, algo="numpy", return_cvg=True)

# Plot the convergence graph
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.semilogy(cvg)
plt.xlabel('N Iteration')
plt.ylabel('Cvg Criterion')
plt.plot(cvg)
plt.show()
```
![](https://raw.githubusercontent.com/CEA-MetroCarac/pyvsnr/main/tests/images/cvg.png)

Here for example, we could set the `cvg_threshold` to 1e-2. This would stop the algorithm after around 20 iterations, instead of 30.


## Examples

The `tests` directory contains a Jupyter notebook with examples and performance tests. You can run it using:

```bash
jupyter notebook tests/notebook.ipynb
```

**Stripes removal example :**
![camera_stripes](https://raw.githubusercontent.com/CEA-MetroCarac/pyvsnr/main/tests/images/camera_stripes.png)
**Gaussian noise removal example :**
![camera_stripes](https://raw.githubusercontent.com/CEA-MetroCarac/pyvsnr/main/tests/images/camera_gaussian.png)
**Curtains removal example :**
![camera_stripes](https://raw.githubusercontent.com/CEA-MetroCarac/pyvsnr/main/tests/images/camera_curtains.png)

You can also create your own tests using `from pyvsnr.utils import curtains_addition, stripes_addition, add_gaussian_noise`

## Shared library re-compilation
If you encounter shared library load errors then you may need
to recompile from source. This requires a working CUDA installation
with `nvcc` compiler. The source code is distributed with this package
and is found in the install directory, find this using:

```bash
python -c 'import pyvsnr; print(pyvsnr.PRECOMPILED_PATH)'
```

Navigate to this directory and re-compile for your system using the following, on linux:

```bash
cd ...
nvcc -lcufft -lcublas --compiler-options '-fPIC' -o libvsnr2d.so --shared vsnr2d.cu
```

and on Windows:

```powershell
cd ...
nvcc -lcufft -lcublas -o libvsnr2d.dll --shared vsnr2d.cu
```

## Authors informations

This Python port of the original code was developed by Killian PAVY, with guidance and direction from Patrick QUEMERE.

All credit for the original code goes to Jean EYMERIE and Pierre WEISS.

In case you use the results of this code in your article, please don't forget to cite:

- Fehrenbach, Jérôme, Pierre Weiss, and Corinne Lorenzo. "*Variational algorithms to remove stationary noise: applications to microscopy imaging.*" IEEE Transactions on Image Processing 21.10 (2012): 4420-4430.
- Fehrenbach, Jérôme, and Pierre Weiss. "*Processing stationary noise: model and parameter selection in variational methods.*" SIAM Journal on Imaging Sciences 7.2 (2014): 613-640.
- Escande, Paul, Pierre Weiss, and Wenxing Zhang. "*A variational model for multiplicative structured noise removal.*" Journal of Mathematical Imaging and Vision 57.1 (2017): 43-55.

<!-- The main source code is located in the `src` directory:

- `vsnr2d.py`: Contains the main `vsnr2d` function for computing VSNR using numpy or cupy.
- `vsnr2d_cuda.py`: Contains the CUDA implementation of VSNR, including the `vsnr2d_cuda` function.
<br>
The `tests` directory contains unit tests and performance tests:

- `test_cuda_equals_py.py`: Tests to ensure the CUDA and Python implementations produce the same results.
- `notebok.ipynb`: Jupyter notebook for examples & performance testing. -->
