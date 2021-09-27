# VSNR

![](examples/data/fib_sem_result.png)

## Description

This repository contains the sources of the CPU/GPU based denoising codes of
the 2D-VSNR algorithm.
 
## Installation

    $ pip install pyvsnr

## Requirements

For CPU execution, the vsnr algorithm requires only the
![numpy](https://numpy.org/) package.

For GPU execution, the ![cupy](https://cupy.dev) library is used.
Follow the ![installaton instructions](https://docs.cupy.dev/en/stable/install.html)
for more details.

To run the ![examples](https://github.com/patquem/pyvsnr/tree/main/examples/examples.py)
, the following additional packages have to be installed:

- os
- time
- matplotlib
- tifffile

## Usage

For a single image treatment:

```python
from pyvsnr import VSNR
from skimage import io

# read image to correct
img = io.imread('my_image.tif')

# vsnr object creation
vsnr = VSNR(img.shape)

# add filter (at least one !)
vsnr.add_filter(alpha=1e-2, name='gabor', sigma=(1, 30), theta=20)
vsnr.add_filter(alpha=5e-2, name='gabor', sigma=(3, 40), theta=20)

# vsnr initialization
vsnr.initialize()

# image processing
img_corr = vsnr.eval(img, maxit=100, cvg_threshold=1e-4)

...
```
Some applicative examples are given in 
![examples.py](https://github.com/patquem/pyvsnr/tree/main/examples/examples.py)
and will return the following results (in addition to the one given as
 illustration at the top) :
 
**stripes removal** :
![](examples/data/camera_stripes_result.png)

**curtains removal** :
![](examples/data/camera_curtains_result.png)

**Note 1 :** in case of images batchs, in particularly in the case of
stacks where successive images are quite similar (FIB-SEM slices for instance),
computation time can be significantly decreased by this way (assuming all
the images have the same size):

```python
import glob
from pyvsnr import VSNR
from skimage import io

fnames = sorted(glob.glob('my_directory/*.tif'))
img0 = io.imread(fnames[0])
vsnr = VSNR(img0.shape)
vsnr.add_filter(alpha=1e-2, name='gabor', sigma=(1, 30), theta=20)
vsnr.add_filter(alpha=5e-2, name='gabor', sigma=(3, 40), theta=20)
vsnr.initialize()

# images processing
for fname in fnames:
    img = io.imread(fname)
    img_corr = vsnr.eval(img, maxit=100, cvg_threshold=1e-4)
    ...
```
**Note 2 :** in case of GPU executions, the first run is always more longer
 than the other ones. Keep it in mind when evaluating processing time.
 
 Running times evolution returned by ex_perf_evluatione in 
![examples.py](https://github.com/patquem/pyvsnr/tree/main/examples/examples.py)
![](examples/data/perf_evaluation_result.png)

## Authors information

This is a port of the original Matlab code by Jerome FEHRENBACH, Pierre
WEISS to python.

All credit goes to the original author.

In case you use the results of this code with your article, please don't forget
to cite:

- Fehrenbach, Jérôme, Pierre Weiss, and Corinne Lorenzo. "*Variational algorithms to remove stationary noise: applications to microscopy imaging.*" IEEE Transactions on Image Processing 21.10 (2012): 4420-4430.
- Fehrenbach, Jérôme, and Pierre Weiss. "*Processing stationary noise: model and parameter selection in variational methods.*" SIAM Journal on Imaging Sciences 7.2 (2014): 613-640.
- *Escande, Paul, Pierre Weiss, and Wenxing Zhang. "*A variational model for multiplicative structured noise removal.*" Journal of Mathematical Imaging and Vision 57.1 (2017): 43-55.

