VSNR 2D (CPU/GPU)
=================

Description
-----------
This repository contains the sources of the CPU/GPU based denoising codes of
 VSNR (2D).
 
Installation
------------
To complete

.. code-block:: python

    pip install pyvsnr (not yet available)

Usage
-----

.. code-block:: python

    from pyvsnr import VSNR
    from skimage import io

    img = io.imread('my_image.tif')

    vsnr = VSNR(img.shape)
    vsnr.add_filter(alpha=1e-2, name='gabor', sigma=(1, 30), theta=20)
    vsnr.add_filter(alpha=5e-2, name='gabor', sigma=(3, 40), theta=20)
    vsnr.initialize()
    img_corr = vsnr.eval(img, maxit=100, cvg_threshold=1e-4)

Report to examples.py in examples directory for more illustrations

Authors information
-------------------
This is a port of the original Matlab code by Jerome FEHRENBACH, Pierre
WEISS to python.
All credit goes to the original author.
In case you use the results of this code with your article, please don't forget
to cite:
- Fehrenbach, Jérôme, Pierre Weiss, and Corinne Lorenzo. "*Variational algorithms to remove stationary noise: applications to microscopy imaging.*" IEEE Transactions on Image Processing 21.10 (2012): 4420-4430.
- Fehrenbach, Jérôme, and Pierre Weiss. "*Processing stationary noise: model and parameter selection in variational methods.*" SIAM Journal on Imaging Sciences 7.2 (2014): 613-640.
- *Escande, Paul, Pierre Weiss, and Wenxing Zhang. "*A variational model for multiplicative structured noise removal.*" Journal of Mathematical Imaging and Vision 57.1 (2017): 43-55.

