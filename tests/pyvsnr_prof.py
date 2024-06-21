# nsys profile -o report python pyvsnr_prof.py
import cupyx
import cupy as cp
from pyvsnr import vsnr2d

imgs = cp.random.rand(20,2048, 2048).astype(cp.float32)
filters=[{'name':'Dirac', 'noise_level':0.35}]

with cupyx.profiler.profile():
    for img in imgs:
        vsnr2d(img, filters)