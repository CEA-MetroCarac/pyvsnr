import importlib.metadata
from .vsnr2d import vsnr2d
from .vsnr2d_cuda import vsnr2d_cuda, PRECOMPILED_PATH


try:
    __version__ = importlib.metadata.version("pyvsnr")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"