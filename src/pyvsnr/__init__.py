import os
import importlib.metadata
from .vsnr2d import vsnr2d
from .vsnr2d_cuda import vsnr2d_cuda, PRECOMPILED_PATH


try:
    __version__ = importlib.metadata.version("pyvsnr")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


_CUPY_INSTALL_FLAG = os.path.join(os.path.dirname(__file__), ".cupy_checked")
if not os.path.exists(_CUPY_INSTALL_FLAG):
    try:
        from . import install_cupy
        print("🚀 First time using pyvsnr - checking for CUDA and CuPy...")
        install_cupy.install_cupy_if_needed()
        with open(_CUPY_INSTALL_FLAG, "w", encoding="utf-8") as f:
            f.write("checked")
    except Exception as e:
        print(f"ℹ️ CuPy auto-installation encountered an issue: {e}")
        print("  You can manually install CuPy for GPU acceleration")
        print("  To try the automatic installation again, run: python -m pyvsnr.install_cupy")
        try:
            with open(_CUPY_INSTALL_FLAG, "w", encoding="utf-8") as f:
                f.write("failed")
        except OSError:
            pass
