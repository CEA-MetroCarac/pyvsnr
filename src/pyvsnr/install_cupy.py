"""
Auto-detection and installation of CuPy for pyvsnr
"""

import os
import sys
import subprocess
import re


def get_cuda_version():
    """Detect CUDA version from various sources"""
    cuda_version = None

    # nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                cuda_version = match.group(1)
                print(f"✅ Detected CUDA version from nvcc: {cuda_version}")
                return cuda_version
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
    ):
        pass

    # nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
            if match:
                cuda_version = match.group(1)
                print(
                    f"✅ Detected CUDA version from nvidia-smi: {cuda_version}"
                )
                return cuda_version
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
    ):
        pass

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        try:
            nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
            if os.path.exists(nvcc_path):
                result = subprocess.run(
                    [nvcc_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    match = re.search(r"release (\d+\.\d+)", result.stdout)
                    if match:
                        cuda_version = match.group(1)
                        print(
                            f"✅ Detected CUDA version from CUDA_HOME: {cuda_version}"
                        )
                        return cuda_version
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

    return None


def get_cupy_package_name(cuda_version):
    """Get the appropriate CuPy package name based on CUDA version"""
    if not cuda_version:
        return None

    try:
        major = int(cuda_version.split(".")[0])

        if major >= 11:
            return f"cupy-cuda{major}x"
        else:
            print(f"ℹ️ CUDA version {cuda_version} is too old for CuPy support")
            return None

    except (ValueError, IndexError):
        print(f"ℹ️ Could not parse CUDA version: {cuda_version}")
        return None


def is_package_installed(package_name):
    """Check if a package is already installed"""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "show", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def install_cupy_if_needed():
    """Try to install appropriate CuPy version if CUDA is detected"""

    if os.environ.get("PYVSNR_SKIP_CUPY_INSTALL", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        print("⏭ Skipping CuPy installation (PYVSNR_SKIP_CUPY_INSTALL is set)")
        return

    cupy_variants = ["cupy", "cupy-cuda11x", "cupy-cuda12x"]
    for variant in cupy_variants:
        if is_package_installed(variant):
            print(
                f"✅ {variant} is already installed, skipping automatic installation"
            )
            return

    print("🔍 Checking for CUDA installation...")
    cuda_version = get_cuda_version()

    if not cuda_version:
        print("ℹ️ Could not detect CUDA installation.")
        print("  For GPU acceleration, please install CuPy manually:")
        print("    pip install cupy-cuda11x  # for CUDA 11.x")
        print("    pip install cupy-cuda12x  # for CUDA 12.x")
        return

    cupy_package = get_cupy_package_name(cuda_version)
    if not cupy_package:
        return

    print(f"📦 Installing {cupy_package} for CUDA {cuda_version}...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                cupy_package,
                "--no-warn-script-location",
            ]
        )
        print(f"✅ Successfully installed {cupy_package}")
        print("   GPU acceleration is now available!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {cupy_package}: {e}")
        print("   You may need to install CuPy manually for GPU acceleration")


if __name__ == "__main__":
    install_cupy_if_needed()
