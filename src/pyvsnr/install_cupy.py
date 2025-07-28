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

    return cuda_version


def get_rocm_version():
    """Detect ROCm version"""
    rocm_version = None

    # /opt/rocm/.info/version-dev or /opt/rocm/.info/version
    for version_file in [
        "/opt/rocm/.info/version",
        "/opt/rocm/.info/version-dev",
    ]:
        if os.path.exists(version_file):
            with open(version_file) as f:
                content = f.read()
                match = re.search(r"(\d+\.\d+)", content)
                if match:
                    rocm_version = match.group(1)
                    print(
                        f"✅ Detected ROCm version from {version_file}: {rocm_version}"
                    )
                    break

    return rocm_version


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
    def print_help():
        print("   You may need to install CuPy manually for GPU acceleration (see https://docs.cupy.dev/en/stable/install.html)")
        print("    pip install cupy-cuda11x  # for CUDA 11.x")
        print("    pip install cupy-cuda12x  # for CUDA 12.x")
        print("    pip install cupy-rocm-4-3   # for ROCm 4.3 (Linux only)")
        print("    pip install cupy-rocm-5-0   # for ROCm 5.0 (Linux only)")

    try:
        import cupy
        print("✅ CuPy is already available")
        return
    except ImportError:
        pass

    print("🔍 Checking for CUDA installation...")
    cuda_version = get_cuda_version()

    if not cuda_version:
        print("ℹ️ Could not detect CUDA installation.")
        print_help()
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
        print("GPU acceleration is now available!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {cupy_package}: {e}")
        print_help()


if __name__ == "__main__":
    install_cupy_if_needed()
