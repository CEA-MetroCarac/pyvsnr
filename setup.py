import os, sys, re, subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.bdist_wheel import bdist_wheel


def tprint(msg):
    try:
        with open('/dev/tty', 'w') as tty:
            tty.write(msg + '\n')
            tty.flush()
    except:
        print(msg, file=sys.stderr, flush=True)


def get_cuda_version():
    """Detect CUDA version from various sources"""
    cuda_version = None

    # nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10, check=True
        )
        match = re.search(r"release (\d+\.\d+)", result.stdout)
        if match:
            cuda_version = match.group(1)
            tprint(f"✅ Detected CUDA version from nvcc: {cuda_version}")
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
            ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=True
        )
        match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
        if match:
            cuda_version = match.group(1)
            tprint(
                f"✅ Detected CUDA version from nvidia-smi: {cuda_version}"
            )
            return cuda_version
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
    ):
        pass

    # Environment variables
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        try:
            nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
            if os.path.exists(nvcc_path):
                result = subprocess.run(
                    [nvcc_path, "--version"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                match = re.search(r"release (\d+\.\d+)", result.stdout)
                if match:
                    cuda_version = match.group(1)
                    tprint(
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
            tprint(f"ℹ️ CUDA version {cuda_version} is too old for CuPy support")
            return None

    except (ValueError, IndexError):
        tprint(f"ℹ️ Could not parse CUDA version: {cuda_version}")
        return None


def is_package_installed(package_name):
    """Check if a package is already installed"""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def install_cupy_if_needed():
    """Try to install appropriate CuPy version if CUDA is detected"""
    try:
        __import__("cupy")
        tprint("✅ CuPy is already installed, skipping automatic installation.")
        return
    except ImportError:
        pass

    tprint("🔍 Checking for CUDA installation...")
    cuda_version = get_cuda_version()

    if not cuda_version:
        tprint("ℹ️ Could not detect CUDA installation.")
        tprint("  For GPU acceleration, please install CuPy manually:")
        tprint("    pip install cupy-cuda11x  # for CUDA 11.x")
        tprint("    pip install cupy-cuda12x  # for CUDA 12.x")
        return

    cupy_package = get_cupy_package_name(cuda_version)
    if not cupy_package:
        return

    tprint(f"📦 Installing {cupy_package} for CUDA {cuda_version}...")
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = ''
        subprocess.run(
            [sys.executable, "-m", "pip", "install", cupy_package],
            check=True,
            env=env,
            cwd=os.path.expanduser("~")
        )
        tprint(f"✅ Successfully installed {cupy_package}")
        tprint("   GPU acceleration is now available!")
    except subprocess.CalledProcessError as e:
        tprint(f"❌ Failed to install {cupy_package}: {e}")
        tprint("   You may need to install CuPy manually for GPU acceleration")


def _run_cupy_install():
    try:
        tprint("🚀 PyVSNR CuPy Auto-Installer")
        install_cupy_if_needed()
    except ImportError as e:
        tprint(f"ℹ️ CuPy auto-installation encountered an issue: {e}")
        tprint("  You can manually install CuPy for GPU acceleration")
        tprint("  To try the automatic installation again, run: python -m pyvsnr.install_cupy")

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        _run_cupy_install()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        _run_cupy_install()

class PostWheelCommand(bdist_wheel):
    """Post-installation for wheel builds."""
    def run(self):
        bdist_wheel.run(self)
        _run_cupy_install()

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },

)
