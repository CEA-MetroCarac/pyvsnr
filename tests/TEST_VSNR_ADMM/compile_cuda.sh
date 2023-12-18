#!/usr/bin/env bash

# Exit on any error
set -e

# Define an error handler
handle_error() {
    echo "An error occurred while compiling vsnr_admm.cu. Make sure vsnr_admm.cu is in the same folder."
    exit 1
}

# Set the error handler
trap 'handle_error' ERR

# Compile the CUDA file
nvcc -o vsnr_admm -lcufft -lcublas --compiler-options "-fPIC" vsnr_admm.cu