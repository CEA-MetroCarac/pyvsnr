#!/usr/bin/env bash

# Exit on any error
set -e

# Define an error handler
handle_error() {
    echo "An error occurred while compiling vsnr2d.cu. Make sure vsnr2d.cu is in the same folder, and also the folder precompiled exists."
    exit 1
}

# Set the error handler
trap 'handle_error' ERR

# Compile the CUDA file
nvcc -o libvsnr2d.so -lcufft -lcublas --compiler-options "-fPIC" --shared ../vsnr2d.cu