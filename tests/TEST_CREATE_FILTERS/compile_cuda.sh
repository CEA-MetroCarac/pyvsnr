#!/usr/bin/env bash

# Exit on any error
set -e

# Define an error handler
handle_error() {
    echo "An error occurred while compiling save_filter.cu. Make sure save_filter.cu is in the same folder."
    exit 1
}

# Set the error handler
trap 'handle_error' ERR

# Compile the CUDA file
nvcc -o save_filter -lcufft -lcublas --compiler-options "-fPIC" save_filter.cu