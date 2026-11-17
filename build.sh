#!/bin/sh
# Builds bwarp: with the GPU variant (linux-cuda preset) if nvcc is found, otherwise the CPU parts
# only (linux preset). "build.sh cpu" always builds the CPU parts only.

set -e
cd "$(dirname "$0")"

if [ "$1" != "cpu" ] && { [ -n "$CUDACXX" ] || command -v nvcc >/dev/null 2>&1; }; then
    echo "CUDA found: building the CPU and GPU parts (linux-cuda preset)..."
    cmake --preset linux-cuda
    cmake --build build-cuda
    echo
    echo "Build complete. Executable in: $(pwd)/build-cuda/bin/Release/"
else
    echo "Building the CPU parts (linux preset)..."
    cmake --preset linux
    cmake --build build
    echo
    echo "Build complete. Executable in: $(pwd)/build/bin/Release/"
fi
