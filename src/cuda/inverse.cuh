/*
 * inverse.cuh - The inverse BWT on the GPU: psi from a stable sort of the BWT, then one chain per sample.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace bwarp::gpu {
    constexpr int kInverseThreads = 256;

    __global__ void iota(uint32_t* out, uint32_t n) {
        const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < n) {
            out[i] = i;
        }
    }

    // Rows are those of the n + 1 rotations; the stored BWT leaves out the sentinel's character, at primary.
    // psi(row of suffix i) = row of suffix i + 1. Row 0, the sentinel's suffix, precedes suffix 0.
    // In: psi[1 + j] = the stored index of the BWT's j-th character in stable sorted order.
    __global__ void finishPsi(uint32_t* psi, uint32_t n, uint32_t primary) {
        const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i == 0) {
            psi[0] = primary;
        } else if (i <= n) {
            const uint32_t s = psi[i];
            psi[i] = s + (s >= primary);
        }
    }

    // starts[c]: the BWT's characters below c. Row r >= 1 starts with the character c with
    // starts[c] <= r - 1 < starts[c + 1].
    __device__ __forceinline__ uint32_t firstCharacter(const uint32_t* starts, uint32_t r) {
        const uint32_t key = r - 1;
        uint32_t c = 0;

#pragma unroll
        for (uint32_t step = 128; step > 0; step >>= 1) {
            c += starts[c + step] <= key ? step : 0;
        }

        return c;
    }

    // Chain j writes text positions [j * 2^logSample, ...), starting at primary for j = 0 and at samples[j - 1]
    // otherwise. text is padded to a multiple of 16 bytes. Rows stay in [0, n] whatever the input, as psi
    // is a permutation of them. Each step waits on a dependent load, so the speed comes from the number of chains.
    __global__ void __launch_bounds__(kInverseThreads) walkChains(const uint32_t* __restrict__ psi,
        const uint32_t* __restrict__ counts, const uint32_t* __restrict__ samples, uint32_t primary, uint32_t n,
        uint32_t logSample, uint32_t chains, uint8_t* __restrict__ text) {
        __shared__ uint32_t starts[257];

        if (threadIdx.x == 0) {
            uint32_t sum = 0;

            for (int c = 0; c < 256; c++) {
                starts[c] = sum;
                sum += counts[c];
            }

            starts[256] = sum;
        }

        __syncthreads();

        const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

        if (j >= chains) {
            return;
        }

        uint32_t row = j == 0 ? primary : samples[j - 1];
        const uint32_t begin = j << logSample;
        const uint32_t end = static_cast<uint32_t>(min(uint64_t(begin) + (uint64_t(1) << logSample), uint64_t(n)));
        uint4* out = reinterpret_cast<uint4*>(text + begin);

        for (uint32_t t = begin; t < end; t += 16) {
            uint32_t w[4];

#pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t word = 0;

#pragma unroll
                for (int b = 0; b < 4; b++) {
                    word |= firstCharacter(starts, row) << (8 * b);
                    row = __ldg(psi + row);
                }

                w[k] = word;
            }

            *out++ = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
}
