/*
 * inverse.cuh - The inverse BWT on the GPU.
 *
 * LF (and psi) from a stable ranking of the BWT, then one chain per sample.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Rows are those of the n + 1 sorted rotations, the sentinel's, row 0, first. The stored BWT leaves out the
// sentinel's character, at row primary, so stored index i is row i + (i >= primary).
// LF(row of suffix i) = row of suffix i - 1, psi its inverse.
namespace bwarp::gpu {
    constexpr int kInverseThreads = 256;
    constexpr int kRankWarps = kInverseThreads / 32;
    constexpr uint32_t kRankTile = 16 * kInverseThreads;

    // Up to this many chains, each walks from both ends, with psi forward and LF backward: twice the loads in
    // flight, for psi's scatter and twice the memory the walk touches. More chains keep enough loads in flight.
    constexpr uint32_t kMaxBidirectionalChains = 2048;

    __device__ __forceinline__ uint32_t rowOf(uint32_t i, uint32_t primary) {
        return i + (i >= primary);
    }

    // counts[c * tiles + tile]: c's occurrences in the tile. bwt is padded to a multiple of 16 bytes.
    __global__ void __launch_bounds__(kInverseThreads) countTiles(const uint8_t* __restrict__ bwt, uint32_t n,
        uint32_t tiles, uint32_t* __restrict__ counts) {
        __shared__ uint32_t histogram[256];

        histogram[threadIdx.x] = 0;
        __syncthreads();

        const uint32_t at = blockIdx.x * kRankTile + threadIdx.x * 16;

        if (at < n) {
            const uint4 v = *reinterpret_cast<const uint4*>(bwt + at);
            const uint32_t words[4] = {v.x, v.y, v.z, v.w};

#pragma unroll
            for (uint32_t b = 0; b < 16; b++) {
                if (at + b < n) {
                    atomicAdd(&histogram[(words[b / 4] >> (8 * (b % 4))) & 255], 1u);
                }
            }
        }

        __syncthreads();
        counts[threadIdx.x * tiles + blockIdx.x] = histogram[threadIdx.x];
    }

    // After an exclusive sum over counts, offsets[c * tiles + tile] is the sorted index of c's first occurrence
    // in the tile. Ranks each tile stably, 256 characters at a time, and writes LF and, if asked, psi.
    __global__ void __launch_bounds__(kInverseThreads) rankTiles(const uint8_t* __restrict__ bwt, uint32_t n,
        uint32_t tiles, const uint32_t* __restrict__ offsets, uint32_t primary, uint32_t* __restrict__ lf,
        uint32_t* __restrict__ psi) {
        __shared__ uint32_t next[256];  // per character, the sorted index of its next occurrence
        __shared__ uint32_t warpBase[kRankWarps][256];

        const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;

        next[threadIdx.x] = offsets[threadIdx.x * tiles + blockIdx.x];

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            lf[primary] = 0;

            if (psi) {
                psi[0] = primary;
            }
        }

        for (uint32_t round = 0; round < kRankTile / kInverseThreads; round++) {
            for (uint32_t w = 0; w < kRankWarps; w++) {
                warpBase[w][threadIdx.x] = 0;
            }

            __syncthreads();

            const uint32_t i = blockIdx.x * kRankTile + round * kInverseThreads + threadIdx.x;
            const bool valid = i < n;
            const uint32_t c = valid ? bwt[i] : 0;
            const uint32_t peers = __match_any_sync(~0u, valid ? c : 256);
            const uint32_t below = __popc(peers & ((1u << lane) - 1));

            if (valid && below == 0) {
                warpBase[warp][c] = __popc(peers);
            }

            __syncthreads();

            // Each warp's counts become its bases, in order.
            uint32_t sum = next[threadIdx.x];

            for (uint32_t w = 0; w < kRankWarps; w++) {
                const uint32_t count = warpBase[w][threadIdx.x];
                warpBase[w][threadIdx.x] = sum;
                sum += count;
            }

            next[threadIdx.x] = sum;
            __syncthreads();

            if (valid) {
                const uint32_t sorted = warpBase[warp][c] + below;  // its row is sorted + 1
                const uint32_t row = rowOf(i, primary);

                lf[row] = sorted + 1;

                if (psi) {
                    psi[sorted + 1] = row;
                }
            }

            __syncthreads();
        }
    }

    // starts[c]: the BWT's characters below c. Row r >= 1 starts with the character c with
    // starts[c] <= r - 1 < starts[c + 1].
    __device__ __forceinline__ void loadStarts(uint32_t* starts, const uint32_t* offsets, uint32_t tiles,
        uint32_t n) {
        starts[threadIdx.x] = offsets[threadIdx.x * tiles];

        if (threadIdx.x == 0) {
            starts[256] = n;
        }

        __syncthreads();
    }

    __device__ __forceinline__ uint32_t firstCharacter(const uint32_t* starts, uint32_t r) {
        const uint32_t key = r - 1;
        uint32_t c = 0;

#pragma unroll
        for (uint32_t step = 128; step > 0; step >>= 1) {
            c += starts[c + step] <= key ? step : 0;
        }

        return c;
    }

    // Writes text [mid, end) backward from row, that of position end. Only the last chain's end may be unaligned;
    // mid is a multiple of 16 unless it is end.
    __device__ __forceinline__ void walkBackward(const uint32_t* __restrict__ lf, const uint32_t* starts,
        uint32_t row, uint32_t mid, uint32_t end, uint8_t* __restrict__ text) {
        uint32_t t = end;

        while (t > mid && (t & 15) != 0) {
            row = __ldg(lf + row);
            text[--t] = static_cast<uint8_t>(firstCharacter(starts, row));
        }

        for (uint4* out = reinterpret_cast<uint4*>(text + t); t > mid; t -= 16) {
            uint32_t w[4];

#pragma unroll
            for (int k = 3; k >= 0; k--) {
                uint32_t word = 0;

#pragma unroll
                for (int b = 3; b >= 0; b--) {
                    row = __ldg(lf + row);
                    word |= firstCharacter(starts, row) << (8 * b);
                }

                w[k] = word;
            }

            *--out = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }

    // Writes text [begin, mid) forward from row, that of position begin; past mid up to a multiple of 16.
    __device__ __forceinline__ void walkForward(const uint32_t* __restrict__ psi, const uint32_t* starts,
        uint32_t row, uint32_t begin, uint32_t mid, uint8_t* __restrict__ text) {
        uint4* out = reinterpret_cast<uint4*>(text + begin);

        for (uint32_t t = begin; t < mid; t += 16) {
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

    // Chain j writes text positions [j * 2^logSample, (j + 1) * 2^logSample), backward from the row of its end:
    // samples[j], or row 0 at position n for the last chain. With psi, the first half goes forward instead,
    // from primary for j = 0 and samples[j - 1] otherwise, and each warp takes one direction of 32 chains, so its
    // threads don't diverge. text is padded to a multiple of 16 bytes. Rows stay in [0, n] whatever the input,
    // as LF and psi are permutations of them. Each step waits on a dependent load, so the speed comes from the
    // number of chains.
    __global__ void __launch_bounds__(kInverseThreads) walkChains(const uint32_t* __restrict__ lf,
        const uint32_t* __restrict__ psi, const uint32_t* __restrict__ offsets, uint32_t tiles,
        const uint32_t* __restrict__ samples, uint32_t primary, uint32_t n, uint32_t logSample, uint32_t chains,
        uint8_t* __restrict__ text) {
        __shared__ uint32_t starts[257];

        loadStarts(starts, offsets, tiles, n);

        const uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
        const bool forward = psi && ((h >> 5) & 1) == 0;
        const uint32_t j = psi ? (h >> 6) * 32 + (h & 31) : h;

        if (j >= chains) {
            return;
        }

        const uint32_t begin = j << logSample;
        const uint32_t end = static_cast<uint32_t>(min(uint64_t(begin) + (uint64_t(1) << logSample), uint64_t(n)));
        const uint32_t mid = psi ? min(begin + (((end - begin) / 2 + 15) & ~15u), end) : begin;

        if (forward) {
            walkForward(psi, starts, j == 0 ? primary : samples[j - 1], begin, mid, text);
        } else {
            walkBackward(lf, starts, j + 1 < chains ? samples[j] : 0, mid, end, text);
        }
    }
}
