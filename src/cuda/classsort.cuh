/*
 * classsort.cuh - Sorts each class: the positions whose next sample position is d characters ahead.
 *
 * Suffix i in class d >= 1 is ordered by (T[i], suffix i + 1), and i + 1 is in class d - 1,
 * so each class is one stable 8-bit radix pass over the predecessors of class d - 1, walked in order.
 * Class d - 1 is walked with n prepended, so that suffix n - 1 sorts first among those starting with
 * T[n - 1]. Keys hold T[q] to T[q - 3]; the next class shifts them, so only classes 1 and 5 read the text.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * From sadcs (https://github.com/acesds/sadcs), BWT path only.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cub/cub.cuh>
#include "devicecommon.cuh"

namespace gpusa {
    // Where each class starts in the concatenated lists, and n.
    struct ClassLists {
        uint32_t start[kClasses + 1];
    };

    inline ClassLists classLists(const CoverTables& tables, int32_t n) {
        ClassLists lists{};

        for (int d = 0; d < kClasses; d++) {
            lists.start[d + 1] = lists.start[d] + static_cast<uint32_t>(tables.classSize(n, d));
        }

        return lists;
    }

    __global__ void scatterSample(const uint32_t* __restrict__ ranks, uint32_t m, uint32_t* __restrict__ list) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < m) {
            list[ranks[t]] = (t / kCoverSize) * kCoverPeriod + cResidue[t % kCoverSize];
        }
    }

    // A large sample is partitioned by rank before scattering (see partitionThreshold()).
    __global__ void samplePairs(const uint32_t* __restrict__ ranks, uint32_t m, uint32_t* __restrict__ keys, uint32_t* __restrict__ positions) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < m) {
            keys[t] = ranks[t];
            positions[t] = (t / kCoverSize) * kCoverPeriod + cResidue[t % kCoverSize];
        }
    }

    __global__ void scatterPairs(const uint32_t* __restrict__ keys, const uint32_t* __restrict__ positions, uint32_t m, uint32_t* __restrict__ list) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < m) {
            list[keys[t]] = positions[t];
        }
    }

    // Element k of class d-1 with n prepended.
    __device__ __forceinline__ uint32_t previousElement(const uint32_t* __restrict__ previous, uint32_t n, uint32_t k) {
        return k == 0 ? n : previous[k - 1];
    }

    __device__ __forceinline__ bool predecessorInClass(uint32_t p, int d) {
        return p > 0 && cClassOf[(p - 1) % kCoverPeriod] == d;
    }

    // T[q], T[q-1], T[q-2], T[q-3] from the top byte down, zero before the text.
    __device__ __forceinline__ uint32_t characterKey(const uint64_t* __restrict__ text, uint32_t q) {
        const uint8_t* const bytes = reinterpret_cast<const uint8_t*>(text);
        uint32_t key = 0;

#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
            key = (key << 8) | (q >= i ? bytes[q - i] : 0);
        }

        return key;
    }

    __global__ void flagPredecessors(const uint32_t* __restrict__ previous, uint32_t count, uint32_t n, int d, uint32_t* __restrict__ flags) {
        const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;

        if (k <= count) {
            flags[k] = predecessorInClass(previousElement(previous, n, k), d) ? 1 : 0;
        }
    }

    // previousKeys: the sorted keys of class d - 1, or null to read the text.
    __global__ void gatherPredecessors(const uint64_t* __restrict__ text, const uint32_t* __restrict__ previous, const uint32_t* __restrict__ previousKeys, uint32_t count, uint32_t n, int d, const uint32_t* __restrict__ index, uint32_t* __restrict__ keys, uint32_t* __restrict__ positions) {
        const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;

        if (k <= count) {
            const uint32_t p = previousElement(previous, n, k);

            if (predecessorInClass(p, d)) {
                const uint32_t q = p - 1;
                keys[index[k]] = previousKeys && k > 0 ? previousKeys[k - 1] << 8 : characterKey(text, q);
                positions[index[k]] = q;
            }
        }
    }

    class ClassSort {
     public:
        static uint32_t largestClass(const ClassLists& lists) {
            uint32_t largest = 0;

            for (int d = 0; d < kClasses; d++) {
                largest = std::max(largest, lists.start[d + 1] - lists.start[d]);
            }

            return largest;
        }

        static uint32_t largestNonSampleClass(const ClassLists& lists) {
            uint32_t largest = 0;

            for (int d = 1; d < kClasses; d++) {
                largest = std::max(largest, lists.start[d + 1] - lists.start[d]);
            }

            return largest;
        }

        // Lives in the merge's output buffer, which is free until the merge.
        static size_t scratchWords(const ClassLists& lists) {
            const size_t m = lists.start[1] - lists.start[0];
            return std::max(4 * static_cast<size_t>(largestNonSampleClass(lists)) + m + 1, 4 * m);
        }

        static cudaError_t tempBytes(const ClassLists& lists, size_t& bytes) {
            const uint32_t largest = largestClass(lists);
            cub::DoubleBuffer<uint32_t> keys(nullptr, nullptr), values(nullptr, nullptr);
            size_t sort = 0, scan = 0;

            GPUSA_TRY(cub::DeviceRadixSort::SortPairs(nullptr, sort, keys, values, largest, 0, 32));
            GPUSA_TRY(cub::DeviceScan::ExclusiveSum(
                nullptr, scan, static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr), largest + 1));
            bytes = std::max(sort, scan);

            return cudaSuccess;
        }

        // Fills lists (n entries) with the sorted classes.
        static cudaError_t run(Construction& c, const ClassLists& layout, uint32_t* lists, uint32_t* scratch) {
            size_t tempBytes = 0;
            uint8_t* temp = nullptr;

            GPUSA_TRY(ClassSort::tempBytes(layout, tempBytes));
            GPUSA_TRY(c.memory.allocate(temp, tempBytes));

            const cudaError_t error = sortAll(c, layout, lists, scratch, temp, tempBytes);
            c.memory.release(temp);

            return error;
        }

     private:
        static cudaError_t sortAll(Construction& c, const ClassLists& layout, uint32_t* lists, uint32_t* scratch, uint8_t* temp, size_t tempBytes) {
            if (c.m >= partitionThreshold()) {
                uint32_t* const keys0 = scratch;
                uint32_t* const keys1 = keys0 + c.m;
                uint32_t* const values0 = keys1 + c.m;
                uint32_t* const values1 = values0 + c.m;
                cub::DoubleBuffer<uint32_t> keys(keys0, keys1), values(values0, values1);
                const int high = static_cast<int>(std::bit_width(c.m - 1));
                size_t bytes = tempBytes;

                samplePairs<<<blocksFor(c.m), threadsPerBlock()>>>(c.ranks, c.m, keys0, values0);
                GPUSA_TRY(cub::DeviceRadixSort::SortPairs(
                    temp, bytes, keys, values, c.m, std::max(0, high - partitionBits()), high));
                scatterPairs<<<blocksFor(c.m), threadsPerBlock()>>>(keys.Current(), values.Current(), c.m, lists + layout.start[0]);
            } else if (c.m > 0) {
                scatterSample<<<blocksFor(c.m), threadsPerBlock()>>>(c.ranks, c.m, lists + layout.start[0]);
            }

            for (int d = 1; d < kClasses; d++) {
                const uint32_t size = layout.start[d + 1] - layout.start[d];

                if (size == 0) {
                    continue;
                }

                const uint32_t* const previous = lists + layout.start[d - 1];
                const uint32_t count = layout.start[d] - layout.start[d - 1];
                uint32_t* const positions = lists + layout.start[d];
                uint32_t* const previousKeys = scratch;
                uint32_t* const index = previousKeys + largestNonSampleClass(layout);
                uint32_t* const keys0 = index + (layout.start[1] - layout.start[0]) + 1;
                uint32_t* const keys1 = keys0 + size;
                uint32_t* const values1 = keys1 + size;
                cub::DoubleBuffer<uint32_t> keys(keys0, keys1);
                cub::DoubleBuffer<uint32_t> values(positions, values1);
                size_t bytes = tempBytes;

                const bool fromText = d % 4 == 1;

                flagPredecessors<<<blocksFor(count + 1ull), threadsPerBlock()>>>(previous, count, c.n, d, index);
                GPUSA_TRY(cub::DeviceScan::ExclusiveSum(temp, bytes, index, index, count + 1));
                gatherPredecessors<<<blocksFor(count + 1ull), threadsPerBlock()>>>(
                    c.text, previous, fromText ? nullptr : previousKeys, count, c.n, d, index, keys0, positions);

                bytes = tempBytes;
                GPUSA_TRY(cub::DeviceRadixSort::SortPairs(temp, bytes, keys, values, size, 24, 32));

                if (values.Current() != positions) {
                    GPUSA_TRY(cudaMemcpyAsync(
                        positions, values.Current(), size * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
                }

                if (d + 1 < kClasses && (d + 1) % 4 != 1) {
                    GPUSA_TRY(cudaMemcpyAsync(
                        previousKeys, keys.Current(), size * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
                }
            }

            return cudaGetLastError();
        }
    };
}
