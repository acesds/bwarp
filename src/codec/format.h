/*
 * format.h - Constants and coding primitives of the bwarp format (docs/format.md), shared by the CPU
 * reference codec and the GPU kernels, which must produce the same bytes.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <bit>
#include <cstdint>

#if defined(__CUDACC__)
#define BWARP_HD __host__ __device__ __forceinline__ constexpr
#define BWARP_HOST_DEVICE __host__ __device__
#else
#define BWARP_HD constexpr
#define BWARP_HOST_DEVICE
#endif

namespace bwarp {
    constexpr uint32_t kMagic = 0x50525742;  // "BWRP"
    constexpr uint8_t kVersion = 0;
    constexpr uint8_t kTransformBwt = 0;
    constexpr int kHeaderBytes = 16;

    constexpr int kLanes = 32;
    constexpr int kMinLogSlice = 10;
    constexpr int kMaxLogSlice = 16;  // at most 2048 bytes per lane, so a run's extra bits fit one rANS op
    constexpr int kMaxLogSegment = 24;
    constexpr int kMaxContexts = 16;

    constexpr int kProbBits = 11;  // 2 KiB decoding tables per context
    constexpr uint32_t kProbScale = 1u << kProbBits;
    constexpr uint32_t kRansLow = 1u << 16;  // states stay in [kRansLow, 2^32); I/O in 16-bit words

    constexpr uint32_t kDirectLengths = 8;
    constexpr uint32_t kMaxLengthCodes = 32;

    BWARP_HD uint32_t floorLog2(uint32_t x) {
#if defined(__CUDA_ARCH__)
        return 31 - __clz(x);
#else
        return 31 - static_cast<uint32_t>(std::countl_zero(x));
#endif
    }

    BWARP_HD uint32_t laneBytes(uint32_t sliceBytes) {
        return (sliceBytes + kLanes - 1) / kLanes;
    }

    // Run lengths 1 to 8 have their own codes; longer ones a code per half octave plus extra bits.
    struct LengthCode {
        uint32_t code;
        uint32_t extraBits;
        uint32_t extra;
    };

    BWARP_HD LengthCode lengthCode(uint32_t length) {
        if (length <= kDirectLengths) {
            return {length - 1, 0, 0};
        }

        const uint32_t lg = floorLog2(length);
        const uint32_t hi = (length >> (lg - 1)) & 1;

        return {kDirectLengths + 2 * (lg - 3) + hi, lg - 1, length & ((1u << (lg - 1)) - 1)};
    }

    BWARP_HD uint32_t lengthExtraBits(uint32_t code) {
        return code < kDirectLengths ? 0 : (code - kDirectLengths) / 2 + 2;
    }

    BWARP_HD uint32_t lengthBase(uint32_t code) {
        if (code < kDirectLengths) {
            return code + 1;
        }

        const uint32_t lg = 3 + (code - kDirectLengths) / 2;
        return (2 | ((code - kDirectLengths) & 1)) << (lg - 1);
    }

    // Number of length codes for slices of 2^logSlice bytes.
    BWARP_HD uint32_t lengthCodeCount(int logSlice) {
        return lengthCode(laneBytes(1u << logSlice)).code + 1;
    }

    // Symbols are coded as their rank in the segment's alphabet. The context of a symbol is the bucket of the
    // previous run's symbol in the same lane, or the last bucket at the lane's start; that of a run length
    // is the bucket of its own symbol.
    BWARP_HD uint32_t contextOf(uint32_t rank, uint32_t contexts) {
        return rank < contexts - 1 ? rank : contexts - 1;
    }

    // Run lengths use only the first contexts.
    constexpr uint32_t kMaxLengthContexts = 4;

#if !defined(__CUDA_ARCH__)  // __clz is not constexpr
    static_assert(lengthCode(laneBytes(1u << kMaxLogSlice)).code < kMaxLengthCodes);
    static_assert(lengthExtraBits(lengthCode(laneBytes(1u << kMaxLogSlice)).code) <= kProbBits,
        "a run's extra bits must fit one rANS op");
#endif

    BWARP_HD uint32_t lengthContexts(uint32_t contexts) {
        return contexts < kMaxLengthContexts ? contexts : kMaxLengthContexts;
    }

    // Before encoding with freq, x must be below this, or a 16-bit word goes out first.
    BWARP_HD bool ransNeedsOutput(uint32_t x, uint32_t freq) {
        return x >= ((static_cast<uint64_t>(kRansLow >> kProbBits) << 16) * freq);
    }

    BWARP_HD uint32_t ransEncode(uint32_t x, uint32_t freq, uint32_t cum) {
        return ((x / freq) << kProbBits) + x % freq + cum;
    }

    BWARP_HD uint32_t ransDecodeAdvance(uint32_t x, uint32_t freq, uint32_t cum) {
        return freq * (x >> kProbBits) + (x & (kProbScale - 1)) - cum;
    }

    // Tables store counts as floats with `mantissa` bits below the top one: exact below 2^(mantissa + 1),
    // then 2^mantissa steps per octave. Dequantizing gives the middle of the step.
    constexpr uint32_t kMaxCountMantissa = 3;

    BWARP_HD uint32_t quantizeCount(uint32_t count, uint32_t mantissa) {
        if (count < (1u << mantissa)) {
            return count;
        }

        const uint32_t lg = floorLog2(count);
        return ((lg - mantissa + 1) << mantissa) + ((count >> (lg - mantissa)) & ((1u << mantissa) - 1));
    }

    BWARP_HD uint32_t dequantizeCount(uint32_t q, uint32_t mantissa) {
        if (q < (1u << mantissa)) {
            return q;
        }

        const uint32_t shift = (q >> mantissa) - 1;
        return (((q & ((1u << mantissa) - 1)) | (1u << mantissa)) << shift) + ((1u << shift) >> 1);
    }

    // The table coder's adaptive binary decisions. p is the probability of a 0 bit, in units of 2^-kProbBits.
    constexpr uint32_t kAdaptShift = 3;

    BWARP_HD uint32_t adaptProbability(uint32_t p, uint32_t bit) {
        return bit ? p - (p >> kAdaptShift) : p + ((kProbScale - p) >> kAdaptShift);
    }

    // Scales counts[0, size), summing to more than 0, to frequencies summing to kProbScale.
    // Every nonzero count gets a nonzero frequency; the rounding error goes to the largest ones.
    BWARP_HD void normalizeFrequencies(const uint32_t* counts, uint32_t size, uint32_t* freqs) {
        uint64_t total = 0;

        for (uint32_t i = 0; i < size; i++) {
            total += counts[i];
        }

        int64_t diff = kProbScale;

        for (uint32_t i = 0; i < size; i++) {
            uint32_t f = 0;

            if (counts[i] > 0) {
                f = static_cast<uint32_t>((static_cast<uint64_t>(counts[i]) * kProbScale + total / 2) / total);
                f = f > 0 ? f : 1;
            }

            freqs[i] = f;
            diff -= f;
        }

        while (diff != 0) {
            uint32_t largest = 0;

            for (uint32_t i = 1; i < size; i++) {
                if (freqs[i] > freqs[largest]) {
                    largest = i;
                }
            }

            if (diff > 0) {
                freqs[largest] += static_cast<uint32_t>(diff);
                diff = 0;
            } else {
                const uint32_t take = -diff < static_cast<int64_t>(freqs[largest] - 1)
                                          ? static_cast<uint32_t>(-diff)
                                          : freqs[largest] - 1;
                freqs[largest] -= take;
                diff += take;
            }
        }
    }
}
