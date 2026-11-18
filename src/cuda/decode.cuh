/*
 * decode.cuh - The decoder kernels: a block's tables and rANS streams back to its BWT.
 *
 * The tables take a thread per segment, their frequencies and slot tables a block per segment, and the streams
 * a warp per slice, lane j decoding the slice's lane j.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstdint>
#include "format.h"
#include "table.h"

namespace bwarp::gpu {
    // What the stream decoder reads of a segment's table.
    struct DeviceTable {
        uint32_t contexts;
        uint8_t symbols[256];  // by rank
        uint8_t symbolUsed[kMaxContexts];
        uint8_t lengthUsed[kMaxLengthContexts];
        uint32_t symbolCode[kMaxContexts][256];  // freq | cum << 16
        uint32_t lengthCode[kMaxLengthContexts][kMaxLengthCodes];
        uint8_t symbolSlot[kMaxContexts][kProbScale];  // slot -> rank
        uint8_t lengthSlot[kMaxLengthContexts][kProbScale];  // slot -> length code
    };

    enum DecodeError : uint32_t {
        kBadTable = 1,
        kBadStream = 2,
    };

    constexpr uint32_t kTableWarps = 4;  // segments per block of the table decoder
    constexpr uint32_t kSymbolEntries = kMaxContexts * 256;
    constexpr uint32_t kPredictedEntries = kSymbolEntries + kMaxLengthContexts * kMaxLengthCodes;

    static_assert(sizeof(TableModels) == sizeof(Probability) * (sizeof(TableModels) / sizeof(Probability)));
    static_assert(sizeof(Probability) == 2);
#if !defined(__CUDA_ARCH__)
    static_assert(maxCountQ(kMaxCountMantissa) < 256, "predicted q must fit a byte");
#endif

    // A warp per segment: lane 0 runs the coder, the whole warp ranks the bytes and predicts the rows in between,
    // as codeTable() does serially. tables must be zeroed.
    __global__ void __launch_bounds__(kTableWarps * 32) decodeTables(const uint8_t* data, const uint64_t* segmentOffsets,
        uint32_t segments, uint32_t lengthCodes, TableData* tables, uint32_t* error) {
        __shared__ __align__(16) uint8_t modelBytes[kTableWarps][sizeof(TableModels)];
        __shared__ __align__(16) uint8_t predictorBytes[kTableWarps][sizeof(Predictor)];
        __shared__ uint8_t predictedQ[kTableWarps][kPredictedEntries];
        __shared__ uint16_t byteQ[kTableWarps][256];

        const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x & 31;
        const uint32_t g = blockIdx.x * kTableWarps + warp;

        if (g >= segments) {
            return;
        }

        TableModels& models = *reinterpret_cast<TableModels*>(modelBytes[warp]);
        Predictor& predictor = *reinterpret_cast<Predictor*>(predictorBytes[warp]);
        uint8_t* const predicted = predictedQ[warp];
        TableData& table = tables[g];
        TableReader reader(data + segmentOffsets[g], data + segmentOffsets[g + 1]);

        for (uint32_t i = lane; i < sizeof(TableModels) / sizeof(Probability); i += 32) {
            reinterpret_cast<Probability*>(&models)[i].p = kProbScale / 2;
        }

        __syncwarp();

        bool ok = true;

        if (lane == 0) {
            table.lengthCodes = lengthCodes;
            ok = codeCounts(reader, table, models);
        }

        if (!__shfl_sync(0xffffffffu, ok, 0)) {
            if (lane == 0) {
                atomicOr(error, kBadTable);
            }

            return;
        }

        // Ranks as rankSymbols(): by count, descending, ties by byte.
        for (uint32_t b = lane; b < 256; b += 32) {
            byteQ[warp][b] = table.byteQ[b];
        }

        __syncwarp();

        uint32_t present = 0;

        for (uint32_t b = lane; b < 256; b += 32) {
            const uint16_t q = byteQ[warp][b];

            if (q == 0) {
                continue;
            }

            uint32_t rank = 0;

            for (uint32_t o = 0; o < 256; o++) {
                const uint16_t other = byteQ[warp][o];
                rank += other > q || (other == q && o < b);
            }

            table.symbols[rank] = static_cast<uint8_t>(b);
            table.rankOf[b] = static_cast<uint8_t>(rank);
            present++;
        }

        const uint32_t alphabet = __reduce_add_sync(0xffffffffu, present);

        if (alphabet == 0) {
            if (lane == 0) {
                atomicOr(error, kBadTable);
            }

            return;
        }

        if (lane == 0) {
            table.alphabet = alphabet;
        }

        __syncwarp();

        if (lane == 0) {
            preparePredictor(table, predictor);
        }

        __syncwarp();

        const uint32_t k = table.contexts, kl = lengthContexts(k);

        for (uint32_t at = lane; at < kPredictedEntries; at += 32) {
            const bool symbol = at < kSymbolEntries;
            const uint32_t c = symbol ? at / 256 : (at - kSymbolEntries) / kMaxLengthCodes;
            const uint32_t i = symbol ? at % 256 : (at - kSymbolEntries) % kMaxLengthCodes;

            if (c < (symbol ? k : kl) && i < (symbol ? alphabet : table.lengthCodes)) {
                predicted[at] = static_cast<uint8_t>(predictQ(table, predictor, symbol, c, i));
            }
        }

        __syncwarp();

        if (lane == 0) {
            const auto lookup = [&](bool symbol, uint32_t c, uint32_t i) {
                return uint32_t(predicted[symbol ? c * 256 + i : kSymbolEntries + c * kMaxLengthCodes + i]);
            };

            if (!codeRows(reader, table, models, lookup) || !reader.finished()) {
                atomicOr(error, kBadTable);
            }
        }
    }

    __device__ void finishRow(const uint16_t* q, uint32_t size, uint32_t mantissa, uint32_t* code) {
        uint32_t counts[256], freqs[256];

        for (uint32_t i = 0; i < size; i++) {
            counts[i] = dequantizeCount(q[i], mantissa);
        }

        normalizeFrequencies(counts, size, freqs);

        for (uint32_t i = 0, cum = 0; i < size; i++) {
            code[i] = freqs[i] | cum << 16;
            cum += freqs[i];
        }
    }

    // The last entry of code[0, size) whose cumulative frequency is at most slot.
    __device__ uint32_t findSlot(const uint32_t* code, uint32_t size, uint32_t slot) {
        uint32_t low = 0, high = size;  // code[low] <= slot < code[high]

        while (high - low > 1) {
            const uint32_t middle = (low + high) / 2;

            if ((code[middle] >> 16) <= slot) {
                low = middle;
            } else {
                high = middle;
            }
        }

        return low;
    }

    // A block per segment. The slot tables only with slots, for decoding.
    __global__ void finishTables(const TableData* tables, DeviceTable* out, bool slots) {
        const TableData& table = tables[blockIdx.x];
        DeviceTable& device = out[blockIdx.x];
        const uint32_t k = table.contexts, kl = lengthContexts(k);

        if (table.alphabet == 0) {
            return;  // decodeTables reported it
        }

        if (threadIdx.x == 0) {
            device.contexts = k;
        }

        for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) {
            device.symbols[i] = table.symbols[i];
        }

        for (uint32_t c = threadIdx.x; c < k; c += blockDim.x) {
            device.symbolUsed[c] = table.symbolUsed[c];

            if (table.symbolUsed[c]) {
                finishRow(table.symbolQ[c], table.alphabet, table.mantissa, device.symbolCode[c]);
            }
        }

        for (uint32_t c = threadIdx.x; c < kl; c += blockDim.x) {
            device.lengthUsed[c] = table.lengthUsed[c];

            if (table.lengthUsed[c]) {
                finishRow(table.lengthQ[c], table.lengthCodes, table.mantissa, device.lengthCode[c]);
            }
        }

        __syncthreads();

        for (uint32_t c = 0; slots && c < k + kl; c++) {
            const bool symbol = c < k;
            const uint32_t row = symbol ? c : c - k;

            if (!(symbol ? table.symbolUsed[row] : table.lengthUsed[row])) {
                continue;
            }

            const uint32_t* code = symbol ? device.symbolCode[row] : device.lengthCode[row];
            const uint32_t size = symbol ? table.alphabet : table.lengthCodes;
            uint8_t* slots = symbol ? device.symbolSlot[row] : device.lengthSlot[row];

            for (uint32_t s = threadIdx.x; s < kProbScale; s += blockDim.x) {
                slots[s] = static_cast<uint8_t>(findSlot(code, size, s));
            }
        }
    }

    constexpr uint32_t kSliceWarps = 4;  // slices per block of the stream decoder

    __device__ __forceinline__ uint32_t loadWord(const uint8_t* p) {
        return uint32_t(p[0]) | uint32_t(p[1]) << 8;
    }

    // The low k bytes of a 16-byte block, as its two halves.
    __device__ __forceinline__ uint64_t lowBytes(uint32_t k) {
        return k >= 8 ? ~0ull : (1ull << (8 * k)) - 1;
    }

    __device__ __forceinline__ uint64_t highBytes(uint32_t k) {
        return k <= 8 ? 0 : k >= 16 ? ~0ull : (1ull << (8 * (k - 8))) - 1;
    }

    // Writes a lane's runs as aligned 16-byte stores, staged in registers, at a cost independent of the run
    // length. Lanes are 1 KiB apart, so byte stores would each cost a memory transaction.
    struct RunWriter {
        uint8_t* out;
        uint64_t low = 0, high = 0;  // the block ending at pos
        uint32_t fill;               // its bytes so far; pos - fill is 16-aligned
        uint32_t skip;               // its leading bytes that belong to the previous lane

        __device__ RunWriter(uint8_t* out, uint32_t pos) : out(out), fill(pos & 15), skip(pos & 15) {}

        __device__ void put(uint32_t& pos, uint8_t symbol, uint32_t length) {
            const uint64_t pattern = symbol * 0x0101010101010101ull;

            while (length > 0) {
                if (fill == 0 && length >= 16) {
                    *reinterpret_cast<uint4*>(out + pos) = make_uint4(static_cast<uint32_t>(pattern),
                        static_cast<uint32_t>(pattern), static_cast<uint32_t>(pattern), static_cast<uint32_t>(pattern));
                    pos += 16;
                    length -= 16;
                    continue;
                }

                const uint32_t take = min(length, 16 - fill), to = fill + take;
                const uint64_t lowMask = lowBytes(to) & ~lowBytes(fill), highMask = highBytes(to) & ~highBytes(fill);

                low = (low & ~lowMask) | (pattern & lowMask);
                high = (high & ~highMask) | (pattern & highMask);
                fill = to;
                pos += take;
                length -= take;

                if (fill == 16) {
                    if (skip == 0) {
                        *reinterpret_cast<uint4*>(out + pos - 16) = make_uint4(static_cast<uint32_t>(low),
                            static_cast<uint32_t>(low >> 32), static_cast<uint32_t>(high), static_cast<uint32_t>(high >> 32));
                    } else {
                        flush(pos);
                    }

                    fill = 0;
                    skip = 0;
                }
            }
        }

        // Writes the staged bytes one by one.
        __device__ void flush(uint32_t pos) {
            for (uint32_t i = skip; i < fill; i++) {
                out[pos - fill + i] = static_cast<uint8_t>(i < 8 ? low >> (8 * i) : high >> (8 * (i - 8)));
            }
        }
    };

    // A warp per slice; the lanes take refill words in lane order, as the format prescribes.
    __global__ void __launch_bounds__(kSliceWarps * 32) decodeSlices(const uint8_t* data, const uint64_t* sliceOffsets,
        uint32_t slices, uint32_t n, int logSegment, int logSlice, const DeviceTable* tables, uint8_t* bwt,
        uint32_t* error) {
        const uint32_t lane = threadIdx.x & 31;
        const uint32_t slice = blockIdx.x * kSliceWarps + threadIdx.x / 32;

        if (slice >= slices) {
            return;
        }

        const DeviceTable& table = tables[slice >> (logSegment - logSlice)];
        const uint32_t begin = slice << logSlice;
        const uint32_t end = min(n - begin, 1u << logSlice) + begin;
        const uint32_t size = end - begin, perLane = laneBytes(size);
        uint32_t pos = begin + min(lane * perLane, size);
        const uint32_t laneEnd = begin + min((lane + 1) * perLane, size);

        const uint8_t* stream = data + sliceOffsets[slice];
        const uint64_t streamBytes = sliceOffsets[slice + 1] - sliceOffsets[slice];

        if (streamBytes < kLanes * 4 || streamBytes % 2 != 0) {
            if (lane == 0) {
                atomicOr(error, kBadStream);
            }

            return;
        }

        const uint8_t* words = stream + kLanes * 4;
        const uint32_t wordCount = static_cast<uint32_t>((streamBytes - kLanes * 4) / 2);
        const uint32_t below = (1u << lane) - 1;
        const uint32_t k = table.contexts, kl = lengthContexts(k);
        uint32_t x = loadWord(stream + 4 * lane) | loadWord(stream + 4 * lane + 2) << 16;
        uint32_t next = 0, context = k - 1;
        bool bad = x < kRansLow || k == 0;

        RunWriter writer(bwt, pos);

        const auto renormalize = [&](bool need) {
            const uint32_t mask = __ballot_sync(0xffffffffu, need);

            if (need) {
                const uint32_t at = next + __popc(mask & below);

                if (at < wordCount) {
                    x = x << 16 | loadWord(words + 2 * at);
                } else {
                    bad = true;
                }
            }

            next += __popc(mask);
        };

        for (;;) {
            const bool active = pos < laneEnd && !bad;

            if (!__any_sync(0xffffffffu, active) || __any_sync(0xffffffffu, bad)) {
                break;
            }

            uint32_t rank = 0, code = 0;

            if (active) {
                if (!table.symbolUsed[context]) {
                    bad = true;
                } else {
                    const uint32_t slot = x & (kProbScale - 1);
                    rank = table.symbolSlot[context][slot];
                    const uint32_t fc = table.symbolCode[context][rank];
                    x = ransDecodeAdvance(x, fc & 0xffff, fc >> 16);
                }
            }

            renormalize(active && x < kRansLow);

            const uint32_t lengthContext = contextOf(rank, kl);

            if (active && !bad) {
                if (!table.lengthUsed[lengthContext]) {
                    bad = true;
                } else {
                    const uint32_t slot = x & (kProbScale - 1);
                    code = table.lengthSlot[lengthContext][slot];
                    const uint32_t fc = table.lengthCode[lengthContext][code];
                    x = ransDecodeAdvance(x, fc & 0xffff, fc >> 16);
                }
            }

            renormalize(active && x < kRansLow);

            uint32_t length = 0;

            if (active && !bad) {
                const uint32_t bits = lengthExtraBits(code);
                const uint32_t extra = (x & (kProbScale - 1)) >> (kProbBits - bits);

                length = lengthBase(code) + extra;
                x = ransDecodeAdvance(x, 1u << (kProbBits - bits), extra << (kProbBits - bits));
                bad = length > laneEnd - pos;
            }

            renormalize(active && x < kRansLow);

            if (active && !bad) {
                writer.put(pos, table.symbols[rank], length);
                context = contextOf(rank, k);
            }
        }

        writer.flush(pos);

        bad = bad || x != kRansLow;

        if (lane == 0 && next != wordCount) {
            bad = true;
        }

        if (__any_sync(0xffffffffu, bad) && lane == 0) {
            atomicOr(error, kBadStream);
        }
    }
}
