/*
 * encode.cuh - The encoder kernels: choosing and coding each segment's table, then the slices' rANS streams.
 *
 * They match the CPU reference encoder byte for byte.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "decode.cuh"
#include "format.h"
#include "table.h"

namespace bwarp::gpu {
    constexpr uint32_t kChooseThreads = 256;

    __device__ __forceinline__ bool isLaneStart(uint32_t p, uint32_t n, int logSlice) {
        const uint32_t sliceBegin = p >> logSlice << logSlice;
        return (p - sliceBegin) % laneBytes(min(n - sliceBegin, 1u << logSlice)) == 0;
    }

    // For each run of slices [first, end), calls start(its rank, laneStart, the previous byte's rank) at its first
    // byte and finish(its rank, length) at its last, in no particular order, a byte's rank being rank(byte). The
    // block's threads take 16 bytes each and rank them all before calling, so that the calls' atomics wait on no
    // loads. A run's end finds its start in the chunk or, through a block-wide max-scan, in an earlier one. bwt is
    // padded to 16 bytes.
    template <class Rank, class Start, class Finish>
    __device__ void forEachRun(const uint8_t* __restrict__ bwt, uint32_t n, int logSlice, uint32_t first, uint32_t end,
        Rank rank, Start start, Finish finish) {
        using BlockScan = cub::BlockScan<uint32_t, kChooseThreads>;
        __shared__ typename BlockScan::TempStorage scan;

        const uint32_t stop = min(n, end << logSlice);
        uint32_t carry = 0;  // 1 + the last run start so far

        for (uint32_t base = first << logSlice; base < stop; base += kChooseThreads * 16) {
            const uint32_t at = base + threadIdx.x * 16;
            uint32_t starts = 0, laneStarts = 0, ends = 0, words[4] = {}, before = 0;

            if (at < stop) {
                const uint4 chunk = *reinterpret_cast<const uint4*>(bwt + at);
                const uint32_t sliceBegin = at >> logSlice << logSlice;  // a chunk never crosses a slice
                const uint32_t lb = laneBytes(min(n - sliceBegin, 1u << logSlice));
                const uint32_t into = (at - sliceBegin) % lb;
                before = at > 0 ? __ldg(bwt + at - 1) : 0;
                uint32_t previous = before << 24;  // in the top byte

                words[0] = chunk.x;
                words[1] = chunk.y;
                words[2] = chunk.z;
                words[3] = chunk.w;

                for (uint32_t k = into == 0 ? 0 : lb - into; k < 16; k += lb) {
                    laneStarts |= 1u << k;
                }

                // A byte differing from the one before, a bit per byte.
#pragma unroll
                for (uint32_t j = 0; j < 4; j++) {
                    const uint32_t differs = __vcmpne4(words[j], words[j] << 8 | previous >> 24) & 0x01010101u;

                    starts |= ((differs | differs >> 7 | differs >> 14 | differs >> 21) & 15) << (4 * j);
                    previous = words[j];
                }

                starts = (starts | laneStarts) & (stop - at >= 16 ? 0xffffu : (1u << (stop - at)) - 1);

                const uint32_t next = at + 16;
                const bool nextStarts = next >= stop || isLaneStart(next, n, logSlice) ||
                                        __ldg(bwt + next) != (words[3] >> 24);

                // A run ends before a start, or at the last byte.
                ends = (starts >> 1 | (nextStarts ? 1u << 15 : 0)) & ((stop - at >= 16 ? 1u << 16 : 1u << (stop - at)) - 1);

                if (stop - at < 16) {
                    ends |= 1u << (stop - at - 1);
                }
            }

            uint32_t ranks[16];
            const uint32_t beforeRank = rank(before);

#pragma unroll
            for (uint32_t k = 0; k < 16; k++) {
                ranks[k] = rank((words[k / 4] >> (8 * (k % 4))) & 255);
            }

            const uint32_t last = starts ? at + 31 - __clz(starts) + 1 : 0;
            uint32_t open, aggregate;

            BlockScan(scan).ExclusiveScan(last, open, 0u, [](uint32_t a, uint32_t b) { return max(a, b); }, aggregate);
            open = max(open, carry);
            carry = max(carry, aggregate);
            __syncthreads();

#pragma unroll
            for (uint32_t k = 0; k < 16; k++) {
                if ((starts >> k) & 1) {
                    start(ranks[k], (laneStarts >> k) & 1, k > 0 ? ranks[k - 1] : beforeRank);
                }
            }

#pragma unroll
            for (uint32_t k = 0; k < 16; k++) {
                if ((ends >> k) & 1) {
                    const uint32_t prior = starts & ((2u << k) - 1);
                    const uint32_t runStart = prior ? at + 31 - __clz(prior) : open - 1;

                    finish(ranks[k], at + k + 1 - runStart);
                }
            }
        }
    }

    constexpr uint32_t kFineContexts = kMaxContexts + 1;  // the previous rank's bucket for K = 16, then lane starts
    constexpr uint32_t kContextChoices = 5;               // K = 1, 2, 4, 8, 16
    constexpr uint32_t kMantissaChoices = 2;
    constexpr uint32_t kLengthChoices = 3;                // length contexts 1, 2, 4

    struct RowCost {
        uint64_t bits;
        uint32_t decisions;
    };

    // codeRow()'s models are independent chains, coded in parts whose costs add up: part 0 is the nonzero flag in
    // zero context 0 (40% of the entries in text), part 1 the used flag and the other nonzero flags, part 2 + r the
    // residuals in residual context r.
    constexpr uint32_t kZeroParts = 2;
    constexpr uint32_t kRowParts = kZeroParts + kResidualContexts;

    // A part's probabilities, held by a warp a lane each: in parts 0 and 1, lane z has nonzero[z] and lane 0 of part
    // 1 also used; in part 2 + r, lane k < 16 has residual[r].length[k] and lane 16 + k residual[r].top[k]. Each sees
    // the bits codeRow() codes with it in the same order, so the integer costs add up to the same sums. bits and
    // decisions are the lane's share.
    static_assert(kZeroContexts == 32 && 2 * kMaxUintBits == 32);

    struct LaneCells {
        uint32_t p = kProbScale / 2;
        uint32_t used = kProbScale / 2;  // part 1, lane 0; costMarginal(): nonzero[lane], lanes 0 and 1
        uint32_t bits = 0;
        uint32_t decisions = 0;

        // Codes b with prob in the active lanes, without branching: the warp's steps form one chain whichever lanes
        // are active, so each is kept short. Idle lanes read cost[0], so that a step loads at most two lines.
        __device__ void code(const uint32_t* __restrict__ cost, uint32_t& prob, bool active, uint32_t b) {
            const uint32_t c = __ldg(cost + (active ? (b ? kProbScale - prob : prob) : 0));

            bits += active ? c : 0;
            decisions += active;
            prob = active ? adaptProbability(prob, b) : prob;
        }

        // codeUint(v) on UintModels a lane each, as in part 2 + r: length[k] for k up to v + 1's bit length (below
        // 15), top[that k], then raw bits (lane 0).
        __device__ void codeUint(const uint32_t* __restrict__ cost, uint32_t v, uint32_t lane) {
            const uint32_t u = v + 1, length = floorLog2(u);
            const uint32_t k = min(length, kMaxUintBits - 1), cell = lane % kMaxUintBits;

            const uint32_t raw = lane == 0 && k > 1 ? k - 1 : 0;

            code(cost, p, lane < kMaxUintBits ? cell < kMaxUintBits - 1 && cell <= length : k >= 1 && cell == k,
                lane < kMaxUintBits ? cell < length : (u >> (k > 0 ? k - 1 : 0)) & 1);
            bits += raw * __ldg(cost + kProbScale / 2);
            decisions += raw;
        }

        __device__ void load(const RowModels& models, uint32_t part, uint32_t lane) {
            if (part < kZeroParts) {
                p = models.nonzero[lane].p;
                used = models.used.p;
            } else {
                const UintModels& values = models.residual[part - kZeroParts];
                p = lane < kMaxUintBits ? values.length[lane].p : values.top[lane - kMaxUintBits].p;
            }
        }

        __device__ void save(RowModels& models, uint32_t part, uint32_t lane) const {
            if (part < kZeroParts) {
                if ((lane == 0) == (part == 0)) {
                    models.nonzero[lane].p = static_cast<uint16_t>(p);
                }

                if (part == 1 && lane == 0) {
                    models.used.p = static_cast<uint16_t>(used);
                }
            } else {
                UintModels& values = models.residual[part - kZeroParts];
                (lane < kMaxUintBits ? values.length[lane] : values.top[lane - kMaxUintBits]).p = static_cast<uint16_t>(p);
            }
        }
    };

    // A row's q held as an array.
    __device__ __forceinline__ auto arrayQ(const uint16_t* q) {
        return [q](uint32_t i) { return uint32_t(q[i]); };
    }

    // codeRow()'s cost in one part, by a warp. q(i): the row's entry i.
    template <class Q>
    __device__ void costRowPart(LaneCells& cells, const uint32_t* __restrict__ cost, bool used, Q q,
        const uint8_t* predicted, uint32_t size, uint32_t skip, uint32_t part, uint32_t lane) {
        cells.code(cost, cells.used, part == 1 && lane == 0, used);

        if (!used) {
            return;
        }

        // 32 entries at a time, lane j reading entry j, then the part's own in order: in parts 0 and 1, each to the
        // lane of its zero context.
        for (uint32_t at = 0; at < size; at += 32) {
            const uint32_t i = at + lane;
            const bool valid = i < size && i != skip;
            const uint32_t pq = valid ? predicted[i] : 0, qi = valid ? q(i) : 0;

            if (part < kZeroParts) {
                const uint32_t zeroContext = min(pq, kZeroContexts - 1);
                const bool match = valid && (zeroContext == 0) == (part == 0);

                for (uint32_t matches = __ballot_sync(~0u, match); matches != 0; matches &= matches - 1) {
                    const uint32_t from = __ffs(matches) - 1;
                    cells.code(cost, cells.p, lane == __shfl_sync(~0u, zeroContext, from), __shfl_sync(~0u, qi, from) != 0);
                }
            } else {
                const int32_t base = static_cast<int32_t>(pq > 1 ? pq : 1);
                const uint32_t v = zigzag(int32_t(qi) - base);
                const bool match =
                    valid && qi != 0 && min(floorLog2(pq + 1), kResidualContexts - 1) == part - kZeroParts;

                for (uint32_t matches = __ballot_sync(~0u, match); matches != 0; matches &= matches - 1) {
                    cells.codeUint(cost, __shfl_sync(~0u, v, __ffs(matches) - 1), lane);
                }
            }
        }
    }

    // codeMarginal()'s cost, by a warp: lanes 0 and 1 hold nonzero[0] and [1], the lanes the value models.
    __device__ void costMarginal(LaneCells& cells, const uint32_t* __restrict__ cost, const uint16_t* q, uint32_t size,
        uint32_t lane) {
        int32_t previous = 0;     // the last nonzero q so far
        uint32_t previousZero = 1;

        // 32 entries at a time, lane j reading entry j and finding what comes before it in the earlier lanes.
        for (uint32_t at = 0; at < size; at += 32) {
            const uint32_t count = min(32u, size - at);
            const uint32_t qi = lane < count ? q[at + lane] : 0;
            const uint32_t nonzeros = __ballot_sync(~0u, qi != 0), before = nonzeros & ((1u << lane) - 1);
            const uint32_t zeroBefore = lane == 0 ? previousZero : !((nonzeros >> (lane - 1)) & 1);
            const int32_t base = static_cast<int32_t>(__shfl_sync(~0u, qi, before ? 31 - __clz(before) : 0));
            const uint32_t v = zigzag(int32_t(qi) - (before ? base : previous));

            // Each nonzero flag to lane 0 or 1, by the one before; then the values.
            for (uint32_t j = 0; j < count; j++) {
                cells.code(cost, cells.used, lane == __shfl_sync(~0u, zeroBefore, j), (nonzeros >> j) & 1);
            }

            for (uint32_t matches = nonzeros; matches != 0; matches &= matches - 1) {
                cells.codeUint(cost, __shfl_sync(~0u, v, __ffs(matches) - 1), lane);
            }

            previousZero = !((nonzeros >> (count - 1)) & 1);
            previous = nonzeros ? static_cast<int32_t>(__shfl_sync(~0u, qi, 31 - __clz(nonzeros))) : previous;
        }
    }

    // Per fine context c and rank, the tokens in contexts c and above; the last row stays zero. Packed, ranks r and
    // r + 128 share a word, r in the low half: exact while no rank has 2^16 tokens in a segment, so for segments of
    // up to 2^16 bytes (a slice's lanes are longer than a byte, so not every byte can start a run).
    template <bool Packed>
    struct SuffixCounts {
        static constexpr uint32_t kWords = Packed ? 128 : 256;
        static constexpr uint32_t kRows = kFineContexts + 1;
        static constexpr uint32_t kZeroRow = kFineContexts;

        uint32_t words[kRows][kWords];

        __device__ __forceinline__ void add(uint32_t c, uint32_t r) {
            atomicAdd(&words[c][Packed ? r % 128 : r], Packed ? 1u << (r / 128 * 16) : 1u);
        }

        __device__ __forceinline__ uint32_t get(uint32_t c, uint32_t r) const {
            return Packed ? (words[c][r % 128] >> (r / 128 * 16)) & 0xffff : words[c][r];
        }

        // Tokens in contexts [c, d).
        __device__ __forceinline__ uint32_t tokens(uint32_t c, uint32_t d, uint32_t r) const {
            return get(c, r) - get(d, r);
        }
    };

    // Rows c < K - 1 are the same for every K, so they are coded once, in order, and each K adds its last row: the
    // merged contexts K - 1 and above. Likewise for length rows. Arrays by K are indexed by log2 K, by length
    // context count by its log2. The shared symbol rows' q are not kept but quantized from suffix when needed.
    template <bool Packed>
    struct ChooseShared {
        uint32_t tokens[256];  // per byte
        uint32_t lengthTotal[kMaxLengthCodes];
        SuffixCounts<Packed> suffix;
        // The same for length contexts by min(rank, 3).
        uint32_t lengthSuffix[kMaxLengthContexts + 1][kMaxLengthCodes];
        uint16_t byteQ[256];
        uint16_t mergedQ[kContextChoices][256];
        uint16_t mergedLengthQ[kLengthChoices][kMaxLengthCodes];
        bool mergedUsed[kContextChoices];
        bool mergedLengthUsed[kLengthChoices];
        // The predictor's counts: per rank and per length code, and the merged rows' context tokens.
        uint32_t rankCount[256];
        uint32_t lengthCount[kMaxLengthCodes];
        uint64_t rankTokens;
        uint64_t lengthTokens;
        uint64_t mergedTokens[kContextChoices];
        uint64_t mergedLengthTokens[kLengthChoices];
        uint8_t predicted[kMaxContexts - 1][256];
        uint8_t lengthPredicted[kMaxLengthContexts - 1][kMaxLengthCodes];
        uint8_t mergedPredicted[kContextChoices][256];
        uint8_t mergedLengthPredicted[kLengthChoices][kMaxLengthCodes];
        // What the stream's tokens cost with each row's frequencies.
        uint64_t rowBits[kMaxContexts - 1];
        uint64_t lengthRowBits[kMaxLengthContexts - 1];
        uint64_t mergedBits[kContextChoices];
        uint64_t mergedLengthBits[kLengthChoices];
        // What coding the rows costs: the shared ones before each K's last row, and that row.
        RowCost prefix[kRowParts][kContextChoices];
        RowCost lengthPrefix[kRowParts][kLengthChoices];
        RowCost merged[kContextChoices];
        RowCost mergedLength[kLengthChoices];
        RowCost counts;  // K, m and the count lists
        uint32_t alphabet;
        uint32_t winner;  // log2 K of this m's best so far, or UINT32_MAX
        TableHead table;  // with the shared length rows
        __align__(16) uint8_t snapshots[sizeof(RowModels) * (kContextChoices + kLengthChoices)];  // the models before
                                                                                                    // each last row
    };

    __device__ __forceinline__ uint64_t warpSum(uint64_t v) {
        for (uint32_t d = 16; d > 0; d /= 2) {
            v += __shfl_xor_sync(~0u, v, d);
        }

        return v;
    }

    // The stream cost of a row of size entries with count(i) tokens, coded with the frequencies of its q(i), by a
    // warp: normalizeFrequencies() with lane j holding entries j, j + 32, ... The rounding error goes to the largest
    // frequency, the first of equals, as there. The cost in every lane.
    template <class Q, class Count>
    __device__ uint64_t streamBits(Q q, uint32_t size, bool used, Count count, uint32_t m,
        const uint32_t* __restrict__ cost, uint32_t lane) {
        if (!used) {
            return 0;
        }

        constexpr uint32_t kPerLane = 256 / 32;
        uint32_t counts[kPerLane], freqs[kPerLane];
        uint64_t total = 0;

#pragma unroll
        for (uint32_t j = 0; j < kPerLane; j++) {
            const uint32_t i = lane + 32 * j;
            counts[j] = i < size ? dequantizeCount(q(i), m) : 0;
            total += counts[j];
        }

        total = warpSum(total);

        int32_t share = 0;

#pragma unroll
        for (uint32_t j = 0; j < kPerLane; j++) {
            uint32_t f = 0;

            if (counts[j] > 0) {
                f = static_cast<uint32_t>((uint64_t(counts[j]) * kProbScale + total / 2) / total);
                f = f > 0 ? f : 1;
            }

            freqs[j] = f;
            share += static_cast<int32_t>(f);
        }

        int32_t diff = static_cast<int32_t>(kProbScale) - __reduce_add_sync(~0u, static_cast<uint32_t>(share));

        while (diff != 0) {
            // The largest frequency, then the lowest index, as a key: frequency above, 255 - index below.
            uint32_t key = 0;

#pragma unroll
            for (uint32_t j = 0; j < kPerLane; j++) {
                const uint32_t i = lane + 32 * j;
                key = i < size ? max(key, freqs[j] << 8 | (255 - i)) : key;
            }

            key = __reduce_max_sync(~0u, key);

            const uint32_t largest = 255 - (key & 255), f = key >> 8;
            const uint32_t change = diff > 0 ? static_cast<uint32_t>(diff) : min(uint32_t(-diff), f - 1);
            const uint32_t updated = diff > 0 ? f + change : f - change;

#pragma unroll
            for (uint32_t j = 0; j < kPerLane; j++) {
                if (lane + 32 * j == largest) {
                    freqs[j] = updated;
                }
            }

            diff += diff > 0 ? -static_cast<int32_t>(change) : static_cast<int32_t>(change);
        }

        uint64_t bits = 0;

#pragma unroll
        for (uint32_t j = 0; j < kPerLane; j++) {
            const uint32_t i = lane + 32 * j;

            if (i < size) {
                bits += uint64_t(count(i)) * __ldg(cost + freqs[j]);
            }
        }

        return warpSum(bits);
    }

    // The predicted q of every row entry, as codeRows() asks for them.
    __device__ void predictRows(const TableData& table, const Predictor& predictor, uint8_t* predicted, uint32_t thread,
        uint32_t threads) {
        const uint32_t k = table.contexts, kl = lengthContexts(k);

        for (uint32_t at = thread; at < kPredictedEntries; at += threads) {
            const bool symbol = at < kSymbolEntries;
            const uint32_t c = symbol ? at / 256 : (at - kSymbolEntries) / kMaxLengthCodes;
            const uint32_t i = symbol ? at % 256 : (at - kSymbolEntries) % kMaxLengthCodes;

            if (c < (symbol ? k : kl) && i < (symbol ? table.alphabet : table.lengthCodes)) {
                predicted[at] = static_cast<uint8_t>(predictQ(table, predictor, symbol, c, i));
            }
        }
    }

    __device__ __forceinline__ uint32_t lookupPredicted(const uint8_t* predicted, bool symbol, uint32_t c, uint32_t i) {
        return predicted[symbol ? c * 256 + i : kSymbolEntries + c * kMaxLengthCodes + i];
    }

    // A block per segment: chooseModel() of the CPU encoder. Leaves the winner in tables[g] and the number of its
    // table's decisions in decisions[g]. cost: encodingCosts(). Packed: segments of at most 2^16 bytes, which
    // leaves room for three blocks per SM.
    template <bool Packed>
    __global__ void __launch_bounds__(kChooseThreads, Packed ? 3 : 2) chooseTables(const uint8_t* __restrict__ bwt,
        uint32_t n, int logSegment, int logSlice, uint32_t maxContexts, const uint32_t* __restrict__ cost,
        TableData* tables, uint64_t* decisions) {
        using Suffix = SuffixCounts<Packed>;
        __shared__ ChooseShared<Packed> s;

        TableHead& table = s.table;
        const uint32_t g = blockIdx.x, tid = threadIdx.x;
        const uint32_t slices = (n + (1u << logSlice) - 1) >> logSlice;
        const uint32_t first = g << (logSegment - logSlice);
        const uint32_t end = min(slices, (g + 1) << (logSegment - logSlice));
        const uint32_t lengthCodes = lengthCodeCount(logSlice);

        for (uint32_t b = tid; b < 256; b += kChooseThreads) {
            s.tokens[b] = 0;
        }

        if (tid < kMaxLengthCodes) {
            s.lengthTotal[tid] = 0;
        }

        __syncthreads();

        forEachRun(
            bwt, n, logSlice, first, end, [](uint32_t byte) { return byte; },
            [&](uint32_t byte, bool, uint32_t) { atomicAdd(&s.tokens[byte], 1u); },
            [&](uint32_t, uint32_t length) { atomicAdd(&s.lengthTotal[lengthCode(length).code], 1u); });

        uint32_t choices = 0;  // the K tried: 1, 2, ... up to maxContexts

        while (choices < kContextChoices && (1u << choices) <= maxContexts) {
            choices++;
        }

        const uint32_t lengthChoices = min(choices, kLengthChoices);
        const uint32_t fineRows = (1u << (choices - 1)) - 1, fineLengthRows = (1u << (lengthChoices - 1)) - 1;
        uint64_t bestBits = UINT64_MAX;
        uint32_t bestIndex = 0, bestDecisions = 0;

        for (uint32_t m = 0; m < kMantissaChoices; m++) {
            __syncthreads();

            for (uint32_t b = tid; b < 256; b += kChooseThreads) {
                s.byteQ[b] = static_cast<uint16_t>(quantizeCount(s.tokens[b], m));
            }

            for (uint32_t e = tid; e < Suffix::kRows * Suffix::kWords; e += kChooseThreads) {
                (&s.suffix.words[0][0])[e] = 0;
            }

            for (uint32_t e = tid; e < (kMaxLengthContexts + 1) * kMaxLengthCodes; e += kChooseThreads) {
                (&s.lengthSuffix[0][0])[e] = 0;
            }

            if (tid == 0) {
                s.alphabet = 0;
            }

            __syncthreads();

            // Ranks as rankSymbols(): by q, descending, ties by byte.
            for (uint32_t b = tid; b < 256; b += kChooseThreads) {
                const uint16_t q = s.byteQ[b];

                if (q == 0) {
                    continue;
                }

                uint32_t rank = 0;

                for (uint32_t o = 0; o < 256; o++) {
                    rank += s.byteQ[o] > q || (s.byteQ[o] == q && o < b);
                }

                table.symbols[rank] = static_cast<uint8_t>(b);
                table.rankOf[b] = static_cast<uint8_t>(rank);
                atomicAdd(&s.alphabet, 1u);
            }

            __syncthreads();

            forEachRun(
                bwt, n, logSlice, first, end, [&](uint32_t byte) { return uint32_t(table.rankOf[byte]); },
                [&](uint32_t rank, bool laneStart, uint32_t previous) {
                    s.suffix.add(laneStart ? kMaxContexts : min(previous, kMaxContexts - 1), rank);
                },
                [&](uint32_t rank, uint32_t length) {
                    atomicAdd(&s.lengthSuffix[min(rank, kMaxLengthContexts - 1)][lengthCode(length).code], 1u);
                });

            __syncthreads();

            // Packed, both halves at once: neither carries.
            for (uint32_t w = tid; w < Suffix::kWords; w += kChooseThreads) {
                for (uint32_t c = kFineContexts - 1, sum = 0; c != UINT32_MAX; c--) {
                    sum += s.suffix.words[c][w];
                    s.suffix.words[c][w] = sum;
                }
            }

            if (tid < kMaxLengthCodes) {
                for (uint32_t c = kMaxLengthContexts - 1, sum = 0; c != UINT32_MAX; c--) {
                    sum += s.lengthSuffix[c][tid];
                    s.lengthSuffix[c][tid] = sum;
                }
            }

            __syncthreads();

            const uint32_t alphabet = s.alphabet;

            for (uint32_t b = tid; b < 256; b += kChooseThreads) {
                table.byteQ[b] = s.byteQ[b];
                s.rankCount[b] = b < alphabet ? dequantizeCount(s.byteQ[table.symbols[b]], m) : 0;
            }

            for (uint32_t l = tid; l < kMaxLengthCodes; l += kChooseThreads) {
                const uint32_t q = l < lengthCodes ? quantizeCount(s.lengthTotal[l], m) : 0;

                table.lengthCodeQ[l] = static_cast<uint16_t>(q);
                s.lengthCount[l] = dequantizeCount(q, m);
            }

            // Quantized tokens of shared symbol row c, by rank; tokens in length contexts [c, d).
            const auto fineQ = [m](uint32_t c) {
                return [c, m](uint32_t i) { return quantizeCount(s.suffix.tokens(c, c + 1, i), m); };
            };
            const auto lengthTokens = [](uint32_t c, uint32_t d, uint32_t l) {
                return s.lengthSuffix[c][l] - s.lengthSuffix[d][l];
            };

            for (uint32_t e = tid; e < choices * alphabet; e += kChooseThreads) {
                const uint32_t ki = e / alphabet, r = e % alphabet;
                s.mergedQ[ki][r] =
                    static_cast<uint16_t>(quantizeCount(s.suffix.tokens((1u << ki) - 1, Suffix::kZeroRow, r), m));
            }

            for (uint32_t e = tid; e < fineLengthRows * lengthCodes; e += kChooseThreads) {
                const uint32_t c = e / lengthCodes, l = e % lengthCodes;
                table.lengthQ[c][l] = static_cast<uint16_t>(quantizeCount(lengthTokens(c, c + 1, l), m));
            }

            for (uint32_t e = tid; e < lengthChoices * lengthCodes; e += kChooseThreads) {
                const uint32_t li = e / lengthCodes, l = e % lengthCodes;
                s.mergedLengthQ[li][l] =
                    static_cast<uint16_t>(quantizeCount(lengthTokens((1u << li) - 1, kMaxLengthContexts, l), m));
            }

            if (tid == 0) {
                table.contexts = 1;
                table.mantissa = m;
                table.lengthCodes = lengthCodes;
                table.alphabet = alphabet;

                for (uint32_t ki = 0; ki < kContextChoices; ki++) {
                    s.merged[ki] = {0, 0};
                }

                for (uint32_t li = 0; li < kLengthChoices; li++) {
                    s.mergedLength[li] = {0, 0};
                }
            }

            __syncthreads();

            // A thread per row's used flag, then per sum of counts.
            if (uint32_t job = tid; job < fineRows) {
                bool used = false;

                for (uint32_t r = 0; r < alphabet && !used; r++) {
                    used = fineQ(job)(r) > 0;
                }

                table.symbolUsed[job] = used;
            } else if ((job -= fineRows) < choices) {
                bool used = false;

                for (uint32_t r = 0; r < alphabet; r++) {
                    used = used || s.mergedQ[job][r] > 0;
                }

                s.mergedUsed[job] = used;
            } else if ((job -= choices) < fineLengthRows) {
                bool used = false;

                for (uint32_t l = 0; l < lengthCodes; l++) {
                    used = used || table.lengthQ[job][l] > 0;
                }

                table.lengthUsed[job] = used;
            } else if ((job -= fineLengthRows) < lengthChoices) {
                bool used = false;

                for (uint32_t l = 0; l < lengthCodes; l++) {
                    used = used || s.mergedLengthQ[job][l] > 0;
                }

                s.mergedLengthUsed[job] = used;
            } else if ((job -= lengthChoices) < choices + lengthChoices + 2) {
                // Ranks from (1 << job) - 1 up, for merged rows; all of them, then the length codes.
                const bool merged = job < choices + lengthChoices;
                const uint32_t from = job < choices ? (1u << job) - 1 : merged ? (1u << (job - choices)) - 1 : 0;
                const uint32_t* count = job == choices + lengthChoices + 1 ? s.lengthCount : s.rankCount;
                uint64_t sum = 0;

                for (uint32_t r = from, size = merged || job == choices + lengthChoices ? 256 : kMaxLengthCodes; r < size;
                     r++) {
                    sum += count[r];
                }

                if (job < choices) {
                    s.mergedTokens[job] = sum;
                } else if (merged) {
                    s.mergedLengthTokens[job - choices] = sum;
                } else if (job == choices + lengthChoices) {
                    s.rankTokens = sum;
                } else {
                    s.lengthTokens = sum;
                }
            }

            __syncthreads();

            // The predicted q, as predictQ(): a shared row c has only its own rank's tokens, and leaves them out of
            // the denominator.
            for (uint32_t e = tid; e < fineRows * alphabet; e += kChooseThreads) {
                const uint32_t c = e / alphabet, i = e % alphabet;
                s.predicted[c][i] =
                    static_cast<uint8_t>(predictCount(s.rankCount[c], s.rankCount[i], s.rankTokens - s.rankCount[c], m));
            }

            for (uint32_t e = tid; e < choices * alphabet; e += kChooseThreads) {
                const uint32_t ki = e / alphabet, i = e % alphabet;
                s.mergedPredicted[ki][i] =
                    static_cast<uint8_t>(predictCount(s.mergedTokens[ki], s.rankCount[i], s.rankTokens, m));
            }

            for (uint32_t e = tid; e < fineLengthRows * lengthCodes; e += kChooseThreads) {
                const uint32_t c = e / lengthCodes, l = e % lengthCodes;
                s.lengthPredicted[c][l] =
                    static_cast<uint8_t>(predictCount(s.rankCount[c], s.lengthCount[l], s.lengthTokens, m));
            }

            for (uint32_t e = tid; e < lengthChoices * lengthCodes; e += kChooseThreads) {
                const uint32_t li = e / lengthCodes, l = e % lengthCodes;
                s.mergedLengthPredicted[li][l] =
                    static_cast<uint8_t>(predictCount(s.mergedLengthTokens[li], s.lengthCount[l], s.lengthTokens, m));
            }

            __syncthreads();

            RowModels* snapshots = reinterpret_cast<RowModels*>(s.snapshots);

            // The stream cost of row job, by a warp: the shared rows, the merged ones, then the same for lengths.
            const auto costStream = [&](uint32_t job, uint32_t lane) {
                uint64_t* to;
                uint64_t bits;

                if (uint32_t at = job; at < fineRows) {
                    to = &s.rowBits[at];
                    bits = streamBits(fineQ(at), alphabet, table.symbolUsed[at],
                        [at](uint32_t i) { return s.suffix.tokens(at, at + 1, i); }, m, cost, lane);
                } else if ((at -= fineRows) < choices) {
                    to = &s.mergedBits[at];
                    bits = streamBits(arrayQ(s.mergedQ[at]), alphabet, s.mergedUsed[at],
                        [at](uint32_t i) { return s.suffix.tokens((1u << at) - 1, Suffix::kZeroRow, i); }, m,
                        cost, lane);
                } else if ((at -= choices) < fineLengthRows) {
                    to = &s.lengthRowBits[at];
                    bits = streamBits(arrayQ(table.lengthQ[at]), lengthCodes, table.lengthUsed[at],
                        [&, at](uint32_t l) { return lengthTokens(at, at + 1, l); }, m, cost, lane);
                } else {
                    at -= fineLengthRows;
                    to = &s.mergedLengthBits[at];
                    bits = streamBits(arrayQ(s.mergedLengthQ[at]), lengthCodes, s.mergedLengthUsed[at],
                        [&, at](uint32_t l) { return lengthTokens((1u << at) - 1, kMaxLengthContexts, l); }, m,
                        cost, lane);
                }

                if (lane == 0) {
                    *to = bits;
                }
            };
            const uint32_t streamJobs = fineRows + choices + fineLengthRows + lengthChoices;

            // Part part of last row row, from its snapshot, by a warp: symbols for K = 2^row, then lengths.
            const auto costMerged = [&](uint32_t row, uint32_t part, uint32_t lane) {
                const bool symbol = row < choices;
                const uint32_t ki = symbol ? row : row - choices;
                LaneCells cells;

                cells.load(snapshots[symbol ? ki : kContextChoices + ki], part, lane);

                if (symbol) {
                    costRowPart(cells, cost, s.mergedUsed[ki], arrayQ(s.mergedQ[ki]), s.mergedPredicted[ki], alphabet,
                        256, part, lane);
                } else {
                    costRowPart(cells, cost, s.mergedLengthUsed[ki], arrayQ(s.mergedLengthQ[ki]),
                        s.mergedLengthPredicted[ki], lengthCodes, kMaxLengthCodes, part, lane);
                }

                const uint32_t bits = __reduce_add_sync(~0u, cells.bits);
                const uint32_t decisions = __reduce_add_sync(~0u, cells.decisions);

                if (lane == 0) {
                    RowCost& to = symbol ? s.merged[ki] : s.mergedLength[ki];
                    atomicAdd(reinterpret_cast<unsigned long long*>(&to.bits), static_cast<unsigned long long>(bits));
                    atomicAdd(&to.decisions, decisions);
                }
            };

            // The serial parts: warp p codes part p of the shared rows, the symbol rows, then the length rows. The
            // residual parts' warps, done sooner, then code their parts of the last rows and cost the streams, and
            // the last one also codes the counts.
            static_assert(kChooseThreads / 32 == kRowParts);

            if (const uint32_t part = tid / 32, lane = tid % 32; part < kRowParts) {
                for (uint32_t kind = 0; kind < 2; kind++) {
                    const bool symbol = kind == 0;
                    const uint32_t rows = symbol ? fineRows : fineLengthRows, last = symbol ? choices : lengthChoices;
                    const uint32_t size = symbol ? alphabet : lengthCodes, stride = symbol ? 256 : kMaxLengthCodes;
                    const uint8_t* predicted = symbol ? s.predicted[0] : s.lengthPredicted[0];
                    const bool* used = symbol ? table.symbolUsed : table.lengthUsed;
                    RowModels* snapshot = snapshots + (symbol ? 0 : kContextChoices);
                    RowCost* prefix = symbol ? s.prefix[part] : s.lengthPrefix[part];
                    LaneCells cells;

                    for (uint32_t c = 0, ki = 0;; c++) {
                        if (ki < last && c == (1u << ki) - 1) {
                            const uint32_t bits = __reduce_add_sync(~0u, cells.bits);
                            const uint32_t decisions = __reduce_add_sync(~0u, cells.decisions);

                            cells.save(snapshot[ki], part, lane);

                            if (lane == 0) {
                                prefix[ki] = {bits, decisions};
                            }

                            ki++;
                        }

                        if (c == rows) {
                            break;
                        }

                        if (symbol) {
                            costRowPart(cells, cost, used[c], fineQ(c), predicted + c * stride, size, c, part, lane);
                        } else {
                            costRowPart(cells, cost, used[c], arrayQ(table.lengthQ[c]), predicted + c * stride, size,
                                kMaxLengthCodes, part, lane);
                        }
                    }
                }

                for (uint32_t row = 0; part >= kZeroParts && row < choices + lengthChoices; row++) {
                    costMerged(row, part, lane);  // from the snapshots this warp wrote
                }

                if (part == kRowParts - 1) {
                    // codeCounts(): K in 4 raw bits, m in 2, then the counts of bytes and of length codes.
                    LaneCells bytes, lengthTotals;

                    costMarginal(bytes, cost, table.byteQ, 256, lane);
                    costMarginal(lengthTotals, cost, table.lengthCodeQ, lengthCodes, lane);

                    const uint32_t bits = __reduce_add_sync(~0u, bytes.bits + lengthTotals.bits);
                    const uint32_t decisions = __reduce_add_sync(~0u, bytes.decisions + lengthTotals.decisions);

                    if (lane == 0) {
                        s.counts = {bits + 6ull * __ldg(cost + kProbScale / 2), decisions + 6};
                    }
                }

                for (uint32_t job = part - kZeroParts; part >= kZeroParts && job < streamJobs; job += kResidualContexts) {
                    costStream(job, lane);
                }
            }

            __syncthreads();

            // The zero parts of each last row, a warp per row and part.
            for (uint32_t job = tid / 32, lane = tid % 32; job < (choices + lengthChoices) * kZeroParts;
                 job += kChooseThreads / 32) {
                costMerged(job / kZeroParts, job % kZeroParts, lane);
            }

            __syncthreads();

            if (tid == 0) {
                s.winner = UINT32_MAX;

                for (uint32_t ki = 0; ki < choices; ki++) {
                    const uint32_t k = 1u << ki, kl = lengthContexts(k), li = ki < lengthChoices ? ki : lengthChoices - 1;
                    const uint32_t index = ki * kMantissaChoices + m;
                    uint64_t bits = s.counts.bits + s.merged[ki].bits + s.mergedLength[li].bits + s.mergedBits[ki] +
                                    s.mergedLengthBits[li];
                    uint32_t decisions = s.counts.decisions + s.merged[ki].decisions + s.mergedLength[li].decisions;

                    for (uint32_t part = 0; part < kRowParts; part++) {
                        bits += s.prefix[part][ki].bits + s.lengthPrefix[part][li].bits;
                        decisions += s.prefix[part][ki].decisions + s.lengthPrefix[part][li].decisions;
                    }

                    for (uint32_t c = 0; c < k - 1; c++) {
                        bits += s.rowBits[c];
                    }

                    for (uint32_t c = 0; c < kl - 1; c++) {
                        bits += s.lengthRowBits[c];
                    }

                    // The CPU tries K outermost, so the first of equals is the one of lowest index.
                    if (bits < bestBits || (bits == bestBits && index < bestIndex)) {
                        bestBits = bits;
                        bestIndex = index;
                        bestDecisions = decisions;
                        s.winner = ki;
                    }
                }
            }

            __syncthreads();

            if (const uint32_t ki = s.winner; ki != UINT32_MAX) {
                const uint32_t k = 1u << ki, kl = lengthContexts(k), li = ki < lengthChoices ? ki : lengthChoices - 1;
                const uint32_t* from = reinterpret_cast<const uint32_t*>(&table);
                TableData& out = tables[g];
                uint32_t* to = reinterpret_cast<uint32_t*>(static_cast<TableHead*>(&out));

                static_assert(sizeof(TableHead) % 4 == 0);

                for (uint32_t i = tid; i < sizeof(TableHead) / 4; i += kChooseThreads) {
                    to[i] = from[i];
                }

                __syncthreads();

                // The symbol rows: the shared ones quantized from the counts, then the merged last one.
                for (uint32_t e = tid; e < k * alphabet; e += kChooseThreads) {
                    const uint32_t c = e / alphabet, r = e % alphabet;
                    out.symbolQ[c][r] = c < k - 1 ? static_cast<uint16_t>(fineQ(c)(r)) : s.mergedQ[ki][r];
                }

                for (uint32_t l = tid; l < lengthCodes; l += kChooseThreads) {
                    out.lengthQ[kl - 1][l] = s.mergedLengthQ[li][l];
                }

                if (tid == 0) {
                    out.contexts = k;
                    out.symbolUsed[k - 1] = s.mergedUsed[ki];
                    out.lengthUsed[kl - 1] = s.mergedLengthUsed[li];
                }
            }
        }

        if (tid == 0) {
            decisions[g] = bestDecisions;
        }
    }

    constexpr uint32_t kReservedOps = 4096;  // per table, in the memory the host reserves for them: typically enough

    // Records a table's decisions as rANS ops, freq | cum << 16, the way TableWriter does.
    struct OpRecorder {
        uint32_t* ops;
        uint32_t count = 0;

        __device__ void push(uint32_t b, uint32_t p) {
            ops[count++] = b ? (kProbScale - p) | p << 16 : p;
        }

        __device__ uint32_t bit(Probability& prob, uint32_t b) {
            push(b, prob.p);
            prob.p = static_cast<uint16_t>(adaptProbability(prob.p, b));
            return b;
        }

        __device__ uint32_t raw(uint32_t v, int bits) {
            for (int i = bits - 1; i >= 0; i--) {
                push((v >> i) & 1, kProbScale / 2);
            }

            return v;
        }
    };

    // A warp per segment: codes tables[g] into the ops at opOffsets[g], then encodes them backwards into the 16-bit
    // words that end where the next segment's ops start. words[g]: how many, states[g]: the final state.
    __global__ void __launch_bounds__(kTableWarps * 32) encodeTables(TableData* tables, uint32_t segments,
        const uint64_t* opOffsets, uint32_t* ops, uint16_t* words, uint32_t* wordCounts, uint32_t* states) {
        __shared__ __align__(16) uint8_t modelBytes[kTableWarps][sizeof(TableModels)];
        __shared__ __align__(16) uint8_t predictorBytes[kTableWarps][sizeof(Predictor)];
        __shared__ uint8_t predictedQ[kTableWarps][kPredictedEntries];

        const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x & 31;
        const uint32_t g = blockIdx.x * kTableWarps + warp;

        if (g >= segments) {
            return;
        }

        TableModels& models = *reinterpret_cast<TableModels*>(modelBytes[warp]);
        Predictor& predictor = *reinterpret_cast<Predictor*>(predictorBytes[warp]);
        TableData& table = tables[g];
        OpRecorder recorder{ops + opOffsets[g]};

        for (uint32_t i = lane; i < sizeof(TableModels) / sizeof(Probability); i += 32) {
            reinterpret_cast<Probability*>(&models)[i].p = kProbScale / 2;
        }

        __syncwarp();

        if (lane == 0) {
            codeCounts(recorder, table, models);
            preparePredictor(table, predictor);
        }

        __syncwarp();
        predictRows(table, predictor, predictedQ[warp], lane, 32);
        __syncwarp();

        if (lane == 0) {
            codeRows(recorder, table, models, [&](bool symbol, uint32_t c, uint32_t i) {
                return lookupPredicted(predictedQ[warp], symbol, c, i);
            });
        }

        __syncwarp();

        // The warp encodes the ops backwards, every lane alike: each batch of 32 is loaded a batch ahead and handed
        // around by shuffles, and lane 0 stores the words. Each lane divides by its op's freq ahead, as a
        // multiplication by ceil(2^64 / freq): exact, the state being below 2^32 and freq at most 2^11.
        const uint32_t total = __shfl_sync(~0u, recorder.count, 0);
        const uint32_t* const op = ops + opOffsets[g];
        uint16_t* const end = words + opOffsets[g + 1];
        uint32_t x = kRansLow, count = 0;
        uint32_t next = lane < total ? op[total - 1 - lane] : 1;

        for (uint32_t done = 0; done < total; done += 32) {
            const uint32_t e = next, steps = min(32u, total - done);
            const uint32_t freq = e & 0xffff;
            const uint32_t limit = static_cast<uint32_t>(min((uint64_t(freq) << 21) - 1, uint64_t(~0u)));
            const uint64_t inverse = freq > 1 ? ~uint64_t(0) / freq + 1 : 0;

            next = done + 32 + lane < total ? op[total - 33 - done - lane] : 1;

            for (uint32_t t = 0; t < steps; t++) {
                const uint32_t f = __shfl_sync(~0u, e, t);
                const uint32_t outLimit = __shfl_sync(~0u, limit, t);
                const uint64_t m = __shfl_sync(~0u, inverse, t);

                if (x > outLimit) {  // ransNeedsOutput()
                    if (lane == 0) {
                        end[-1 - int64_t(count)] = static_cast<uint16_t>(x);
                    }

                    count++;
                    x >>= 16;
                }

                // ransEncode()
                const uint32_t opFreq = f & 0xffff;
                const uint32_t q = opFreq > 1 ? static_cast<uint32_t>(__umul64hi(x, m)) : x;
                x = (q << kProbBits) + (x - q * opFreq) + (f >> 16);
            }
        }

        if (lane == 0) {
            wordCounts[g] = count;
            states[g] = x;
        }
    }

    __device__ __forceinline__ void put16(uint8_t* p, uint32_t v) {
        *reinterpret_cast<uint16_t*>(p) = static_cast<uint16_t>(v);  // every offset is even
    }

    // A block per segment: the final state, then the words, at tableOffsets[g].
    __global__ void placeTables(const uint16_t* words, const uint64_t* opOffsets, const uint32_t* wordCounts,
        const uint32_t* states, const uint64_t* tableOffsets, uint8_t* out) {
        const uint32_t g = blockIdx.x, count = wordCounts[g];
        uint8_t* const to = out + tableOffsets[g];
        const uint16_t* const from = words + opOffsets[g + 1] - count;

        if (threadIdx.x == 0) {
            put16(to, states[g]);
            put16(to + 2, states[g] >> 16);
        }

        for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
            put16(to + 4 + 2 * i, from[i]);
        }
    }

    constexpr uint32_t kEncodeWarps = 4;  // slices per block of the stream encoder

    // Where encodeSlices() puts the streams. Into scratch, slice s's words end at (s + 1) * capacity and those beyond
    // capacity are dropped, its final states go to states[32 s + lane], and words[s] counts all its words; the
    // tokens add up in tokens. Into out, given words[s], the stream goes to offsets[s].
    struct SliceOutput {
        uint32_t* words;
        uint16_t* scratch;
        uint32_t capacity;
        uint32_t* states;
        unsigned long long* tokens;
        const uint64_t* offsets;
        uint8_t* out;
    };

    // A warp per slice, as encodeSlice(): lanes walk their runs backwards over the decoder's (step, op, lane) order;
    // the words a step's lanes push go out in descending lane order. Into out with kWrite, else into scratch.
    template <bool kWrite>
    __global__ void __launch_bounds__(kEncodeWarps * 32) encodeSlices(const uint8_t* __restrict__ bwt, uint32_t n,
        uint32_t slices, int logSegment, int logSlice, const TableData* __restrict__ tables,
        const DeviceTable* __restrict__ codes, SliceOutput to) {
        const uint32_t sl = blockIdx.x * kEncodeWarps + threadIdx.x / 32, j = threadIdx.x & 31;

        if (sl >= slices) {
            return;
        }

        const uint32_t begin = sl << logSlice;
        const uint32_t size = min(n - begin, 1u << logSlice), lb = laneBytes(size);
        const uint32_t laneBegin = begin + min(j * lb, size), laneEnd = begin + min((j + 1) * lb, size);
        const uint32_t g = sl >> (logSegment - logSlice);
        const TableData& table = tables[g];
        const DeviceTable& code = codes[g];
        const uint32_t k = table.contexts, kl = lengthContexts(k);

        uint32_t count = laneEnd > laneBegin;

        for (uint32_t p = laneBegin + 1; p < laneEnd; p++) {
            count += __ldg(bwt + p) != __ldg(bwt + p - 1);
        }

        const uint32_t steps = __reduce_max_sync(~0u, count);
        uint8_t* const stream = kWrite ? to.out + to.offsets[sl] : nullptr;
        const uint32_t total = kWrite ? to.words[sl] : 0;
        uint16_t* const scratchEnd = kWrite ? nullptr : to.scratch + size_t(sl + 1) * to.capacity;
        uint32_t x = kRansLow, pos = laneEnd, pushed = 0;

        for (uint32_t t = steps; t-- > 0;) {
            const bool active = t < count;
            uint32_t rank = 0, context = 0, length = 0;

            if (active) {
                const uint8_t byte = __ldg(bwt + pos - 1);
                uint32_t start = pos - 1;

                while (start > laneBegin && __ldg(bwt + start - 1) == byte) {
                    start--;
                }

                rank = table.rankOf[byte];
                context = start == laneBegin ? k - 1 : contextOf(table.rankOf[__ldg(bwt + start - 1)], k);
                length = pos - start;
                pos = start;
            }

            const LengthCode lc = lengthCode(active ? length : 1);

            for (int op = 2; op >= 0; op--) {
                uint32_t freq, cum;

                if (op == 0) {
                    const uint32_t e = code.symbolCode[context][rank];
                    freq = e & 0xffff;
                    cum = e >> 16;
                } else if (op == 1) {
                    const uint32_t e = code.lengthCode[contextOf(rank, kl)][lc.code];
                    freq = e & 0xffff;
                    cum = e >> 16;
                } else {
                    freq = 1u << (kProbBits - lc.extraBits);
                    cum = lc.extra << (kProbBits - lc.extraBits);
                }

                const bool out16 = active && ransNeedsOutput(x, freq);
                const uint32_t mask = __ballot_sync(~0u, out16);

                if (out16) {
                    const uint32_t index = pushed + __popc(j == 31 ? 0 : mask >> (j + 1));

                    if constexpr (kWrite) {
                        put16(stream + kLanes * 4 + 2 * size_t(total - 1 - index), x);
                    } else if (index < to.capacity) {
                        scratchEnd[-1 - int64_t(index)] = static_cast<uint16_t>(x);
                    }

                    x >>= 16;
                }

                pushed += __popc(mask);

                if (active) {
                    x = ransEncode(x, freq, cum);
                }
            }
        }

        if constexpr (kWrite) {
            put16(stream + 4 * j, x);
            put16(stream + 4 * j + 2, x >> 16);
        } else {
            const uint32_t sum = __reduce_add_sync(~0u, count);

            to.states[sl * kLanes + j] = x;

            if (j == 0) {
                to.words[sl] = pushed;
                atomicAdd(to.tokens, static_cast<unsigned long long>(sum));
            }
        }
    }

    // A block per slice: moves a stream encodeSlices() put into scratch whole to offsets[s].
    __global__ void placeSlices(SliceOutput to) {
        const uint32_t sl = blockIdx.x, total = to.words[sl];
        uint8_t* const stream = to.out + to.offsets[sl];
        const uint16_t* const from = to.scratch + size_t(sl + 1) * to.capacity - total;

        if (threadIdx.x < kLanes) {
            const uint32_t x = to.states[sl * kLanes + threadIdx.x];
            put16(stream + 4 * threadIdx.x, x);
            put16(stream + 4 * threadIdx.x + 2, x >> 16);
        }

        for (uint32_t i = threadIdx.x; i < total; i += blockDim.x) {
            put16(stream + kLanes * 4 + 2 * i, from[i]);
        }
    }
}
