/*
 * encode.cuh - The encoder kernels: choosing and coding each segment's table, then the slices' rANS streams.
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

    // For each run of slices [first, end), calls start(byte, laneStart, previous byte) at its first byte and
    // finish(byte, length) at its last, in no particular order. The block's threads take 16 bytes each; a run's
    // end finds its start in the chunk or, through a block-wide max-scan, in an earlier one. bwt is padded to
    // 16 bytes.
    template <class Start, class Finish>
    __device__ void forEachRun(const uint8_t* __restrict__ bwt, uint32_t n, int logSlice, uint32_t first, uint32_t end,
        Start start, Finish finish) {
        using BlockScan = cub::BlockScan<uint32_t, kChooseThreads>;
        __shared__ typename BlockScan::TempStorage scan;

        const uint32_t stop = min(n, end << logSlice);
        uint32_t carry = 0;  // 1 + the last run start so far

        for (uint32_t base = first << logSlice; base < stop; base += kChooseThreads * 16) {
            const uint32_t at = base + threadIdx.x * 16;
            uint32_t starts = 0, laneStarts = 0, ends = 0, words[4] = {};

            if (at < stop) {
                const uint4 chunk = *reinterpret_cast<const uint4*>(bwt + at);
                const uint32_t sliceBegin = at >> logSlice << logSlice;  // a chunk never crosses a slice
                const uint32_t lb = laneBytes(min(n - sliceBegin, 1u << logSlice));
                const uint32_t into = (at - sliceBegin) % lb;
                uint32_t previous = at > 0 ? uint32_t(__ldg(bwt + at - 1)) << 24 : 0;  // in the top byte

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

            const uint32_t last = starts ? at + 31 - __clz(starts) + 1 : 0;
            uint32_t open, aggregate;

            BlockScan(scan).ExclusiveScan(last, open, 0u, [](uint32_t a, uint32_t b) { return max(a, b); }, aggregate);
            open = max(open, carry);
            carry = max(carry, aggregate);
            __syncthreads();

            for (uint32_t bits = starts; bits != 0; bits &= bits - 1) {
                const uint32_t k = __ffs(bits) - 1;
                const uint32_t byte = (words[k / 4] >> (8 * (k % 4))) & 255;
                const bool laneStart = (laneStarts >> k) & 1;
                const uint32_t previous = laneStart ? 0
                                          : k > 0   ? (words[(k - 1) / 4] >> (8 * ((k - 1) % 4))) & 255
                                                    : __ldg(bwt + at - 1);

                start(static_cast<uint8_t>(byte), laneStart, static_cast<uint8_t>(previous));
            }

            for (uint32_t bits = ends; bits != 0; bits &= bits - 1) {
                const uint32_t k = __ffs(bits) - 1;
                const uint32_t before = starts & ((2u << k) - 1);
                const uint32_t runStart = before ? at + 31 - __clz(before) : open - 1;

                finish(static_cast<uint8_t>((words[k / 4] >> (8 * (k % 4))) & 255), at + k + 1 - runStart);
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

    // codeRow()'s models are independent chains: part 0 is the used and nonzero flags, part 1 + r the residuals in
    // residual context r. Their costs add up.
    constexpr uint32_t kRowParts = 1 + kResidualContexts;

    // codeRow()'s cost in one part.
    __device__ void costRowPart(CostCoder& coder, RowModels& models, bool used, const uint16_t* q,
        const uint8_t* predicted, uint32_t size, uint32_t skip, uint32_t part) {
        if (part == 0) {
            coder.bit(models.used, used);
        }

        if (!used) {
            return;
        }

        for (uint32_t i = 0; i < size; i++) {
            if (i == skip) {
                continue;
            }

            const uint32_t pq = predicted[i];

            if (part == 0) {
                coder.bit(models.nonzero[pq < kZeroContexts - 1 ? pq : kZeroContexts - 1], q[i] != 0);
            } else if (q[i] != 0) {
                const uint32_t residualContext = min(floorLog2(pq + 1), kResidualContexts - 1);

                if (residualContext == part - 1) {
                    const int32_t base = static_cast<int32_t>(pq > 1 ? pq : 1);
                    codeUint(coder, models.residual[residualContext], zigzag(int32_t(q[i]) - base));
                }
            }
        }
    }

    // Copies one part's models.
    __device__ void copyRowPart(RowModels& to, const RowModels& from, uint32_t part) {
        if (part == 0) {
            to.used = from.used;

            for (uint32_t i = 0; i < kZeroContexts; i++) {
                to.nonzero[i] = from.nonzero[i];
            }
        } else {
            to.residual[part - 1] = from.residual[part - 1];
        }
    }

    // Rows c < K - 1 are the same for every K, so they are coded once, in order, and each K adds its last row: the
    // merged contexts K - 1 and above. Likewise for length rows. Arrays by K are indexed by log2 K, by length
    // context count by its log2.
    struct ChooseShared {
        uint32_t tokens[256];  // per byte
        uint32_t lengthTotal[kMaxLengthCodes];
        // Per fine context c and rank, the tokens in contexts c and above; the last row stays zero. The same for
        // length contexts by min(rank, 3).
        uint32_t suffix[kFineContexts + 1][256];
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
        __align__(16) uint8_t table[sizeof(TableData)];  // with the shared rows
        __align__(16) uint8_t models[sizeof(TableModels)];
        __align__(16) uint8_t snapshots[sizeof(RowModels) * (kContextChoices + kLengthChoices)];  // the models before
                                                                                                    // each last row
    };

    // Quantized tokens in contexts [c, c + 1) or, merged, [c, ...): hi = row c, lo = row c + 1 or the zero row.
    __device__ __forceinline__ uint32_t rowQ(const uint32_t* hi, const uint32_t* lo, uint32_t i, uint32_t m) {
        return quantizeCount(hi[i] - lo[i], m);
    }

    // The stream cost of a row of size entries with counts hi - lo, coded with the frequencies of q.
    __device__ uint64_t streamBits(const uint16_t* q, uint32_t size, bool used, const uint32_t* hi, const uint32_t* lo,
        uint32_t m, const uint32_t* __restrict__ cost) {
        if (!used) {
            return 0;
        }

        uint32_t counts[256], freqs[256];
        uint64_t bits = 0;

        for (uint32_t i = 0; i < size; i++) {
            counts[i] = dequantizeCount(q[i], m);
        }

        normalizeFrequencies(counts, size, freqs);

        for (uint32_t i = 0; i < size; i++) {
            bits += uint64_t(hi[i] - lo[i]) * __ldg(cost + freqs[i]);
        }

        return bits;
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
    // table's decisions in decisions[g]. cost: encodingCosts().
    __global__ void __launch_bounds__(kChooseThreads, 2) chooseTables(const uint8_t* __restrict__ bwt, uint32_t n,
        int logSegment, int logSlice, uint32_t maxContexts, const uint32_t* __restrict__ cost, TableData* tables,
        uint64_t* decisions) {
        __shared__ ChooseShared s;

        TableData& table = *reinterpret_cast<TableData*>(s.table);
        TableModels& models = *reinterpret_cast<TableModels*>(s.models);
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
            bwt, n, logSlice, first, end, [&](uint8_t byte, bool, uint8_t) { atomicAdd(&s.tokens[byte], 1u); },
            [&](uint8_t, uint32_t length) { atomicAdd(&s.lengthTotal[lengthCode(length).code], 1u); });

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

            for (uint32_t e = tid; e < (kFineContexts + 1) * 256; e += kChooseThreads) {
                (&s.suffix[0][0])[e] = 0;
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
                bwt, n, logSlice, first, end,
                [&](uint8_t byte, bool laneStart, uint8_t previous) {
                    const uint32_t context =
                        laneStart ? kMaxContexts : min(uint32_t(table.rankOf[previous]), kMaxContexts - 1);
                    atomicAdd(&s.suffix[context][table.rankOf[byte]], 1u);
                },
                [&](uint8_t byte, uint32_t length) {
                    const uint32_t rank = table.rankOf[byte];
                    atomicAdd(&s.lengthSuffix[min(rank, kMaxLengthContexts - 1)][lengthCode(length).code], 1u);
                });

            __syncthreads();

            for (uint32_t r = tid; r < 256; r += kChooseThreads) {
                for (uint32_t c = kFineContexts - 1, sum = 0; c != UINT32_MAX; c--) {
                    sum += s.suffix[c][r];
                    s.suffix[c][r] = sum;
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
            const uint32_t* zero = s.suffix[kFineContexts];
            const uint32_t* lengthZero = s.lengthSuffix[kMaxLengthContexts];

            for (uint32_t b = tid; b < 256; b += kChooseThreads) {
                table.byteQ[b] = s.byteQ[b];
                s.rankCount[b] = b < alphabet ? dequantizeCount(s.byteQ[table.symbols[b]], m) : 0;
            }

            for (uint32_t l = tid; l < kMaxLengthCodes; l += kChooseThreads) {
                const uint32_t q = l < lengthCodes ? quantizeCount(s.lengthTotal[l], m) : 0;

                table.lengthCodeQ[l] = static_cast<uint16_t>(q);
                s.lengthCount[l] = dequantizeCount(q, m);
            }

            for (uint32_t e = tid; e < fineRows * alphabet; e += kChooseThreads) {
                const uint32_t c = e / alphabet, r = e % alphabet;
                table.symbolQ[c][r] = static_cast<uint16_t>(rowQ(s.suffix[c], s.suffix[c + 1], r, m));
            }

            for (uint32_t e = tid; e < choices * alphabet; e += kChooseThreads) {
                const uint32_t ki = e / alphabet, r = e % alphabet;
                s.mergedQ[ki][r] = static_cast<uint16_t>(rowQ(s.suffix[(1u << ki) - 1], zero, r, m));
            }

            for (uint32_t e = tid; e < fineLengthRows * lengthCodes; e += kChooseThreads) {
                const uint32_t c = e / lengthCodes, l = e % lengthCodes;
                table.lengthQ[c][l] = static_cast<uint16_t>(rowQ(s.lengthSuffix[c], s.lengthSuffix[c + 1], l, m));
            }

            for (uint32_t e = tid; e < lengthChoices * lengthCodes; e += kChooseThreads) {
                const uint32_t li = e / lengthCodes, l = e % lengthCodes;
                s.mergedLengthQ[li][l] = static_cast<uint16_t>(rowQ(s.lengthSuffix[(1u << li) - 1], lengthZero, l, m));
            }

            for (uint32_t i = tid; i < sizeof(TableModels) / sizeof(Probability); i += kChooseThreads) {
                reinterpret_cast<Probability*>(&models)[i].p = kProbScale / 2;
            }

            if (tid == 0) {
                table.contexts = 1;
                table.mantissa = m;
                table.lengthCodes = lengthCodes;
                table.alphabet = alphabet;
            }

            __syncthreads();

            // A thread per row's used flag, then per sum of counts.
            if (uint32_t job = tid; job < fineRows) {
                bool used = false;

                for (uint32_t r = 0; r < alphabet; r++) {
                    used = used || table.symbolQ[job][r] > 0;
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
            const uint32_t maxQ = maxCountQ(m);

            // The serial parts: warp p codes part p of the shared rows, symbols on lane 0 and lengths on lane 1.
            // The last warp codes the counts, then costs the stream per row.
            static_assert(kChooseThreads / 32 > kRowParts);

            if (const uint32_t part = tid / 32; part < kRowParts && tid % 32 < 2) {
                const bool symbol = tid % 32 == 0;
                const uint32_t rows = symbol ? fineRows : fineLengthRows, last = symbol ? choices : lengthChoices;
                const uint32_t size = symbol ? alphabet : lengthCodes, stride = symbol ? 256 : kMaxLengthCodes;
                const uint16_t* q = symbol ? table.symbolQ[0] : table.lengthQ[0];
                const uint8_t* predicted = symbol ? s.predicted[0] : s.lengthPredicted[0];
                const bool* used = symbol ? table.symbolUsed : table.lengthUsed;
                RowModels& rowModels = models.rows[symbol ? 0 : 1];
                RowModels* snapshot = snapshots + (symbol ? 0 : kContextChoices);
                RowCost* prefix = symbol ? s.prefix[part] : s.lengthPrefix[part];
                CostCoder coder(cost);

                for (uint32_t c = 0, ki = 0;; c++) {
                    if (ki < last && c == (1u << ki) - 1) {
                        copyRowPart(snapshot[ki], rowModels, part);
                        prefix[ki] = {coder.bits(), coder.decisions()};
                        ki++;
                    }

                    if (c == rows) {
                        break;
                    }

                    costRowPart(coder, rowModels, used[c], q + c * stride, predicted + c * stride, size,
                        symbol ? c : kMaxLengthCodes, part);
                }
            } else if (tid == kChooseThreads - 32) {
                CostCoder coder(cost);

                codeCounts(coder, table, models);
                s.counts = {coder.bits(), coder.decisions()};
            } else if (uint32_t job = tid - (kChooseThreads - 31); tid > kChooseThreads - 32 && job < fineRows) {
                s.rowBits[job] = streamBits(table.symbolQ[job], alphabet, table.symbolUsed[job], s.suffix[job],
                    s.suffix[job + 1], m, cost);
            } else if (tid > kChooseThreads - 32 && (job -= fineRows) < choices) {
                s.mergedBits[job] =
                    streamBits(s.mergedQ[job], alphabet, s.mergedUsed[job], s.suffix[(1u << job) - 1], zero, m, cost);
            } else if (tid > kChooseThreads - 32 && (job -= choices) < fineLengthRows) {
                s.lengthRowBits[job] = streamBits(table.lengthQ[job], lengthCodes, table.lengthUsed[job],
                    s.lengthSuffix[job], s.lengthSuffix[job + 1], m, cost);
            } else if (tid > kChooseThreads - 32 && (job -= fineLengthRows) < lengthChoices) {
                s.mergedLengthBits[job] = streamBits(s.mergedLengthQ[job], lengthCodes, s.mergedLengthUsed[job],
                    s.lengthSuffix[(1u << job) - 1], lengthZero, m, cost);
            }

            __syncthreads();

            // Each last row from its snapshot, a thread each, spread over the warps.
            if (const uint32_t job = tid % 32 * (kChooseThreads / 32) + tid / 32; job < choices + lengthChoices) {
                const bool symbol = job < choices;
                const uint32_t ki = symbol ? job : job - choices;
                CostCoder coder(cost);

                if (symbol) {
                    codeRow(coder, snapshots[ki], s.mergedUsed[ki], s.mergedQ[ki], alphabet, 256,
                        [&](uint32_t i) { return uint32_t(s.mergedPredicted[ki][i]); }, maxQ);
                    s.merged[ki] = {coder.bits(), coder.decisions()};
                } else {
                    codeRow(coder, snapshots[kContextChoices + ki], s.mergedLengthUsed[ki], s.mergedLengthQ[ki],
                        lengthCodes, kMaxLengthCodes, [&](uint32_t i) { return uint32_t(s.mergedLengthPredicted[ki][i]); },
                        maxQ);
                    s.mergedLength[ki] = {coder.bits(), coder.decisions()};
                }
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
                uint32_t* to = reinterpret_cast<uint32_t*>(tables + g);
                TableData& out = tables[g];

                for (uint32_t i = tid; i < sizeof(TableData) / 4; i += kChooseThreads) {
                    to[i] = from[i];
                }

                __syncthreads();

                for (uint32_t r = tid; r < alphabet; r += kChooseThreads) {
                    out.symbolQ[k - 1][r] = s.mergedQ[ki][r];
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

            const uint32_t* const op = ops + opOffsets[g];
            uint16_t* const end = words + opOffsets[g + 1];
            uint32_t x = kRansLow, count = 0;

            for (uint32_t i = recorder.count; i-- > 0;) {
                const uint32_t freq = op[i] & 0xffff, cum = op[i] >> 16;

                if (ransNeedsOutput(x, freq)) {
                    end[-1 - int64_t(count++)] = static_cast<uint16_t>(x);
                    x >>= 16;
                }

                x = ransEncode(x, freq, cum);
            }

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

    // A warp per slice, as encodeSlice(): lanes walk their runs backwards over the decoder's (step, op, lane) order;
    // the words a step's lanes push go out in descending lane order. Without kWrite, counts each slice's words in
    // words[s] and the tokens; with it, writes the stream at offsets[s], given words[s].
    template <bool kWrite>
    __global__ void __launch_bounds__(kEncodeWarps * 32) encodeSlices(const uint8_t* __restrict__ bwt, uint32_t n,
        uint32_t slices, int logSegment, int logSlice, const TableData* __restrict__ tables,
        const DeviceTable* __restrict__ codes, uint32_t* words, const uint64_t* offsets, uint8_t* out,
        unsigned long long* tokens) {
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
        uint8_t* const stream = kWrite ? out + offsets[sl] : nullptr;
        const uint32_t total = kWrite ? words[sl] : 0;
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
                    if constexpr (kWrite) {
                        const uint32_t index = pushed + __popc(j == 31 ? 0 : mask >> (j + 1));
                        put16(stream + kLanes * 4 + 2 * size_t(total - 1 - index), x);
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

            if (j == 0) {
                words[sl] = pushed;
                atomicAdd(tokens, static_cast<unsigned long long>(sum));
            }
        }
    }
}
