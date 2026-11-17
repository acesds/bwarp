/*
 * table.h - Coding of a segment's table (docs/format.md, Tables), shared by the CPU codec and the GPU decoder.
 * Each function is written once for both directions: a Coder's bit() and raw() return the bit they are given
 * when encoding and the decoded bit when decoding.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstdint>
#include "format.h"

namespace bwarp {
    struct Probability {
        uint16_t p = kProbScale / 2;
    };

    class TableReader {
     public:
        BWARP_HOST_DEVICE TableReader(const uint8_t* begin, const uint8_t* end) : m_w(end), m_end(end) {
            if (end - begin >= 4 && (end - begin) % 2 == 0) {
                m_x = uint32_t(begin[0]) | uint32_t(begin[1]) << 8 | uint32_t(begin[2]) << 16 | uint32_t(begin[3]) << 24;
                m_w = begin + 4;
                m_ok = m_x >= kRansLow;
            }
        }

        BWARP_HOST_DEVICE uint32_t bit(Probability& prob, uint32_t) {
            const uint32_t b = decide(prob.p);
            prob.p = static_cast<uint16_t>(adaptProbability(prob.p, b));
            return b;
        }

        BWARP_HOST_DEVICE uint32_t raw(uint32_t, int bits) {
            uint32_t v = 0;

            for (int i = 0; i < bits; i++) {
                v = v << 1 | decide(kProbScale / 2);
            }

            return v;
        }

        BWARP_HOST_DEVICE bool finished() const {
            return m_ok && m_w == m_end && m_x == kRansLow;
        }

     private:
        BWARP_HOST_DEVICE uint32_t decide(uint32_t p) {
            const uint32_t b = (m_x & (kProbScale - 1)) >= p;

            m_x = b ? ransDecodeAdvance(m_x, kProbScale - p, p) : ransDecodeAdvance(m_x, p, 0);

            if (m_x < kRansLow) {
                if (m_w == m_end) {
                    m_ok = false;
                    m_x = kRansLow;
                } else {
                    m_x = m_x << 16 | uint32_t(m_w[0]) | uint32_t(m_w[1]) << 8;
                    m_w += 2;
                }
            }

            return b;
        }

        const uint8_t* m_w;
        const uint8_t* m_end;
        uint32_t m_x = kRansLow;
        bool m_ok = false;
    };

    // Sums what a table's decisions cost, in 1/kCostScale bits, from a table of -log2(f / kProbScale) by f.
    constexpr uint32_t kCostScale = 256;

    class CostCoder {
     public:
        BWARP_HOST_DEVICE explicit CostCoder(const uint32_t* cost) : m_cost(cost) {}

        BWARP_HOST_DEVICE uint32_t bit(Probability& prob, uint32_t b) {
            m_bits += m_cost[b ? kProbScale - prob.p : prob.p];
            m_decisions++;
            prob.p = static_cast<uint16_t>(adaptProbability(prob.p, b));
            return b;
        }

        BWARP_HOST_DEVICE uint32_t raw(uint32_t v, int bits) {
            m_bits += uint64_t(bits) * m_cost[kProbScale / 2];
            m_decisions += static_cast<uint32_t>(bits);
            return v;
        }

        BWARP_HOST_DEVICE uint64_t bits() const {
            return m_bits;
        }

        BWARP_HOST_DEVICE uint32_t decisions() const {
            return m_decisions;
        }

     private:
        const uint32_t* m_cost;
        uint64_t m_bits = 0;
        uint32_t m_decisions = 0;
    };

    constexpr uint32_t kMaxUintBits = 16;

    struct UintModels {
        Probability length[kMaxUintBits];
        Probability top[kMaxUintBits];
    };

    // Exp-Golomb on v + 1: its bit length in unary, then the bits below the top one. v < 2^15 - 1.
    template <class Coder>
    BWARP_HOST_DEVICE uint32_t codeUint(Coder& coder, UintModels& models, uint32_t v) {
        const uint32_t u = v + 1, bits = floorLog2(u);
        uint32_t k = 0;

        while (k < kMaxUintBits - 1 && coder.bit(models.length[k], k < bits)) {
            k++;
        }

        uint32_t r = 1;

        for (uint32_t i = k; i-- > 0;) {
            const uint32_t b = (u >> i) & 1;
            r = r << 1 | (i == k - 1 ? coder.bit(models.top[k], b) : coder.raw(b, 1));
        }

        return r - 1;
    }

    BWARP_HD uint32_t zigzag(int32_t v) {
        return v >= 0 ? 2 * static_cast<uint32_t>(v) : 2 * static_cast<uint32_t>(-v) - 1;
    }

    BWARP_HD int32_t unzigzag(uint32_t u) {
        return u & 1 ? -static_cast<int32_t>((u + 1) / 2) : static_cast<int32_t>(u / 2);
    }

    // What a table holds but its symbol rows, which the GPU encoder keeps as counts while choosing it.
    struct TableHead {
        uint32_t contexts;
        uint32_t mantissa;
        uint32_t lengthCodes;
        uint32_t alphabet;
        uint16_t byteQ[256];  // tokens per byte
        uint16_t lengthCodeQ[kMaxLengthCodes];  // tokens per length code
        uint8_t symbols[256];  // in rank order
        uint8_t rankOf[256];
        bool symbolUsed[kMaxContexts];
        bool lengthUsed[kMaxLengthContexts];
        uint16_t lengthQ[kMaxLengthContexts][kMaxLengthCodes];
    };

    // What a table holds: quantized counts, and the ranks derived from them.
    struct TableData : TableHead {
        uint16_t symbolQ[kMaxContexts][256];  // by rank
    };

    // Ranks by quantized token count, descending, ties by byte.
    BWARP_HOST_DEVICE inline void rankSymbols(TableData& table) {
        uint32_t alphabet = 0;

        for (uint32_t b = 0; b < 256; b++) {
            const uint16_t q = table.byteQ[b];

            if (q == 0) {
                continue;
            }

            uint32_t at = alphabet++;

            for (; at > 0 && table.byteQ[table.symbols[at - 1]] < q; at--) {
                table.symbols[at] = table.symbols[at - 1];
            }

            table.symbols[at] = static_cast<uint8_t>(b);
        }

        table.alphabet = alphabet;

        for (uint32_t r = 0; r < alphabet; r++) {
            table.rankOf[table.symbols[r]] = static_cast<uint8_t>(r);
        }
    }

    constexpr uint32_t kZeroContexts = 32;
    constexpr uint32_t kResidualContexts = 6;

    struct MarginalModels {
        Probability nonzero[2];  // by whether the previous entry was zero
        UintModels value;
    };

    struct RowModels {
        Probability used;
        Probability nonzero[kZeroContexts];  // by predicted q
        UintModels residual[kResidualContexts];  // by predicted q's bit length
    };

    struct TableModels {
        MarginalModels marginal[2];  // bytes, length codes
        RowModels rows[2];  // symbols, lengths
    };

    // Each q as the difference from the previous nonzero one.
    template <class Coder>
    BWARP_HOST_DEVICE bool codeMarginal(Coder& coder, MarginalModels& models, uint16_t* q, uint32_t size, uint32_t maxQ) {
        int32_t previous = 0;
        uint32_t previousZero = 1;

        for (uint32_t i = 0; i < size; i++) {
            if (!coder.bit(models.nonzero[previousZero], q[i] != 0)) {
                q[i] = 0;
                previousZero = 1;
                continue;
            }

            const int32_t value = previous + unzigzag(codeUint(coder, models.value, zigzag(int32_t(q[i]) - previous)));

            if (value < 1 || value > static_cast<int32_t>(maxQ)) {
                return false;
            }

            q[i] = static_cast<uint16_t>(value);
            previous = value;
            previousZero = 0;
        }

        return true;
    }

    // One context's q, each relative to its predicted q, predicted(i). Entry skip cannot occur.
    template <class Coder, class Predicted>
    BWARP_HOST_DEVICE bool codeRow(Coder& coder, RowModels& models, bool& used, uint16_t* q, uint32_t size,
        uint32_t skip, const Predicted& predicted, uint32_t maxQ) {
        used = coder.bit(models.used, used) != 0;

        if (!used) {
            for (uint32_t i = 0; i < size; i++) {
                q[i] = 0;
            }

            return true;
        }

        bool any = false;

        for (uint32_t i = 0; i < size; i++) {
            if (i == skip) {
                q[i] = 0;
                continue;
            }

            const uint32_t pq = predicted(i);
            const uint32_t zeroContext = pq < kZeroContexts - 1 ? pq : kZeroContexts - 1;

            if (!coder.bit(models.nonzero[zeroContext], q[i] != 0)) {
                q[i] = 0;
                continue;
            }

            const int32_t base = static_cast<int32_t>(pq > 1 ? pq : 1);
            const uint32_t residualContext = floorLog2(pq + 1) < kResidualContexts - 1 ? floorLog2(pq + 1)
                                                                                      : kResidualContexts - 1;
            const int32_t value =
                base + unzigzag(codeUint(coder, models.residual[residualContext], zigzag(int32_t(q[i]) - base)));

            if (value < 1 || value > static_cast<int32_t>(maxQ)) {
                return false;
            }

            q[i] = static_cast<uint16_t>(value);
            any = true;
        }

        return any;
    }

    BWARP_HD uint32_t maxCountQ(uint32_t mantissa) {
        return quantizeCount(1u << kMaxLogSegment, mantissa);
    }

    // The head of a table: K, the count precision, the tokens per byte and per length code.
    // table.lengthCodes must be set.
    template <class Coder>
    BWARP_HOST_DEVICE bool codeCounts(Coder& coder, TableData& table, TableModels& models) {
        table.contexts = coder.raw(table.contexts - 1, 4) + 1;
        table.mantissa = coder.raw(table.mantissa, 2);

        const uint32_t maxQ = maxCountQ(table.mantissa);

        return codeMarginal(coder, models.marginal[0], table.byteQ, 256, maxQ) &&
               codeMarginal(coder, models.marginal[1], table.lengthCodeQ, table.lengthCodes, maxQ);
    }

    // What the rows are predicted from: the counts, as if symbol and context were independent.
    struct Predictor {
        uint64_t rankCount[256];
        uint64_t lengthCount[kMaxLengthCodes];
        uint64_t tokens;
        uint64_t lengthTokens;
        uint64_t contextTokens[kMaxContexts];  // tokens whose own symbol, or whose predecessor's, is in context c
        uint64_t lengthContextTokens[kMaxLengthContexts];
    };

    // After rankSymbols().
    BWARP_HOST_DEVICE inline void preparePredictor(const TableData& table, Predictor& p) {
        const uint32_t k = table.contexts, kl = lengthContexts(k);

        p.tokens = 0;
        p.lengthTokens = 0;

        for (uint32_t c = 0; c < kMaxContexts; c++) {
            p.contextTokens[c] = 0;
        }

        for (uint32_t c = 0; c < kMaxLengthContexts; c++) {
            p.lengthContextTokens[c] = 0;
        }

        for (uint32_t r = 0; r < table.alphabet; r++) {
            p.rankCount[r] = dequantizeCount(table.byteQ[table.symbols[r]], table.mantissa);
            p.tokens += p.rankCount[r];
            p.contextTokens[contextOf(r, k)] += p.rankCount[r];
            p.lengthContextTokens[contextOf(r, kl)] += p.rankCount[r];
        }

        for (uint32_t l = 0; l < table.lengthCodes; l++) {
            p.lengthCount[l] = dequantizeCount(table.lengthCodeQ[l], table.mantissa);
            p.lengthTokens += p.lengthCount[l];
        }
    }

    // q of round(total * weight / denominator), at most 2^24.
    BWARP_HD uint32_t predictCount(uint64_t total, uint64_t weight, uint64_t denominator, uint32_t mantissa) {
        uint64_t predicted = denominator > 0 ? (total * weight + denominator / 2) / denominator : 0;
        predicted = predicted < (1u << kMaxLogSegment) ? predicted : (1u << kMaxLogSegment);

        return quantizeCount(static_cast<uint32_t>(predicted), mantissa);
    }

    // The predicted q of entry i in symbol context c, or in length context c.
    BWARP_HOST_DEVICE inline uint32_t predictQ(const TableData& table, const Predictor& p, bool symbol, uint32_t c,
        uint32_t i) {
        uint64_t total, weight, denominator;

        if (symbol) {
            total = p.contextTokens[c];
            weight = p.rankCount[i];
            denominator = c < table.contexts - 1 ? p.tokens - p.rankCount[c] : p.tokens;
        } else {
            total = p.lengthContextTokens[c];
            weight = p.lengthCount[i];
            denominator = p.lengthTokens;
        }

        return predictCount(total, weight, denominator, table.mantissa);
    }

    // The rows, after codeCounts() and rankSymbols(): each context's counts relative to predicted(symbol, c, i).
    // A symbol never follows itself in a lane, so symbol context c < K - 1 has no entry c.
    template <class Coder, class Predicted>
    BWARP_HOST_DEVICE bool codeRows(Coder& coder, TableData& table, TableModels& models, const Predicted& predicted) {
        const uint32_t k = table.contexts, maxQ = maxCountQ(table.mantissa);

        for (uint32_t c = 0; c < k; c++) {
            const auto row = [&](uint32_t i) { return predicted(true, c, i); };

            if (!codeRow(coder, models.rows[0], table.symbolUsed[c], table.symbolQ[c], table.alphabet,
                    c < k - 1 ? c : 256, row, maxQ)) {
                return false;
            }
        }

        for (uint32_t c = 0; c < lengthContexts(k); c++) {
            const auto row = [&](uint32_t i) { return predicted(false, c, i); };

            if (!codeRow(coder, models.rows[1], table.lengthUsed[c], table.lengthQ[c], table.lengthCodes,
                    kMaxLengthCodes, row, maxQ)) {
                return false;
            }
        }

        return true;
    }

    // A whole table, serially. table.lengthCodes must be set.
    template <class Coder>
    BWARP_HOST_DEVICE bool codeTable(Coder& coder, TableData& table, TableModels& models, Predictor& predictor) {
        if (!codeCounts(coder, table, models)) {
            return false;
        }

        rankSymbols(table);

        if (table.alphabet == 0) {
            return false;
        }

        preparePredictor(table, predictor);

        return codeRows(coder, table, models, [&](bool symbol, uint32_t c, uint32_t i) {
            return predictQ(table, predictor, symbol, c, i);
        });
    }
}
