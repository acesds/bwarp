/*
 * cpucodec.cpp - The reference encoder and decoder of the bwarp format.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "cpucodec.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include "format.h"

using namespace bwarp;

namespace {
    constexpr uint32_t kMaxLengthCodes = 32;

    static_assert(lengthCode(laneBytes(1u << kMaxLogSlice)).code < kMaxLengthCodes);
    static_assert(lengthExtraBits(lengthCode(laneBytes(1u << kMaxLogSlice)).code) <= kProbBits,
        "a run's extra bits must fit one rANS op");

    struct Layout {
        uint32_t n;
        int logSegment;
        int logSlice;
        uint32_t segments;
        uint32_t slices;
        uint32_t slicesPerSegment;

        Layout(uint32_t n, int logSegment, int logSlice)
            : n(n),
              logSegment(logSegment),
              logSlice(logSlice),
              segments(static_cast<uint32_t>((uint64_t(n) + (uint64_t(1) << logSegment) - 1) >> logSegment)),
              slices(static_cast<uint32_t>((uint64_t(n) + (uint64_t(1) << logSlice) - 1) >> logSlice)),
              slicesPerSegment(1u << (logSegment - logSlice)) {}

        uint32_t sliceBegin(uint32_t s) const {
            return s << logSlice;
        }

        uint32_t sliceEnd(uint32_t s) const {
            return static_cast<uint32_t>(std::min<uint64_t>(uint64_t(s + 1) << logSlice, n));
        }

        uint32_t firstSlice(uint32_t segment) const {
            return segment * slicesPerSegment;
        }

        uint32_t endSlice(uint32_t segment) const {
            return std::min(slices, (segment + 1) * slicesPerSegment);
        }

        size_t indexBytes() const {
            return kHeaderBytes + (size_t(segments) + 1 + slices + 1) * 8;
        }
    };

    bool validParams(int logSegment, int logSlice, int contexts) {
        return logSlice >= kMinLogSlice && logSlice <= kMaxLogSlice && logSegment >= logSlice &&
               logSegment <= kMaxLogSegment && contexts >= 1 && contexts <= kMaxContexts;
    }

    void put32(uint8_t* p, uint32_t v) {
        for (int i = 0; i < 4; i++) {
            p[i] = static_cast<uint8_t>(v >> (8 * i));
        }
    }

    void put64(uint8_t* p, uint64_t v) {
        for (int i = 0; i < 8; i++) {
            p[i] = static_cast<uint8_t>(v >> (8 * i));
        }
    }

    uint32_t get32(const uint8_t* p) {
        return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
    }

    uint64_t get64(const uint8_t* p) {
        return uint64_t(get32(p)) | uint64_t(get32(p + 4)) << 32;
    }

    struct Probability {
        uint32_t p = kProbScale / 2;
    };

    // Records the table's decisions, then encodes them backwards with rANS: the final state, then the words.
    class TableWriter {
     public:
        uint32_t bit(Probability& prob, uint32_t b) {
            push(b, prob.p);
            prob.p = adaptProbability(prob.p, b);
            return b;
        }

        uint32_t raw(uint32_t v, int bits) {
            for (int i = bits - 1; i >= 0; i--) {
                push((v >> i) & 1, kProbScale / 2);
            }

            return v;
        }

        void finish(std::vector<uint8_t>& out) const {
            uint32_t x = kRansLow;
            std::vector<uint16_t> words;

            for (size_t i = m_ops.size(); i-- > 0;) {
                if (ransNeedsOutput(x, m_ops[i].freq)) {
                    words.push_back(static_cast<uint16_t>(x));
                    x >>= 16;
                }

                x = ransEncode(x, m_ops[i].freq, m_ops[i].cum);
            }

            const size_t at = out.size();
            out.resize(at + 4 + 2 * words.size());
            put32(&out[at], x);

            for (size_t i = 0; i < words.size(); i++) {
                const uint16_t w = words[words.size() - 1 - i];
                out[at + 4 + 2 * i] = static_cast<uint8_t>(w);
                out[at + 5 + 2 * i] = static_cast<uint8_t>(w >> 8);
            }
        }

     private:
        struct Op {
            uint16_t freq;
            uint16_t cum;
        };

        void push(uint32_t b, uint32_t p) {
            m_ops.push_back(b ? Op{static_cast<uint16_t>(kProbScale - p), static_cast<uint16_t>(p)}
                              : Op{static_cast<uint16_t>(p), 0});
        }

        std::vector<Op> m_ops;
    };

    class TableReader {
     public:
        TableReader(const uint8_t* begin, const uint8_t* end) : m_w(end), m_end(end) {
            if (end - begin >= 4 && (end - begin) % 2 == 0) {
                m_x = get32(begin);
                m_w = begin + 4;
                m_ok = m_x >= kRansLow;
            }
        }

        uint32_t bit(Probability& prob, uint32_t) {
            const uint32_t b = decide(prob.p);
            prob.p = adaptProbability(prob.p, b);
            return b;
        }

        uint32_t raw(uint32_t, int bits) {
            uint32_t v = 0;

            for (int i = 0; i < bits; i++) {
                v = v << 1 | decide(kProbScale / 2);
            }

            return v;
        }

        bool finished() const {
            return m_ok && m_w == m_end && m_x == kRansLow;
        }

     private:
        uint32_t decide(uint32_t p) {
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

    constexpr uint32_t kMaxUintBits = 16;

    struct UintModels {
        Probability length[kMaxUintBits];
        Probability top[kMaxUintBits];
    };

    // Exp-Golomb on v + 1: its bit length in unary, then the bits below the top one. v < 2^15 - 1.
    template <class Coder>
    uint32_t codeUint(Coder& coder, UintModels& models, uint32_t v) {
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

    uint32_t zigzag(int32_t v) {
        return v >= 0 ? 2 * static_cast<uint32_t>(v) : 2 * static_cast<uint32_t>(-v) - 1;
    }

    int32_t unzigzag(uint32_t u) {
        return u & 1 ? -static_cast<int32_t>((u + 1) / 2) : static_cast<int32_t>(u / 2);
    }

    struct Run {
        uint8_t byte;
        bool laneStart;
        uint32_t length;
    };

    void tokenize(const uint8_t* bwt, uint32_t begin, uint32_t end, std::vector<Run>& runs) {
        for (uint32_t p = begin; p < end;) {
            uint32_t q = p + 1;

            while (q < end && bwt[q] == bwt[p]) {
                q++;
            }

            runs.push_back({bwt[p], p == begin, q - p});
            p = q;
        }
    }

    // Lane j of a slice [begin, end).
    void laneRange(uint32_t begin, uint32_t end, int j, uint32_t& laneBegin, uint32_t& laneEnd) {
        const uint32_t size = end - begin, lb = laneBytes(size);
        laneBegin = begin + std::min(uint32_t(j) * lb, size);
        laneEnd = begin + std::min(uint32_t(j + 1) * lb, size);
    }

    // A segment's table. The q arrays hold quantized counts; both sides derive the ranks and frequencies from them.
    struct Model {
        uint32_t contexts = 1;
        uint32_t mantissa = 2;
        uint32_t lengthCodes = 0;
        uint16_t byteQ[256] = {};  // tokens per byte
        uint16_t lengthCodeQ[kMaxLengthCodes] = {};  // tokens per length code
        uint32_t alphabet = 0;
        uint8_t symbols[256] = {};  // in rank order
        uint8_t rankOf[256] = {};
        bool symbolUsed[kMaxContexts] = {};
        bool lengthUsed[kMaxContexts] = {};
        uint16_t symbolQ[kMaxContexts][256] = {};  // by rank
        uint16_t lengthQ[kMaxContexts][kMaxLengthCodes] = {};
        uint32_t symbolCount[kMaxContexts][256] = {};  // reconstructed from q
        uint32_t lengthCount[kMaxContexts][kMaxLengthCodes] = {};
        uint32_t symbolFreq[kMaxContexts][256] = {};
        uint32_t symbolCum[kMaxContexts][257] = {};
        uint32_t lengthFreq[kMaxContexts][kMaxLengthCodes] = {};
        uint32_t lengthCum[kMaxContexts][kMaxLengthCodes + 1] = {};
    };

    // Ranks by quantized token count, descending, ties by byte.
    void rankSymbols(Model& model) {
        model.alphabet = 0;

        for (uint32_t b = 0; b < 256; b++) {
            if (model.byteQ[b] > 0) {
                model.symbols[model.alphabet++] = static_cast<uint8_t>(b);
            }
        }

        std::stable_sort(model.symbols, model.symbols + model.alphabet,
            [&](uint8_t a, uint8_t b) { return model.byteQ[a] > model.byteQ[b]; });

        for (uint32_t r = 0; r < model.alphabet; r++) {
            model.rankOf[model.symbols[r]] = static_cast<uint8_t>(r);
        }
    }

    void toFrequencies(const uint32_t* counts, uint32_t size, uint32_t* freq, uint32_t* cum) {
        normalizeFrequencies(counts, size, freq);
        cum[0] = 0;

        for (uint32_t i = 0; i < size; i++) {
            cum[i + 1] = cum[i] + freq[i];
        }
    }

    void finishModel(Model& model) {
        for (uint32_t c = 0; c < model.contexts; c++) {
            if (model.symbolUsed[c]) {
                toFrequencies(model.symbolCount[c], model.alphabet, model.symbolFreq[c], model.symbolCum[c]);
            }

            if (model.lengthUsed[c]) {
                toFrequencies(model.lengthCount[c], model.lengthCodes, model.lengthFreq[c], model.lengthCum[c]);
            }
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
    bool codeMarginal(Coder& coder, MarginalModels& models, uint16_t* q, uint32_t size, uint32_t maxQ) {
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

    // One context's q, each relative to the prediction total * weights[i] / denominator. Entry skip cannot occur.
    template <class Coder>
    bool codeRow(Coder& coder, RowModels& models, bool& used, uint16_t* q, uint32_t* counts, uint32_t size,
        uint32_t skip, const uint64_t* weights, uint64_t total, uint64_t denominator, uint32_t mantissa, uint32_t maxQ) {
        used = coder.bit(models.used, used) != 0;
        std::fill(counts, counts + size, 0u);

        if (!used) {
            std::fill(q, q + size, uint16_t(0));
            return true;
        }

        bool any = false;

        for (uint32_t i = 0; i < size; i++) {
            if (i == skip) {
                q[i] = 0;
                continue;
            }

            const uint64_t predicted = denominator > 0 ? (total * weights[i] + denominator / 2) / denominator : 0;
            const uint32_t pq =
                quantizeCount(static_cast<uint32_t>(std::min<uint64_t>(predicted, 1u << kMaxLogSegment)), mantissa);

            if (!coder.bit(models.nonzero[std::min(pq, kZeroContexts - 1)], q[i] != 0)) {
                q[i] = 0;
                continue;
            }

            const int32_t base = static_cast<int32_t>(std::max(pq, 1u));
            UintModels& residual = models.residual[std::min(floorLog2(pq + 1), kResidualContexts - 1)];
            const int32_t value = base + unzigzag(codeUint(coder, residual, zigzag(int32_t(q[i]) - base)));

            if (value < 1 || value > static_cast<int32_t>(maxQ)) {
                return false;
            }

            q[i] = static_cast<uint16_t>(value);
            counts[i] = dequantizeCount(q[i], mantissa);
            any = true;
        }

        return any;
    }

    // The table: K, the count precision, the tokens per byte and per length code, then each context's counts
    // predicted from these as if symbol and context were independent. A symbol never follows itself in a lane.
    template <class Coder>
    bool codeTable(Coder& coder, Model& model, TableModels& models) {
        model.contexts = coder.raw(model.contexts - 1, 4) + 1;
        model.mantissa = coder.raw(model.mantissa, 2);

        const uint32_t maxQ = quantizeCount(1u << kMaxLogSegment, model.mantissa);
        if (!codeMarginal(coder, models.marginal[0], model.byteQ, 256, maxQ) ||
            !codeMarginal(coder, models.marginal[1], model.lengthCodeQ, model.lengthCodes, maxQ)) {
            return false;
        }

        rankSymbols(model);

        if (model.alphabet == 0) {
            return false;
        }

        const uint32_t k = model.contexts;
        uint64_t rankCount[256] = {}, lengthCount[kMaxLengthCodes] = {}, tokens = 0, lengthTokens = 0;

        for (uint32_t r = 0; r < model.alphabet; r++) {
            rankCount[r] = dequantizeCount(model.byteQ[model.symbols[r]], model.mantissa);
            tokens += rankCount[r];
        }

        for (uint32_t l = 0; l < model.lengthCodes; l++) {
            lengthCount[l] = dequantizeCount(model.lengthCodeQ[l], model.mantissa);
            lengthTokens += lengthCount[l];
        }

        // Tokens whose own symbol, or whose predecessor's, falls into context c.
        uint64_t contextTokens[kMaxContexts] = {}, lengthContextTokens[kMaxContexts] = {};

        for (uint32_t r = 0; r < model.alphabet; r++) {
            contextTokens[contextOf(r, k)] += rankCount[r];
            lengthContextTokens[contextOf(r, lengthContexts(k))] += rankCount[r];
        }

        RowModels* const rows = models.rows;

        for (uint32_t c = 0; c < k; c++) {
            const bool own = c < k - 1;

            if (!codeRow(coder, rows[0], model.symbolUsed[c], model.symbolQ[c], model.symbolCount[c], model.alphabet,
                    own ? c : 256, rankCount, contextTokens[c], own ? tokens - rankCount[c] : tokens, model.mantissa,
                    maxQ)) {
                return false;
            }
        }

        for (uint32_t c = 0; c < lengthContexts(k); c++) {
            if (!codeRow(coder, rows[1], model.lengthUsed[c], model.lengthQ[c], model.lengthCount[c], model.lengthCodes,
                    kMaxLengthCodes, lengthCount, lengthContextTokens[c], lengthTokens, model.mantissa, maxQ)) {
                return false;
            }
        }

        return true;
    }

    void buildModel(const std::vector<Run>& runs, uint32_t contexts, uint32_t mantissa, uint32_t lengthCodes,
        Model& model) {
        uint32_t tokenCount[256] = {};

        for (const Run& run : runs) {
            tokenCount[run.byte]++;
        }

        model.contexts = contexts;
        model.mantissa = mantissa;
        model.lengthCodes = lengthCodes;

        for (uint32_t b = 0; b < 256; b++) {
            model.byteQ[b] = static_cast<uint16_t>(quantizeCount(tokenCount[b], mantissa));
        }

        rankSymbols(model);

        auto symbolCounts = std::make_unique<uint32_t[]>(kMaxContexts * 256);
        auto lengthCounts = std::make_unique<uint32_t[]>(kMaxContexts * kMaxLengthCodes);
        uint32_t lengthTotal[kMaxLengthCodes] = {};
        uint32_t previous = 0;

        for (const Run& run : runs) {
            const uint32_t rank = model.rankOf[run.byte];
            const uint32_t context = run.laneStart ? contexts - 1 : contextOf(previous, contexts);
            const uint32_t code = lengthCode(run.length).code;

            symbolCounts[context * 256 + rank]++;
            lengthCounts[contextOf(rank, lengthContexts(contexts)) * kMaxLengthCodes + code]++;
            lengthTotal[code]++;
            previous = rank;
        }

        for (uint32_t l = 0; l < lengthCodes; l++) {
            model.lengthCodeQ[l] = static_cast<uint16_t>(quantizeCount(lengthTotal[l], mantissa));
        }

        for (uint32_t c = 0; c < contexts; c++) {
            for (uint32_t i = 0; i < model.alphabet; i++) {
                model.symbolQ[c][i] = static_cast<uint16_t>(quantizeCount(symbolCounts[c * 256 + i], mantissa));
                model.symbolUsed[c] = model.symbolUsed[c] || model.symbolQ[c][i] > 0;
            }

            for (uint32_t l = 0; l < lengthCodes; l++) {
                model.lengthQ[c][l] = static_cast<uint16_t>(quantizeCount(lengthCounts[c * kMaxLengthCodes + l], mantissa));
                model.lengthUsed[c] = model.lengthUsed[c] || model.lengthQ[c][l] > 0;
            }
        }
    }

    // Also derives the frequencies, as the decoder will.
    void writeModel(Model& model, std::vector<uint8_t>& out) {
        TableWriter writer;

        auto models = std::make_unique<TableModels>();
        codeTable(writer, model, *models);
        writer.finish(out);
        finishModel(model);
    }

    // The bits the runs' symbols and length codes take with the model.
    double streamBits(const Model& model, const std::vector<Run>& runs) {
        static const auto cost = [] {
            std::vector<double> c(kProbScale + 1);

            for (uint32_t f = 1; f <= kProbScale; f++) {
                c[f] = kProbBits - std::log2(double(f));
            }

            return c;
        }();

        const uint32_t k = model.contexts;
        uint32_t previous = 0;
        double bits = 0;

        for (const Run& run : runs) {
            const uint32_t rank = model.rankOf[run.byte];
            const uint32_t context = run.laneStart ? k - 1 : contextOf(previous, k);

            bits += cost[model.symbolFreq[context][rank]];
            bits += cost[model.lengthFreq[contextOf(rank, lengthContexts(k))][lengthCode(run.length).code]];
            previous = rank;
        }

        return bits;
    }

    // Writes the segment's table for the context count up to maxContexts and the count precision that minimize
    // the table and stream bits together.
    void chooseModel(const std::vector<Run>& runs, uint32_t maxContexts, uint32_t lengthCodes, Model& best,
        std::vector<uint8_t>& out) {
        constexpr uint32_t kContextCounts[] = {1, 2, 4, 8, 16};
        constexpr uint32_t kMantissas = 2;  // finer counts did not pay on Silesia
        auto model = std::make_unique<Model>();
        std::vector<uint8_t> table, bestTable;
        double bestBits = 0;

        for (const uint32_t contexts : kContextCounts) {
            for (uint32_t mantissa = 0; contexts <= maxContexts && mantissa < kMantissas; mantissa++) {
                *model = Model{};
                buildModel(runs, contexts, mantissa, lengthCodes, *model);
                table.clear();
                writeModel(*model, table);

                const double bits = 8.0 * table.size() + streamBits(*model, runs);

                if (bestTable.empty() || bits < bestBits) {
                    best = *model;
                    bestTable.swap(table);
                    bestBits = bits;
                }
            }
        }

        out.insert(out.end(), bestTable.begin(), bestTable.end());
    }

    bool readModel(const uint8_t* begin, const uint8_t* end, uint32_t lengthCodes, Model& model) {
        TableReader reader(begin, end);

        model.lengthCodes = lengthCodes;

        auto models = std::make_unique<TableModels>();

        if (!codeTable(reader, model, *models) || !reader.finished()) {
            return false;
        }

        finishModel(model);

        return true;
    }

    struct Token {
        uint8_t rank;
        uint8_t context;
        uint32_t length;
    };

    // Appends the slice's stream: the 32 final encoder states, then the 16-bit words in decoding order.
    // Encoding runs backwards over the decoder's (step, op, lane) order, so the words are reversed at the end.
    void encodeSlice(const uint8_t* bwt, uint32_t begin, uint32_t end, const Model& model, std::vector<uint8_t>& out,
        int64_t& tokenTotal) {
        std::vector<Token> lanes[kLanes];
        std::vector<Run> runs;
        size_t steps = 0;

        for (int j = 0; j < kLanes; j++) {
            uint32_t laneBegin, laneEnd;

            laneRange(begin, end, j, laneBegin, laneEnd);
            runs.clear();
            tokenize(bwt, laneBegin, laneEnd, runs);

            uint32_t previous = 0;

            for (const Run& run : runs) {
                const uint32_t rank = model.rankOf[run.byte];
                const uint32_t context = run.laneStart ? model.contexts - 1 : contextOf(previous, model.contexts);

                lanes[j].push_back({static_cast<uint8_t>(rank), static_cast<uint8_t>(context), run.length});
                previous = rank;
            }

            steps = std::max(steps, lanes[j].size());
            tokenTotal += static_cast<int64_t>(lanes[j].size());
        }

        uint32_t x[kLanes];
        std::vector<uint16_t> words;

        std::fill(x, x + kLanes, kRansLow);

        for (size_t t = steps; t-- > 0;) {
            for (int op = 2; op >= 0; op--) {
                for (int j = kLanes - 1; j >= 0; j--) {
                    if (t >= lanes[j].size()) {
                        continue;
                    }

                    const Token& token = lanes[j][t];
                    const LengthCode length = lengthCode(token.length);
                    uint32_t freq, cum;

                    if (op == 0) {
                        freq = model.symbolFreq[token.context][token.rank];
                        cum = model.symbolCum[token.context][token.rank];
                    } else if (op == 1) {
                        const uint32_t c = contextOf(token.rank, lengthContexts(model.contexts));
                        freq = model.lengthFreq[c][length.code];
                        cum = model.lengthCum[c][length.code];
                    } else {
                        freq = 1u << (kProbBits - length.extraBits);
                        cum = length.extra << (kProbBits - length.extraBits);
                    }

                    if (ransNeedsOutput(x[j], freq)) {
                        words.push_back(static_cast<uint16_t>(x[j]));
                        x[j] >>= 16;
                    }

                    x[j] = ransEncode(x[j], freq, cum);
                }
            }
        }

        std::reverse(words.begin(), words.end());

        const size_t at = out.size();
        out.resize(at + kLanes * 4 + words.size() * 2);

        for (int j = 0; j < kLanes; j++) {
            put32(&out[at + 4 * j], x[j]);
        }

        for (size_t i = 0; i < words.size(); i++) {
            out[at + kLanes * 4 + 2 * i] = static_cast<uint8_t>(words[i]);
            out[at + kLanes * 4 + 2 * i + 1] = static_cast<uint8_t>(words[i] >> 8);
        }
    }

    // Slot -> index tables of one model, for decoding.
    struct SlotTables {
        uint8_t symbol[kMaxContexts][kProbScale];
        uint8_t length[kMaxContexts][kProbScale];

        explicit SlotTables(const Model& model) {
            for (uint32_t c = 0; c < model.contexts; c++) {
                for (uint32_t i = 0; model.symbolUsed[c] && i < model.alphabet; i++) {
                    std::fill(symbol[c] + model.symbolCum[c][i], symbol[c] + model.symbolCum[c][i + 1], uint8_t(i));
                }

                for (uint32_t i = 0; model.lengthUsed[c] && i < model.lengthCodes; i++) {
                    std::fill(length[c] + model.lengthCum[c][i], length[c] + model.lengthCum[c][i + 1], uint8_t(i));
                }
            }
        }
    };

    bool decodeSlice(const uint8_t* stream, const uint8_t* streamEnd, uint32_t begin, uint32_t end, const Model& model,
        const SlotTables& slots, uint8_t* bwt) {
        if (streamEnd - stream < kLanes * 4 || (streamEnd - stream) % 2 != 0) {
            return false;
        }

        uint32_t x[kLanes], pos[kLanes], laneEnd[kLanes], context[kLanes], rank[kLanes], code[kLanes];
        bool active[kLanes];
        const uint8_t* w = stream + kLanes * 4;

        for (int j = 0; j < kLanes; j++) {
            x[j] = get32(stream + 4 * j);
            laneRange(begin, end, j, pos[j], laneEnd[j]);
            context[j] = model.contexts - 1;

            if (x[j] < kRansLow) {
                return false;
            }
        }

        const auto renormalize = [&](uint32_t& state) {
            if (state < kRansLow) {
                if (w == streamEnd) {
                    return false;
                }

                state = state << 16 | uint32_t(w[0]) | uint32_t(w[1]) << 8;
                w += 2;
            }

            return true;
        };

        for (;;) {
            bool any = false;

            for (int j = 0; j < kLanes; j++) {
                active[j] = pos[j] < laneEnd[j];
                any = any || active[j];
            }

            if (!any) {
                break;
            }

            for (int j = 0; j < kLanes; j++) {
                if (!active[j]) {
                    continue;
                }

                const uint32_t c = context[j];

                if (!model.symbolUsed[c]) {
                    return false;
                }

                rank[j] = slots.symbol[c][x[j] & (kProbScale - 1)];
                x[j] = ransDecodeAdvance(x[j], model.symbolFreq[c][rank[j]], model.symbolCum[c][rank[j]]);

                if (!renormalize(x[j])) {
                    return false;
                }
            }

            for (int j = 0; j < kLanes; j++) {
                if (!active[j]) {
                    continue;
                }

                const uint32_t c = contextOf(rank[j], lengthContexts(model.contexts));

                if (!model.lengthUsed[c]) {
                    return false;
                }

                code[j] = slots.length[c][x[j] & (kProbScale - 1)];
                x[j] = ransDecodeAdvance(x[j], model.lengthFreq[c][code[j]], model.lengthCum[c][code[j]]);

                if (!renormalize(x[j])) {
                    return false;
                }
            }

            for (int j = 0; j < kLanes; j++) {
                if (!active[j]) {
                    continue;
                }

                const uint32_t bits = lengthExtraBits(code[j]);
                const uint32_t extra = (x[j] & (kProbScale - 1)) >> (kProbBits - bits);
                const uint32_t length = lengthBase(code[j]) + extra;

                x[j] = ransDecodeAdvance(x[j], 1u << (kProbBits - bits), extra << (kProbBits - bits));

                if (!renormalize(x[j]) || length > laneEnd[j] - pos[j]) {
                    return false;
                }

                memset(bwt + pos[j], model.symbols[rank[j]], length);
                pos[j] += length;
                context[j] = contextOf(rank[j], model.contexts);
            }
        }

        return w == streamEnd && std::all_of(x, x + kLanes, [](uint32_t s) { return s == kRansLow; });
    }
}

bool encodeBwt(const uint8_t* bwt, int32_t n, int32_t primary, const EncodeParams& params, std::vector<uint8_t>& out,
    EncodeStats* stats) {
    EncodeStats local;
    EncodeStats& s = stats ? *stats : local;

    s = EncodeStats{};

    if (n < 0 || !validParams(params.logSegment, params.logSlice, params.contexts)) {
        return false;
    }

    const Layout layout(static_cast<uint32_t>(n), params.logSegment, params.logSlice);
    const uint32_t lengthCodes = lengthCodeCount(params.logSlice);
    std::vector<uint64_t> segmentOffsets(layout.segments + 1), sliceOffsets(layout.slices + 1);
    std::vector<std::unique_ptr<Model>> models(layout.segments);
    std::vector<Run> runs;

    out.assign(layout.indexBytes(), 0);
    put32(&out[0], kMagic);
    out[4] = kVersion;
    out[5] = kTransformBwt;
    out[6] = static_cast<uint8_t>(params.logSegment);
    out[7] = static_cast<uint8_t>(params.logSlice);
    put32(&out[8], static_cast<uint32_t>(n));
    put32(&out[12], static_cast<uint32_t>(primary));

    for (uint32_t g = 0; g < layout.segments; g++) {
        runs.clear();

        for (uint32_t sl = layout.firstSlice(g); sl < layout.endSlice(g); sl++) {
            for (int j = 0; j < kLanes; j++) {
                uint32_t laneBegin, laneEnd;

                laneRange(layout.sliceBegin(sl), layout.sliceEnd(sl), j, laneBegin, laneEnd);
                tokenize(bwt, laneBegin, laneEnd, runs);
            }
        }

        models[g] = std::make_unique<Model>();
        segmentOffsets[g] = out.size();
        chooseModel(runs, static_cast<uint32_t>(params.contexts), lengthCodes, *models[g], out);
    }

    segmentOffsets[layout.segments] = out.size();

    for (uint32_t g = 0; g < layout.segments; g++) {
        for (uint32_t sl = layout.firstSlice(g); sl < layout.endSlice(g); sl++) {
            sliceOffsets[sl] = out.size();
            encodeSlice(bwt, layout.sliceBegin(sl), layout.sliceEnd(sl), *models[g], out, s.tokens);
        }
    }

    sliceOffsets[layout.slices] = out.size();

    uint8_t* index = &out[kHeaderBytes];

    for (const uint64_t offset : segmentOffsets) {
        put64(index, offset);
        index += 8;
    }

    for (const uint64_t offset : sliceOffsets) {
        put64(index, offset);
        index += 8;
    }

    s.indexBytes = static_cast<int64_t>(layout.indexBytes());
    s.tableBytes = static_cast<int64_t>(segmentOffsets[layout.segments] - segmentOffsets[0]);
    s.streamBytes = static_cast<int64_t>(sliceOffsets[layout.slices] - sliceOffsets[0]);

    return true;
}

bool decodeBwt(const uint8_t* data, size_t size, std::vector<uint8_t>& bwt, int32_t& primary) {
    if (size < kHeaderBytes || get32(data) != kMagic || data[4] != kVersion || data[5] != kTransformBwt) {
        return false;
    }

    const int logSegment = data[6], logSlice = data[7];
    const uint32_t n = get32(data + 8);

    if (!validParams(logSegment, logSlice, 1) || n > INT32_MAX) {
        return false;
    }

    const Layout layout(n, logSegment, logSlice);

    if (size < layout.indexBytes()) {
        return false;
    }

    std::vector<uint64_t> segmentOffsets(layout.segments + 1), sliceOffsets(layout.slices + 1);
    const uint8_t* index = data + kHeaderBytes;

    for (uint64_t& offset : segmentOffsets) {
        offset = get64(index);
        index += 8;
    }

    for (uint64_t& offset : sliceOffsets) {
        offset = get64(index);
        index += 8;
    }

    if (segmentOffsets.front() != layout.indexBytes() || segmentOffsets.back() != sliceOffsets.front() ||
        sliceOffsets.back() != size || !std::is_sorted(segmentOffsets.begin(), segmentOffsets.end()) ||
        !std::is_sorted(sliceOffsets.begin(), sliceOffsets.end())) {
        return false;
    }

    const uint32_t lengthCodes = lengthCodeCount(logSlice);
    auto model = std::make_unique<Model>();
    std::unique_ptr<SlotTables> slots;

    bwt.assign(n, 0);
    primary = static_cast<int32_t>(get32(data + 12));

    for (uint32_t g = 0; g < layout.segments; g++) {
        *model = Model{};

        if (!readModel(data + segmentOffsets[g], data + segmentOffsets[g + 1], lengthCodes, *model)) {
            return false;
        }

        slots = std::make_unique<SlotTables>(*model);

        for (uint32_t sl = layout.firstSlice(g); sl < layout.endSlice(g); sl++) {
            if (!decodeSlice(data + sliceOffsets[sl], data + sliceOffsets[sl + 1], layout.sliceBegin(sl),
                    layout.sliceEnd(sl), *model, *slots, bwt.data())) {
                return false;
            }
        }
    }

    return true;
}
