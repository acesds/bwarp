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
#include "container.h"
#include "format.h"
#include "table.h"

using namespace bwarp;

namespace {
    // Records the table's decisions, then encodes them backwards with rANS: the final state, then the words.
    class TableWriter {
     public:
        uint32_t bit(Probability& prob, uint32_t b) {
            push(b, prob.p);
            prob.p = static_cast<uint16_t>(adaptProbability(prob.p, b));
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

    // A table, with the frequencies both sides derive from it.
    struct Model : TableData {
        uint32_t symbolFreq[kMaxContexts][256] = {};
        uint32_t symbolCum[kMaxContexts][257] = {};
        uint32_t lengthFreq[kMaxLengthContexts][kMaxLengthCodes] = {};
        uint32_t lengthCum[kMaxLengthContexts][kMaxLengthCodes + 1] = {};
    };

    void toFrequencies(const uint16_t* q, uint32_t size, uint32_t mantissa, uint32_t* freq, uint32_t* cum) {
        uint32_t counts[256];

        for (uint32_t i = 0; i < size; i++) {
            counts[i] = dequantizeCount(q[i], mantissa);
        }

        normalizeFrequencies(counts, size, freq);
        cum[0] = 0;

        for (uint32_t i = 0; i < size; i++) {
            cum[i + 1] = cum[i] + freq[i];
        }
    }

    void finishModel(Model& model) {
        for (uint32_t c = 0; c < model.contexts; c++) {
            if (model.symbolUsed[c]) {
                toFrequencies(model.symbolQ[c], model.alphabet, model.mantissa, model.symbolFreq[c], model.symbolCum[c]);
            }
        }

        for (uint32_t c = 0; c < lengthContexts(model.contexts); c++) {
            if (model.lengthUsed[c]) {
                toFrequencies(model.lengthQ[c], model.lengthCodes, model.mantissa, model.lengthFreq[c], model.lengthCum[c]);
            }
        }
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
        auto lengthCounts = std::make_unique<uint32_t[]>(kMaxLengthContexts * kMaxLengthCodes);
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
        }

        for (uint32_t c = 0; c < lengthContexts(contexts); c++) {
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
        Predictor predictor;
        codeTable(writer, model, *models, predictor);
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

        Predictor predictor;

        if (!codeTable(reader, model, *models, predictor) || !reader.finished()) {
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
        uint8_t length[kMaxLengthContexts][kProbScale];

        explicit SlotTables(const Model& model) {
            for (uint32_t c = 0; c < model.contexts; c++) {
                for (uint32_t i = 0; model.symbolUsed[c] && i < model.alphabet; i++) {
                    std::fill(symbol[c] + model.symbolCum[c][i], symbol[c] + model.symbolCum[c][i + 1], uint8_t(i));
                }
            }

            for (uint32_t c = 0; c < lengthContexts(model.contexts); c++) {
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
    Container container;

    if (!readContainer(data, size, container)) {
        return false;
    }

    const Layout& layout = container.layout;
    const std::vector<uint64_t>& segmentOffsets = container.segmentOffsets;
    const std::vector<uint64_t>& sliceOffsets = container.sliceOffsets;
    const uint32_t lengthCodes = lengthCodeCount(layout.logSlice);
    auto model = std::make_unique<Model>();
    std::unique_ptr<SlotTables> slots;

    bwt.assign(layout.n, 0);
    primary = container.primary;

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
