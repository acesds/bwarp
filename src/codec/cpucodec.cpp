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

    // LSB first.
    class BitWriter {
     public:
        explicit BitWriter(std::vector<uint8_t>& out) : m_out(out) {}

        void put(uint32_t value, int bits) {
            m_acc |= uint64_t(value) << m_count;
            m_count += bits;

            while (m_count >= 8) {
                m_out.push_back(static_cast<uint8_t>(m_acc));
                m_acc >>= 8;
                m_count -= 8;
            }
        }

        // Elias gamma, v >= 1.
        void gamma(uint32_t v) {
            const int bits = static_cast<int>(floorLog2(v));
            put(0, bits);
            put(1, 1);
            put(v & ((1u << bits) - 1), bits);
        }

        void flush() {
            if (m_count > 0) {
                m_out.push_back(static_cast<uint8_t>(m_acc));
            }

            m_acc = 0;
            m_count = 0;
        }

     private:
        std::vector<uint8_t>& m_out;
        uint64_t m_acc = 0;
        int m_count = 0;
    };

    class BitReader {
     public:
        BitReader(const uint8_t* begin, const uint8_t* end) : m_p(begin), m_end(end) {}

        uint32_t get(int bits) {
            while (m_count < bits) {
                if (m_p == m_end) {
                    m_ok = false;
                    return 0;
                }

                m_acc |= uint64_t(*m_p++) << m_count;
                m_count += 8;
            }

            const uint32_t v = static_cast<uint32_t>(m_acc & ((uint64_t(1) << bits) - 1));
            m_acc >>= bits;
            m_count -= bits;

            return v;
        }

        uint32_t gamma() {
            int bits = 0;

            while (get(1) == 0) {
                if (!m_ok || ++bits > 16) {
                    m_ok = false;
                    return 0;
                }
            }

            return (1u << bits) | get(bits);
        }

        bool ok() const {
            return m_ok;
        }

     private:
        const uint8_t* m_p;
        const uint8_t* m_end;
        uint64_t m_acc = 0;
        int m_count = 0;
        bool m_ok = true;
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

    struct Model {
        uint32_t alphabet = 0;
        uint8_t symbols[256] = {};  // in rank order
        uint8_t rankOf[256] = {};
        uint32_t contexts = 1;
        uint32_t lengthCodes = 0;
        bool symbolUsed[kMaxContexts] = {};
        bool lengthUsed[kMaxContexts] = {};
        uint32_t symbolFreq[kMaxContexts][256] = {};
        uint32_t symbolCum[kMaxContexts][257] = {};
        uint32_t lengthFreq[kMaxContexts][kMaxLengthCodes] = {};
        uint32_t lengthCum[kMaxContexts][kMaxLengthCodes + 1] = {};

        void finish() {
            for (uint32_t c = 0; c < contexts; c++) {
                for (uint32_t i = 0; i < alphabet; i++) {
                    symbolCum[c][i + 1] = symbolCum[c][i] + symbolFreq[c][i];
                }

                for (uint32_t i = 0; i < lengthCodes; i++) {
                    lengthCum[c][i + 1] = lengthCum[c][i] + lengthFreq[c][i];
                }
            }
        }
    };

    void buildModel(const std::vector<Run>& runs, uint32_t contexts, uint32_t lengthCodes, Model& model) {
        uint32_t tokenCount[256] = {};

        for (const Run& run : runs) {
            tokenCount[run.byte]++;
        }

        for (uint32_t b = 0; b < 256; b++) {
            if (tokenCount[b] > 0) {
                model.symbols[model.alphabet++] = static_cast<uint8_t>(b);
            }
        }

        std::stable_sort(model.symbols, model.symbols + model.alphabet,
            [&](uint8_t a, uint8_t b) { return tokenCount[a] > tokenCount[b]; });

        for (uint32_t r = 0; r < model.alphabet; r++) {
            model.rankOf[model.symbols[r]] = static_cast<uint8_t>(r);
        }

        model.contexts = contexts;
        model.lengthCodes = lengthCodes;

        auto symbolCounts = std::make_unique<uint32_t[]>(kMaxContexts * 256);
        auto lengthCounts = std::make_unique<uint32_t[]>(kMaxContexts * kMaxLengthCodes);
        uint32_t previous = 0;

        for (const Run& run : runs) {
            const uint32_t rank = model.rankOf[run.byte];
            const uint32_t context = run.laneStart ? contexts - 1 : contextOf(previous, contexts);

            symbolCounts[context * 256 + rank]++;
            lengthCounts[contextOf(rank, contexts) * kMaxLengthCodes + lengthCode(run.length).code]++;
            previous = rank;
        }

        for (uint32_t c = 0; c < contexts; c++) {
            const uint32_t* const s = &symbolCounts[c * 256];
            const uint32_t* const l = &lengthCounts[c * kMaxLengthCodes];

            model.symbolUsed[c] = std::any_of(s, s + model.alphabet, [](uint32_t x) { return x > 0; });
            model.lengthUsed[c] = std::any_of(l, l + lengthCodes, [](uint32_t x) { return x > 0; });

            if (model.symbolUsed[c]) {
                normalizeFrequencies(s, model.alphabet, model.symbolFreq[c]);
            }

            if (model.lengthUsed[c]) {
                normalizeFrequencies(l, lengthCodes, model.lengthFreq[c]);
            }
        }

        model.finish();
    }

    void writeModel(const Model& model, std::vector<uint8_t>& out) {
        BitWriter bits(out);

        bits.put(model.alphabet - 1, 8);

        for (uint32_t r = 0; r < model.alphabet; r++) {
            bits.put(model.symbols[r], 8);
        }

        bits.put(model.contexts - 1, 4);

        for (uint32_t c = 0; c < model.contexts; c++) {
            bits.put(model.symbolUsed[c], 1);

            for (uint32_t i = 0; model.symbolUsed[c] && i < model.alphabet; i++) {
                bits.gamma(model.symbolFreq[c][i] + 1);
            }
        }

        for (uint32_t c = 0; c < model.contexts; c++) {
            bits.put(model.lengthUsed[c], 1);

            for (uint32_t i = 0; model.lengthUsed[c] && i < model.lengthCodes; i++) {
                bits.gamma(model.lengthFreq[c][i] + 1);
            }
        }

        bits.flush();
    }

    // Reads frequencies summing to kProbScale.
    bool readFrequencies(BitReader& bits, uint32_t size, uint32_t* freqs) {
        uint32_t sum = 0;

        for (uint32_t i = 0; i < size; i++) {
            const uint32_t v = bits.gamma();

            if (v == 0 || v > kProbScale + 1) {
                return false;
            }

            freqs[i] = v - 1;
            sum += v - 1;
        }

        return bits.ok() && sum == kProbScale;
    }

    bool readModel(const uint8_t* begin, const uint8_t* end, uint32_t lengthCodes, Model& model) {
        BitReader bits(begin, end);

        model.alphabet = bits.get(8) + 1;

        for (uint32_t r = 0; r < model.alphabet; r++) {
            model.symbols[r] = static_cast<uint8_t>(bits.get(8));
        }

        model.contexts = bits.get(4) + 1;
        model.lengthCodes = lengthCodes;

        for (uint32_t c = 0; c < model.contexts; c++) {
            model.symbolUsed[c] = bits.get(1) != 0;

            if (model.symbolUsed[c] && !readFrequencies(bits, model.alphabet, model.symbolFreq[c])) {
                return false;
            }
        }

        for (uint32_t c = 0; c < model.contexts; c++) {
            model.lengthUsed[c] = bits.get(1) != 0;

            if (model.lengthUsed[c] && !readFrequencies(bits, lengthCodes, model.lengthFreq[c])) {
                return false;
            }
        }

        model.finish();

        return bits.ok();
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
                        const uint32_t c = contextOf(token.rank, model.contexts);
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

                const uint32_t c = contextOf(rank[j], model.contexts);

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
        buildModel(runs, static_cast<uint32_t>(params.contexts), lengthCodes, *models[g]);
        segmentOffsets[g] = out.size();
        writeModel(*models[g], out);
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
