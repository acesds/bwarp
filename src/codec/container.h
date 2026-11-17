/*
 * container.h - The bwarp container: header, segment and slice layout, offset index (docs/format.md, Layout).
 * Host only; shared by the CPU codec and the GPU decoder.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "format.h"

namespace bwarp {
    inline uint32_t get32(const uint8_t* p) {
        return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
    }

    inline uint64_t get64(const uint8_t* p) {
        return uint64_t(get32(p)) | uint64_t(get32(p + 4)) << 32;
    }

    inline void put32(uint8_t* p, uint32_t v) {
        for (int i = 0; i < 4; i++) {
            p[i] = static_cast<uint8_t>(v >> (8 * i));
        }
    }

    inline void put64(uint8_t* p, uint64_t v) {
        for (int i = 0; i < 8; i++) {
            p[i] = static_cast<uint8_t>(v >> (8 * i));
        }
    }

    inline bool validParams(int logSegment, int logSlice, int contexts, int logSample) {
        return logSlice >= kMinLogSlice && logSlice <= kMaxLogSlice && logSegment >= logSlice &&
               logSegment <= kMaxLogSegment && contexts >= 1 && contexts <= kMaxContexts &&
               logSample >= kMinLogSample && logSample <= kMaxLogSample;
    }

    struct Layout {
        uint32_t n = 0;
        int logSegment = 0;
        int logSlice = 0;
        int logSample = 0;
        uint32_t samples = 0;
        uint32_t segments = 0;
        uint32_t slices = 0;
        uint32_t slicesPerSegment = 0;

        Layout() = default;

        Layout(uint32_t n, int logSegment, int logSlice, int logSample)
            : n(n),
              logSegment(logSegment),
              logSlice(logSlice),
              logSample(logSample),
              samples(sampleCount(n, logSample)),
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

        size_t samplesAt() const {
            return kHeaderBytes;
        }

        size_t offsetsAt() const {
            return kHeaderBytes + size_t(samples) * 4;
        }

        size_t indexBytes() const {
            return offsetsAt() + (size_t(segments) + 1 + slices + 1) * 8;
        }
    };

    struct Container {
        Layout layout;
        int32_t primary = 0;
        uint32_t checksum = 0;  // CRC-32C of the text
        std::vector<uint32_t> samples;
        std::vector<uint64_t> segmentOffsets;  // tables, then the end of the last
        std::vector<uint64_t> sliceOffsets;    // streams, then the end of the last
    };

    // Rows of the n + 1 sorted rotations: the sentinel's is 0, so the others are in [1, n].
    inline bool validRow(uint32_t row, uint32_t n) {
        return n == 0 ? row == 0 : row >= 1 && row <= n;
    }

    struct BlockHeader {
        Layout layout;
        uint32_t primary = 0;
        uint32_t checksum = 0;
    };

    // The fixed kHeaderBytes at a block's start: false unless they are valid. Its index ends at
    // layout.indexBytes(), with the block's size, the last slice offset, in its last 8 bytes.
    inline bool readBlockHeader(const uint8_t* data, BlockHeader& header) {
        if (get32(data) != kMagic || data[4] != kVersion || data[5] != kTransformBwt || data[9] != 0 ||
            data[10] != 0 || data[11] != 0) {
            return false;
        }

        const int logSegment = data[6], logSlice = data[7], logSample = data[8];
        const uint32_t n = get32(data + 12);

        header.primary = get32(data + 16);
        header.checksum = get32(data + 20);

        if (!validParams(logSegment, logSlice, 1, logSample) || n > kMaxBlockBytes || !validRow(header.primary, n)) {
            return false;
        }

        header.layout = Layout(n, logSegment, logSlice, logSample);

        return true;
    }

    // False unless the header is valid, the rows in range and the offsets ascend over exactly the rest of the data.
    inline bool readContainer(const uint8_t* data, size_t size, Container& container) {
        BlockHeader header;

        if (size < kHeaderBytes || !readBlockHeader(data, header)) {
            return false;
        }

        const Layout& layout = header.layout;
        const uint32_t n = layout.n;

        if (size < layout.indexBytes()) {
            return false;
        }

        container.layout = layout;
        container.primary = static_cast<int32_t>(header.primary);
        container.checksum = header.checksum;
        container.samples.resize(layout.samples);
        container.segmentOffsets.resize(layout.segments + 1);
        container.sliceOffsets.resize(layout.slices + 1);

        const uint8_t* index = data + layout.samplesAt();

        for (uint32_t& sample : container.samples) {
            sample = get32(index);
            index += 4;

            if (!validRow(sample, n)) {
                return false;
            }
        }

        for (uint64_t& offset : container.segmentOffsets) {
            offset = get64(index);
            index += 8;
        }

        for (uint64_t& offset : container.sliceOffsets) {
            offset = get64(index);
            index += 8;
        }

        const auto& tables = container.segmentOffsets;
        const auto& streams = container.sliceOffsets;

        return tables.front() == layout.indexBytes() && tables.back() == streams.front() && streams.back() == size &&
               std::is_sorted(tables.begin(), tables.end()) && std::is_sorted(streams.begin(), streams.end());
    }
}
