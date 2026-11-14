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

    inline bool validParams(int logSegment, int logSlice, int contexts) {
        return logSlice >= kMinLogSlice && logSlice <= kMaxLogSlice && logSegment >= logSlice &&
               logSegment <= kMaxLogSegment && contexts >= 1 && contexts <= kMaxContexts;
    }

    struct Layout {
        uint32_t n = 0;
        int logSegment = 0;
        int logSlice = 0;
        uint32_t segments = 0;
        uint32_t slices = 0;
        uint32_t slicesPerSegment = 0;

        Layout() = default;

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

    struct Container {
        Layout layout;
        int32_t primary = 0;
        std::vector<uint64_t> segmentOffsets;  // tables, then the end of the last
        std::vector<uint64_t> sliceOffsets;    // streams, then the end of the last
    };

    // False unless the header is valid and the offsets ascend over exactly the rest of the data.
    inline bool readContainer(const uint8_t* data, size_t size, Container& container) {
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

        container.layout = layout;
        container.primary = static_cast<int32_t>(get32(data + 12));
        container.segmentOffsets.resize(layout.segments + 1);
        container.sliceOffsets.resize(layout.slices + 1);

        const uint8_t* index = data + kHeaderBytes;

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
