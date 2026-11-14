/*
 * cpucodec.h - The reference encoder and decoder of the bwarp format (docs/format.md), sequential on the CPU.
 * The GPU kernels must produce the same bytes.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct EncodeParams {
    int logSegment = 15;  // frequency tables per 2^logSegment bytes
    int logSlice = 15;    // one 32-lane rANS stream per 2^logSlice bytes
    int contexts = 16;    // at most; each segment's table picks its own
};

struct EncodeStats {
    int64_t indexBytes = 0;  // header and offsets
    int64_t tableBytes = 0;
    int64_t streamBytes = 0;
    int64_t tokens = 0;
};

// False if params are out of range.
bool encodeBwt(const uint8_t* bwt, int32_t n, int32_t primary, const EncodeParams& params, std::vector<uint8_t>& out,
    EncodeStats* stats = nullptr);

// False on malformed input.
bool decodeBwt(const uint8_t* data, size_t size, std::vector<uint8_t>& bwt, int32_t& primary);
