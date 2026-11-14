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
    int logSample = 0;    // the inverse BWT's chains start every 2^logSample text positions; 0: chooseLogSample()
};

// The automatic sample interval: the smallest 2^k, k in [10, 16], with at most 2^13 chains or whose samples
// take at most 1/128 of the tables and streams.
constexpr int kAutoMinLogSample = 10;
constexpr int kAutoMaxLogSample = 16;
constexpr int kAutoLogChains = 13;
constexpr uint64_t kAutoSampleShare = 128;

int chooseLogSample(uint32_t n, uint64_t codedBytes, int recordedLogSample);

// -log2(f / 2^11) for f in [1, 2^11], in 1/256 bits (entry 0 unused): what the encoders weigh tables by, the CPU
// and the GPU one alike.
const uint32_t* encodingCosts();

struct EncodeStats {
    int64_t indexBytes = 0;  // header, samples and offsets
    int64_t tableBytes = 0;
    int64_t streamBytes = 0;
    int64_t tokens = 0;
    int logSample = 0;
};

// samples: sampleCount(n, recordedLogSample) rows, those of text positions j * 2^recordedLogSample for j >= 1,
// as gpuBwt() records them; the file keeps every 2^(params.logSample - recordedLogSample)-th. False if params or
// rows are out of range.
bool encodeBwt(const uint8_t* bwt, int32_t n, int32_t primary, const uint32_t* samples, int recordedLogSample,
    const EncodeParams& params, std::vector<uint8_t>& out, EncodeStats* stats = nullptr);

// For other encoders: whether encodeBwt() accepts the input, and the header, samples and offsets it writes before
// the tables and streams, given their offsets relative to the first table (the last slice offset is their size).
// Sets s's logSample, indexBytes, tableBytes and streamBytes.
bool validEncodeInput(int32_t n, int32_t primary, const uint32_t* samples, int recordedLogSample,
    const EncodeParams& params);
void writeIndex(int32_t n, int32_t primary, const uint32_t* samples, int recordedLogSample, const EncodeParams& params,
    const std::vector<uint64_t>& segmentOffsets, const std::vector<uint64_t>& sliceOffsets, std::vector<uint8_t>& out,
    EncodeStats& s);

// False on malformed input.
bool decodeBwt(const uint8_t* data, size_t size, std::vector<uint8_t>& bwt, int32_t& primary,
    std::vector<uint32_t>* samples = nullptr);
