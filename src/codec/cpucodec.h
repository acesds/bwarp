/*
 * cpucodec.h - The reference encoder and decoder.
 *
 * Sequential, on the CPU. The GPU kernels must produce the same bytes.
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
    int logSegment = 15;    // frequency tables per 2^logSegment bytes
    int logSlice = 15;      // one 32-lane rANS stream per 2^logSlice bytes
    int contexts = 16;      // at most; each segment's table picks its own
    int logSample = 0;      // the inverse BWT's chains start every 2^logSample text positions; 0: chooseLogSample()
    bool lastBlock = true;  // the file's last block: its header says so
};

// The automatic sample interval: the smallest 2^k, k in [10, 16], with at most 2^13 chains or
// whose samples take at most 1/128 of the tables and streams.
constexpr int kAutoMinLogSample = 10;
constexpr int kAutoMaxLogSample = 16;
constexpr int kAutoLogChains = 13;
constexpr uint64_t kAutoSampleShare = 128;

int chooseLogSample(uint32_t n, uint64_t codedBytes, int recordedLogSample);

// -log2(f / 2^11) for f in [1, 2^11], in 1/256 bits (entry 0 unused)
// The coding costs are the same for the CPU and GPU encoders, so they share this table.
const uint32_t* encodingCosts();

struct EncodeStats {
    int64_t indexBytes = 0;   // header, samples and offsets
    int64_t tableBytes = 0;
    int64_t streamBytes = 0;
    int64_t tokens = 0;       // runs coded, each a symbol and a length
    int logSample = 0;        // the one used, chosen if params.logSample is 0
};

// Encodes one block of at most kMaxBlockBytes: the BWT, and checksum, the text's CRC-32C, which is only stored.
// samples are the rows gpuBwt() records every 2^recordedLogSample positions; the block keeps every
// 2^(logSample - recordedLogSample)-th. Returns false if n, params or rows are out of range.
bool encodeBwt(const uint8_t* bwt, int32_t n, int32_t primary, uint32_t checksum, const uint32_t* samples,
    int recordedLogSample, const EncodeParams& params, std::vector<uint8_t>& out, EncodeStats* stats = nullptr);

// For other encoders: returns whether encodeBwt() accepts the input.
bool validEncodeInput(int32_t n, int32_t primary, const uint32_t* samples, int recordedLogSample, const EncodeParams& params);

// For other encoders: writes the header, samples and offsets that precede the tables and streams, given their
// offsets relative to the first table. Sets s's logSample, indexBytes, tableBytes and streamBytes.
void writeIndex(int32_t n, int32_t primary, uint32_t checksum, const uint32_t* samples, int recordedLogSample,
    const EncodeParams& params, const std::vector<uint64_t>& segmentOffsets, const std::vector<uint64_t>& sliceOffsets,
    std::vector<uint8_t>& out, EncodeStats& s);

// Decodes one block to its BWT and primary index. Returns false if the data is malformed.
// The checksum is only passed on: checking it needs the text, which the BWT doesn't give.
bool decodeBwt(const uint8_t* data, size_t size, std::vector<uint8_t>& bwt, int32_t& primary,
    std::vector<uint32_t>* samples = nullptr, uint32_t* checksum = nullptr);
