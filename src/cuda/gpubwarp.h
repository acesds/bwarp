/*
 * gpubwarp.h - The GPU interface: BWT construction, encoding, decoding and decompression.
 *
 * Plain C++, so bwarp builds without CUDA: gpubwarp_stub.cpp then implements it, every call Unavailable.
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
#include "cpucodec.h"

enum class GpuStatus {
    Ok,
    Unavailable,       // no CUDA device, or built without CUDA
    NotEnoughMemory,
    Invalid,           // malformed input
    Failed,            // see gpuLastError()
    ChecksumMismatch,  // decompressed, but the text's CRC-32C isn't the stored one
};

struct GpuDevice {
    char name[256];
    int computeMajor;
    int computeMinor;
    int multiprocessors;
    int64_t l2Bytes;
    int64_t totalBytes;
    int64_t freeBytes;
};

// Describes the first CUDA device. Returns Unavailable if there is none.
GpuStatus gpuQueryDevice(GpuDevice& device);

// Returns the message of the last error.
const char* gpuLastError();

// Allocates page-locked host memory, which transfers to and from the device about twice as fast.
// Returns null on failure.
void* gpuAllocatePinned(size_t bytes);
void gpuFreePinned(void* pointer);

// Device memory left free, so the Windows driver doesn't fall back to host memory.
constexpr int64_t kGpuMemoryReserve = int64_t(256) << 20;

// ---- BWT construction --------------------------------------------------------------------------

struct GpuBwtStats {
    int64_t deviceBytes = 0;     // peak, and below each phase's
    int64_t rankingBytes = 0;
    int64_t classSortBytes = 0;
    int64_t mergeBytes = 0;
    bool pinned = false;         // transfers went through the pinned staging buffers
    double setupSeconds = 0.0;   // creating a temporary context
    double uploadSeconds = 0.0;
    double deviceSeconds = 0.0;  // the phases below
    double rankingSeconds = 0.0;
    double classSortSeconds = 0.0;
    double mergeSeconds = 0.0;
    double bwtSeconds = 0.0;
    double downloadSeconds = 0.0;
    int32_t characterPasses = 0;
    int32_t doublingRounds = 0;
};

// Returns the peak device memory gpuBwt() needs for n bytes (about 10n), or -1 on a CUDA error.
int64_t gpuBwtRequiredBytes(int32_t n);

// Device memory and pinned staging buffers for gpuBwt() and gpuCompress(), reused across calls.
struct GpuBwtContext;

// Creates a context for texts of up to capacity bytes. Only one context per device at a time.
// Returns NotEnoughMemory if gpuBwtRequiredBytes(capacity) + kGpuMemoryReserve exceeds the free memory.
GpuStatus gpuCreateBwtContext(int32_t capacity, GpuBwtContext*& context);

void gpuDestroyBwtContext(GpuBwtContext* context);

// Builds the BWT of text into bwt, which may be text, and sets primary to the row of text position 0.
// With samples, also records the rows of positions j * 2^logSample for j >= 1: sampleCount(n, logSample) of them,
// logSample at least 8. Rows count the sentinel's as row 0.
GpuStatus gpuBwt(GpuBwtContext* context, const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary,
    GpuBwtStats* stats = nullptr, uint32_t* samples = nullptr, int logSample = 0);

// Same, with a temporary context.
GpuStatus gpuBwt(const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, GpuBwtStats* stats = nullptr,
    uint32_t* samples = nullptr, int logSample = 0);

// ---- Decoding ----------------------------------------------------------------------------------

struct GpuDecodeStats {
    int64_t deviceBytes = 0;
    double setupSeconds = 0.0;   // allocating
    double uploadSeconds = 0.0;
    double tableSeconds = 0.0;   // decoding the tables
    double finishSeconds = 0.0;  // their frequencies and slot tables
    double streamSeconds = 0.0;
    double rankSeconds = 0.0;    // gpuDecompress(): the inverse BWT's LF
    double chainSeconds = 0.0;   // and its chains
    double downloadSeconds = 0.0;
    double checkSeconds = 0.0;   // gpuDecompress(): the text's checksum
};

// Decodes one block to its BWT and primary index, with the same result as decodeBwt().
GpuStatus gpuDecode(const uint8_t* data, size_t size, std::vector<uint8_t>& bwt, int32_t& primary,
    GpuDecodeStats* stats = nullptr);

// Decompresses one block to its text: decodes it and inverts the BWT.
// Returns ChecksumMismatch if the text's CRC-32C isn't the stored one.
GpuStatus gpuDecompress(const uint8_t* data, size_t size, std::vector<uint8_t>& text, GpuDecodeStats* stats = nullptr);

// Same, into text of capacity bytes, and sets n to the text's size. Page-locked text downloads faster.
// Returns NotEnoughMemory if the text doesn't fit.
GpuStatus gpuDecompress(const uint8_t* data, size_t size, uint8_t* text, size_t capacity, size_t& n,
    GpuDecodeStats* stats = nullptr);

// Device memory for gpuDecompress(), reserved up front and reused across calls.
struct GpuDecompressContext;

// Returns the device memory needed to decompress a block of size bytes, given its header, or -1 if the header
// is invalid. A context reserving the most any block of a file needs allocates nothing more for that file.
int64_t gpuDecompressRequiredBytes(const uint8_t* header, size_t size);

// Creates a context that reserves reserve bytes of device memory.
// Returns NotEnoughMemory if reserve + kGpuMemoryReserve exceeds the free memory.
GpuStatus gpuCreateDecompressContext(int64_t reserve, GpuDecompressContext*& context);

void gpuDestroyDecompressContext(GpuDecompressContext* context);

// Same as the buffer overload, with a context, which may be null. With storedChecksum, doesn't check the text
// but returns the stored CRC-32C there, so the caller can check it while writing the text.
GpuStatus gpuDecompress(GpuDecompressContext* context, const uint8_t* data, size_t size, uint8_t* text,
    size_t capacity, size_t& n, GpuDecodeStats* stats = nullptr, uint32_t* storedChecksum = nullptr);

// ---- Encoding ----------------------------------------------------------------------------------

struct GpuEncodeStats {
    int64_t deviceBytes = 0;
    double setupSeconds = 0.0;   // allocating
    double uploadSeconds = 0.0;
    double chooseSeconds = 0.0;  // choosing the tables
    double tableSeconds = 0.0;   // coding them
    double streamSeconds = 0.0;  // encoding the streams
    double placeSeconds = 0.0;   // placing the tables and streams
    double downloadSeconds = 0.0;
};

// Encodes one block on the GPU, byte for byte as encodeBwt() does.
// Returns Invalid where encodeBwt() returns false.
GpuStatus gpuEncode(const uint8_t* bwt, int32_t n, int32_t primary, uint32_t checksum, const uint32_t* samples,
    int recordedLogSample, const EncodeParams& params, std::vector<uint8_t>& out, EncodeStats* encodeStats = nullptr,
    GpuEncodeStats* stats = nullptr);

// ---- Compression -------------------------------------------------------------------------------

struct GpuCompressStats {
    GpuBwtStats bwt;
    GpuEncodeStats encode;
};

// Compresses one block of text, BWT and encoding all on the GPU, to the same bytes as gpuBwt() then encodeBwt().
// out points into one of the context's two output buffers, used in turn, so it stays valid until the call after
// next. Returns Invalid where encodeBwt() returns false.
GpuStatus gpuCompress(GpuBwtContext* context, const uint8_t* text, int32_t n, const EncodeParams& params,
    const uint8_t*& out, size_t& size, EncodeStats* encodeStats = nullptr, GpuCompressStats* stats = nullptr);
