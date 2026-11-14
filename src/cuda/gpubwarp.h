/*
 * gpubwarp.h - The GPU interface. Plain C++, so bwarp builds without CUDA (see gpubwarp_stub.cpp).
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>

enum class GpuStatus {
    Ok,
    Unavailable,
    NotEnoughMemory,
    Failed,  // see gpuLastError()
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

// The first CUDA device.
GpuStatus gpuQueryDevice(GpuDevice& device);

const char* gpuLastError();

// Page-locked host memory, or null. Transfers from it skip the staging buffers, at twice the speed.
void* gpuAllocatePinned(size_t bytes);
void gpuFreePinned(void* pointer);

// Left free, so the Windows driver doesn't fall back to host memory.
constexpr int64_t kGpuMemoryReserve = int64_t(256) << 20;

// ---- BWT construction (from sadcs) -------------------------------------------------------------

struct GpuBwtStats {
    int64_t deviceBytes = 0;     // peak; the next three are each phase's
    int64_t rankingBytes = 0;
    int64_t classSortBytes = 0;
    int64_t mergeBytes = 0;
    bool pinned = false;         // through the pinned staging buffers
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

// Peak device memory for n bytes (about 10n), or -1.
int64_t gpuBwtRequiredBytes(int32_t n);

// Device memory (a pool reserved up front) and pinned staging buffers, reused across calls.
// Without one, gpuBwt() creates a temporary context, about 18 ms on enwik8.
struct GpuBwtContext;

// NotEnoughMemory if gpuBwtRequiredBytes(capacity) + kGpuMemoryReserve exceeds the free memory.
// Only one context at a time per device.
GpuStatus gpuCreateBwtContext(int32_t capacity, GpuBwtContext*& context);

void gpuDestroyBwtContext(GpuBwtContext* context);

// The BWT of text plus a sentinel, without the sentinel; primary is the sentinel's row. bwt may be text.
GpuStatus gpuBwt(GpuBwtContext* context, const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, GpuBwtStats* stats = nullptr);
GpuStatus gpuBwt(const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, GpuBwtStats* stats = nullptr);
