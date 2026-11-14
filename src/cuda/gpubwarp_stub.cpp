/*
 * gpubwarp_stub.cpp - gpubwarp.h without CUDA: every call is unavailable.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "gpubwarp.h"

GpuStatus gpuQueryDevice(GpuDevice&) {
    return GpuStatus::Unavailable;
}

const char* gpuLastError() {
    return "bwarp was built without CUDA (configure with the cuda preset)";
}

void* gpuAllocatePinned(size_t) {
    return nullptr;
}

void gpuFreePinned(void*) {}

int64_t gpuBwtRequiredBytes(int32_t) {
    return 0;
}

GpuStatus gpuCreateBwtContext(int32_t, GpuBwtContext*& context) {
    context = nullptr;
    return GpuStatus::Unavailable;
}

void gpuDestroyBwtContext(GpuBwtContext*) {}

GpuStatus gpuBwt(GpuBwtContext*, const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, GpuBwtStats* stats,
    uint32_t* samples, int logSample) {
    return gpuBwt(text, n, bwt, primary, stats, samples, logSample);
}

GpuStatus gpuBwt(const uint8_t*, int32_t, uint8_t*, int32_t& primary, GpuBwtStats* stats, uint32_t*, int) {
    primary = -1;

    if (stats) {
        *stats = GpuBwtStats{};
    }

    return GpuStatus::Unavailable;
}

GpuStatus gpuDecode(const uint8_t*, size_t, std::vector<uint8_t>&, int32_t& primary, GpuDecodeStats* stats) {
    primary = -1;

    if (stats) {
        *stats = GpuDecodeStats{};
    }

    return GpuStatus::Unavailable;
}

GpuStatus gpuDecompress(const uint8_t*, size_t, std::vector<uint8_t>&, GpuDecodeStats* stats) {
    if (stats) {
        *stats = GpuDecodeStats{};
    }

    return GpuStatus::Unavailable;
}
