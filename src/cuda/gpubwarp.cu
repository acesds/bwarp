/*
 * gpubwarp.cu - Host side of the GPU variant.
 *
 * The BWT construction is from sadcs (https://github.com/acesds/sadcs), BWT path only.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "gpubwarp.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <new>
#include <cuda_runtime.h>
#include "classsort.cuh"
#include "container.h"
#include "crc32c.h"
#include "decode.cuh"
#include "encode.cuh"
#include "inverse.cuh"
#include "devicecommon.cuh"
#include "merge.cuh"
#include "samplerank.cuh"

namespace {
    char lastError[512] = "";

    void setError(const char* what, cudaError_t error) {
        snprintf(lastError, sizeof lastError, "%s: %s", what, cudaGetErrorString(error));
    }

    double secondsSince(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    // Transfers through two pinned chunks, double-buffered. Cheaper than pinning the caller's buffers,
    // which costs about 36 ms/GB to register and 11 ms/GB to unregister.
    class Staging {
     public:
        static constexpr size_t kChunk = size_t(8) << 20;

        Staging() = default;
        Staging(const Staging&) = delete;
        Staging& operator=(const Staging&) = delete;

        ~Staging() {
            release();
        }

        bool create() {
            for (int i = 0; i < 2; i++) {
                if (cudaMallocHost(&m_buffer[i], kChunk) != cudaSuccess ||
                    cudaEventCreateWithFlags(&m_done[i], cudaEventDisableTiming) != cudaSuccess) {
                    cudaGetLastError();  // clear it
                    release();
                    return false;
                }
            }

            return true;
        }

        static bool isPinned(const void* host) {
            cudaPointerAttributes attributes;

            if (cudaPointerGetAttributes(&attributes, host) != cudaSuccess) {
                cudaGetLastError();  // clear it
                return false;
            }

            return attributes.type == cudaMemoryTypeHost;
        }

        bool pinned() const {
            return m_buffer[1] != nullptr;
        }

        // Ordered on the default stream: later kernels see the data without a sync.
        cudaError_t upload(void* device, const void* host, size_t bytes) {
            if (!pinned() || isPinned(host)) {
                return cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
            }

            for (size_t k = 0, offset = 0; offset < bytes; k++, offset += kChunk) {
                const size_t length = std::min(kChunk, bytes - offset);
                const int b = static_cast<int>(k & 1);

                GPUSA_TRY(cudaEventSynchronize(m_done[b]));  // its previous DMA is done
                memcpy(m_buffer[b], static_cast<const uint8_t*>(host) + offset, length);
                GPUSA_TRY(cudaMemcpyAsync(static_cast<uint8_t*>(device) + offset, m_buffer[b], length,
                    cudaMemcpyHostToDevice, 0));
                GPUSA_TRY(cudaEventRecord(m_done[b], 0));
            }

            return cudaSuccess;
        }

        cudaError_t download(void* host, const void* device, size_t bytes) {
            if (!pinned() || isPinned(host)) {
                return cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost);
            }

            const size_t chunks = (bytes + kChunk - 1) / kChunk;
            const auto issue = [&](size_t k) -> cudaError_t {
                const size_t offset = k * kChunk;
                const int b = static_cast<int>(k & 1);

                GPUSA_TRY(cudaMemcpyAsync(m_buffer[b], static_cast<const uint8_t*>(device) + offset,
                    std::min(kChunk, bytes - offset), cudaMemcpyDeviceToHost, 0));

                return cudaEventRecord(m_done[b], 0);
            };

            if (chunks > 0) {
                GPUSA_TRY(issue(0));
            }

            for (size_t k = 0; k < chunks; k++) {
                // Chunk k+1 goes into the buffer copied out in the previous iteration.
                if (k + 1 < chunks) {
                    GPUSA_TRY(issue(k + 1));
                }

                const int b = static_cast<int>(k & 1);
                const size_t offset = k * kChunk;

                GPUSA_TRY(cudaEventSynchronize(m_done[b]));
                memcpy(static_cast<uint8_t*>(host) + offset, m_buffer[b], std::min(kChunk, bytes - offset));
            }

            return cudaSuccess;
        }

     private:
        uint8_t* m_buffer[2] = {};
        cudaEvent_t m_done[2] = {};

        void release() {
            for (int i = 0; i < 2; i++) {
                if (m_done[i]) {
                    cudaEventSynchronize(m_done[i]);
                    cudaEventDestroy(m_done[i]);
                    m_done[i] = nullptr;
                }

                if (m_buffer[i]) {
                    cudaFreeHost(m_buffer[i]);
                    m_buffer[i] = nullptr;
                }
            }
        }
    };

    struct MemoryPlan {
        int64_t ranking = 0;
        int64_t classSort = 0;
        int64_t merge = 0;
        size_t outputWords = 0;  // the merge output, also the class sorts' scratch

        int64_t peak() const {
            return std::max({ranking, classSort, merge});
        }
    };

    cudaError_t planMemory(const gpusa::CoverTables& tables, int32_t n, gpusa::Merger& merger, MemoryPlan& plan) {
        using namespace gpusa;

        const uint32_t m = static_cast<uint32_t>(gpuSampleSize(n));
        const int64_t base = static_cast<int64_t>(textWords(n) * 8 + rankEntries(n) * 4);
        int64_t rankingWork = 0;

        GPUSA_TRY(SampleRanking::workingBytes(m, rankingWork));
        plan.ranking = base + rankingWork;

        const ClassLists layout = classLists(tables, n);

        merger.plan(layout, MergeParams{}, MergeParams{}.levels);
        GPUSA_TRY(merger.queryTemp());

        plan.outputWords = std::max({static_cast<size_t>(n), ClassSort::scratchWords(layout), merger.scratchWords()});
        const int64_t listsAndOutput = static_cast<int64_t>(n) * 4 + static_cast<int64_t>(plan.outputWords) * 4;

        size_t classTemp = 0;

        GPUSA_TRY(ClassSort::tempBytes(layout, classTemp));
        plan.classSort = base + listsAndOutput + static_cast<int64_t>(std::max<size_t>(classTemp, 1));
        plan.merge = base + listsAndOutput + merger.bytes();

        return cudaSuccess;
    }

    GpuStatus checkMemory(int64_t required) {
        GpuDevice device;

        if (gpuQueryDevice(device) != GpuStatus::Ok) {
            return GpuStatus::Unavailable;
        }

        if (required + kGpuMemoryReserve > device.freeBytes) {
            snprintf(lastError, sizeof lastError, "needs %lld MiB of device memory plus %lld MiB reserve, %lld MiB free",
                static_cast<long long>(required >> 20), static_cast<long long>(kGpuMemoryReserve >> 20),
                static_cast<long long>(device.freeBytes >> 20));
            return GpuStatus::NotEnoughMemory;
        }

        return GpuStatus::Ok;
    }

    GpuStatus failed(const char* what, cudaError_t error) {
        setError(what, error);
        return error == cudaErrorMemoryAllocation ? GpuStatus::NotEnoughMemory : GpuStatus::Failed;
    }

    // uploaded, if given, runs once the text is on the device.
    GpuStatus uploadAndRank(gpusa::Construction& c, Staging& staging, const uint8_t* text, int32_t n, GpuBwtStats& s,
        const std::function<void()>* uploaded) {
        using namespace gpusa;

        cudaError_t error;
        auto start = std::chrono::steady_clock::now();

        c.n = static_cast<uint32_t>(n);
        c.m = static_cast<uint32_t>(gpuSampleSize(n));

        if ((error = c.tables.upload()) != cudaSuccess) {
            return failed("cover tables", error);
        }

        if ((error = c.memory.allocate(c.text, textWords(n))) != cudaSuccess ||
            (error = c.memory.allocate(c.ranks, rankEntries(n))) != cudaSuccess) {
            return failed("cudaMalloc", error);
        }

        s.setupSeconds += secondsSince(start);
        s.pinned = staging.pinned();
        start = std::chrono::steady_clock::now();

        if ((error = cudaMemsetAsync(c.text, 0, textWords(n) * 8)) != cudaSuccess ||
            (error = cudaMemsetAsync(c.ranks, 0, rankEntries(n) * 4)) != cudaSuccess ||
            (error = staging.upload(c.text, text, static_cast<size_t>(n))) != cudaSuccess ||
            (error = cudaDeviceSynchronize()) != cudaSuccess) {
            return failed("upload", error);
        }

        s.uploadSeconds = secondsSince(start);

        if (uploaded) {
            (*uploaded)();
        }

        start = std::chrono::steady_clock::now();

        if ((error = SampleRanking(c).run(s)) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess) {
            return failed("sample ranking", error);
        }

        s.rankingBytes = c.memory.peak();
        s.rankingSeconds = secondsSince(start);
        s.deviceSeconds = s.rankingSeconds;

        return GpuStatus::Ok;
    }
}

GpuStatus gpuQueryDevice(GpuDevice& device) {
    device = GpuDevice{};
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);

    if (error != cudaSuccess || count == 0) {
        if (error != cudaSuccess) {
            setError("cudaGetDeviceCount", error);
            cudaGetLastError();  // clear it
        } else {
            snprintf(lastError, sizeof lastError, "no CUDA device");
        }

        return GpuStatus::Unavailable;
    }

    cudaDeviceProp properties;

    if ((error = cudaGetDeviceProperties(&properties, 0)) != cudaSuccess) {
        setError("cudaGetDeviceProperties", error);
        return GpuStatus::Unavailable;
    }

    size_t freeBytes = 0, totalBytes = 0;

    if ((error = cudaMemGetInfo(&freeBytes, &totalBytes)) != cudaSuccess) {
        setError("cudaMemGetInfo", error);
        return GpuStatus::Unavailable;
    }

    snprintf(device.name, sizeof device.name, "%s", properties.name);
    device.computeMajor = properties.major;
    device.computeMinor = properties.minor;
    device.multiprocessors = properties.multiProcessorCount;
    device.l2Bytes = properties.l2CacheSize;
    device.totalBytes = static_cast<int64_t>(totalBytes);
    device.freeBytes = static_cast<int64_t>(freeBytes);

    return GpuStatus::Ok;
}

int64_t gpuBwtRequiredBytes(int32_t n) {
    if (n <= 0) {
        return 0;
    }

    const gpusa::CoverTables tables;
    gpusa::Merger merger;
    MemoryPlan plan;

    return planMemory(tables, n, merger, plan) == cudaSuccess ? plan.peak() : -1;
}

namespace {
    // A device memory pool that keeps all it is given, from reserve bytes on: allocating from it again is cheap.
    cudaError_t createPool(cudaMemPool_t& pool, size_t reserve) {
        cudaMemPoolProps properties = {};

        properties.allocType = cudaMemAllocationTypePinned;
        properties.handleTypes = cudaMemHandleTypeNone;
        properties.location.type = cudaMemLocationTypeDevice;
        properties.location.id = 0;

        uint64_t keepAll = UINT64_MAX;  // don't hand memory back to the driver at syncs
        cudaError_t error;
        void* reserved = nullptr;

        if ((error = cudaMemPoolCreate(&pool, &properties)) != cudaSuccess ||
            (error = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keepAll)) != cudaSuccess) {
            return error;
        }

        if (reserve > 0 && ((error = cudaMallocFromPoolAsync(&reserved, reserve, pool, 0)) != cudaSuccess ||
                               (error = cudaFreeAsync(reserved, 0)) != cudaSuccess ||
                               (error = cudaStreamSynchronize(0)) != cudaSuccess)) {
            return error;
        }

        return cudaSuccess;
    }
}

struct GpuBwtContext {
    cudaMemPool_t pool = nullptr;
    int32_t capacity = 0;
    Staging staging;
    uint8_t* output[2] = {};  // gpuCompress()'s, page-locked, used in turn
    size_t outputCapacity[2] = {};
    int nextOutput = 0;

    GpuBwtContext() = default;
    GpuBwtContext(const GpuBwtContext&) = delete;
    GpuBwtContext& operator=(const GpuBwtContext&) = delete;

    ~GpuBwtContext() {
        cudaStreamSynchronize(0);  // the frees of the last construction

        if (pool) {
            cudaMemPoolDestroy(pool);
        }

        for (uint8_t* buffer : output) {
            if (buffer) {
                cudaFreeHost(buffer);
            }
        }
    }
};

GpuStatus gpuCreateBwtContext(int32_t capacity, GpuBwtContext*& context) {
    lastError[0] = '\0';
    context = nullptr;
    GpuDevice device;

    if (const GpuStatus status = gpuQueryDevice(device); status != GpuStatus::Ok) {
        return status;
    }

    const int64_t peak = gpuBwtRequiredBytes(std::max(capacity, 0));

    if (peak < 0) {
        snprintf(lastError, sizeof lastError, "planning the device memory failed");
        return GpuStatus::Failed;
    }

    if (const GpuStatus status = checkMemory(peak); status != GpuStatus::Ok) {
        return status;
    }

    std::unique_ptr<GpuBwtContext> created(new (std::nothrow) GpuBwtContext);

    if (!created) {
        snprintf(lastError, sizeof lastError, "out of host memory");
        return GpuStatus::Failed;
    }

    // Reserve the peak once; the pool keeps it after the free.
    if (const cudaError_t error = createPool(created->pool, static_cast<size_t>(peak)); error != cudaSuccess) {
        return failed("memory pool", error);
    }

    created->staging.create();
    created->capacity = std::max(capacity, 0);
    context = created.release();

    return GpuStatus::Ok;
}

void gpuDestroyBwtContext(GpuBwtContext* context) {
    delete context;
}

namespace {
    // Leaves the BWT in output as MergeOutput::Bwt describes it.
    // With samples, also records the rows of text positions j * 2^logSample, j >= 1.
    GpuStatus buildOnDevice(GpuBwtContext& context, gpusa::Construction& c, const uint8_t* text, int32_t n, GpuBwtStats& s, uint32_t*& lists, uint32_t*& output, gpusa::Merger& merger, uint32_t& suffixZeroRow, uint32_t* samples, int logSample, const std::function<void()>* uploaded = nullptr) {
        using namespace gpusa;

        MemoryPlan plan;
        cudaError_t error;

        if (n > context.capacity) {
            snprintf(lastError, sizeof lastError, "input of %d bytes exceeds the context's capacity of %d bytes", n,
                context.capacity);
            return GpuStatus::NotEnoughMemory;
        }

        if ((error = planMemory(c.tables, n, merger, plan)) != cudaSuccess) {
            return failed("planning", error);
        }

        if (const GpuStatus status = uploadAndRank(c, context.staging, text, n, s, uploaded);
            status != GpuStatus::Ok) {
            return status;
        }

        auto start = std::chrono::steady_clock::now();
        const ClassLists layout = classLists(c.tables, n);

        c.memory.resetPeak();

        if ((error = c.memory.allocate(lists, static_cast<size_t>(n))) != cudaSuccess ||
            (error = c.memory.allocate(output, plan.outputWords)) != cudaSuccess) {
            return failed("cudaMalloc", error);
        }

        if ((error = ClassSort::run(c, layout, lists, output)) != cudaSuccess ||
            (error = cudaDeviceSynchronize()) != cudaSuccess) {
            return failed("class sorts", error);
        }

        s.classSortBytes = c.memory.peak();
        s.classSortSeconds = secondsSince(start);
        start = std::chrono::steady_clock::now();
        c.memory.resetPeak();

        if ((error = merger.allocate(c.memory, output)) != cudaSuccess) {
            return failed("cudaMalloc", error);
        }

        MergeInput input{c.text, c.ranks, lists, c.n, layout};
        const uint32_t sampleCount = samples ? bwarp::sampleCount(static_cast<uint32_t>(n), logSample) : 0;

        if (sampleCount > 0) {
            if ((error = c.memory.allocate(input.samples, sampleCount)) != cudaSuccess) {
                return failed("cudaMalloc", error);
            }

            input.logSample = static_cast<uint32_t>(logSample);
        }

        if ((error = merger.run(input, output, MergeOutput::Bwt)) != cudaSuccess ||
            (error = cudaDeviceSynchronize()) != cudaSuccess ||
            (error = cudaMemcpy(&suffixZeroRow, merger.suffixZeroRow(), sizeof(uint32_t), cudaMemcpyDeviceToHost)) !=
                cudaSuccess ||
            (sampleCount > 0 && (error = cudaMemcpy(samples, input.samples, sampleCount * sizeof(uint32_t),
                                     cudaMemcpyDeviceToHost)) != cudaSuccess)) {
            return failed("merge", error);
        }

        if (input.samples) {
            c.memory.release(input.samples);
        }

        s.mergeBytes = c.memory.peak();
        merger.release(c.memory);
        s.mergeSeconds = secondsSince(start);
        s.deviceSeconds += s.classSortSeconds + s.mergeSeconds;
        s.deviceBytes = std::max({s.rankingBytes, s.classSortBytes, s.mergeBytes});

        return GpuStatus::Ok;
    }

    template <class Build>
    GpuStatus withTemporaryContext(int32_t n, GpuBwtStats& s, Build build) {
        const auto start = std::chrono::steady_clock::now();
        GpuBwtContext* context = nullptr;

        if (const GpuStatus status = gpuCreateBwtContext(n, context); status != GpuStatus::Ok) {
            return status;
        }

        const double setup = secondsSince(start);
        const GpuStatus status = build(context);

        s.setupSeconds += setup;
        gpuDestroyBwtContext(context);

        return status;
    }
}

GpuStatus gpuBwt(GpuBwtContext* context, const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, GpuBwtStats* stats,
    uint32_t* samples, int logSample) {
    using namespace gpusa;

    if (!context) {
        return gpuBwt(text, n, bwt, primary, stats, samples, logSample);
    }

    lastError[0] = '\0';

    GpuBwtStats local;
    GpuBwtStats& s = stats ? *stats : local;

    s = GpuBwtStats{};
    primary = n > 0 ? -1 : 0;

    if (n <= 1) {
        if (n == 1) {
            bwt[0] = text[0];  // the sentinel's row; suffix 0's is left out
            primary = 1;
        }

        return GpuStatus::Ok;
    }

    Construction c(context->pool);
    Merger merger;
    uint32_t* lists = nullptr;
    uint32_t* output = nullptr;
    uint32_t row = 0;

    if (const GpuStatus status = buildOnDevice(*context, c, text, n, s, lists, output, merger, row, samples, logSample);
        status != GpuStatus::Ok) {
        return status;
    }

    // Byte 0 is the sentinel's row, T[n - 1]; byte row + 1, suffix 0's, is left out.
    auto start = std::chrono::steady_clock::now();
    uint8_t* const deviceBwt = reinterpret_cast<uint8_t*>(output);
    cudaError_t error;

    if ((error = cudaMemcpy(deviceBwt, reinterpret_cast<const uint8_t*>(c.text) + n - 1, 1, cudaMemcpyDeviceToDevice)) !=
        cudaSuccess) {
        return failed("BWT", error);
    }

    s.bwtSeconds = secondsSince(start);
    s.deviceSeconds += s.bwtSeconds;
    start = std::chrono::steady_clock::now();
    const size_t head = static_cast<size_t>(row) + 1;

    if ((error = context->staging.download(bwt, deviceBwt, head)) != cudaSuccess ||
        (error = context->staging.download(bwt + head, deviceBwt + head + 1, static_cast<size_t>(n) - head)) !=
            cudaSuccess) {
        return failed("download", error);
    }

    s.downloadSeconds = secondsSince(start);
    primary = static_cast<int32_t>(row) + 1;

    return GpuStatus::Ok;
}

GpuStatus gpuBwt(const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, GpuBwtStats* stats, uint32_t* samples,
    int logSample) {
    lastError[0] = '\0';

    GpuBwtStats local;
    GpuBwtStats& s = stats ? *stats : local;

    s = GpuBwtStats{};
    primary = n > 0 ? -1 : 0;

    if (n <= 1) {
        GpuDevice device;
        const GpuStatus status = gpuQueryDevice(device);

        if (status == GpuStatus::Ok && n == 1) {
            bwt[0] = text[0];  // the sentinel's row; suffix 0's is left out
            primary = 1;
        }

        return status;
    }

    return withTemporaryContext(
        n, s, [&](GpuBwtContext* context) { return gpuBwt(context, text, n, bwt, primary, &s, samples, logSample); });
}

const char* gpuLastError() {
    return lastError;
}

void* gpuAllocatePinned(size_t bytes) {
    void* pointer = nullptr;

    if (cudaMallocHost(&pointer, std::max<size_t>(bytes, 1)) != cudaSuccess) {
        cudaGetLastError();  // clear it
        return nullptr;
    }

    return pointer;
}

void gpuFreePinned(void* pointer) {
    if (pointer) {
        cudaFreeHost(pointer);
    }
}

// ---- Decoding ----------------------------------------------------------------------------------

namespace {
    template <class T>
    class DeviceArray {
     public:
        DeviceArray() = default;
        DeviceArray(const DeviceArray&) = delete;
        DeviceArray& operator=(const DeviceArray&) = delete;

        ~DeviceArray() {
            if (m_pointer) {
                m_pooled ? cudaFreeAsync(m_pointer, 0) : cudaFree(m_pointer);
            }
        }

        // From pool if given, freed into it in stream order.
        cudaError_t allocate(size_t count, int64_t& total, cudaMemPool_t pool = nullptr) {
            m_bytes = std::max<size_t>(count, 1) * sizeof(T);
            m_pooled = pool != nullptr;
            total += static_cast<int64_t>(m_bytes);
            return pool ? cudaMallocFromPoolAsync(reinterpret_cast<void**>(&m_pointer), m_bytes, pool, 0)
                        : cudaMalloc(&m_pointer, m_bytes);
        }

        T* get() const {
            return m_pointer;
        }

        size_t bytes() const {
            return m_bytes;
        }

     private:
        T* m_pointer = nullptr;
        size_t m_bytes = 0;
        bool m_pooled = false;
    };

    // Owns a createPool() pool; declared before the arrays allocated from it, so it outlives them.
    class DevicePool {
     public:
        DevicePool() = default;
        DevicePool(const DevicePool&) = delete;
        DevicePool& operator=(const DevicePool&) = delete;

        ~DevicePool() {
            if (m_pool) {
                cudaStreamSynchronize(0);  // the arrays' frees
                cudaMemPoolDestroy(m_pool);
            }
        }

        cudaError_t create(size_t reserve) {
            return createPool(m_pool, reserve);
        }

        cudaMemPool_t get() const {
            return m_pool;
        }

     private:
        cudaMemPool_t m_pool = nullptr;
    };

    GpuStatus invalid(const char* what) {
        snprintf(lastError, sizeof lastError, "malformed input: %s", what);
        return GpuStatus::Invalid;
    }

    cudaError_t readError(const uint32_t* device, uint32_t& error) {
        return cudaMemcpy(&error, device, sizeof error, cudaMemcpyDeviceToHost);
    }
}

namespace {
    struct DeviceDecode {
        bwarp::Container container;
        DeviceArray<uint8_t> data, bwt;  // bwt padded to a multiple of 16 bytes
        DeviceArray<uint64_t> segmentOffsets, sliceOffsets;
        DeviceArray<bwarp::TableData> tables;
        DeviceArray<bwarp::gpu::DeviceTable> deviceTables;
        DeviceArray<uint32_t> error;
    };

    // Leaves the BWT in d.bwt.
    GpuStatus decodeOnDevice(const uint8_t* data, size_t size, DeviceDecode& d, GpuDecodeStats& s) {
        using namespace bwarp;

        if (!readContainer(data, size, d.container)) {
            return invalid("header or index");
        }

        const Layout& layout = d.container.layout;

        if (layout.n == 0) {
            return GpuStatus::Ok;
        }

        cudaError_t error;
        auto start = std::chrono::steady_clock::now();

        if ((error = d.data.allocate(size, s.deviceBytes)) != cudaSuccess ||
            (error = d.bwt.allocate((size_t(layout.n) + 15) & ~size_t(15), s.deviceBytes)) != cudaSuccess ||
            (error = d.segmentOffsets.allocate(layout.segments + 1, s.deviceBytes)) != cudaSuccess ||
            (error = d.sliceOffsets.allocate(layout.slices + 1, s.deviceBytes)) != cudaSuccess ||
            (error = d.tables.allocate(layout.segments, s.deviceBytes)) != cudaSuccess ||
            (error = d.deviceTables.allocate(layout.segments, s.deviceBytes)) != cudaSuccess ||
            (error = d.error.allocate(1, s.deviceBytes)) != cudaSuccess) {
            return failed("cudaMalloc", error);
        }

        s.setupSeconds = secondsSince(start);
        start = std::chrono::steady_clock::now();

        if ((error = cudaMemcpy(d.data.get(), data, size, cudaMemcpyHostToDevice)) != cudaSuccess ||
            (error = cudaMemcpy(d.segmentOffsets.get(), d.container.segmentOffsets.data(), d.segmentOffsets.bytes(),
                 cudaMemcpyHostToDevice)) != cudaSuccess ||
            (error = cudaMemcpy(d.sliceOffsets.get(), d.container.sliceOffsets.data(), d.sliceOffsets.bytes(),
                 cudaMemcpyHostToDevice)) != cudaSuccess ||
            (error = cudaMemset(d.tables.get(), 0, d.tables.bytes())) != cudaSuccess ||
            (error = cudaMemset(d.error.get(), 0, d.error.bytes())) != cudaSuccess ||
            (error = cudaDeviceSynchronize()) != cudaSuccess) {
            return failed("upload", error);
        }

        s.uploadSeconds = secondsSince(start);
        start = std::chrono::steady_clock::now();

        // The stream decoder trusts the tables, so they are checked first.
        uint32_t decodeError = 0;

        gpu::decodeTables<<<(layout.segments + gpu::kTableWarps - 1) / gpu::kTableWarps, gpu::kTableWarps * 32>>>(
            d.data.get(), d.segmentOffsets.get(), layout.segments, lengthCodeCount(layout.logSlice), d.tables.get(),
            d.error.get());

        if ((error = cudaGetLastError()) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess ||
            (error = readError(d.error.get(), decodeError)) != cudaSuccess) {
            return failed("table decoding", error);
        }

        if (decodeError != 0) {
            return invalid("table");
        }

        s.tableSeconds = secondsSince(start);
        start = std::chrono::steady_clock::now();

        gpu::finishTables<<<layout.segments, 256>>>(d.tables.get(), d.deviceTables.get(), true);

        if ((error = cudaGetLastError()) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess) {
            return failed("table finishing", error);
        }

        s.finishSeconds = secondsSince(start);
        start = std::chrono::steady_clock::now();

        gpu::decodeSlices<<<(layout.slices + gpu::kSliceWarps - 1) / gpu::kSliceWarps, gpu::kSliceWarps * 32>>>(
            d.data.get(), d.sliceOffsets.get(), layout.slices, layout.n, layout.logSegment, layout.logSlice,
            d.deviceTables.get(), d.bwt.get(), d.error.get());

        if ((error = cudaGetLastError()) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess ||
            (error = readError(d.error.get(), decodeError)) != cudaSuccess) {
            return failed("stream decoding", error);
        }

        if (decodeError != 0) {
            return invalid("stream");
        }

        s.streamSeconds = secondsSince(start);

        return GpuStatus::Ok;
    }

    // The text of d.bwt into text (at least 16 bytes, for its 16-byte stores). n > 0.
    GpuStatus invertOnDevice(DeviceDecode& d, DeviceArray<uint32_t>& text, GpuDecodeStats& s) {
        using namespace bwarp;

        const Layout& layout = d.container.layout;
        const uint32_t n = layout.n;
        const uint32_t chains = layout.samples + 1;
        const uint32_t primary = static_cast<uint32_t>(d.container.primary);
        const uint32_t tiles = (n + gpu::kRankTile - 1) / gpu::kRankTile;
        const bool both = chains <= gpu::kMaxBidirectionalChains;
        auto start = std::chrono::steady_clock::now();
        cudaError_t error;
        DeviceArray<uint8_t> temp;
        DeviceArray<uint32_t> lf, psi, offsets, samples;
        size_t scanBytes = 0;

        if ((error = cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, offsets.get(), offsets.get(),
                 256 * tiles)) != cudaSuccess ||
            (error = lf.allocate(size_t(n) + 1, s.deviceBytes)) != cudaSuccess ||
            (both && (error = psi.allocate(size_t(n) + 1, s.deviceBytes)) != cudaSuccess) ||
            (error = offsets.allocate(size_t(256) * tiles, s.deviceBytes)) != cudaSuccess ||
            (error = samples.allocate(layout.samples, s.deviceBytes)) != cudaSuccess ||
            (error = temp.allocate(scanBytes, s.deviceBytes)) != cudaSuccess ||
            (error = text.allocate((size_t(n) + 15) / 4 & ~size_t(3), s.deviceBytes)) != cudaSuccess ||
            (layout.samples > 0 &&
                (error = cudaMemcpy(samples.get(), d.container.samples.data(), size_t(layout.samples) * 4,
                     cudaMemcpyHostToDevice)) != cudaSuccess) ||
            (error = cudaDeviceSynchronize()) != cudaSuccess) {
            return failed("inverse setup", error);
        }

        s.setupSeconds += secondsSince(start);
        start = std::chrono::steady_clock::now();

        gpu::countTiles<<<tiles, gpu::kInverseThreads>>>(d.bwt.get(), n, tiles, offsets.get());

        if ((error = cudaGetLastError()) != cudaSuccess ||
            (error = cub::DeviceScan::ExclusiveSum(temp.get(), scanBytes, offsets.get(), offsets.get(),
                 256 * tiles)) != cudaSuccess) {
            return failed("inverse ranking", error);
        }

        gpu::rankTiles<<<tiles, gpu::kInverseThreads>>>(d.bwt.get(), n, tiles, offsets.get(), primary, lf.get(),
            both ? psi.get() : nullptr);

        if ((error = cudaGetLastError()) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess) {
            return failed("inverse ranking", error);
        }

        s.rankSeconds = secondsSince(start);
        start = std::chrono::steady_clock::now();

        const uint32_t threads = both ? (chains + 31) / 32 * 64 : chains;

        gpu::walkChains<<<(threads + gpu::kInverseThreads - 1) / gpu::kInverseThreads, gpu::kInverseThreads>>>(
            lf.get(), both ? psi.get() : nullptr, offsets.get(), tiles, samples.get(), primary, n,
            static_cast<uint32_t>(layout.logSample), chains, reinterpret_cast<uint8_t*>(text.get()));

        if ((error = cudaGetLastError()) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess) {
            return failed("inverse chains", error);
        }

        s.chainSeconds = secondsSince(start);

        return GpuStatus::Ok;
    }
}

GpuStatus gpuDecode(const uint8_t* data, size_t size, std::vector<uint8_t>& bwt, int32_t& primary,
    GpuDecodeStats* stats) {
    lastError[0] = '\0';

    GpuDecodeStats local;
    GpuDecodeStats& s = stats ? *stats : local;
    DeviceDecode d;

    s = GpuDecodeStats{};
    primary = -1;

    if (const GpuStatus status = decodeOnDevice(data, size, d, s); status != GpuStatus::Ok) {
        return status;
    }

    const uint32_t n = d.container.layout.n;
    const auto start = std::chrono::steady_clock::now();
    cudaError_t error;

    bwt.resize(n);

    if (n > 0 && (error = cudaMemcpy(bwt.data(), d.bwt.get(), n, cudaMemcpyDeviceToHost)) != cudaSuccess) {
        return failed("download", error);
    }

    s.downloadSeconds = secondsSince(start);
    primary = d.container.primary;

    return GpuStatus::Ok;
}

namespace {
    // gpuDecompress(), downloading to room(n): where the n bytes of text go, or null if they don't fit.
    template <class Room>
    GpuStatus decompress(const uint8_t* data, size_t size, GpuDecodeStats* stats, Room room) {
        lastError[0] = '\0';

        GpuDecodeStats local;
        GpuDecodeStats& s = stats ? *stats : local;
        DeviceDecode d;
        DeviceArray<uint32_t> deviceText;

        s = GpuDecodeStats{};

        if (const GpuStatus status = decodeOnDevice(data, size, d, s); status != GpuStatus::Ok) {
            return status;
        }

        const uint32_t n = d.container.layout.n;

        if (n > 0) {
            if (const GpuStatus status = invertOnDevice(d, deviceText, s); status != GpuStatus::Ok) {
                return status;
            }
        }

        const auto start = std::chrono::steady_clock::now();
        uint8_t* const text = room(n);
        cudaError_t error;

        if (n > 0 && !text) {
            snprintf(lastError, sizeof lastError, "the text of %u bytes does not fit the buffer", n);
            return GpuStatus::NotEnoughMemory;
        }

        if (n > 0 && (error = cudaMemcpy(text, deviceText.get(), n, cudaMemcpyDeviceToHost)) != cudaSuccess) {
            return failed("download", error);
        }

        s.downloadSeconds = secondsSince(start);

        const auto checking = std::chrono::steady_clock::now();
        const uint32_t checksum = crc32c(text, n);

        s.checkSeconds = secondsSince(checking);

        if (checksum != d.container.checksum) {
            snprintf(lastError, sizeof lastError, "the text's CRC-32C is %08x, the block says %08x", checksum,
                d.container.checksum);
            return GpuStatus::ChecksumMismatch;
        }

        return GpuStatus::Ok;
    }
}

GpuStatus gpuDecompress(const uint8_t* data, size_t size, std::vector<uint8_t>& text, GpuDecodeStats* stats) {
    return decompress(data, size, stats, [&](uint32_t n) {
        text.resize(n);
        return text.data();
    });
}

GpuStatus gpuDecompress(const uint8_t* data, size_t size, uint8_t* text, size_t capacity, size_t& n,
    GpuDecodeStats* stats) {
    n = 0;

    return decompress(data, size, stats, [&](uint32_t bytes) {
        n = bytes;
        return bytes <= capacity ? text : nullptr;
    });
}

// ---- Encoding ----------------------------------------------------------------------------------

namespace {
    // Device memory a pool for encodeOnDevice() reserves ahead: the BWT, the per-segment and per-slice arrays, and
    // typically enough for the ops and the body, sized on the way.
    size_t encodeReserve(const bwarp::Layout& coding) {
        const size_t perSegment =
            bwarp::gpu::kReservedOps * 6 + sizeof(bwarp::TableData) + sizeof(bwarp::gpu::DeviceTable) + 64;
        const size_t perSlice = (size_t(1) << coding.logSlice) + bwarp::kLanes * 4 + 32;

        return size_t(coding.n) * 2 + 16 + coding.segments * perSegment + coding.slices * perSlice + (size_t(1) << 20);
    }

    // encodeBwt() of a valid input with n > 0 whose BWT is on the device, padded to 16 bytes: leaves the tables and
    // streams in body, bodyBytes of them, and writes the header, samples and offsets to index. Everything comes
    // from pool. prepare(bodyBytes) runs while the device places the body, which is done on return.
    template <class Prepare>
    GpuStatus encodeOnDevice(const uint8_t* deviceBwt, int32_t n, int32_t primary, uint32_t checksum,
        const uint32_t* samples, int recordedLogSample, const EncodeParams& params, cudaMemPool_t pool,
        DeviceArray<uint8_t>& body,
        uint64_t& bodyBytes, std::vector<uint8_t>& index, EncodeStats& e, GpuEncodeStats& s, Prepare prepare) {
        using namespace bwarp;

        const Layout coding(static_cast<uint32_t>(n), params.logSegment, params.logSlice, kMaxLogSample);
        std::vector<uint64_t> segmentOffsets(coding.segments + 1, 0), sliceOffsets(coding.slices + 1, 0);
        cudaError_t error;
        auto start = std::chrono::steady_clock::now();
        DeviceArray<uint32_t> cost, ops, wordCounts, states, sliceWords, sliceStates;
        DeviceArray<uint16_t> words, scratch;
        const uint32_t scratchWords = 1u << (coding.logSlice - 1);  // per slice: as many bytes as it codes, at most
        DeviceArray<uint64_t> decisions, opOffsets, deviceSegmentOffsets, deviceSliceOffsets;
        DeviceArray<unsigned long long> tokens;
        DeviceArray<TableData> tables;
        DeviceArray<gpu::DeviceTable> codes;
        DeviceArray<uint8_t> temp;
        size_t scanBytes = 0;

        if ((error = cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, decisions.get(), opOffsets.get(),
                 coding.segments + 1)) != cudaSuccess ||
            (error = cost.allocate(kProbScale + 1, s.deviceBytes, pool)) != cudaSuccess ||
            (error = decisions.allocate(coding.segments + 1, s.deviceBytes, pool)) != cudaSuccess ||
            (error = opOffsets.allocate(coding.segments + 1, s.deviceBytes, pool)) != cudaSuccess ||
            (error = temp.allocate(scanBytes, s.deviceBytes, pool)) != cudaSuccess ||
            (error = tables.allocate(coding.segments, s.deviceBytes, pool)) != cudaSuccess ||
            (error = codes.allocate(coding.segments, s.deviceBytes, pool)) != cudaSuccess ||
            (error = wordCounts.allocate(coding.segments, s.deviceBytes, pool)) != cudaSuccess ||
            (error = states.allocate(coding.segments, s.deviceBytes, pool)) != cudaSuccess ||
            (error = sliceWords.allocate(coding.slices, s.deviceBytes, pool)) != cudaSuccess ||
            (error = sliceStates.allocate(size_t(coding.slices) * kLanes, s.deviceBytes, pool)) != cudaSuccess ||
            (error = scratch.allocate(size_t(coding.slices) * scratchWords, s.deviceBytes, pool)) != cudaSuccess ||
            (error = tokens.allocate(1, s.deviceBytes, pool)) != cudaSuccess) {
            return failed("cudaMalloc", error);
        }

        s.setupSeconds += secondsSince(start);
        start = std::chrono::steady_clock::now();

        if ((error = cudaMemcpyAsync(cost.get(), encodingCosts(), cost.bytes(), cudaMemcpyHostToDevice)) !=
                cudaSuccess ||
            (error = cudaMemsetAsync(decisions.get(), 0, decisions.bytes())) != cudaSuccess ||
            (error = cudaMemsetAsync(tokens.get(), 0, tokens.bytes())) != cudaSuccess ||
            (error = cudaDeviceSynchronize()) != cudaSuccess) {
            return failed("upload", error);
        }

        s.uploadSeconds += secondsSince(start);
        start = std::chrono::steady_clock::now();

        // Up to 2^16 bytes per segment, the counts pack into 16 bits.
        (coding.logSegment <= 16 ? gpu::chooseTables<true> : gpu::chooseTables<false>)<<<coding.segments,
            gpu::kChooseThreads>>>(deviceBwt, coding.n, coding.logSegment, coding.logSlice,
            static_cast<uint32_t>(params.contexts), cost.get(), tables.get(), decisions.get());

        uint64_t totalOps = 0;

        if ((error = cudaGetLastError()) != cudaSuccess ||
            (error = cub::DeviceScan::ExclusiveSum(temp.get(), scanBytes, decisions.get(), opOffsets.get(),
                 coding.segments + 1)) != cudaSuccess ||
            (error = cudaMemcpy(&totalOps, opOffsets.get() + coding.segments, sizeof totalOps,
                 cudaMemcpyDeviceToHost)) != cudaSuccess) {
            return failed("choosing tables", error);
        }

        s.chooseSeconds = secondsSince(start);
        start = std::chrono::steady_clock::now();

        if ((error = ops.allocate(totalOps, s.deviceBytes, pool)) != cudaSuccess ||
            (error = words.allocate(totalOps, s.deviceBytes, pool)) != cudaSuccess) {
            return failed("cudaMalloc", error);
        }

        gpu::encodeTables<<<(coding.segments + gpu::kTableWarps - 1) / gpu::kTableWarps, gpu::kTableWarps * 32>>>(
            tables.get(), coding.segments, opOffsets.get(), ops.get(), words.get(), wordCounts.get(), states.get());
        gpu::finishTables<<<coding.segments, 256>>>(tables.get(), codes.get(), false);

        if ((error = cudaGetLastError()) != cudaSuccess || (error = cudaDeviceSynchronize()) != cudaSuccess) {
            return failed("coding tables", error);
        }

        s.tableSeconds = secondsSince(start);
        start = std::chrono::steady_clock::now();

        const uint32_t sliceBlocks = (coding.slices + gpu::kEncodeWarps - 1) / gpu::kEncodeWarps;
        std::vector<uint32_t> tableWords(coding.segments), streamWords(coding.slices);
        unsigned long long tokenCount = 0;
        gpu::SliceOutput to{sliceWords.get(), scratch.get(), scratchWords, sliceStates.get(), tokens.get(), nullptr,
            nullptr};

        gpu::encodeSlices<false><<<sliceBlocks, gpu::kEncodeWarps * 32>>>(deviceBwt, coding.n, coding.slices,
            coding.logSegment, coding.logSlice, tables.get(), codes.get(), to);

        if ((error = cudaGetLastError()) != cudaSuccess ||
            (error = cudaMemcpy(tableWords.data(), wordCounts.get(), tableWords.size() * 4, cudaMemcpyDeviceToHost)) !=
                cudaSuccess ||
            (error = cudaMemcpy(streamWords.data(), sliceWords.get(), streamWords.size() * 4,
                 cudaMemcpyDeviceToHost)) != cudaSuccess ||
            (error = cudaMemcpy(&tokenCount, tokens.get(), sizeof tokenCount, cudaMemcpyDeviceToHost)) !=
                cudaSuccess) {
            return failed("encoding streams", error);
        }

        s.streamSeconds = secondsSince(start);
        start = std::chrono::steady_clock::now();

        for (uint32_t g = 0; g < coding.segments; g++) {
            segmentOffsets[g + 1] = segmentOffsets[g] + 4 + 2 * uint64_t(tableWords[g]);
        }

        sliceOffsets[0] = segmentOffsets[coding.segments];

        for (uint32_t sl = 0; sl < coding.slices; sl++) {
            sliceOffsets[sl + 1] = sliceOffsets[sl] + kLanes * 4 + 2 * uint64_t(streamWords[sl]);
        }

        // The device writes the body, the tables and streams, while the host writes the index and prepares.
        bodyBytes = sliceOffsets.back();

        if ((error = body.allocate(bodyBytes, s.deviceBytes, pool)) != cudaSuccess ||
            (error = deviceSegmentOffsets.allocate(segmentOffsets.size(), s.deviceBytes, pool)) != cudaSuccess ||
            (error = deviceSliceOffsets.allocate(sliceOffsets.size(), s.deviceBytes, pool)) != cudaSuccess ||
            (error = cudaMemcpyAsync(deviceSegmentOffsets.get(), segmentOffsets.data(), deviceSegmentOffsets.bytes(),
                 cudaMemcpyHostToDevice)) != cudaSuccess ||
            (error = cudaMemcpyAsync(deviceSliceOffsets.get(), sliceOffsets.data(), deviceSliceOffsets.bytes(),
                 cudaMemcpyHostToDevice)) != cudaSuccess) {
            return failed("placing", error);
        }

        // If a stream outgrew its scratch (incompressible data), all are encoded again, in place.
        to.offsets = deviceSliceOffsets.get();
        to.out = body.get();
        gpu::placeTables<<<coding.segments, 256>>>(words.get(), opOffsets.get(), wordCounts.get(), states.get(),
            deviceSegmentOffsets.get(), body.get());

        if (*std::max_element(streamWords.begin(), streamWords.end()) <= scratchWords) {
            gpu::placeSlices<<<coding.slices, 256>>>(to);
        } else {
            gpu::encodeSlices<true><<<sliceBlocks, gpu::kEncodeWarps * 32>>>(deviceBwt, coding.n, coding.slices,
                coding.logSegment, coding.logSlice, tables.get(), codes.get(), to);
        }

        writeIndex(n, primary, checksum, samples, recordedLogSample, params, segmentOffsets, sliceOffsets, index, e);
        e.tokens = static_cast<int64_t>(tokenCount);

        if ((error = cudaGetLastError()) != cudaSuccess) {
            return failed("placing", error);
        }

        if (const GpuStatus status = prepare(bodyBytes); status != GpuStatus::Ok) {
            cudaDeviceSynchronize();  // before the arrays go
            return status;
        }

        if ((error = cudaDeviceSynchronize()) != cudaSuccess) {
            return failed("placing", error);
        }

        s.placeSeconds = secondsSince(start);

        return GpuStatus::Ok;
    }
}

GpuStatus gpuEncode(const uint8_t* bwt, int32_t n, int32_t primary, uint32_t checksum, const uint32_t* samples,
    int recordedLogSample, const EncodeParams& params, std::vector<uint8_t>& out, EncodeStats* encodeStats,
    GpuEncodeStats* stats) {
    using namespace bwarp;

    lastError[0] = '\0';

    GpuEncodeStats local;
    GpuEncodeStats& s = stats ? *stats : local;
    EncodeStats localEncode;
    EncodeStats& e = encodeStats ? *encodeStats : localEncode;

    s = GpuEncodeStats{};
    e = EncodeStats{};
    out.clear();

    if (!validEncodeInput(n, primary, samples, recordedLogSample, params)) {
        snprintf(lastError, sizeof lastError, "invalid encoding parameters or rows");
        return GpuStatus::Invalid;
    }

    if (n == 0) {
        const Layout coding(0, params.logSegment, params.logSlice, kMaxLogSample);
        const std::vector<uint64_t> segmentOffsets(coding.segments + 1, 0), sliceOffsets(coding.slices + 1, 0);

        writeIndex(n, primary, checksum, samples, recordedLogSample, params, segmentOffsets, sliceOffsets, out, e);
        return GpuStatus::Ok;
    }

    cudaError_t error;
    auto start = std::chrono::steady_clock::now();
    DevicePool pool;  // declared first: it outlives the arrays
    DeviceArray<uint8_t> deviceBwt, body;
    uint64_t bodyBytes = 0;

    if ((error = pool.create(encodeReserve(Layout(uint32_t(n), params.logSegment, params.logSlice, kMaxLogSample)))) !=
            cudaSuccess ||
        (error = deviceBwt.allocate(size_t(n) + 16, s.deviceBytes, pool.get())) != cudaSuccess) {
        return failed("cudaMalloc", error);
    }

    s.setupSeconds = secondsSince(start);
    start = std::chrono::steady_clock::now();

    if ((error = cudaMemcpy(deviceBwt.get(), bwt, size_t(n), cudaMemcpyHostToDevice)) != cudaSuccess) {
        return failed("upload", error);
    }

    s.uploadSeconds = secondsSince(start);

    // While the device places the body, the host makes room for it.
    const GpuStatus status = encodeOnDevice(deviceBwt.get(), n, primary, checksum, samples, recordedLogSample,
        params, pool.get(), body, bodyBytes, out, e, s, [&](uint64_t bytes) {
            out.resize(out.size() + bytes);
            return GpuStatus::Ok;
        });

    if (status != GpuStatus::Ok) {
        return status;
    }

    start = std::chrono::steady_clock::now();

    if ((error = cudaMemcpy(out.data() + out.size() - bodyBytes, body.get(), bodyBytes, cudaMemcpyDeviceToHost)) !=
        cudaSuccess) {
        return failed("download", error);
    }

    s.downloadSeconds = secondsSince(start);

    return GpuStatus::Ok;
}

// ---- Compression -------------------------------------------------------------------------------

namespace {
    // The BWT as gpuBwt() downloads it, padded with zeros to 16 more bytes: byte 0, the sentinel's row, is
    // T[n - 1], and the merge's byte head, suffix 0's, is left out.
    __global__ void packBwt(const uint8_t* __restrict__ merged, const uint8_t* __restrict__ text, uint32_t n,
        uint32_t head, uint8_t* __restrict__ bwt) {
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n + 16; i += gridDim.x * blockDim.x) {
            bwt[i] = i == 0 ? text[n - 1] : i < head ? merged[i] : i < n ? merged[i + 1] : 0;
        }
    }

    // Grows the context's page-locked output buffer b to at least bytes, with some room to spare.
    GpuStatus reserveOutput(GpuBwtContext& context, int b, size_t bytes) {
        if (bytes <= context.outputCapacity[b]) {
            return GpuStatus::Ok;
        }

        gpuFreePinned(context.output[b]);
        context.outputCapacity[b] = bytes + bytes / 8;
        context.output[b] = static_cast<uint8_t*>(gpuAllocatePinned(context.outputCapacity[b]));

        if (!context.output[b]) {
            context.outputCapacity[b] = 0;
            snprintf(lastError, sizeof lastError, "out of page-locked host memory");
            return GpuStatus::NotEnoughMemory;
        }

        return GpuStatus::Ok;
    }
}

GpuStatus gpuCompress(GpuBwtContext* context, const uint8_t* text, int32_t n, const EncodeParams& params,
    const uint8_t*& out, size_t& size, EncodeStats* encodeStats, GpuCompressStats* stats) {
    using namespace bwarp;

    lastError[0] = '\0';

    GpuCompressStats local;
    GpuCompressStats& s = stats ? *stats : local;
    EncodeStats localEncode;
    EncodeStats& e = encodeStats ? *encodeStats : localEncode;

    s = GpuCompressStats{};
    e = EncodeStats{};
    out = nullptr;
    size = 0;

    if (!context) {
        snprintf(lastError, sizeof lastError, "gpuCompress() needs a context");
        return GpuStatus::Invalid;
    }

    // The size and parameters, before building anything: as for an empty input, which has no rows to check.
    if (n < 0 || uint32_t(n) > kMaxBlockBytes || !validEncodeInput(0, 0, nullptr, kAutoMinLogSample, params)) {
        snprintf(lastError, sizeof lastError, "invalid encoding parameters");
        return GpuStatus::Invalid;
    }

    std::vector<uint32_t> samples(sampleCount(static_cast<uint32_t>(n), kAutoMinLogSample));
    std::vector<uint8_t> index;
    int32_t primary = 0;
    const int b = context->nextOutput;

    context->nextOutput ^= 1;

    // Too small to build on the device: gpuBwt() and gpuEncode().
    if (n <= 1) {
        uint8_t bwt = 0;
        GpuStatus status;

        if ((status = gpuBwt(context, text, n, &bwt, primary, &s.bwt)) != GpuStatus::Ok ||
            (status = gpuEncode(&bwt, n, primary, crc32c(text, size_t(n)), samples.data(), kAutoMinLogSample, params,
                 index, &e, &s.encode)) != GpuStatus::Ok ||
            (status = reserveOutput(*context, b, index.size())) != GpuStatus::Ok) {
            return status;
        }

        memcpy(context->output[b], index.data(), index.size());
        out = context->output[b];
        size = index.size();

        return GpuStatus::Ok;
    }

    // The checksum on the CPU while the device builds the BWT, after the upload, which it would slow down.
    std::future<uint32_t> checksum;
    const std::function<void()> uploaded = [&] {
        checksum = std::async(std::launch::async, [=] { return crc32c(text, size_t(n)); });
    };

    DeviceArray<uint8_t> deviceBwt, body;  // from the context's pool, which outlives them
    cudaError_t error;
    std::chrono::steady_clock::time_point start;

    // The construction's memory goes back to the pool, in stream order, before the encoder takes from it.
    {
        gpusa::Construction c(context->pool);
        gpusa::Merger merger;
        uint32_t* lists = nullptr;
        uint32_t* output = nullptr;
        uint32_t row = 0;

        if (const GpuStatus status = buildOnDevice(*context, c, text, n, s.bwt, lists, output, merger, row,
                samples.data(), kAutoMinLogSample, &uploaded);
            status != GpuStatus::Ok) {
            return status;
        }

        start = std::chrono::steady_clock::now();

        if ((error = deviceBwt.allocate(size_t(n) + 16, s.encode.deviceBytes, context->pool)) != cudaSuccess) {
            return failed("cudaMalloc", error);
        }

        packBwt<<<std::min<uint32_t>((uint32_t(n) + 16 + 255) / 256, 4096), 256>>>(
            reinterpret_cast<const uint8_t*>(output), reinterpret_cast<const uint8_t*>(c.text), uint32_t(n), row + 1,
            deviceBwt.get());

        if ((error = cudaGetLastError()) != cudaSuccess) {
            return failed("BWT", error);
        }

        primary = static_cast<int32_t>(row) + 1;
    }

    s.encode.uploadSeconds = secondsSince(start);

    if (!validEncodeInput(n, primary, samples.data(), kAutoMinLogSample, params)) {
        snprintf(lastError, sizeof lastError, "the construction's rows are out of range");
        return GpuStatus::Failed;
    }

    // While the device places the body, the host copies the index into the output buffer.
    uint64_t bodyBytes = 0;
    const GpuStatus status = encodeOnDevice(deviceBwt.get(), n, primary, checksum.get(), samples.data(),
        kAutoMinLogSample, params, context->pool, body, bodyBytes, index, e, s.encode, [&](uint64_t bytes) {
            const auto growing = std::chrono::steady_clock::now();
            const GpuStatus reserved = reserveOutput(*context, b, index.size() + bytes);

            s.encode.setupSeconds += secondsSince(growing);

            if (reserved == GpuStatus::Ok) {
                memcpy(context->output[b], index.data(), index.size());
            }

            return reserved;
        });

    if (status != GpuStatus::Ok) {
        return status;
    }

    start = std::chrono::steady_clock::now();

    if ((error = cudaMemcpy(context->output[b] + index.size(), body.get(), bodyBytes, cudaMemcpyDeviceToHost)) !=
        cudaSuccess) {
        return failed("download", error);
    }

    s.encode.downloadSeconds = secondsSince(start);
    out = context->output[b];
    size = index.size() + bodyBytes;

    return GpuStatus::Ok;
}
