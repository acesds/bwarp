/*
 * tuning.h - Host-side launch and partitioning parameters of the GPU variant.
 *
 * By default they are the constants below. Built with -DBWARP_TUNING=ON, each can be overridden by an
 * environment variable, read once, so one binary can sweep them (tools/tune.sh). The effective values
 * are printed to stderr on first use. An invalid value is reported and the default kept.
 *
 *   BWARP_TUNE_THREADS                   threads per block of the BWT's element-wise kernels   (256)
 *   BWARP_TUNE_PARTITION_THRESHOLD       random writes above which they are partitioned first  (2^22)
 *   BWARP_TUNE_PARTITION_BITS            radix bits of that partitioning pass                  (8)
 *   BWARP_TUNE_MERGE_G                   splitter spacing                                      (128)
 *   BWARP_TUNE_MERGE_TILE                elements per merge tile                               (256)
 *   BWARP_TUNE_MERGE_CAPACITY            records per tile in shared memory                     (480)
 *   BWARP_TUNE_MERGE_THREADS             threads per merge block                               (256)
 *   BWARP_TUNE_MERGE_OVERFLOW_CAPACITY   records per tile of the overflow merge                (960)
 *   BWARP_TUNE_MERGE_OVERFLOW_THREADS    threads per overflow merge block                      (512)
 *   BWARP_TUNE_MERGE_LEVELS              levels of splitter merging below the top one          (1)
 *   BWARP_TUNE_STAGING_MB                pinned staging chunk for transfers, in MiB            (8)
 *
 * The constraints (capacity <= 2 * threads, threads a multiple of 32 from 64 to 1024) are checked here. A setting
 * a kernel can't launch with (the overflow merge at 1024 threads: too many registers) fails with a CUDA error.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef BWARP_TUNING
#include <cstdio>
#include <cstdlib>
#endif

namespace gpusa {
    struct Tuning {
        uint32_t threads = 256;
        uint32_t partitionThreshold = 1u << 22;
        int partitionBits = 8;
        uint32_t mergeG = 128;
        uint32_t mergeTile = 256;
        uint32_t mergeCapacity = 480;            // 23.4 KiB, four blocks per SM
        uint32_t mergeThreads = 256;
        uint32_t mergeOverflowCapacity = 960;    // two blocks per SM
        uint32_t mergeOverflowThreads = 512;
        int mergeLevels = 1;
        size_t stagingBytes = size_t(8) << 20;
    };

#ifdef BWARP_TUNING
    namespace tuningDetail {
        // Reads name into value if it is set and an integer in [low, high].
        template <typename T>
        void read(const char* name, T& value, long long low, long long high) {
            const char* text = std::getenv(name);

            if (!text || !*text) {
                return;
            }

            char* end = nullptr;
            const long long parsed = std::strtoll(text, &end, 0);

            if (*end != '\0' || parsed < low || parsed > high) {
                std::fprintf(stderr, "bwarp: ignoring %s=%s (must be %lld to %lld)\n", name, text, low, high);
                return;
            }

            value = static_cast<T>(parsed);
        }

        inline bool validThreads(uint32_t threads) {
            return threads >= 64 && threads <= 1024 && threads % 32 == 0;
        }

        inline Tuning load() {
            const Tuning defaults;
            Tuning t;
            uint32_t stagingMiB = static_cast<uint32_t>(t.stagingBytes >> 20);

            read("BWARP_TUNE_THREADS", t.threads, 64, 1024);
            read("BWARP_TUNE_PARTITION_THRESHOLD", t.partitionThreshold, 0, 0xffffffffll);
            read("BWARP_TUNE_PARTITION_BITS", t.partitionBits, 1, 32);
            read("BWARP_TUNE_MERGE_G", t.mergeG, 2, 1u << 20);
            read("BWARP_TUNE_MERGE_TILE", t.mergeTile, 16, 1u << 20);
            read("BWARP_TUNE_MERGE_CAPACITY", t.mergeCapacity, 32, 2048);
            read("BWARP_TUNE_MERGE_THREADS", t.mergeThreads, 64, 1024);
            read("BWARP_TUNE_MERGE_OVERFLOW_CAPACITY", t.mergeOverflowCapacity, 32, 2048);
            read("BWARP_TUNE_MERGE_OVERFLOW_THREADS", t.mergeOverflowThreads, 64, 1024);
            read("BWARP_TUNE_MERGE_LEVELS", t.mergeLevels, 0, 4);
            read("BWARP_TUNE_STAGING_MB", stagingMiB, 1, 1024);
            t.stagingBytes = size_t(stagingMiB) << 20;

            if (!validThreads(t.threads)) {
                std::fprintf(stderr, "bwarp: BWARP_TUNE_THREADS must be a multiple of 32, using %u\n", defaults.threads);
                t.threads = defaults.threads;
            }

            if (!validThreads(t.mergeThreads) || t.mergeCapacity > 2 * t.mergeThreads) {
                std::fprintf(stderr, "bwarp: merge threads must be a multiple of 32 and capacity <= 2 * threads, "
                                     "using %u records, %u threads\n", defaults.mergeCapacity, defaults.mergeThreads);
                t.mergeCapacity = defaults.mergeCapacity;
                t.mergeThreads = defaults.mergeThreads;
            }

            if (!validThreads(t.mergeOverflowThreads) || t.mergeOverflowCapacity > 2 * t.mergeOverflowThreads) {
                std::fprintf(stderr, "bwarp: overflow merge threads must be a multiple of 32 and capacity <= "
                                     "2 * threads, using %u records, %u threads\n",
                    defaults.mergeOverflowCapacity, defaults.mergeOverflowThreads);
                t.mergeOverflowCapacity = defaults.mergeOverflowCapacity;
                t.mergeOverflowThreads = defaults.mergeOverflowThreads;
            }

            std::fprintf(stderr,
                "bwarp: tuning threads=%u partition_threshold=%u partition_bits=%d merge_g=%u merge_tile=%u "
                "merge_capacity=%u merge_threads=%u merge_overflow_capacity=%u merge_overflow_threads=%u "
                "merge_levels=%d staging_mb=%u\n",
                t.threads, t.partitionThreshold, t.partitionBits, t.mergeG, t.mergeTile, t.mergeCapacity,
                t.mergeThreads, t.mergeOverflowCapacity, t.mergeOverflowThreads, t.mergeLevels, stagingMiB);

            return t;
        }
    }

    inline const Tuning& tuning() {
        static const Tuning t = tuningDetail::load();
        return t;
    }
#else
    inline const Tuning& tuning() {
        static constexpr Tuning t{};
        return t;
    }
#endif
}
