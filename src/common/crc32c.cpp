/*
 * crc32c.cpp - Implementation of the CRC-32C checksum helper functions.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "crc32c.h"
#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <nmmintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#define BWARP_CRC32C_SSE42 1
#endif

#if defined(__clang__) || defined(__GNUC__)
#define BWARP_TARGET_SSE42 __attribute__((target("sse4.2")))
#else
#define BWARP_TARGET_SSE42
#endif

namespace {
    constexpr uint32_t kPolynomial = 0x82F63B78;  // reflected

    struct Tables {
        uint32_t byte[256];   // one byte
        uint32_t x2n[32];     // x^(2^k) mod P, for combining

        Tables() {
            for (uint32_t i = 0; i < 256; i++) {
                uint32_t c = i;

                for (int k = 0; k < 8; k++) {
                    c = c & 1 ? (c >> 1) ^ kPolynomial : c >> 1;
                }

                byte[i] = c;
            }
        }
    };

    // a * b mod P, polynomials over GF(2), reflected as CRCs are (zlib's multmodp()).
    uint32_t multiplyModP(uint32_t a, uint32_t b) {
        uint32_t m = 1u << 31, p = 0;

        for (;;) {
            if (a & m) {
                p ^= b;

                if ((a & (m - 1)) == 0) {
                    break;
                }
            }

            m >>= 1;
            b = b & 1 ? (b >> 1) ^ kPolynomial : b >> 1;
        }

        return p;
    }

    const Tables& tables() {
        static const Tables t = [] {
            Tables made;
            uint32_t p = 1u << 30;  // x^1

            for (uint32_t& entry : made.x2n) {
                entry = p;
                p = multiplyModP(p, p);
            }

            return made;
        }();

        return t;
    }

    // x^(n * 2^k) mod P.
    uint32_t x2nModP(uint64_t n, uint32_t k) {
        uint32_t p = 1u << 31;  // x^0

        while (n) {
            if (n & 1) {
                p = multiplyModP(tables().x2n[k & 31], p);
            }

            n >>= 1;
            k++;
        }

        return p;
    }

    // The raw register, without the pre- and post-inversion.
    uint32_t updateTable(uint32_t reg, const uint8_t* p, size_t size) {
        const Tables& t = tables();

        for (size_t i = 0; i < size; i++) {
            reg = t.byte[(reg ^ p[i]) & 255] ^ (reg >> 8);
        }

        return reg;
    }

#if defined(BWARP_CRC32C_SSE42)
    bool hasSse42() {
        static const bool has = [] {
#if defined(_MSC_VER)
            int info[4];
            __cpuid(info, 1);
            return (info[2] & (1 << 20)) != 0;
#else
            unsigned a, b, c, d;
            return __get_cpuid(1, &a, &b, &c, &d) && (c & (1u << 20)) != 0;
#endif
        }();

        return has;
    }

    // Three independent streams over thirds of the data, as the instruction's latency is three times its
    // throughput, combined at the end.
    BWARP_TARGET_SSE42 uint32_t updateSse42(uint32_t reg, const uint8_t* p, size_t size) {
        uint64_t r = reg;

        for (; size > 0 && reinterpret_cast<uintptr_t>(p) % 8 != 0; size--) {
            r = _mm_crc32_u8(static_cast<uint32_t>(r), *p++);
        }

        constexpr size_t kMinStriped = 3 * 4096;

        if (size >= kMinStriped) {
            const size_t third = size / 24 * 8;
            uint64_t r1 = 0, r2 = 0;
            const uint8_t* p1 = p + third;
            const uint8_t* p2 = p + 2 * third;

            for (size_t i = 0; i < third; i += 8) {
                uint64_t v0, v1, v2;
                memcpy(&v0, p + i, 8);
                memcpy(&v1, p1 + i, 8);
                memcpy(&v2, p2 + i, 8);
                r = _mm_crc32_u64(r, v0);
                r1 = _mm_crc32_u64(r1, v1);
                r2 = _mm_crc32_u64(r2, v2);
            }

            // Raw registers combine as CRCs do without the inversions: shift the first past the others.
            const uint32_t shift = x2nModP(third, 3);
            r = multiplyModP(shift, multiplyModP(shift, static_cast<uint32_t>(r)) ^ static_cast<uint32_t>(r1)) ^
                static_cast<uint32_t>(r2);
            p += 3 * third;
            size -= 3 * third;
        }

        for (; size >= 8; size -= 8, p += 8) {
            uint64_t v;
            memcpy(&v, p, 8);
            r = _mm_crc32_u64(r, v);
        }

        for (; size > 0; size--) {
            r = _mm_crc32_u8(static_cast<uint32_t>(r), *p++);
        }

        return static_cast<uint32_t>(r);
    }
#endif

    uint32_t update(uint32_t reg, const uint8_t* p, size_t size) {
#if defined(BWARP_CRC32C_SSE42)
        if (hasSse42()) {
            return updateSse42(reg, p, size);
        }
#endif

        return updateTable(reg, p, size);
    }

    uint32_t crcOf(const uint8_t* p, size_t size, uint32_t crc) {
        return ~update(~crc, p, size);
    }
}

uint32_t crc32cCombine(uint32_t crcA, uint32_t crcB, uint64_t sizeB) {
    return multiplyModP(x2nModP(sizeB, 3), crcA) ^ crcB;
}

uint32_t crc32c(const void* data, size_t size, uint32_t crc) {
    const auto* p = static_cast<const uint8_t*>(data);

    // Large inputs in parts on as many threads, at most 8, as memory bandwidth runs out beyond.
    constexpr size_t kMinPart = size_t(16) << 20;
    const size_t threads = std::min<size_t>({8, std::max(1u, std::thread::hardware_concurrency()), size / kMinPart});

    if (threads < 2) {
        return crcOf(p, size, crc);
    }

    const size_t part = size / threads;
    std::vector<uint32_t> crcs(threads);
    std::vector<std::thread> workers;

    for (size_t t = 1; t < threads; t++) {
        workers.emplace_back([&, t] {
            crcs[t] = crcOf(p + t * part, t + 1 < threads ? part : size - t * part, 0);
        });
    }

    crcs[0] = crcOf(p, part, crc);

    for (std::thread& worker : workers) {
        worker.join();
    }

    for (size_t t = 1; t < threads; t++) {
        crc = crc32cCombine(t == 1 ? crcs[0] : crc, crcs[t], t + 1 < threads ? part : size - t * part);
    }

    return crc;
}
