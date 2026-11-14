/*
 * selftest.cpp - The checker on known BWTs, then the GPU construction verified by the checker.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "selftest.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "checker.h"
#include "cpucodec.h"
#include "gpubwarp.h"

namespace {
    constexpr int32_t kLargeTextSize = 1 << 22;

    // splitmix64, with fixed seeds.
    struct Rng {
        uint64_t state;

        uint64_t next() {
            uint64_t z = (state += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        }

        uint32_t nextBelow(uint32_t bound) {
            return static_cast<uint32_t>(next() % bound);
        }
    };

    struct Suite {
        bool verbose;
        int cases = 0;
        int failures = 0;

        void reportFailure(const std::string& name, const char* what, int64_t index) {
            failures++;
            printf("  FAIL %s: %s (index %lld)\n", name.c_str(), what, static_cast<long long>(index));
        }
    };

    struct Bwt {
        std::vector<uint8_t> bytes;
        int32_t primary = 0;
    };

    std::vector<uint8_t> randomText(Rng& rng, int32_t n, uint32_t alphabetSize) {
        std::vector<uint8_t> text(n);

        for (uint8_t& c : text) {
            c = static_cast<uint8_t>(alphabetSize == 256 ? rng.nextBelow(256) : 'a' + rng.nextBelow(alphabetSize));
        }

        return text;
    }

    std::vector<uint8_t> periodicText(Rng& rng, int32_t n, int32_t period) {
        const std::vector<uint8_t> pattern = randomText(rng, period, 4);
        std::vector<uint8_t> text(n);

        for (int32_t i = 0; i < n; i++) {
            text[i] = pattern[i % period];
        }

        return text;
    }

    std::vector<uint8_t> fibonacciText(int32_t n, uint8_t a, uint8_t b) {
        std::vector<uint8_t> shorter{a};
        std::vector<uint8_t> longer{a, b};

        while (static_cast<int32_t>(longer.size()) < n) {
            std::vector<uint8_t> next(longer);
            next.insert(next.end(), shorter.begin(), shorter.end());
            shorter = std::move(longer);
            longer = std::move(next);
        }

        longer.resize(n);

        return longer;
    }

    std::vector<uint8_t> thueMorseText(int32_t n) {
        std::vector<uint8_t> text(n);

        for (int32_t i = 0; i < n; i++) {
            text[i] = static_cast<uint8_t>('a' + (std::popcount(static_cast<uint32_t>(i)) & 1));
        }

        return text;
    }

    std::vector<uint8_t> textOf(const char* s) {
        return std::vector<uint8_t>(s, s + strlen(s));
    }

    // ---- checker -------------------------------------------------------------------------------

    // Worked out by hand.
    struct KnownBwt {
        const char* text;
        const char* bwt;
        int32_t primary;
    };

    constexpr KnownBwt kKnownBwts[] = {
        {"", "", 0},
        {"a", "a", 1},
        {"banana", "annbaa", 4},
        {"mississippi", "ipssmpissii", 5},
    };

    void testChecker(Suite& suite, const std::string& name, const std::vector<uint8_t>& text, const Bwt& bwt, bool valid) {
        const CheckResult result = checkBwt(text.data(), static_cast<int32_t>(text.size()), bwt.bytes.data(), bwt.primary);

        suite.cases++;

        if (result.ok != valid) {
            suite.reportFailure(name, valid ? result.reason : "not caught", result.index);
        } else if (suite.verbose) {
            printf("  ok   %s\n", name.c_str());
        }
    }

    void runCheckerCases(Suite& suite) {
        for (const KnownBwt& known : kKnownBwts) {
            const std::vector<uint8_t> text = textOf(known.text);
            const Bwt good{textOf(known.bwt), known.primary};
            const std::string name = "checker, \"" + std::string(known.text) + "\"";

            testChecker(suite, name, text, good, true);

            if (text.size() < 2) {
                continue;
            }

            Bwt swapped = good;
            const auto other = std::find_if(swapped.bytes.begin() + 1, swapped.bytes.end(),
                [&](uint8_t c) { return c != swapped.bytes[0]; });
            std::iter_swap(swapped.bytes.begin(), other);
            testChecker(suite, name + ", swapped bytes", text, swapped, false);

            for (int32_t primary = -1; primary <= static_cast<int32_t>(text.size()) + 1; primary++) {
                if (primary != good.primary) {
                    testChecker(suite, name + ", primary index " + std::to_string(primary), text,
                        Bwt{good.bytes, primary}, false);
                }
            }
        }
    }

    // ---- codec ---------------------------------------------------------------------------------

    void testCodec(Suite& suite, const std::string& name, const std::vector<uint8_t>& input, const EncodeParams& params) {
        const int32_t n = static_cast<int32_t>(input.size());
        const std::string label = name + " (2^" + std::to_string(params.logSegment) + ", 2^" +
                                  std::to_string(params.logSlice) + ", " + std::to_string(params.contexts) + ")";
        std::vector<uint8_t> encoded, decoded;
        int32_t primary = -1;

        suite.cases++;

        if (!encodeBwt(input.data(), n, 12345, params, encoded)) {
            suite.reportFailure(label, "encoder refused", -1);
            return;
        }

        if (!decodeBwt(encoded.data(), encoded.size(), decoded, primary) || primary != 12345 || decoded != input) {
            suite.reportFailure(label, "round trip differs", -1);
            return;
        }

        // Every truncation must be refused.
        suite.cases++;

        for (size_t size = 0; size < encoded.size(); size += std::max<size_t>(1, encoded.size() / 97)) {
            if (decodeBwt(encoded.data(), size, decoded, primary)) {
                suite.reportFailure(label + ", truncated", "not refused", static_cast<int64_t>(size));
                return;
            }
        }

        // A flipped byte must be refused or decode to as many bytes as the header says, never crash.
        suite.cases++;

        for (size_t at = 0; at < encoded.size(); at += std::max<size_t>(1, encoded.size() / 89)) {
            std::vector<uint8_t> corrupt = encoded;
            corrupt[at] ^= 0x5a;

            const size_t n = corrupt[8] | corrupt[9] << 8 | corrupt[10] << 16 | size_t(corrupt[11]) << 24;

            if (decodeBwt(corrupt.data(), corrupt.size(), decoded, primary) && decoded.size() != n) {
                suite.reportFailure(label + ", corrupted", "wrong size", static_cast<int64_t>(at));
                return;
            }
        }

        if (suite.verbose) {
            printf("  ok   %s, %zu -> %zu bytes\n", label.c_str(), input.size(), encoded.size());
        }
    }

    void runCodecCases(Suite& suite) {
        const EncodeParams paramSets[] = {{15, 15, 16}, {20, 15, 8}, {10, 10, 1}, {12, 10, 4}, {17, 17, 16}, {24, 12, 2}};
        Rng rng{5};
        std::vector<std::pair<std::string, std::vector<uint8_t>>> inputs = {
            {"\"\"", {}},
            {"\"a\"", textOf("a")},
            {"\"mississippi\"", textOf("mississippi")},
            {"run of 0xff", std::vector<uint8_t>(300000, 0xff)},
            {"random sigma=2", randomText(rng, 100000, 2)},
            {"random sigma=256", randomText(rng, 100000, 256)},
            {"period 7", periodicText(rng, 70001, 7)},
        };

        // Runs of random lengths up to 5000, longer than a lane at small slices.
        std::vector<uint8_t> runs;

        while (runs.size() < 400000) {
            runs.insert(runs.end(), 1 + rng.nextBelow(rng.nextBelow(2) ? 8 : 5000), static_cast<uint8_t>(rng.nextBelow(6)));
        }

        inputs.push_back({"random runs", runs});

        for (const EncodeParams& params : paramSets) {
            for (const auto& [name, input] : inputs) {
                testCodec(suite, name, input, params);
            }
        }
    }

    // ---- GPU -----------------------------------------------------------------------------------

    void testGpu(Suite& suite, const std::string& name, const std::vector<uint8_t>& input, GpuBwtContext* context = nullptr) {
        const int32_t n = static_cast<int32_t>(input.size());
        Bwt bwt;

        suite.cases++;
        bwt.bytes = input;

        if (gpuBwt(context, bwt.bytes.data(), n, bwt.bytes.data(), bwt.primary) != GpuStatus::Ok) {
            suite.reportFailure(name + " GPU", gpuLastError(), -1);
            return;
        }

        if (const CheckResult result = checkBwt(input.data(), n, bwt.bytes.data(), bwt.primary); !result.ok) {
            suite.reportFailure(name + " GPU", result.reason, result.index);
            return;
        }

        if (suite.verbose) {
            printf("  ok   %s GPU (n=%d)\n", name.c_str(), n);
        }
    }

    // Largest first, so each construction runs in memory an earlier one left dirty.
    void runGpuContextCases(Suite& suite, Rng& rng) {
        constexpr int32_t kCapacity = 20000000;
        GpuBwtContext* context = nullptr;

        suite.cases++;

        if (gpuCreateBwtContext(kCapacity, context) != GpuStatus::Ok) {
            suite.reportFailure("GPU context", gpuLastError(), -1);
            return;
        }

        testGpu(suite, "context, random sigma=4, n=20M", randomText(rng, kCapacity, 4), context);
        testGpu(suite, "context, large random sigma=256", randomText(rng, kLargeTextSize, 256), context);
        testGpu(suite, "context, \"banana\"", textOf("banana"), context);
        testGpu(suite, "context, \"\"", {}, context);
        testGpu(suite, "context, \"a\"", textOf("a"), context);
        testGpu(suite, "context, large fibonacci", fibonacciText(kLargeTextSize, 'a', 'b'), context);
        testGpu(suite, "context, period 33", periodicText(rng, 5000, 33), context);

        for (int i = 0; i < 5; i++) {
            const int32_t n = 1 + static_cast<int32_t>(rng.nextBelow(20000));
            testGpu(suite, "context, random sigma=2 #" + std::to_string(i), randomText(rng, n, 2), context);
        }

        // Larger than the capacity: refused, and the context still works afterwards.
        std::vector<uint8_t> tooLarge(kCapacity + 1, 'a');
        int32_t primary = 0;

        suite.cases++;

        if (gpuBwt(context, tooLarge.data(), kCapacity + 1, tooLarge.data(), primary) != GpuStatus::NotEnoughMemory) {
            suite.reportFailure("GPU context, over capacity", "not refused with NotEnoughMemory", -1);
        }

        testGpu(suite, "context, after refusal", randomText(rng, 5000, 26), context);
        gpuDestroyBwtContext(context);
    }

    void runGpuCases(Suite& suite) {
        GpuDevice device;

        if (gpuQueryDevice(device) != GpuStatus::Ok) {
            printf("selftest: GPU cases skipped (%s)\n", gpuLastError());
            return;
        }

        printf("selftest: GPU BWT on %s (sm_%d%d, %lld MiB free)...\n", device.name, device.computeMajor,
            device.computeMinor, static_cast<long long>(device.freeBytes >> 20));
        Rng rng{3};

        testGpu(suite, "\"\"", {});
        testGpu(suite, "\"a\"", textOf("a"));
        testGpu(suite, "\"banana\"", textOf("banana"));
        testGpu(suite, "\"mississippi\"", textOf("mississippi"));
        testGpu(suite, "run of a", std::vector<uint8_t>(1000, 'a'));
        testGpu(suite, "fibonacci", fibonacciText(5000, 'a', 'b'));
        testGpu(suite, "fibonacci 0/1", fibonacciText(5000, 0, 1));
        testGpu(suite, "thue-morse", thueMorseText(4096));

        for (const int32_t period : {1, 2, 3, 5, 7, 31, 32, 33, 63, 64, 65, 1000}) {
            testGpu(suite, "period " + std::to_string(period), periodicText(rng, 5000, period));
        }

        for (int32_t n = 0; n <= 80; n++) {
            testGpu(suite, "run of a, n=" + std::to_string(n), std::vector<uint8_t>(n, 'a'));
            testGpu(suite, "random sigma=2, n=" + std::to_string(n), randomText(rng, n, 2));
        }

        for (const uint32_t alphabetSize : {1u, 2u, 4u, 26u, 256u}) {
            for (int i = 0; i < 10; i++) {
                const int32_t n = 1 + static_cast<int32_t>(rng.nextBelow(20000));
                testGpu(suite, "random sigma=" + std::to_string(alphabetSize) + " #" + std::to_string(i),
                    randomText(rng, n, alphabetSize));
            }
        }

        testGpu(suite, "large run of a", std::vector<uint8_t>(kLargeTextSize, 'a'));
        testGpu(suite, "large period 33", periodicText(rng, kLargeTextSize, 33));
        testGpu(suite, "large fibonacci", fibonacciText(kLargeTextSize, 'a', 'b'));
        testGpu(suite, "large thue-morse", thueMorseText(kLargeTextSize));
        testGpu(suite, "large random sigma=4", randomText(rng, kLargeTextSize, 4));
        testGpu(suite, "large random sigma=256", randomText(rng, kLargeTextSize, 256));

        runGpuContextCases(suite, rng);
    }
}

int runSelfTest(bool verbose) {
    Suite suite{verbose};

    printf("selftest: checker on known BWTs and planted errors...\n");
    runCheckerCases(suite);
    printf("selftest: codec round trips and truncations...\n");
    runCodecCases(suite);
    runGpuCases(suite);

    printf("selftest: %d of %d cases passed\n", suite.cases - suite.failures, suite.cases);

    return suite.failures;
}
