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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include "checker.h"
#include "container.h"
#include "cpucodec.h"
#include "crc32c.h"
#include "fileio.h"
#include "format.h"
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

    // ---- checksum ------------------------------------------------------------------------------

    uint32_t crc32cBitwise(const uint8_t* p, size_t size) {
        uint32_t c = ~0u;

        for (size_t i = 0; i < size; i++) {
            c ^= p[i];

            for (int k = 0; k < 8; k++) {
                c = c & 1 ? (c >> 1) ^ 0x82F63B78u : c >> 1;
            }
        }

        return ~c;
    }

    // Against the bit-by-bit definition at every alignment and around the striped loop's threshold, then a large
    // input, split over threads, against its parts chained.
    void runChecksumCases(Suite& suite) {
        printf("selftest: CRC-32C...\n");

        Rng rng{11};
        const std::vector<uint8_t> data = randomText(rng, 100000, 256);

        suite.cases++;

        if (crc32c("123456789", 9) != 0xE3069283u) {
            suite.reportFailure("CRC-32C, \"123456789\"", "differs from the check value", -1);
        }

        for (const size_t start : {0, 1, 3, 7}) {
            for (const size_t size : {0, 1, 7, 8, 9, 100, 4095, 12287, 12288, 12289, 12300, 40000, 99990}) {
                const uint8_t* p = data.data() + start;
                const size_t split = size / 3;
                const uint32_t whole = crc32c(p, size);

                suite.cases++;

                if (whole != crc32cBitwise(p, size) || crc32c(p + split, size - split, crc32c(p, split)) != whole ||
                    crc32cCombine(crc32c(p, split), crc32c(p + split, size - split), size - split) != whole) {
                    suite.reportFailure("CRC-32C, " + std::to_string(size) + " bytes at " + std::to_string(start),
                        "differs from the definition", -1);
                }
            }
        }

        std::vector<uint8_t> large((size_t(40) << 20) + 13);

        for (size_t i = 0; i < large.size(); i += 8) {
            const uint64_t v = rng.next();
            memcpy(large.data() + i, &v, std::min<size_t>(8, large.size() - i));
        }

        uint32_t chained = 0;

        for (size_t at = 1; at < large.size(); at += size_t(1) << 20) {
            chained = crc32c(large.data() + at, std::min(size_t(1) << 20, large.size() - at), chained);
        }

        suite.cases++;

        if (crc32c(large.data() + 1, large.size() - 1) != chained) {
            suite.reportFailure("CRC-32C, 40 MiB on threads", "differs from its parts chained", -1);
        }
    }

    // ---- codec ---------------------------------------------------------------------------------

    // The GPU decoder must accept exactly what the CPU decoder accepts, with the same result, and so must the
    // GPU decompressor, whose output must have the same size. The CPU decoder doesn't invert the BWT, so the
    // decompressor's checksum mismatches count as accepted.
    bool sameAsCpu(const std::vector<uint8_t>& encoded, size_t size) {
        std::vector<uint8_t> cpu, gpu, text;
        int32_t cpuPrimary = -1, gpuPrimary = -1;
        const bool cpuOk = decodeBwt(encoded.data(), size, cpu, cpuPrimary);
        const GpuStatus status = gpuDecode(encoded.data(), size, gpu, gpuPrimary);
        GpuStatus inverseStatus = gpuDecompress(encoded.data(), size, text);

        if (inverseStatus == GpuStatus::ChecksumMismatch) {
            inverseStatus = GpuStatus::Ok;
        }

        for (const GpuStatus st : {status, inverseStatus}) {
            if (st != GpuStatus::Ok && st != GpuStatus::Invalid) {
                printf("  GPU decoder: %s\n", gpuLastError());
                return false;
            }
        }

        return cpuOk == (status == GpuStatus::Ok) && cpuOk == (inverseStatus == GpuStatus::Ok) &&
               (!cpuOk || (cpu == gpu && cpuPrimary == gpuPrimary && text.size() == cpu.size()));
    }

    void testCodec(Suite& suite, const std::string& name, const std::vector<uint8_t>& input, const EncodeParams& params,
        bool gpu) {
        const int32_t n = static_cast<int32_t>(input.size());
        const std::string label = name + " (2^" + std::to_string(params.logSegment) + ", 2^" +
                                  std::to_string(params.logSlice) + ", " + std::to_string(params.contexts) + ", 2^" +
                                  std::to_string(params.logSample) + ")";
        std::vector<uint8_t> encoded, decoded;
        std::vector<uint32_t> decodedSamples;
        int32_t primary = -1;

        // Not a BWT, so any rows in range will do.
        const int32_t inputPrimary = n / 2 + (n > 0 ? 1 : 0);
        std::vector<uint32_t> samples(bwarp::sampleCount(static_cast<uint32_t>(n), bwarp::kMinLogSample));

        for (size_t j = 0; j < samples.size(); j++) {
            samples[j] = static_cast<uint32_t>((j * 7919) % static_cast<size_t>(n) + 1);
        }

        suite.cases++;

        // Not a BWT either, so any checksum will do: decodeBwt() only passes it on.
        const uint32_t checksum = crc32c(input.data(), input.size());

        if (!encodeBwt(input.data(), n, inputPrimary, checksum, samples.data(), bwarp::kMinLogSample, params,
                encoded)) {
            suite.reportFailure(label, "encoder refused", -1);
            return;
        }

        uint32_t decodedChecksum = 0;
        bool same = decodeBwt(encoded.data(), encoded.size(), decoded, primary, &decodedSamples, &decodedChecksum) &&
                    primary == inputPrimary && decoded == input && decodedChecksum == checksum;
        const int logSample = encoded.size() > 8 ? encoded[8] : 0;

        for (size_t j = 0; same && j < decodedSamples.size(); j++) {
            same = decodedSamples[j] == samples[((j + 1) << (logSample - bwarp::kMinLogSample)) - 1];
        }

        if (!same) {
            suite.reportFailure(label, "round trip differs", -1);
            return;
        }

        if (gpu && !sameAsCpu(encoded, encoded.size())) {
            suite.reportFailure(label, "GPU decoder differs", -1);
            return;
        }

        if (std::vector<uint8_t> gpuEncoded;
            gpu && (gpuEncode(input.data(), n, inputPrimary, checksum, samples.data(), bwarp::kMinLogSample, params,
                        gpuEncoded) != GpuStatus::Ok ||
                       gpuEncoded != encoded)) {
            suite.reportFailure(label, "GPU encoder differs", -1);
            return;
        }

        // Every truncation must be refused.
        suite.cases++;

        for (size_t size = 0; size < encoded.size(); size += std::max<size_t>(1, encoded.size() / 97)) {
            if (decodeBwt(encoded.data(), size, decoded, primary) || (gpu && !sameAsCpu(encoded, size))) {
                suite.reportFailure(label + ", truncated", "not refused", static_cast<int64_t>(size));
                return;
            }
        }

        // A flipped byte must be refused or decode to as many bytes as the header says, never crash.
        suite.cases++;

        for (size_t at = 0; at < encoded.size(); at += std::max<size_t>(1, encoded.size() / 89)) {
            std::vector<uint8_t> corrupt = encoded;
            corrupt[at] ^= 0x5a;

            const size_t n = corrupt[12] | corrupt[13] << 8 | corrupt[14] << 16 | size_t(corrupt[15]) << 24;

            if (decodeBwt(corrupt.data(), corrupt.size(), decoded, primary) && decoded.size() != n) {
                suite.reportFailure(label + ", corrupted", "wrong size", static_cast<int64_t>(at));
                return;
            }

            if (gpu && !sameAsCpu(corrupt, corrupt.size())) {
                suite.reportFailure(label + ", corrupted", "GPU decoder differs", static_cast<int64_t>(at));
                return;
            }
        }

        if (suite.verbose) {
            printf("  ok   %s, %zu -> %zu bytes\n", label.c_str(), input.size(), encoded.size());
        }
    }

    void runCodecCases(Suite& suite) {
        const EncodeParams paramSets[] = {{15, 15, 16, 0}, {20, 15, 8, 8}, {10, 10, 1, 31}, {12, 10, 4, 9},
            {16, 16, 16, 16}, {24, 12, 2, 0}};
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

        GpuDevice device;
        const bool gpu = gpuQueryDevice(device) == GpuStatus::Ok;

        printf("selftest: codec round trips, truncations and corruptions%s...\n", gpu ? ", CPU and GPU decoders" : "");

        for (const EncodeParams& params : paramSets) {
            for (const auto& [name, input] : inputs) {
                testCodec(suite, name, input, params, gpu);
            }
        }
    }

    // ---- commands ------------------------------------------------------------------------------

    namespace fs = std::filesystem;

    // Runs "bwarp <args>": true if it succeeded.
    bool runArgs(CommandRunner command, std::vector<std::string> args) {
        std::vector<char*> argv;

        args.insert(args.begin(), "bwarp");

        for (std::string& arg : args) {
            argv.push_back(arg.data());
        }

        return command(static_cast<int>(argv.size()), argv.data()) == 0;
    }

    bool fileIs(const fs::path& path, const std::vector<uint8_t>& expected) {
        std::vector<uint8_t> bytes;
        return readWholeFile(path.string().c_str(), bytes) && bytes == expected;
    }

    // compress and decompress on a directory of temporary files, in blocks of 1 GiB and of 64 KiB (several per file,
    // one file exactly two), then the failures: an existing output, a wrong checksum in a later block, a truncation.
    // Failed outputs must be gone.
    void runCommandCases(Suite& suite, CommandRunner command) {
        GpuDevice device;

        if (gpuQueryDevice(device) != GpuStatus::Ok) {
            printf("selftest: command cases skipped (%s)\n", gpuLastError());
            return;
        }

        printf("selftest: compress and decompress commands on temporary files (errors below are expected)...\n");

        std::error_code error;
        const fs::path root = fs::temp_directory_path(error) /
                              ("bwarp-selftest-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        Rng rng{7};
        std::vector<uint8_t> runs;

        while (runs.size() < 200000) {
            runs.insert(runs.end(), 1 + rng.nextBelow(300), static_cast<uint8_t>('a' + rng.nextBelow(3)));
        }

        const std::vector<std::pair<std::string, std::vector<uint8_t>>> files = {
            {"empty", {}},
            {"one", textOf("a")},
            {"text", randomText(rng, 300000, 26)},
            {"sub/runs", runs},
            {"sub/exact", periodicText(rng, 131072, 33)},
        };

        bool ok = fs::create_directories(root / "in" / "sub", error);

        for (const auto& [name, bytes] : files) {
            ok = ok && writeWholeFile((root / "in" / name).string().c_str(), bytes.data(), bytes.size());
        }

        if (!ok) {
            suite.cases++;
            suite.reportFailure("commands", "cannot write the temporary files", -1);
            fs::remove_all(root, error);
            return;
        }

        const std::string in = (root / "in").string(), packed = (root / "packed").string(),
                          out = (root / "out").string();

        for (const char* block : {"1073741824", "65536"}) {
            const std::string label = std::string("commands, blocks of ") + block;

            suite.cases++;

            if (!runArgs(command, {"compress", in, packed, "--recursive", "--force", "--quiet", "--block", block}) ||
                !runArgs(command, {"decompress", packed, out, "--recursive", "--force", "--quiet"})) {
                suite.reportFailure(label, "a command failed", -1);
                continue;
            }

            for (const auto& [name, bytes] : files) {
                if (!fileIs(root / "out" / name, bytes)) {
                    suite.reportFailure(label + ", " + name, "round trip differs", -1);
                }
            }

            if (suite.verbose) {
                printf("  ok   %s\n", label.c_str());
            }
        }

        // packed holds blocks of 64 KiB: text.bwarp five of them.
        const fs::path text = root / "packed" / "text.bwarp", broken = root / "broken.bwarp", result = root / "result";
        std::vector<uint8_t> bytes;
        bwarp::BlockHeader header;

        suite.cases++;

        if (runArgs(command, {"compress", in, packed, "--recursive", "--quiet"})) {
            suite.reportFailure("commands, existing outputs", "overwritten without --force", -1);
        }

        suite.cases++;

        if (!readWholeFile(text.string().c_str(), bytes) || bytes.size() < bwarp::kHeaderBytes ||
            !bwarp::readBlockHeader(bytes.data(), header)) {
            suite.reportFailure("commands, wrong checksum", "cannot read the compressed file", -1);
        } else {
            // The second block's checksum.
            const size_t second = bwarp::get64(bytes.data() + header.layout.indexBytes() - 8);
            std::vector<uint8_t> wrong = bytes;

            wrong[second + 20] ^= 0x10;

            if (!writeWholeFile(broken.string().c_str(), wrong.data(), wrong.size()) ||
                runArgs(command, {"decompress", broken.string(), result.string(), "--quiet"}) || fs::exists(result)) {
                suite.reportFailure("commands, wrong checksum", "not refused, or the output left", -1);
            }

            // Cut by a byte, at the end of the first block, and with a whole file after its last block.
            std::vector<uint8_t> appended = bytes;

            appended.insert(appended.end(), bytes.begin(), bytes.end());

            const std::pair<const char*, std::vector<uint8_t>> brokenFiles[] = {
                {"commands, truncated by a byte", std::vector<uint8_t>(bytes.begin(), bytes.end() - 1)},
                {"commands, truncated after a block", std::vector<uint8_t>(bytes.begin(), bytes.begin() + second)},
                {"commands, bytes after the last block", appended},
            };

            for (const auto& [label, contents] : brokenFiles) {
                suite.cases++;

                if (!writeWholeFile(broken.string().c_str(), contents.data(), contents.size()) ||
                    runArgs(command, {"decompress", broken.string(), result.string(), "--quiet", "--force"}) ||
                    fs::exists(result)) {
                    suite.reportFailure(label, "not refused, or the output left", -1);
                }
            }
        }

        fs::remove_all(root, error);
    }

    // ---- GPU -----------------------------------------------------------------------------------

    // The construction, checked, then compressed and decompressed on the GPU.
    void testGpu(Suite& suite, const std::string& name, const std::vector<uint8_t>& input, GpuBwtContext* context = nullptr) {
        const int32_t n = static_cast<int32_t>(input.size());
        EncodeParams params;
        Bwt bwt;

        params.logSample = bwarp::kMinLogSample;  // the most chains

        std::vector<uint32_t> samples(bwarp::sampleCount(static_cast<uint32_t>(n), params.logSample));

        suite.cases++;
        bwt.bytes = input;

        if (gpuBwt(context, bwt.bytes.data(), n, bwt.bytes.data(), bwt.primary, nullptr, samples.data(),
                params.logSample) != GpuStatus::Ok) {
            suite.reportFailure(name + " GPU", gpuLastError(), -1);
            return;
        }

        if (const CheckResult result = checkBwt(input.data(), n, bwt.bytes.data(), bwt.primary); !result.ok) {
            suite.reportFailure(name + " GPU", result.reason, result.index);
            return;
        }

        std::vector<uint8_t> encoded, text;
        const uint32_t checksum = crc32c(input.data(), input.size());

        suite.cases++;

        if (!encodeBwt(bwt.bytes.data(), n, bwt.primary, checksum, samples.data(), params.logSample, params,
                encoded)) {
            suite.reportFailure(name + " GPU round trip", "encoder refused", -1);
            return;
        }

        if (std::vector<uint8_t> gpuEncoded;
            gpuEncode(bwt.bytes.data(), n, bwt.primary, checksum, samples.data(), params.logSample, params,
                gpuEncoded) != GpuStatus::Ok ||
            gpuEncoded != encoded) {
            suite.reportFailure(name + " GPU round trip", "GPU encoder differs", -1);
            return;
        }

        // The whole compression on the device, which records samples every 2^kAutoMinLogSample positions: with
        // automatic sampling, the CPU encoder picks the same from those recorded here.
        if (context) {
            EncodeParams automatic = params;
            std::vector<uint8_t> expected;
            const uint8_t* compressed = nullptr;
            size_t size = 0;

            automatic.logSample = 0;
            suite.cases++;

            if (!encodeBwt(bwt.bytes.data(), n, bwt.primary, checksum, samples.data(), params.logSample, automatic,
                    expected) ||
                gpuCompress(context, input.data(), n, automatic, compressed, size) != GpuStatus::Ok ||
                size != expected.size() || !std::equal(compressed, compressed + size, expected.begin())) {
                suite.reportFailure(name + " GPU compression", "differs from the CPU encoder", -1);
                return;
            }
        }

        if (gpuDecompress(encoded.data(), encoded.size(), text) != GpuStatus::Ok) {
            suite.reportFailure(name + " GPU round trip", gpuLastError(), -1);
            return;
        }

        if (text != input) {
            const auto diff = std::mismatch(text.begin(), text.end(), input.begin(), input.end());
            suite.reportFailure(name + " GPU round trip", "text differs", diff.first - text.begin());
            return;
        }

        // A stored checksum one bit off.
        std::vector<uint8_t> wrong = encoded;

        wrong[20] ^= 1;
        suite.cases++;

        if (gpuDecompress(wrong.data(), wrong.size(), text) != GpuStatus::ChecksumMismatch) {
            suite.reportFailure(name + " GPU round trip", "wrong checksum not caught", -1);
            return;
        }

        // Into a buffer of exactly the text's size, then one byte short of it.
        if (n > 0) {
            std::vector<uint8_t> buffer(input.size());
            size_t size = 0;

            suite.cases++;

            if (gpuDecompress(encoded.data(), encoded.size(), buffer.data(), buffer.size(), size) != GpuStatus::Ok ||
                size != input.size() || buffer != input ||
                gpuDecompress(encoded.data(), encoded.size(), buffer.data(), buffer.size() - 1, size) !=
                    GpuStatus::NotEnoughMemory) {
                suite.reportFailure(name + " GPU round trip", "decompressing into a buffer", -1);
                return;
            }
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

int runSelfTest(bool verbose, CommandRunner command) {
    Suite suite{verbose};

    printf("selftest: checker on known BWTs and planted errors...\n");
    runCheckerCases(suite);
    runChecksumCases(suite);
    runCodecCases(suite);
    runGpuCases(suite);
    runCommandCases(suite, command);

    printf("selftest: %d of %d cases passed\n", suite.cases - suite.failures, suite.cases);

    return suite.failures;
}
