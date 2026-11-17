/*
 * main.cpp - The bwarp command line.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <system_error>
#include <vector>
#include "checker.h"
#include "container.h"
#include "cpucodec.h"
#include "crc32c.h"
#include "format.h"
#include "fileio.h"
#include "gpubwarp.h"
#include "selftest.h"

namespace {
    constexpr int kExitOk = 0;
    constexpr int kExitError = 1;
    constexpr int kMaxPaths = 2;
    constexpr int64_t kMaxTextSize = INT32_MAX - 1024;  // sadcs's limit

    struct Options {
        const char* paths[kMaxPaths] = {};
        int pathCount = 0;
        bool verbose = false;
        bool recursive = false;
        int runs = 5;
        bool context = false;
        bool pinned = false;
        bool gpu = false;
        bool force = false;
        bool quiet = false;
        int64_t blockBytes = bwarp::kMaxBlockBytes;
        EncodeParams encode;
    };

    struct TestRun {
        int64_t bytes = 0;
        double gpuSeconds = 0.0;  // including transfers
        GpuBwtStats gpu;
        double checkSeconds = 0.0;
        double peakPerByte = 0.0;  // device bytes per input byte; the largest in a sum
        bool ok = true;

        void add(const TestRun& other) {
            bytes += other.bytes;
            gpuSeconds += other.gpuSeconds;
            gpu.deviceBytes = std::max(gpu.deviceBytes, other.gpu.deviceBytes);
            peakPerByte = std::max(peakPerByte, other.peakPerByte);
            checkSeconds += other.checkSeconds;
        }
    };

    double secondsSince(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    void printUsage() {
        printf(
            "bwarp - a GPU-based BWT file compressor\n"
            "\n"
            "usage:\n"
            "  bwarp compress <in> [<out>] | <dir> <outdir> [--segment LOG2] [--slice LOG2] [--contexts K]\n"
            "              [--sample LOG2] [--block BYTES] [--recursive] [--force] [--quiet | --verbose]\n"
            "  bwarp decompress <in.bwarp> [<out>] | <dir> <outdir> [--recursive] [--force]\n"
            "              [--quiet | --verbose]\n"
            "  bwarp info\n"
            "  bwarp bwt <in> <out.bwt>\n"
            "  bwarp check <in> <in.bwt>\n"
            "  bwarp test <in|dir> [--recursive]\n"
            "  bwarp bench <in|dir> [--runs N] [--context] [--pinned] [--recursive]\n"
            "  bwarp ratio <in|dir> [--segment LOG2] [--slice LOG2] [--contexts K] [--sample LOG2]\n"
            "              [--gpu [--verbose]] [--recursive]\n"
            "  bwarp selftest [--verbose]\n"
            "\n"
            "compress compresses on the GPU, BWT and encoder, into <in>.bwarp unless <out> is given; a\n"
            "directory's files go to the same names plus .bwarp under <outdir>. decompress does the\n"
            "reverse, of a directory's .bwarp files. A file is compressed in blocks of --block bytes,\n"
            "at most and by default 2^30 (1 GiB), each with the CRC-32C of its text, which decompress\n"
            "checks. Both read the next block while the GPU works on the current one and write the\n"
            "one before meanwhile, refuse to overwrite files without --force and remove incomplete\n"
            "outputs. Their times include the setup before the first file, as a whole run's would\n"
            "(ns/byte: as the Large Text Compression Benchmark reports). --quiet prints only errors;\n"
            "--verbose adds each block's phases.\n"
            "The .bwt file is divsufsort's bwt format as one block: n, the primary index, the n bytes.\n"
            "info prints the GPU, or that bwarp was built without CUDA.\n"
            "bwt builds the BWT on the GPU; check verifies a BWT file against its text by inverting it\n"
            "on the CPU.\n"
            "test builds and checks in memory; on a directory it runs every file in it (--recursive:\n"
            "and below) and prints a timing table.\n"
            "bench times the GPU build: a warmup, then the best of --runs (default 5) host-to-host runs,\n"
            "with each phase of the best one. It doesn't check the output; test does. --context reuses\n"
            "one GPU context per file, created before the runs and not timed. --pinned uses page-locked\n"
            "host buffers, allocated before the runs (--context --pinned is how libcubwt measures).\n"
            "ratio builds the BWT on the GPU, encodes and decodes it with the CPU reference codec,\n"
            "checks the round trip and prints the compressed sizes. --segment (default 15): frequency\n"
            "tables per 2^LOG2 bytes; --slice (default 15, 10 to 16): one 32-lane stream per 2^LOG2\n"
            "bytes; --contexts (default 16, 1 to 16): the most symbol contexts a table may use;\n"
            "--sample (10 to 31; default automatic, see chooseLogSample()): an inverse BWT chain per\n"
            "2^LOG2 bytes.\n"
            "--gpu also encodes on the GPU and decompresses there, checks both, and prints their\n"
            "speeds without transfers (the faster of two runs), that of the stream decoder and that\n"
            "of the inverse BWT; gcmp is the whole compression on the GPU, BWT and encoder, host to\n"
            "host. --verbose adds the phases.\n");
    }

    bool parseOptions(int argc, char** argv, Options& options) {
        for (int i = 2; i < argc; i++) {
            const char* arg = argv[i];

            if (strcmp(arg, "--runs") == 0 && i + 1 < argc) {
                options.runs = atoi(argv[++i]);

                if (options.runs < 1) {
                    printf("bwarp: --runs must be at least 1\n");
                    return false;
                }
            } else if (strcmp(arg, "--segment") == 0 && i + 1 < argc) {
                options.encode.logSegment = atoi(argv[++i]);
            } else if (strcmp(arg, "--slice") == 0 && i + 1 < argc) {
                options.encode.logSlice = atoi(argv[++i]);
            } else if (strcmp(arg, "--contexts") == 0 && i + 1 < argc) {
                options.encode.contexts = atoi(argv[++i]);
            } else if (strcmp(arg, "--sample") == 0 && i + 1 < argc) {
                options.encode.logSample = atoi(argv[++i]);
            } else if (strcmp(arg, "--context") == 0) {
                options.context = true;
            } else if (strcmp(arg, "--gpu") == 0) {
                options.gpu = true;
            } else if (strcmp(arg, "--block") == 0 && i + 1 < argc) {
                options.blockBytes = strtoll(argv[++i], nullptr, 10);

                if (options.blockBytes < 1 || options.blockBytes > bwarp::kMaxBlockBytes) {
                    printf("bwarp: --block must be 1 to %u\n", bwarp::kMaxBlockBytes);
                    return false;
                }
            } else if (strcmp(arg, "--quiet") == 0) {
                options.quiet = true;
            } else if (strcmp(arg, "--force") == 0) {
                options.force = true;
            } else if (strcmp(arg, "--pinned") == 0) {
                options.pinned = true;
            } else if (strcmp(arg, "--recursive") == 0) {
                options.recursive = true;
            } else if (strcmp(arg, "--verbose") == 0) {
                options.verbose = true;
            } else if (strncmp(arg, "--", 2) == 0) {
                printf("bwarp: unknown option '%s'\n", arg);
                return false;
            } else if (options.pathCount < kMaxPaths) {
                options.paths[options.pathCount++] = arg;
            } else {
                printf("bwarp: unexpected argument '%s'\n", arg);
                return false;
            }
        }

        return true;
    }

    bool loadText(const char* path, std::vector<uint8_t>& text, int32_t& n) {
        if (!readWholeFile(path, text)) {
            printf("bwarp: cannot read '%s'\n", path);
            return false;
        }

        const int64_t size = static_cast<int64_t>(text.size());

        if (size > kMaxTextSize) {
            printf("bwarp: '%s' is too large (%lld bytes, at most %lld supported)\n", path, static_cast<long long>(size),
                static_cast<long long>(kMaxTextSize));
            return false;
        }

        n = static_cast<int32_t>(size);

        return true;
    }

    double megabytesPerSecond(int64_t bytes, double seconds) {
        return seconds > 0.0 ? static_cast<double>(bytes) / (1024.0 * 1024.0) / seconds : 0.0;
    }

    // where: what failed, as "'file', block 1: ", or nothing.
    bool reportGpu(GpuStatus status, const std::string& where = "") {
        if (status != GpuStatus::Ok) {
            printf("bwarp: %s%s: %s\n", where.c_str(),
                status == GpuStatus::Unavailable        ? "GPU unavailable"
                : status == GpuStatus::NotEnoughMemory  ? "GPU out of memory"
                : status == GpuStatus::Invalid          ? "input rejected"
                : status == GpuStatus::ChecksumMismatch ? "corrupt data"
                                                        : "GPU failed",
                gpuLastError());
            return false;
        }

        return true;
    }

    // bwt may be text.
    bool runGpuBwt(const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, double& seconds, GpuBwtStats& stats,
        uint32_t* samples = nullptr, int logSample = 0) {
        const GpuStatus status = gpuBwt(text, n, bwt, primary, &stats, samples, logSample);
        seconds = stats.uploadSeconds + stats.deviceSeconds + stats.downloadSeconds;
        return reportGpu(status);
    }

    bool runBwtCheck(const uint8_t* text, int32_t n, const uint8_t* bwt, int32_t primary, double& seconds) {
        const auto start = std::chrono::steady_clock::now();
        const CheckResult result = checkBwt(text, n, bwt, primary);
        seconds = secondsSince(start);

        if (!result.ok) {
            printf("bwarp: BWT check FAILED (primary index %d): %s, index %lld\n", primary, result.reason,
                static_cast<long long>(result.index));
        }

        return result.ok;
    }

    TestRun runTest(const std::vector<uint8_t>& text, int32_t n) {
        TestRun run;
        std::vector<uint8_t> bwt(n);
        int32_t primary = -1;

        run.bytes = n;

        if (!runGpuBwt(text.data(), n, bwt.data(), primary, run.gpuSeconds, run.gpu)) {
            run.ok = false;
            return run;
        }

        run.peakPerByte = n > 0 ? static_cast<double>(run.gpu.deviceBytes) / n : 0.0;
        run.ok = runBwtCheck(text.data(), n, bwt.data(), primary, run.checkSeconds);

        return run;
    }

    int cmdInfo() {
        GpuDevice device;

        if (!reportGpu(gpuQueryDevice(device))) {
            return kExitError;
        }

        printf("%s, sm_%d%d, %d SMs, %.0f MiB L2, %.2f GiB (%.2f GiB free)\n", device.name, device.computeMajor,
            device.computeMinor, device.multiprocessors, device.l2Bytes / (1024.0 * 1024.0),
            device.totalBytes / (1024.0 * 1024.0 * 1024.0), device.freeBytes / (1024.0 * 1024.0 * 1024.0));

        return kExitOk;
    }

    int cmdBwt(const Options& options) {
        if (options.pathCount != 2) {
            printUsage();
            return kExitError;
        }

        const char* inPath = options.paths[0];
        const char* outPath = options.paths[1];
        std::vector<uint8_t> text;
        int32_t n;

        if (!loadText(inPath, text, n)) {
            return kExitError;
        }

        // In place of the text.
        const auto start = std::chrono::steady_clock::now();
        int32_t primary = 0;
        double gpuSeconds = 0.0;
        GpuBwtStats gpuStats;

        if (!runGpuBwt(text.data(), n, text.data(), primary, gpuSeconds, gpuStats)) {
            return kExitError;
        }

        const double seconds = secondsSince(start);

        if (!writeBwtFile(outPath, text.data(), n, primary)) {
            printf("bwarp: cannot write '%s'\n", outPath);
            return kExitError;
        }

        printf("%s -> %s: %d bytes, GPU, %.3fs (%.1f MB/s), peak device memory %.1f MiB (%.2f bytes per input byte)\n",
            inPath, outPath, n, seconds, megabytesPerSecond(n, seconds), gpuStats.deviceBytes / (1024.0 * 1024.0),
            n > 0 ? static_cast<double>(gpuStats.deviceBytes) / n : 0.0);

        return kExitOk;
    }

    int cmdCheck(const Options& options) {
        if (options.pathCount != 2) {
            printUsage();
            return kExitError;
        }

        const char* textPath = options.paths[0];
        const char* bwtPath = options.paths[1];
        std::vector<uint8_t> text;
        int32_t n;

        if (!loadText(textPath, text, n)) {
            return kExitError;
        }

        std::vector<uint8_t> bwt;
        int32_t primary;

        if (!readBwtFile(bwtPath, bwt, primary)) {
            printf("bwarp: cannot read '%s' as a single-block BWT file\n", bwtPath);
            return kExitError;
        }

        if (static_cast<int64_t>(bwt.size()) != n) {
            printf("bwarp: '%s' has %zu bytes of BWT, but '%s' has %d bytes\n", bwtPath, bwt.size(), textPath, n);
            return kExitError;
        }

        double seconds;

        if (!runBwtCheck(text.data(), n, bwt.data(), primary, seconds)) {
            return kExitError;
        }

        printf("%s: BWT OK (%d bytes, primary index %d, %.3fs)\n", bwtPath, n, primary, seconds);

        return kExitOk;
    }

    int testFile(const Options& options) {
        const char* path = options.paths[0];
        std::vector<uint8_t> text;
        int32_t n;

        if (!loadText(path, text, n)) {
            return kExitError;
        }

        printf("%s: %d bytes, GPU BWT\n", path, n);

        const TestRun run = runTest(text, n);

        if (run.gpu.deviceBytes > 0) {
            const auto perByte = [n](int64_t bytes) { return n > 0 ? static_cast<double>(bytes) / n : 0.0; };

            printf("  GPU             %.3fs, %.1f MB/s (upload %.3fs, device %.3fs, download %.3fs, %s host memory; "
                   "%d character passes, %d doubling rounds)\n",
                run.gpuSeconds, megabytesPerSecond(n, run.gpuSeconds), run.gpu.uploadSeconds, run.gpu.deviceSeconds,
                run.gpu.downloadSeconds, run.gpu.pinned ? "pinned" : "pageable", run.gpu.characterPasses,
                run.gpu.doublingRounds);
            printf("  GPU memory      peak %.1f MiB, %.2f bytes per input byte (sample ranking %.2f, class sorts %.2f, "
                   "merge %.2f)\n",
                run.gpu.deviceBytes / (1024.0 * 1024.0), perByte(run.gpu.deviceBytes), perByte(run.gpu.rankingBytes),
                perByte(run.gpu.classSortBytes), perByte(run.gpu.mergeBytes));
        }

        if (!run.ok) {
            return kExitError;
        }

        printf("  check           %.3fs, OK\n", run.checkSeconds);

        return kExitOk;
    }

    // error_code overloads: bwarp builds without exceptions.
    bool listFiles(const std::filesystem::path& dir, bool recursive, std::vector<std::filesystem::path>& files) {
        std::error_code error;
        auto collect = [&](auto iterator) {
            for (const decltype(iterator) end{}; !error && iterator != end; iterator.increment(error)) {
                std::error_code ignored;

                if (iterator->is_regular_file(ignored)) {
                    files.push_back(iterator->path());
                }
            }
        };

        if (recursive) {
            collect(std::filesystem::recursive_directory_iterator(dir, error));
        } else {
            collect(std::filesystem::directory_iterator(dir, error));
        }

        std::sort(files.begin(), files.end());

        return !error;
    }

    void printTableRow(const char* name, int nameWidth, const TestRun& run, const char* verdict) {
        printf("%-*s %12lld %8.3f %8.1f %6.2f %8.3f  %s\n", nameWidth, name, static_cast<long long>(run.bytes),
            run.gpuSeconds, megabytesPerSecond(run.bytes, run.gpuSeconds), run.peakPerByte, run.checkSeconds, verdict);
    }

    int testDirectory(const Options& options) {
        const std::filesystem::path dir(options.paths[0]);
        std::vector<std::filesystem::path> files;

        if (!listFiles(dir, options.recursive, files)) {
            printf("bwarp: cannot list directory '%s'\n", options.paths[0]);
            return kExitError;
        }

        std::vector<std::string> names;
        int nameWidth = 5;

        for (const auto& file : files) {
            names.push_back(file.lexically_relative(dir).generic_string());
            nameWidth = std::max(nameWidth, static_cast<int>(names.back().size()));
        }

        printf("%s: %zu files, GPU BWT (times in seconds)\n", options.paths[0], files.size());
        printf("%-*s %12s %8s %8s %6s %8s\n", nameWidth, "file", "bytes", "gpu", "MB/s", "peak/n", "check");

        TestRun total;
        size_t passed = 0;

        for (size_t i = 0; i < files.size(); i++) {
            std::vector<uint8_t> text;
            int32_t n;

            if (!loadText(files[i].string().c_str(), text, n)) {
                continue;
            }

            const TestRun run = runTest(text, n);

            printTableRow(names[i].c_str(), nameWidth, run, run.ok ? "OK" : "FAILED");
            total.add(run);
            passed += run.ok ? 1 : 0;
        }

        const std::string verdict = std::to_string(passed) + "/" + std::to_string(files.size()) + " OK";
        printTableRow("total", nameWidth, total, verdict.c_str());

        return passed == files.size() ? kExitOk : kExitError;
    }

    int cmdTest(const Options& options) {
        if (options.pathCount != 1) {
            printUsage();
            return kExitError;
        }

        std::error_code error;

        if (std::filesystem::is_directory(options.paths[0], error)) {
            return testDirectory(options);
        }

        return testFile(options);
    }

    // The best of options.runs builds after a warmup, host to host.
    struct BenchRun {
        int64_t bytes = 0;
        double seconds = 0.0;
        GpuBwtStats stats;
        double peakPerByte = 0.0;

        void add(const BenchRun& other) {
            bytes += other.bytes;
            seconds += other.seconds;
            stats.setupSeconds += other.stats.setupSeconds;
            stats.uploadSeconds += other.stats.uploadSeconds;
            stats.rankingSeconds += other.stats.rankingSeconds;
            stats.classSortSeconds += other.stats.classSortSeconds;
            stats.mergeSeconds += other.stats.mergeSeconds;
            stats.bwtSeconds += other.stats.bwtSeconds;
            stats.downloadSeconds += other.stats.downloadSeconds;
            peakPerByte = std::max(peakPerByte, other.peakPerByte);
        }
    };

    bool benchGpu(const uint8_t* text, int32_t n, const Options& options, BenchRun& best) {
        std::vector<uint8_t> bwt(options.pinned ? 0 : n);
        GpuBwtContext* context = nullptr;

        best = BenchRun{};
        best.bytes = n;

        if (options.context && !reportGpu(gpuCreateBwtContext(n, context))) {
            return false;
        }

        const std::unique_ptr<GpuBwtContext, void (*)(GpuBwtContext*)> owner(context, gpuDestroyBwtContext);

        using Pinned = std::unique_ptr<void, void (*)(void*)>;
        const Pinned pinnedText(options.pinned ? gpuAllocatePinned(static_cast<size_t>(n)) : nullptr, gpuFreePinned);
        const Pinned pinnedOutput(options.pinned ? gpuAllocatePinned(static_cast<size_t>(n)) : nullptr, gpuFreePinned);
        const uint8_t* input = text;
        uint8_t* output = bwt.data();

        if (options.pinned) {
            if (!pinnedText || !pinnedOutput) {
                printf("bwarp: cannot allocate pinned host memory\n");
                return false;
            }

            memcpy(pinnedText.get(), text, static_cast<size_t>(n));
            input = static_cast<const uint8_t*>(pinnedText.get());
            output = static_cast<uint8_t*>(pinnedOutput.get());
        }

        for (int run = 0; run <= options.runs; run++) {
            GpuBwtStats stats;
            int32_t primary = 0;
            const auto start = std::chrono::steady_clock::now();
            const GpuStatus status = gpuBwt(context, input, n, output, primary, &stats);
            const double seconds = secondsSince(start);

            if (!reportGpu(status)) {
                return false;
            }

            if (run > 0 && (run == 1 || seconds < best.seconds)) {  // run 0 is the warmup
                best.seconds = seconds;
                best.stats = stats;
                best.peakPerByte = n > 0 ? static_cast<double>(stats.deviceBytes) / n : 0.0;
            }
        }

        return true;
    }

    void printBenchHeader(int nameWidth) {
        printf("%-*s %12s %7s %7s %7s %7s %7s %7s %7s %8s %7s %6s\n", nameWidth, "file", "bytes", "setup", "upload",
            "rank", "class", "merge", "bwt", "down", "total", "MB/s", "peak/n");
    }

    void printBenchRow(const char* name, int nameWidth, const BenchRun& run) {
        const auto ms = [](double seconds) { return seconds * 1000.0; };
        const GpuBwtStats& s = run.stats;

        printf("%-*s %12lld %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %8.1f %7.1f %6.2f\n", nameWidth, name,
            static_cast<long long>(run.bytes), ms(s.setupSeconds), ms(s.uploadSeconds), ms(s.rankingSeconds),
            ms(s.classSortSeconds), ms(s.mergeSeconds), ms(s.bwtSeconds), ms(s.downloadSeconds), ms(run.seconds),
            megabytesPerSecond(run.bytes, run.seconds), run.peakPerByte);
    }

    // The file at path, or the files in it; names for the table rows.
    bool collectFiles(const Options& options, std::vector<std::filesystem::path>& files, std::vector<std::string>& names,
        int& nameWidth) {
        const std::filesystem::path path(options.paths[0]);
        std::error_code error;
        const bool directory = std::filesystem::is_directory(path, error);

        if (directory && !listFiles(path, options.recursive, files)) {
            printf("bwarp: cannot list directory '%s'\n", options.paths[0]);
            return false;
        }

        if (!directory) {
            files.push_back(path);
        }

        nameWidth = 5;

        for (const auto& file : files) {
            names.push_back(directory ? file.lexically_relative(path).generic_string() : file.filename().string());
            nameWidth = std::max(nameWidth, static_cast<int>(names.back().size()));
        }

        return true;
    }

    int cmdBench(const Options& options) {
        std::vector<std::filesystem::path> files;
        std::vector<std::string> names;
        int nameWidth;

        if (options.pathCount != 1) {
            printUsage();
            return kExitError;
        }

        if (!collectFiles(options, files, names, nameWidth)) {
            return kExitError;
        }

        GpuDevice device;

        if (!reportGpu(gpuQueryDevice(device))) {
            return kExitError;
        }

        printf("%s: %zu file%s, GPU %s, BWT, best of %d runs%s%s (times in ms)\n", options.paths[0], files.size(),
            files.size() == 1 ? "" : "s", device.name, options.runs, options.context ? ", context reused" : "",
            options.pinned ? ", pinned host buffers" : "");
        printBenchHeader(nameWidth);

        BenchRun total;

        for (size_t i = 0; i < files.size(); i++) {
            std::vector<uint8_t> text;
            int32_t n;
            BenchRun run;

            if (!loadText(files[i].string().c_str(), text, n) || !benchGpu(text.data(), n, options, run)) {
                return kExitError;
            }

            printBenchRow(names[i].c_str(), nameWidth, run);
            total.add(run);
        }

        if (files.size() > 1) {
            printBenchRow("total", nameWidth, total);
        }

        return kExitOk;
    }

    struct RatioRun {
        int64_t bytes = 0;
        EncodeStats stats;
        int64_t compressed = 0;
        double encodeSeconds = 0.0;
        double decodeSeconds = 0.0;
        double gpuEncodeSeconds = 0.0;  // the encoder's device phases
        double gpuCompressSeconds = 0.0;  // gpuCompress(), host to host
        double gpuSeconds = 0.0;        // the decompressor's
        double gpuStreamSeconds = 0.0;
        double gpuInverseSeconds = 0.0;

        void add(const RatioRun& other) {
            bytes += other.bytes;
            stats.indexBytes += other.stats.indexBytes;
            stats.tableBytes += other.stats.tableBytes;
            stats.streamBytes += other.stats.streamBytes;
            stats.tokens += other.stats.tokens;
            compressed += other.compressed;
            encodeSeconds += other.encodeSeconds;
            decodeSeconds += other.decodeSeconds;
            gpuEncodeSeconds += other.gpuEncodeSeconds;
            gpuCompressSeconds += other.gpuCompressSeconds;
            gpuSeconds += other.gpuSeconds;
            gpuStreamSeconds += other.gpuStreamSeconds;
            gpuInverseSeconds += other.gpuInverseSeconds;
        }
    };

    void printRatioRow(const char* name, int nameWidth, const RatioRun& run, bool gpu, const char* verdict) {
        const auto percent = [&](int64_t part) { return run.compressed > 0 ? 100.0 * part / run.compressed : 0.0; };

        printf("%-*s %12lld %12lld %6.3f %8.4f %6.2f %6.2f %8.1f %8.1f", nameWidth, name,
            static_cast<long long>(run.bytes), static_cast<long long>(run.compressed),
            run.bytes > 0 ? 8.0 * run.compressed / run.bytes : 0.0,
            run.bytes > 0 ? static_cast<double>(run.stats.tokens) / run.bytes : 0.0, percent(run.stats.tableBytes),
            percent(run.stats.indexBytes), megabytesPerSecond(run.bytes, run.encodeSeconds),
            megabytesPerSecond(run.bytes, run.decodeSeconds));

        if (gpu) {
            printf(" %8.1f %8.1f %8.1f %8.1f %8.1f", megabytesPerSecond(run.bytes, run.gpuEncodeSeconds),
                megabytesPerSecond(run.bytes, run.gpuCompressSeconds), megabytesPerSecond(run.bytes, run.gpuSeconds),
                megabytesPerSecond(run.bytes, run.gpuStreamSeconds), megabytesPerSecond(run.bytes, run.gpuInverseSeconds));
        }

        printf("  %s\n", verdict);
    }

    int cmdRatio(const Options& options) {
        std::vector<std::filesystem::path> files;
        std::vector<std::string> names;
        int nameWidth;
        const EncodeParams& params = options.encode;

        if (options.pathCount != 1) {
            printUsage();
            return kExitError;
        }

        if (!collectFiles(options, files, names, nameWidth)) {
            return kExitError;
        }

        const std::string sampling = params.logSample != 0 ? "2^" + std::to_string(params.logSample) : "auto";

        printf("%s: %zu file%s, segments 2^%d, slices 2^%d, %d contexts, samples %s (CPU reference codec)\n",
            options.paths[0], files.size(), files.size() == 1 ? "" : "s", params.logSegment, params.logSlice,
            params.contexts, sampling.c_str());
        printf("%-*s %12s %12s %6s %8s %6s %6s %8s %8s", nameWidth, "file", "bytes", "compressed", "bpc", "runs/n",
            "tab%", "idx%", "enc MB/s", "dec MB/s");
        printf(options.gpu ? " %8s %8s %8s %8s %8s\n" : "\n", "genc MB/s", "gcmp MB/s", "gdec MB/s", "str MB/s",
            "inv MB/s");

        RatioRun total;
        size_t passed = 0;

        for (size_t i = 0; i < files.size(); i++) {
            std::vector<uint8_t> text, bwt, encoded, decoded;
            std::vector<uint32_t> samples, decodedSamples;
            int32_t n, primary = 0, decodedPrimary = -1;
            double gpuSeconds;
            GpuBwtStats gpuStats;
            RatioRun run;

            if (params.logSample != 0 && (params.logSample < kAutoMinLogSample || params.logSample > bwarp::kMaxLogSample)) {
                printf("bwarp: invalid --sample\n");
                return kExitError;
            }

            if (!loadText(files[i].string().c_str(), text, n)) {
                continue;
            }

            if (uint32_t(n) > bwarp::kMaxBlockBytes) {
                printf("bwarp: '%s' is larger than a block (2^%u bytes), skipped; compress splits it\n",
                    files[i].string().c_str(), bwarp::kMaxLogBlock);
                continue;
            }

            const uint32_t checksum = crc32c(text.data(), size_t(n));

            bwt.resize(n);
            samples.resize(bwarp::sampleCount(static_cast<uint32_t>(n), kAutoMinLogSample));

            if (!runGpuBwt(text.data(), n, bwt.data(), primary, gpuSeconds, gpuStats, samples.data(), kAutoMinLogSample)) {
                continue;
            }

            auto start = std::chrono::steady_clock::now();

            if (!encodeBwt(bwt.data(), n, primary, checksum, samples.data(), kAutoMinLogSample, params, encoded,
                    &run.stats)) {
                printf("bwarp: invalid --segment, --slice or --contexts\n");
                return kExitError;
            }

            run.encodeSeconds = secondsSince(start);
            start = std::chrono::steady_clock::now();
            const bool decodedOk = decodeBwt(encoded.data(), encoded.size(), decoded, decodedPrimary, &decodedSamples);
            run.decodeSeconds = secondsSince(start);
            run.bytes = n;
            run.compressed = static_cast<int64_t>(encoded.size());

            bool ok = decodedOk && decodedPrimary == primary && decoded == bwt;

            for (size_t j = 0; ok && j < decodedSamples.size(); j++) {
                ok = decodedSamples[j] == samples[((j + 1) << (run.stats.logSample - kAutoMinLogSample)) - 1];
            }

            // The faster of two runs, the first one warming up.
            for (int r = 0; options.gpu && ok && r < 2; r++) {
                std::vector<uint8_t> gpuEncoded;
                GpuEncodeStats stats;
                const GpuStatus status = gpuEncode(bwt.data(), n, primary, checksum, samples.data(), kAutoMinLogSample,
                    params, gpuEncoded, nullptr, &stats);

                if (!reportGpu(status)) {
                    ok = false;
                    break;
                }

                ok = gpuEncoded == encoded;

                const double device = stats.chooseSeconds + stats.tableSeconds + stats.streamSeconds + stats.placeSeconds;

                if (r == 0 || device < run.gpuEncodeSeconds) {
                    run.gpuEncodeSeconds = device;
                }

                if (options.verbose && r == 1) {
                    printf("  GPU encoder: upload %.2f ms, choosing %.2f ms, tables %.2f ms, streams %.2f ms, placing "
                           "%.2f ms, download %.2f ms, %lld MiB\n",
                        1e3 * stats.uploadSeconds, 1e3 * stats.chooseSeconds, 1e3 * stats.tableSeconds,
                        1e3 * stats.streamSeconds, 1e3 * stats.placeSeconds, 1e3 * stats.downloadSeconds,
                        static_cast<long long>(stats.deviceBytes >> 20));
                }
            }

            // The whole compression on the device, host to host, with a context made beforehand. --verbose also
            // times building the BWT and encoding it as two calls, gpuBwt() and gpuEncode(), for comparison.
            GpuBwtContext* context = nullptr;

            if (options.gpu && ok && !reportGpu(gpuCreateBwtContext(n, context))) {
                ok = false;
            }

            for (int r = 0; options.gpu && ok && r < 2; r++) {
                const uint8_t* compressed = nullptr;
                size_t size = 0;
                GpuCompressStats stats;
                auto start = std::chrono::steady_clock::now();
                const GpuStatus status = gpuCompress(context, text.data(), n, params, compressed, size, nullptr, &stats);
                const double seconds = secondsSince(start);

                if (!reportGpu(status)) {
                    ok = false;
                    break;
                }

                ok = size == encoded.size() && std::equal(compressed, compressed + size, encoded.begin());

                if (r == 0 || seconds < run.gpuCompressSeconds) {
                    run.gpuCompressSeconds = seconds;
                }

                if (options.verbose && r == 1) {
                    std::vector<uint8_t> separateBwt(n), separateEncoded;
                    int32_t separatePrimary = 0;

                    start = std::chrono::steady_clock::now();

                    if (!reportGpu(gpuBwt(context, text.data(), n, separateBwt.data(), separatePrimary, nullptr,
                            samples.data(), kAutoMinLogSample)) ||
                        !reportGpu(gpuEncode(separateBwt.data(), n, separatePrimary, crc32c(text.data(), size_t(n)),
                            samples.data(), kAutoMinLogSample, params, separateEncoded))) {
                        ok = false;
                        break;
                    }

                    const double separate = secondsSince(start);
                    const GpuBwtStats& b = stats.bwt;
                    const GpuEncodeStats& g = stats.encode;

                    printf("  GPU compressor: %.2f ms (gpuBwt() then gpuEncode(): %.2f ms); BWT upload %.2f ms, "
                           "construction %.2f ms; encoder setup %.2f ms, handover %.2f ms, choosing %.2f ms, tables "
                           "%.2f ms, streams %.2f ms, placing %.2f ms, download %.2f ms\n",
                        1e3 * seconds, 1e3 * separate, 1e3 * b.uploadSeconds, 1e3 * b.deviceSeconds,
                        1e3 * g.setupSeconds, 1e3 * g.uploadSeconds, 1e3 * g.chooseSeconds, 1e3 * g.tableSeconds,
                        1e3 * g.streamSeconds, 1e3 * g.placeSeconds, 1e3 * g.downloadSeconds);
                }
            }

            gpuDestroyBwtContext(context);

            for (int r = 0; options.gpu && ok && r < 2; r++) {
                GpuDecodeStats stats;
                const GpuStatus status = gpuDecompress(encoded.data(), encoded.size(), decoded, &stats);

                if (!reportGpu(status)) {
                    ok = false;
                    break;
                }

                ok = decoded == text;

                const double inverse = stats.rankSeconds + stats.chainSeconds;
                const double device = stats.tableSeconds + stats.finishSeconds + stats.streamSeconds + inverse;

                if (r == 0 || device < run.gpuSeconds) {
                    run.gpuSeconds = device;
                    run.gpuStreamSeconds = stats.streamSeconds;
                    run.gpuInverseSeconds = inverse;
                }

                if (options.verbose && r == 1) {
                    printf("  GPU decompressor: upload %.2f ms, tables %.2f ms, finishing %.2f ms, streams %.2f ms, LF %.2f ms, "
                           "chains %.2f ms, download %.2f ms, %lld MiB\n",
                        1e3 * stats.uploadSeconds, 1e3 * stats.tableSeconds, 1e3 * stats.finishSeconds,
                        1e3 * stats.streamSeconds, 1e3 * stats.rankSeconds, 1e3 * stats.chainSeconds,
                        1e3 * stats.downloadSeconds, static_cast<long long>(stats.deviceBytes >> 20));
                }
            }

            printRatioRow(names[i].c_str(), nameWidth, run, options.gpu, ok ? "OK" : "FAILED");
            total.add(run);
            passed += ok ? 1 : 0;
        }

        if (files.size() > 1) {
            const std::string verdict = std::to_string(passed) + "/" + std::to_string(files.size()) + " OK";
            printRatioRow("total", nameWidth, total, options.gpu, verdict.c_str());
        }

        return passed == files.size() ? kExitOk : kExitError;
    }

    // ---- compress and decompress ---------------------------------------------------------------

    // Page-locked host memory: transfers from and to it skip the staging copies.
    class PinnedBuffer {
     public:
        PinnedBuffer() = default;
        PinnedBuffer(const PinnedBuffer&) = delete;
        PinnedBuffer& operator=(const PinnedBuffer&) = delete;

        ~PinnedBuffer() {
            gpuFreePinned(m_data);
        }

        bool allocate(size_t bytes) {
            gpuFreePinned(m_data);
            m_data = static_cast<uint8_t*>(gpuAllocatePinned(bytes));
            m_capacity = m_data ? bytes : 0;
            return m_data != nullptr;
        }

        uint8_t* data() const {
            return m_data;
        }

        size_t capacity() const {
            return m_capacity;
        }

     private:
        uint8_t* m_data = nullptr;
        size_t m_capacity = 0;
    };

    constexpr const char* kExtension = ".bwarp";

    struct FileJob {
        std::filesystem::path in, out;
        std::string name;
        int64_t size = 0;       // of the input
        int64_t text = 0;       // of the text: compressing, the input's; decompressing, its blocks' together
        int64_t outBytes = 0;   // written so far
        double seconds = 0.0;   // in the (de)compression calls
        bool started = false;   // its output written to
        bool failed = false;
    };

    // Compressing, a piece of a file's text, at most --block bytes; decompressing, one of the file's blocks.
    struct Block {
        size_t job = 0;
        int64_t offset = 0;  // in the input
        int64_t size = 0;
        int64_t n = 0;  // its text's size
        int index = 0;  // in its file
        int64_t deviceBytes = 0;  // decompressing: what gpuDecompress() needs
        bool last = false;
    };

    bool hasExtension(const std::filesystem::path& path) {
        return path.extension() == kExtension;
    }

    // Pairs each input with its output: a file with paths[1], or with its name plus or minus .bwarp; a directory's
    // files (compressing all, decompressing those ending in .bwarp) with the same names under paths[1]. Existing
    // outputs are an error without --force.
    bool planJobs(const Options& options, bool compressing, std::vector<FileJob>& jobs, int& nameWidth) {
        if (options.pathCount < 1) {
            printUsage();
            return false;
        }

        const std::filesystem::path in(options.paths[0]);
        std::error_code error;
        const bool directory = std::filesystem::is_directory(in, error);
        std::vector<std::filesystem::path> files;

        if (directory) {
            if (options.pathCount != 2) {
                printf("bwarp: a directory needs an output directory\n");
                return false;
            }

            if (!listFiles(in, options.recursive, files)) {
                printf("bwarp: cannot list directory '%s'\n", options.paths[0]);
                return false;
            }
        } else {
            files.push_back(in);
        }

        nameWidth = 5;

        for (const auto& file : files) {
            if (!compressing && directory && !hasExtension(file)) {
                continue;
            }

            FileJob job;

            job.in = file;
            job.name = directory ? file.lexically_relative(in).generic_string() : file.filename().string();

            if (directory) {
                job.out = std::filesystem::path(options.paths[1]) / file.lexically_relative(in);
            } else if (options.pathCount == 2) {
                job.out = options.paths[1];
            } else if (compressing) {
                job.out = file;
            } else if (hasExtension(file)) {
                job.out = file;
                job.out.replace_extension();
            } else {
                printf("bwarp: '%s' doesn't end in %s; name the output\n", options.paths[0], kExtension);
                return false;
            }

            if (directory || options.pathCount != 2) {
                if (compressing) {
                    job.out += kExtension;
                } else if (directory) {
                    job.out.replace_extension();
                }
            }

            job.size = static_cast<int64_t>(std::filesystem::file_size(file, error));

            if (error) {
                printf("bwarp: cannot read '%s'\n", file.string().c_str());
                return false;
            }

            if (std::filesystem::equivalent(job.in, job.out, error)) {
                printf("bwarp: '%s' would overwrite its own input\n", job.out.string().c_str());
                return false;
            }

            if (!options.force && std::filesystem::exists(job.out, error)) {
                printf("bwarp: '%s' exists (--force overwrites it)\n", job.out.string().c_str());
                return false;
            }

            nameWidth = std::max(nameWidth, static_cast<int>(job.name.size()));
            jobs.push_back(std::move(job));
        }

        if (jobs.empty()) {
            printf("bwarp: no %sfiles in '%s'\n", compressing ? "" : ".bwarp ", options.paths[0]);
            return false;
        }

        return true;
    }

    // Each file's text in pieces of blockBytes, the last one shorter; an empty file as one empty block.
    void splitBlocks(std::vector<FileJob>& jobs, int64_t blockBytes, std::vector<Block>& blocks) {
        for (size_t j = 0; j < jobs.size(); j++) {
            FileJob& job = jobs[j];
            Block block;

            job.text = job.size;
            block.job = j;

            do {
                block.size = block.n = std::min(blockBytes, job.size - block.offset);
                block.last = block.offset + block.size == job.size;
                blocks.push_back(block);
                block.offset += block.size;
                block.index++;
            } while (block.offset < job.size);
        }
    }

    // Each file's blocks, from their headers and the sizes that end their indexes (see docs/format.md): false
    // unless they cover the file exactly.
    bool scanBlocks(std::vector<FileJob>& jobs, std::vector<Block>& blocks) {
        for (size_t j = 0; j < jobs.size(); j++) {
            FileJob& job = jobs[j];
            const std::string path = job.in.string();
            Block block;

            block.job = j;

            do {
                const uint64_t left = static_cast<uint64_t>(job.size - block.offset);
                uint8_t header[bwarp::kHeaderBytes], end[8];
                bwarp::BlockHeader parsed;
                bool ok = left >= sizeof header &&
                          readFileInto(path.c_str(), header, sizeof header, static_cast<uint64_t>(block.offset)) &&
                          bwarp::readBlockHeader(header, parsed);
                const uint64_t indexBytes = ok ? parsed.layout.indexBytes() : 0;

                ok = ok && left >= indexBytes &&
                     readFileInto(path.c_str(), end, sizeof end, static_cast<uint64_t>(block.offset) + indexBytes - 8);

                const uint64_t size = ok ? bwarp::get64(end) : 0;

                if (!ok || size < indexBytes || size > left) {
                    printf(block.index == 0 ? "bwarp: '%s' is not a bwarp file\n"
                                            : "bwarp: '%s' is corrupt: block %d doesn't fit\n",
                        path.c_str(), block.index);
                    return false;
                }

                // Only the last block has the flag, so whole blocks missing at the end are caught too.
                if (parsed.last != (size == left)) {
                    printf(parsed.last ? "bwarp: '%s' is corrupt: data after its last block, block %d\n"
                                       : "bwarp: '%s' is truncated: it ends after block %d, without its last\n",
                        path.c_str(), block.index);
                    return false;
                }

                block.size = static_cast<int64_t>(size);
                block.n = parsed.layout.n;
                block.deviceBytes = gpuDecompressRequiredBytes(header, static_cast<size_t>(size));
                block.last = parsed.last;
                job.text += block.n;
                blocks.push_back(block);
                block.offset += block.size;
                block.index++;
            } while (block.offset < job.size);
        }

        return true;
    }

    bool writeBlock(const FileJob& job, bool first, const uint8_t* data, size_t size) {
        std::error_code error;

        if (first && job.out.has_parent_path()) {
            std::filesystem::create_directories(job.out.parent_path(), error);
        }

        if (!writeFileDirect(job.out.string().c_str(), data, size, !first)) {
            printf("bwarp: cannot write '%s'\n", job.out.string().c_str());
            return false;
        }

        return true;
    }

    struct CodingTotals {
        int64_t in = 0;
        int64_t out = 0;
        double readWait = 0.0;  // waiting for the reader
        double gpu = 0.0;       // the (de)compression calls, host to host
        double write = 0.0;     // waiting for the writer
    };

    // Writes one block at a time on another thread while the next one is made: its data must stay put until the
    // next call, or wait().
    class BlockWriter {
     public:
        BlockWriter(std::vector<FileJob>& jobs, CodingTotals& totals) : m_jobs(jobs), m_totals(totals) {}

        // False if the block before failed. With expected, also checks the data's CRC-32C against it meanwhile,
        // reporting a mismatch as the block called name: the block fails, after it was written.
        bool write(const Block& block, const uint8_t* data, size_t size, const uint32_t* expected = nullptr,
            const std::string& name = "") {
            if (!wait()) {
                return false;
            }

            FileJob& job = m_jobs[block.job];
            const bool check = expected != nullptr;
            const uint32_t checksum = check ? *expected : 0;

            job.started = true;
            job.outBytes += static_cast<int64_t>(size);
            m_job = block.job;
            m_pending = std::async(std::launch::async, [&job, first = block.index == 0, data, size, check, checksum,
                                                           name] {
                std::future<uint32_t> computing;

                if (check) {
                    computing = std::async(std::launch::async, [data, size] { return crc32c(data, size); });
                }

                bool ok = writeBlock(job, first, data, size);

                if (check) {
                    if (const uint32_t computed = computing.get(); computed != checksum) {
                        printf("bwarp: %s: corrupt data: the text's CRC-32C is %08x, the block says %08x\n",
                            name.c_str(), computed, checksum);
                        ok = false;
                    }
                }

                return ok;
            });

            return true;
        }

        bool wait() {
            if (!m_pending.valid()) {
                return true;
            }

            const auto start = std::chrono::steady_clock::now();
            const bool ok = m_pending.get();

            m_totals.write += secondsSince(start);

            if (!ok) {
                m_jobs[m_job].failed = true;
            }

            return ok;
        }

     private:
        std::vector<FileJob>& m_jobs;
        CodingTotals& m_totals;
        std::future<bool> m_pending;
        size_t m_job = 0;
    };

    // Reads block i into buffers[i % 2] on another thread.
    std::future<bool> startRead(const std::vector<FileJob>& jobs, const std::vector<Block>& blocks,
        PinnedBuffer* buffers, size_t i) {
        return std::async(std::launch::async, [&jobs, &blocks, buffers, i] {
            const Block& block = blocks[i];

            return readFileInto(jobs[block.job].in.string().c_str(), buffers[i % 2].data(),
                static_cast<size_t>(block.size), static_cast<uint64_t>(block.offset));
        });
    }

    // Reads block i + 1 while process(i) runs, from pending, block 0's read, which the caller started. Stops at
    // the first failure, marking that block's file as failed.
    template <class Process>
    bool pipelineBlocks(std::vector<FileJob>& jobs, const std::vector<Block>& blocks, PinnedBuffer* buffers,
        CodingTotals& totals, std::future<bool> pending, Process process) {

        for (size_t i = 0; i < blocks.size(); i++) {
            const auto start = std::chrono::steady_clock::now();
            const bool readOk = pending.get();

            totals.readWait += secondsSince(start);

            if (i + 1 < blocks.size()) {
                pending = startRead(jobs, blocks, buffers, i + 1);
            }

            FileJob& job = jobs[blocks[i].job];

            if (!readOk) {
                printf("bwarp: cannot read '%s'\n", job.in.string().c_str());
            }

            if (!readOk || !process(i)) {
                job.failed = true;
                return false;  // the pending read finishes in the future's destructor
            }
        }

        return true;
    }

    // Outputs of failed files that were written to are incomplete: removed.
    void removeFailed(const std::vector<FileJob>& jobs) {
        for (const FileJob& job : jobs) {
            std::error_code error;

            if (job.failed && job.started && std::filesystem::remove(job.out, error)) {
                printf("bwarp: removed the incomplete '%s'\n", job.out.string().c_str());
            }
        }
    }

    double nanosecondsPerByte(int64_t bytes, double seconds) {
        return bytes > 0 ? 1e9 * seconds / static_cast<double>(bytes) : 0.0;
    }

    void printCodingHeader(int nameWidth, const char* outName) {
        printf("%-*s %12s %12s %6s %8s %8s %7s\n", nameWidth, "file", "bytes", outName, "bpc", "GPU ms", "MB/s",
            "ns/B");
    }

    // Bits per byte of the text, its megabytes per second and nanoseconds per byte.
    void printCodingRow(const char* name, int nameWidth, int64_t in, int64_t out, int64_t compressed, int64_t text,
        double seconds) {
        printf("%-*s %12lld %12lld %6.3f %8.2f %8.1f %7.3f\n", nameWidth, name, static_cast<long long>(in),
            static_cast<long long>(out), text > 0 ? 8.0 * compressed / text : 0.0, 1e3 * seconds,
            megabytesPerSecond(text, seconds), nanosecondsPerByte(text, seconds));
    }

    // Prints the totals and the phases: the time over all files counts the setup before the first, as a whole
    // run's time does (the Large Text Compression Benchmark's ns/byte).
    void printCodingTotals(const char* verb, int nameWidth, size_t files, const CodingTotals& t, int64_t text,
        double setup, double wall) {
        if (files > 1) {
            printCodingRow("total", nameWidth, t.in, t.out, t.in + t.out - text, text, t.gpu);
        }

        const double all = setup + wall;

        printf("%s %zu file%s in %.1f ms, %.2f ns/byte (%.1f MB/s) with setup, reading and writing: setup %.1f ms, "
               "GPU %.1f ms, waiting to read %.1f ms, to write %.1f ms\n",
            verb, files, files == 1 ? "" : "s", 1e3 * all, nanosecondsPerByte(text, all), megabytesPerSecond(text, all),
            1e3 * setup, 1e3 * t.gpu, 1e3 * t.readWait, 1e3 * t.write);
    }

    // "'file'" or "'file', block k" for files of several blocks.
    std::string blockName(const FileJob& job, const Block& block) {
        return "'" + job.in.string() + "'" + (block.index > 0 || !block.last ? ", block " + std::to_string(block.index) : "");
    }

    int cmdCompress(const Options& options) {
        std::vector<FileJob> jobs;
        std::vector<Block> blocks;
        int nameWidth;
        const EncodeParams& params = options.encode;

        if (!planJobs(options, true, jobs, nameWidth)) {
            return kExitError;
        }

        splitBlocks(jobs, options.blockBytes, blocks);

        // Two input buffers and one context for the largest block, the first block read meanwhile.
        auto start = std::chrono::steady_clock::now();
        int64_t largest = 1;

        for (const Block& block : blocks) {
            largest = std::max(largest, block.size);
        }

        PinnedBuffer buffers[2];

        if (!buffers[0].allocate(static_cast<size_t>(largest))) {
            printf("bwarp: cannot allocate %lld bytes of page-locked memory\n", static_cast<long long>(largest));
            return kExitError;
        }

        std::future<bool> first = startRead(jobs, blocks, buffers, 0);
        GpuBwtContext* created = nullptr;

        if (!reportGpu(gpuCreateBwtContext(static_cast<int32_t>(largest), created))) {
            return kExitError;
        }

        const std::unique_ptr<GpuBwtContext, void (*)(GpuBwtContext*)> context(created, gpuDestroyBwtContext);

        if (blocks.size() > 1 && !buffers[1].allocate(static_cast<size_t>(largest))) {
            printf("bwarp: cannot allocate %lld bytes of page-locked memory\n", static_cast<long long>(largest));
            return kExitError;
        }

        const double setup = secondsSince(start);

        if (!options.quiet) {
            printf("%s: %zu file%s, %zu block%s, segments 2^%d, slices 2^%d, %d contexts, samples %s\n",
                options.paths[0], jobs.size(), jobs.size() == 1 ? "" : "s", blocks.size(), blocks.size() == 1 ? "" : "s",
                params.logSegment, params.logSlice, params.contexts,
                params.logSample != 0 ? ("2^" + std::to_string(params.logSample)).c_str() : "auto");
            printCodingHeader(nameWidth, "compressed");
        }

        CodingTotals totals;
        BlockWriter writer(jobs, totals);

        start = std::chrono::steady_clock::now();

        // gpuCompress() alternates between two output buffers, so the writer has block i's while i + 1 is made.
        bool ok = pipelineBlocks(jobs, blocks, buffers, totals, std::move(first), [&](size_t i) {
            const Block& block = blocks[i];
            FileJob& job = jobs[block.job];
            const uint8_t* compressed = nullptr;
            size_t size = 0;
            GpuCompressStats stats;
            EncodeParams blockParams = params;

            blockParams.lastBlock = block.last;

            const auto phase = std::chrono::steady_clock::now();
            const GpuStatus status = gpuCompress(context.get(), buffers[i % 2].data(), static_cast<int32_t>(block.size),
                blockParams, compressed, size, nullptr, &stats);
            const double seconds = secondsSince(phase);

            if (status != GpuStatus::Ok) {
                return reportGpu(status, blockName(job, block) + ": ");
            }

            totals.gpu += seconds;
            job.seconds += seconds;

            if (!writer.write(block, compressed, size)) {
                return false;
            }

            totals.in += block.size;
            totals.out += static_cast<int64_t>(size);

            if (options.verbose) {
                const GpuBwtStats& b = stats.bwt;
                const GpuEncodeStats& g = stats.encode;

                printf("  %supload %.2f ms, BWT %.2f ms, handover %.2f ms, encoding %.2f ms, download %.2f ms\n",
                    block.index > 0 || !block.last ? ("block " + std::to_string(block.index) + ": ").c_str() : "",
                    1e3 * b.uploadSeconds, 1e3 * b.deviceSeconds, 1e3 * g.uploadSeconds,
                    1e3 * (g.setupSeconds + g.chooseSeconds + g.tableSeconds + g.streamSeconds + g.placeSeconds),
                    1e3 * g.downloadSeconds);
            }

            if (block.last && !options.quiet) {
                printCodingRow(job.name.c_str(), nameWidth, job.size, job.outBytes, job.outBytes, job.size, job.seconds);
            }

            return true;
        });

        ok = writer.wait() && ok;
        removeFailed(jobs);

        if (!options.quiet) {
            printCodingTotals("compressed", nameWidth, jobs.size(), totals, totals.in, setup, secondsSince(start));
        }

        return ok ? kExitOk : kExitError;
    }

    int cmdDecompress(const Options& options) {
        std::vector<FileJob> jobs;
        std::vector<Block> blocks;
        int nameWidth;

        if (!planJobs(options, false, jobs, nameWidth)) {
            return kExitError;
        }

        // The blocks' sizes, from their headers, so that the buffers are made before the first is read.
        auto start = std::chrono::steady_clock::now();

        if (!scanBlocks(jobs, blocks)) {
            return kExitError;
        }

        int64_t largestIn = 1, largestText = 1, deviceBytes = 0;

        for (const Block& block : blocks) {
            largestIn = std::max(largestIn, block.size);
            largestText = std::max(largestText, block.n);
            deviceBytes = std::max(deviceBytes, block.deviceBytes);
        }

        // Two of each: block i is read while i - 1 is decompressed, which is written while i is decompressed. The
        // first block is read while the rest are made.
        PinnedBuffer buffers[2], texts[2];
        const auto failed = [&] {
            printf("bwarp: cannot allocate %lld bytes of page-locked memory\n",
                static_cast<long long>(largestIn + largestText));
            return kExitError;
        };

        if (!buffers[0].allocate(static_cast<size_t>(largestIn))) {
            return failed();
        }

        std::future<bool> first = startRead(jobs, blocks, buffers, 0);

        for (size_t b = 0; b < std::min<size_t>(2, blocks.size()); b++) {
            if ((b > 0 && !buffers[b].allocate(static_cast<size_t>(largestIn))) ||
                !texts[b].allocate(static_cast<size_t>(largestText))) {
                return failed();
            }
        }

        // Device memory for the largest block, reused by all.
        GpuDecompressContext* created = nullptr;

        if (!reportGpu(gpuCreateDecompressContext(deviceBytes, created))) {
            return kExitError;
        }

        const std::unique_ptr<GpuDecompressContext, void (*)(GpuDecompressContext*)> context(
            created, gpuDestroyDecompressContext);

        const double setup = secondsSince(start);

        if (!options.quiet) {
            printf("%s: %zu file%s, %zu block%s\n", options.paths[0], jobs.size(), jobs.size() == 1 ? "" : "s",
                blocks.size(), blocks.size() == 1 ? "" : "s");
            printCodingHeader(nameWidth, "text");
        }

        CodingTotals totals;
        BlockWriter writer(jobs, totals);

        start = std::chrono::steady_clock::now();

        bool ok = pipelineBlocks(jobs, blocks, buffers, totals, std::move(first), [&](size_t i) {
            const Block& block = blocks[i];
            FileJob& job = jobs[block.job];
            PinnedBuffer& text = texts[i % 2];  // block i - 2's, written before block i - 1's write started
            size_t n = 0;
            GpuDecodeStats stats;
            const auto phase = std::chrono::steady_clock::now();
            uint32_t checksum = 0;
            const GpuStatus status = gpuDecompress(context.get(), buffers[i % 2].data(),
                static_cast<size_t>(block.size), text.data(), text.capacity(), n, &stats, &checksum);
            const double seconds = secondsSince(phase);

            if (status != GpuStatus::Ok) {
                return reportGpu(status, blockName(job, block) + ": ");
            }

            totals.gpu += seconds;
            job.seconds += seconds;

            if (!writer.write(block, text.data(), n, &checksum, blockName(job, block))) {
                return false;
            }

            totals.in += block.size;
            totals.out += static_cast<int64_t>(n);

            if (options.verbose) {
                printf("  %ssetup %.2f ms, upload %.2f ms, tables %.2f ms, streams %.2f ms, inverse BWT %.2f ms, "
                       "download %.2f ms\n",
                    block.index > 0 || !block.last ? ("block " + std::to_string(block.index) + ": ").c_str() : "",
                    1e3 * stats.setupSeconds, 1e3 * stats.uploadSeconds,
                    1e3 * (stats.tableSeconds + stats.finishSeconds), 1e3 * stats.streamSeconds,
                    1e3 * (stats.rankSeconds + stats.chainSeconds), 1e3 * stats.downloadSeconds);
            }

            if (block.last && !options.quiet) {
                printCodingRow(job.name.c_str(), nameWidth, job.size, job.outBytes, job.size, job.outBytes, job.seconds);
            }

            return true;
        });

        ok = writer.wait() && ok;
        removeFailed(jobs);

        if (!options.quiet) {
            printCodingTotals("decompressed", nameWidth, jobs.size(), totals, totals.out, setup, secondsSince(start));
        }

        return ok ? kExitOk : kExitError;
    }
}

// The command line as main() gets it: an exit code. The self-test runs commands through it.
int runCommand(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return kExitError;
    }

    const char* command = argv[1];
    Options options;

    if (!parseOptions(argc, argv, options)) {
        return kExitError;
    }

    if (strcmp(command, "compress") == 0) {
        return cmdCompress(options);
    }

    if (strcmp(command, "decompress") == 0) {
        return cmdDecompress(options);
    }

    if (strcmp(command, "info") == 0) {
        return cmdInfo();
    }

    if (strcmp(command, "bwt") == 0) {
        return cmdBwt(options);
    }

    if (strcmp(command, "check") == 0) {
        return cmdCheck(options);
    }

    if (strcmp(command, "test") == 0) {
        return cmdTest(options);
    }

    if (strcmp(command, "bench") == 0) {
        return cmdBench(options);
    }

    if (strcmp(command, "ratio") == 0) {
        return cmdRatio(options);
    }

    if (strcmp(command, "selftest") == 0) {
        return runSelfTest(options.verbose, runCommand) == 0 ? kExitOk : kExitError;
    }

    printUsage();

    return kExitError;
}

int main(int argc, char** argv) {
    return runCommand(argc, argv);
}
