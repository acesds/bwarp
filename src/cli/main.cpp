/*
 * main.cpp - The bwarp command line.
 *
 * Part of bwarp, a GPU entropy coder and inverse transform for BWT output.
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
#include <memory>
#include <string>
#include <system_error>
#include <vector>
#include "checker.h"
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
            "bwarp - GPU entropy coding and inverse transform for BWT output\n"
            "\n"
            "usage:\n"
            "  bwarp info\n"
            "  bwarp bwt <in> <out.bwt>\n"
            "  bwarp check <in> <in.bwt>\n"
            "  bwarp test <in|dir> [--recursive]\n"
            "  bwarp bench <in|dir> [--runs N] [--context] [--pinned] [--recursive]\n"
            "  bwarp selftest [--verbose]\n"
            "\n"
            "The .bwt file is divsufsort's bwt format as one block: n, the primary index, the n bytes\n"
            "(the same as sadcs build --bwt).\n"
            "info prints the GPU, or that bwarp was built without CUDA.\n"
            "bwt builds the BWT on the GPU (sadcs's construction); check verifies a BWT file against\n"
            "its text by inverting it on the CPU.\n"
            "test builds and checks in memory; on a directory it runs every file in it (--recursive:\n"
            "and below) and prints a timing table.\n"
            "bench times the GPU build: a warmup, then the best of --runs (default 5) host-to-host runs,\n"
            "with each phase of the best one. It doesn't check the output; test does. --context reuses\n"
            "one GPU context per file, created before the runs and not timed. --pinned uses page-locked\n"
            "host buffers, allocated before the runs (--context --pinned is how libcubwt measures).\n");
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
            } else if (strcmp(arg, "--context") == 0) {
                options.context = true;
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

    bool reportGpu(GpuStatus status) {
        if (status != GpuStatus::Ok) {
            printf("bwarp: GPU %s: %s\n",
                status == GpuStatus::Unavailable       ? "unavailable"
                : status == GpuStatus::NotEnoughMemory ? "out of memory"
                                                       : "failed",
                gpuLastError());
            return false;
        }

        return true;
    }

    // bwt may be text.
    bool runGpuBwt(const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, double& seconds, GpuBwtStats& stats) {
        const GpuStatus status = gpuBwt(text, n, bwt, primary, &stats);
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

    int cmdBench(const Options& options) {
        if (options.pathCount != 1) {
            printUsage();
            return kExitError;
        }

        const std::filesystem::path path(options.paths[0]);
        std::vector<std::filesystem::path> files;
        std::error_code error;
        const bool directory = std::filesystem::is_directory(path, error);

        if (directory && !listFiles(path, options.recursive, files)) {
            printf("bwarp: cannot list directory '%s'\n", options.paths[0]);
            return kExitError;
        }

        if (!directory) {
            files.push_back(path);
        }

        std::vector<std::string> names;
        int nameWidth = 5;

        for (const auto& file : files) {
            names.push_back(directory ? file.lexically_relative(path).generic_string() : file.filename().string());
            nameWidth = std::max(nameWidth, static_cast<int>(names.back().size()));
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
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return kExitError;
    }

    const char* command = argv[1];
    Options options;

    if (!parseOptions(argc, argv, options)) {
        return kExitError;
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

    if (strcmp(command, "selftest") == 0) {
        return runSelfTest(options.verbose) == 0 ? kExitOk : kExitError;
    }

    printUsage();

    return kExitError;
}
