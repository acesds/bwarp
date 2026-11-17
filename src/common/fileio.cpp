/*
 * fileio.cpp - File read and write helpers for the CLI.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "fileio.h"
#include <algorithm>
#include <cstdio>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {
#if defined(_WIN32)
    constexpr size_t kDirectAlignment = 4096;  // a multiple of any disk's sector size
    // Smaller writes are faster through the cache (16 MiB: 3.4 ms, against 5.2), larger ones past it (64 MiB: 20 ms,
    // against 36): measured with an NVMe drive, at 3.3 GB/s.
    constexpr size_t kDirectMinimum = size_t(32) << 20;

    // The first size bytes of data (a multiple of kDirectAlignment, data aligned to it) past the file cache, into a
    // new file at path. Unavailable (as on some network shares) if the file can't be opened that way.
    enum class Direct { Done, Unavailable, Failed };

    Direct writeUncached(const char* path, const uint8_t* data, size_t size) {
        const HANDLE file =
            CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING, nullptr);

        if (file == INVALID_HANDLE_VALUE) {
            return Direct::Unavailable;
        }

        // In pieces of at most 64 MiB, a multiple of the alignment that WriteFile's DWORD takes.
        constexpr size_t kPiece = size_t(64) << 20;
        size_t done = 0;

        while (done < size) {
            DWORD written = 0;

            if (!WriteFile(file, data + done, static_cast<DWORD>(std::min(kPiece, size - done)), &written, nullptr) ||
                written == 0) {
                break;
            }

            done += written;
        }

        return CloseHandle(file) && done == size ? Direct::Done : Direct::Failed;
    }
#endif
    // File size in bytes, or -1; leaves fp at the start.
    int64_t fileSize(FILE* fp) {
        _fseeki64(fp, 0, SEEK_END);
        const int64_t size = _ftelli64(fp);
        _fseeki64(fp, 0, SEEK_SET);
        return size;
    }
}

bool readWholeFile(const char* path, std::vector<uint8_t>& out, size_t padding) {
    FILE* fp = fopen(path, "rb");

    if (!fp) {
        return false;
    }

    const int64_t size = fileSize(fp);

    if (size < 0) {
        fclose(fp);
        return false;
    }

    out.assign(static_cast<size_t>(size) + padding, 0);
    const size_t bytesRead = fread(out.data(), 1, static_cast<size_t>(size), fp);
    fclose(fp);

    return bytesRead == static_cast<size_t>(size);
}

bool writeWholeFile(const char* path, const void* data, size_t size) {
    FILE* fp = fopen(path, "wb");

    if (!fp) {
        return false;
    }

    const size_t bytesWritten = fwrite(data, 1, size, fp);
    const bool closed = fclose(fp) == 0;

    return bytesWritten == size && closed;
}

bool writeFileDirect(const char* path, const void* data, size_t size) {
#if defined(_WIN32)
    const auto* bytes = static_cast<const uint8_t*>(data);
    const size_t head = size / kDirectAlignment * kDirectAlignment;

    if (size >= kDirectMinimum && reinterpret_cast<uintptr_t>(bytes) % kDirectAlignment == 0) {
        switch (writeUncached(path, bytes, head)) {
            case Direct::Done: {
                // The tail, less than a sector, through the cache.
                if (head == size) {
                    return true;
                }

                FILE* fp = fopen(path, "ab");

                if (!fp) {
                    return false;
                }

                const size_t bytesWritten = fwrite(bytes + head, 1, size - head, fp);
                const bool closed = fclose(fp) == 0;

                return bytesWritten == size - head && closed;
            }
            case Direct::Failed:
                return false;
            case Direct::Unavailable:
                break;
        }
    }
#endif

    return writeWholeFile(path, data, size);
}

bool readFileInto(const char* path, void* buffer, size_t size) {
    FILE* fp = fopen(path, "rb");

    if (!fp) {
        return false;
    }

    // In pieces of at most 1 GiB, each one call underneath.
    constexpr size_t kPiece = size_t(1) << 30;
    size_t done = 0;

    while (done < size) {
        const size_t got = fread(static_cast<uint8_t*>(buffer) + done, 1, std::min(kPiece, size - done), fp);

        if (got == 0) {
            break;
        }

        done += got;
    }

    fclose(fp);

    return done == size;
}

bool readBwtFile(const char* path, std::vector<uint8_t>& out, int32_t& primary) {
    FILE* fp = fopen(path, "rb");

    if (!fp) {
        return false;
    }

    const int64_t size = fileSize(fp);
    int32_t blockSize = -1;
    primary = 0;
    bool ok = size >= 4 && fread(&blockSize, sizeof(blockSize), 1, fp) == 1 && blockSize >= 0;

    // A single block: the whole text, or nothing at all for an empty one.
    if (ok && blockSize == 0) {
        ok = size == 4;
        out.clear();
    } else if (ok) {
        ok = size == 8 + static_cast<int64_t>(blockSize) && fread(&primary, sizeof(primary), 1, fp) == 1;

        if (ok) {
            out.resize(static_cast<size_t>(blockSize));
            ok = fread(out.data(), 1, out.size(), fp) == out.size();
        }
    }

    fclose(fp);

    return ok;
}

bool writeBwtFile(const char* path, const uint8_t* bwt, int32_t n, int32_t primary) {
    FILE* fp = fopen(path, "wb");

    if (!fp) {
        return false;
    }

    bool ok = fwrite(&n, sizeof(n), 1, fp) == 1;

    if (ok && n > 0) {
        ok = fwrite(&primary, sizeof(primary), 1, fp) == 1 &&
             fwrite(bwt, 1, static_cast<size_t>(n), fp) == static_cast<size_t>(n);
    }

    const bool closed = fclose(fp) == 0;

    return ok && closed;
}
