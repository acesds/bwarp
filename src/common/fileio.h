/*
 * fileio.h - File read and write helpers for the CLI.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// out.size() is the file size plus padding zero bytes.
bool readWholeFile(const char* path, std::vector<uint8_t>& out, size_t padding = 0);

// append: to the end of an existing file.
bool writeWholeFile(const char* path, const void* data, size_t size, bool append = false);

// writeWholeFile(), but on Windows past the file cache, about twice as fast for large files (100 MB: 30 ms instead
// of 55) and on the disk on return, if data is aligned to 4 KiB (as page-locked memory is) and size at least 32 MiB,
// and when appending, the file's size a multiple of 4 KiB. Smaller files are faster through the cache.
bool writeFileDirect(const char* path, const void* data, size_t size, bool append = false);

// size bytes of the file from offset into buffer: false if the file has fewer.
bool readFileInto(const char* path, void* buffer, size_t size, uint64_t offset = 0);

// divsufsort's bwt format as one block, as sadcs writes it: n and the primary index as int32, then the n bytes.
bool readBwtFile(const char* path, std::vector<uint8_t>& out, int32_t& primary);

bool writeBwtFile(const char* path, const uint8_t* bwt, int32_t n, int32_t primary);
