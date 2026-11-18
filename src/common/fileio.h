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

// Reads a whole file into out, with padding zero bytes at the end.
// Returns false if the file can't be opened or read.
bool readWholeFile(const char* path, std::vector<uint8_t>& out, size_t padding = 0);

// Writes a new file, or with append, appends to an existing one.
// Returns false if the file can't be opened or written.
bool writeWholeFile(const char* path, const void* data, size_t size, bool append = false);

// Same as writeWholeFile(), but on Windows past the file cache, which is faster for large files: if size is at
// least 32 MiB and data aligned to 4 KiB, as page-locked memory is. Otherwise writes through the cache.
bool writeFileDirect(const char* path, const void* data, size_t size, bool append = false);

// Reads size bytes from a file into buffer, starting at offset.
// Returns false if the file can't be opened or read, or if the file has fewer than size bytes.
bool readFileInto(const char* path, void* buffer, size_t size, uint64_t offset = 0);

// Reads a BWT file, divsufsort's format: n and the primary index as int32, then the n bytes.
// Returns false if the file can't be opened or read, or is malformed.
bool readBwtFile(const char* path, std::vector<uint8_t>& out, int32_t& primary);

// Writes a BWT file in the same format.
// Returns false if the file can't be opened or written.
bool writeBwtFile(const char* path, const uint8_t* bwt, int32_t n, int32_t primary);
