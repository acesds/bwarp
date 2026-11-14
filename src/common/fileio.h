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

bool writeWholeFile(const char* path, const void* data, size_t size);

// divsufsort's bwt format as one block, as sadcs writes it: n and the primary index as int32, then the n bytes.
bool readBwtFile(const char* path, std::vector<uint8_t>& out, int32_t& primary);

bool writeBwtFile(const char* path, const uint8_t* bwt, int32_t n, int32_t primary);
