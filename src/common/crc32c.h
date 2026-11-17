/*
 * crc32c.h - CRC-32C (Castagnoli), the checksum bwarp stores for each block's text.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>

// The CRC-32C of size bytes, continuing from crc, that of the bytes before (0 for none): crc32c("123456789") is
// 0xE3069283. With SSE4.2's crc32 instruction, and on several threads for large inputs.
uint32_t crc32c(const void* data, size_t size, uint32_t crc = 0);

// The CRC-32C of A then B, from that of A, that of B and B's size.
uint32_t crc32cCombine(uint32_t crcA, uint32_t crcB, uint64_t sizeB);
