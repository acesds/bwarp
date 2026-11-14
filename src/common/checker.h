/*
 * checker.h - BWT verification.
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * From sadcs (https://github.com/acesds/sadcs), BWT path only.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstdint>

struct CheckResult {
    bool ok;
    int64_t index;      // text position of the first failure, -1 if ok or not tied to a position
    const char* reason; // nullptr if ok
};

// Inverts a BWT and compares it with text.
// The BWT is a bijection, so passing means bwt is the BWT of text.
CheckResult checkBwt(const uint8_t* text, int32_t n, const uint8_t* bwt, int32_t primary);
