/*
 * checker.cpp - Implementation of the BWT checker.
 *
 * Part of bwarp, a GPU entropy coder and inverse transform for BWT output.
 * From sadcs (https://github.com/acesds/sadcs), BWT path only.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "checker.h"
#include <vector>

namespace {
    CheckResult pass() {
        return CheckResult{true, -1, nullptr};
    }

    CheckResult fail(int64_t index, const char* reason) {
        return CheckResult{false, index, reason};
    }
}

CheckResult checkBwt(const uint8_t* text, int32_t n, const uint8_t* bwt, int32_t primary) {
    if (n == 0) {
        return primary == 0 ? pass() : fail(-1, "primary index out of range");
    }

    if (primary < 1 || primary > n) {
        return fail(-1, "primary index out of range");
    }

    // n + 1 rows; row primary is the sentinel's, the others hold bwt[u] in order. lf[u]: where u's row maps to.
    std::vector<int32_t> lf(static_cast<size_t>(n));
    int32_t next[256] = {};

    for (int32_t u = 0; u < n; u++) {
        next[bwt[u]]++;
    }

    for (int32_t c = 0, sum = 1; c < 256; c++) {
        const int32_t count = next[c];
        next[c] = sum;
        sum += count;
    }

    for (int32_t u = 0; u < n; u++) {
        lf[u] = next[bwt[u]]++;
    }

    // From the sentinel's suffix (row 0), LF spells the text backwards.
    int32_t row = 0;

    for (int32_t i = n - 1; i >= 0; i--) {
        if (row == primary) {
            return fail(i, "inverse BWT ends early at text position index");
        }

        const int32_t u = row < primary ? row : row - 1;

        if (bwt[u] != text[i]) {
            return fail(i, "inverse BWT differs from the text at position index");
        }

        row = lf[u];
    }

    return row == primary ? pass() : fail(0, "inverse BWT doesn't end at the primary index");
}
