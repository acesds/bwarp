/*
 * selftest.h - The built-in test suite (bwarp selftest).
 *
 * Part of bwarp, a GPU entropy coder and inverse transform for BWT output.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

// Returns the number of failures.
int runSelfTest(bool verbose);
