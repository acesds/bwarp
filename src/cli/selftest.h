/*
 * selftest.h - The built-in test suite (bwarp selftest).
 *
 * Part of bwarp, a GPU-based BWT file compressor.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

// Runs a command line as main() gets it: an exit code.
using CommandRunner = int (*)(int argc, char** argv);

// Returns the number of failures. command runs the compress and decompress commands on temporary files.
int runSelfTest(bool verbose, CommandRunner command);
