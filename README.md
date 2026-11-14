# bwarp

Fast BWT compression on the GPU: interleaved rANS coding and parallel inverse BWT.

bwarp is the entropy stage and inverse transform for BWTs built by [sadcs](https://github.com/acesds/sadcs). The goal is throughput in both directions with the data staying on the device; ratio comes second. [docs/handoff.md](docs/handoff.md) has the design and the reasoning behind it.

## Status

Scaffold only. What exists so far:

- The build: clang-cl or MSVC for the C++ sources, nvcc for the GPU variant.
- sadcs's GPU BWT construction and its BWT checker, BWT path only (see [From sadcs](#from-sadcs)).
- The command line and a selftest.

Next are the steps in [First steps](docs/handoff.md#first-steps): the GPU inverse BWT microbenchmark, then the CPU ratio prototype.

## Command line

```
bwarp info
bwarp bwt <in> <out.bwt>
bwarp check <in> <in.bwt>
bwarp test <in|dir> [--recursive]
bwarp bench <in|dir> [--runs N] [--context] [--pinned] [--recursive]
bwarp selftest [--verbose]
```

The `.bwt` file is in divsufsort's `bwt` format as a single block, the same as `sadcs build --bwt` writes: n and the primary index as little-endian int32 values, then the n bytes.

- `info`: print the GPU, or that bwarp was built without CUDA.
- `bwt`: build the BWT on the GPU. Same output as `sadcs build --bwt --gpu`, and the same speed.
- `check`: verify a BWT file against its text by inverting it on the CPU (sequential, about 12 MB/s on enwik8).
- `test`: build and check in memory, with the GPU's phases and memory. On a directory it runs every file in it (`--recursive` to include subdirectories), one row per file plus a total.
- `bench`: time the GPU build, the best of `--runs` (default 5) after a warmup, with each phase. `--context` reuses one GPU context per file, `--pinned` uses page-locked host buffers. `--context --pinned` measures the way libcubwt's benchmark does.
- `selftest`: the checker on known BWTs and planted errors, and the GPU construction on adversarial inputs, verified by the checker (GPU cases skipped without a device).

## From sadcs

bwarp needs only sadcs's BWT, so it takes only that path:

- `src/cuda/samplerank.cuh`, `classsort.cuh`, `merge.cuh`, `devicecommon.cuh`, `differencecover.h`: the GPU construction. The merge writes only the BWT (and the splitter indices it needs internally), not the suffix array. Only the cover modulo 32 is kept.
- `src/cuda/gpubwarp.cu`: sadcs's host driver (staging, contexts, memory planning), with `gpuBwt()` only. No `gpuSuffixArray()` and no `gpuRankSample()`.
- `src/common/checker.*`: `checkBwt()` only, none of the suffix array checks.

sadcs's CPU construction is not included. [sadcs's gpu-design.md](https://github.com/acesds/sadcs/blob/main/docs/gpu-design.md) documents the construction.

## Build

bwarp builds on Windows x64 only for now. You'll need:

- Visual Studio 2022 with the "Desktop development with C++" workload.
- CMake 3.23 or newer.
- For the `clang` and `cuda` presets, the standalone LLVM package installed to `C:\Program Files\LLVM`.
- For the GPU variant, the CUDA toolkit 13 or newer (tested with 13.5).

Then:

```
build.bat
```

If the CUDA toolkit is installed (`CUDA_PATH` is set), this builds the CPU and GPU parts into `build-cuda\bin\Release\`, otherwise the CPU parts only into `build\bin\Release\`. `build.bat cpu` always builds the CPU parts only.

By hand:

```
cmake --preset cuda -S .           # Ninja, clang-cl + nvcc, from a Developer Command Prompt
cmake --build build-cuda

cmake --preset msvc -S .           # MSVC; with the toolkit installed, also the GPU variant
cmake --build build-msvc --config Release

cmake --preset clang -S .          # clang-cl, CPU parts only
cmake --build build --config Release
```

With the ClangCL toolset, the CUDA 13.5 Visual Studio integration fails CMake's compiler check, so the `clang` preset builds without the GPU variant and `cuda` uses Ninja instead. Ninja needs the MSVC environment for `cl.exe`, nvcc's host compiler; `build.bat` sets it up itself.

## Where things live

- `src/cuda/`: the GPU variant; its interface is `gpubwarp.h`. Builds without CUDA use `gpubwarp_stub.cpp` instead.
- `src/common/`: the BWT checker and file I/O.
- `src/cli/`: the command line and the selftest.
- `docs/`: the design records, starting with [handoff.md](docs/handoff.md).

## License

MIT, see [LICENSE](LICENSE). Builds with the GPU variant include NVIDIA's CUB (BSD-3-Clause); [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) has its notice. `cmake --install` puts both files next to `bwarp.exe`.
