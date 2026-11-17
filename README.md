# bwarp

A GPU-based BWT file compressor: the Burrows–Wheeler transform, entropy coding and the inverse transform all run on the GPU.

The BWT construction comes from [sadcs](https://github.com/acesds/sadcs); the entropy coder is interleaved rANS, one state per warp lane. The goal is throughput in both directions with the data staying on the device; ratio comes second. [docs/handoff.md](docs/handoff.md) has the design and the reasoning behind it.

## Status

What exists so far:

- The build: clang-cl or MSVC for the C++ sources, nvcc for the GPU variant.
- sadcs's GPU BWT construction and its BWT checker, BWT path only (see [From sadcs](#from-sadcs)).
- Format v0 ([docs/format.md](docs/format.md)) with a sequential CPU reference encoder and decoder. It is the ratio prototype and the oracle the GPU coder must match byte for byte.
- The GPU decoder, byte for byte the same as the CPU one, valid input or not.
- The GPU inverse BWT, so decompression runs entirely on the device: LF from a stable ranking of the BWT, then one chain per sampled row that the construction records, walked backward (from both ends, with psi too, when there are few).
- The command line and a selftest.

Next: the GPU encoder.

## Command line

```
bwarp info
bwarp bwt <in> <out.bwt>
bwarp check <in> <in.bwt>
bwarp test <in|dir> [--recursive]
bwarp bench <in|dir> [--runs N] [--context] [--pinned] [--recursive]
bwarp compress <in> [<out>] | <dir> <outdir> [--segment LOG2] [--slice LOG2] [--contexts K] [--sample LOG2] [--recursive] [--force] [--verbose]
bwarp decompress <in.bwarp> [<out>] | <dir> <outdir> [--recursive] [--force] [--verbose]
bwarp ratio <in|dir> [--segment LOG2] [--slice LOG2] [--contexts K] [--sample LOG2] [--gpu [--verbose]] [--recursive]
bwarp selftest [--verbose]
```

The `.bwt` file is in divsufsort's `bwt` format as a single block, the same as `sadcs build --bwt` writes: n and the primary index as little-endian int32 values, then the n bytes.

- `info`: print the GPU, or that bwarp was built without CUDA.
- `bwt`: build the BWT on the GPU. Same output as `sadcs build --bwt --gpu`, and the same speed.
- `check`: verify a BWT file against its text by inverting it on the CPU (sequential, about 12 MB/s on enwik8).
- `test`: build and check in memory, with the GPU's phases and memory. On a directory it runs every file in it (`--recursive` to include subdirectories), one row per file plus a total.
- `bench`: time the GPU build, the best of `--runs` (default 5) after a warmup, with each phase. `--context` reuses one GPU context per file, `--pinned` uses page-locked host buffers. `--context --pinned` measures the way libcubwt's benchmark does.
- `compress`: compress on the GPU, BWT and encoder without a round trip through the host, into `<in>.bwarp` unless `<out>` is given; a directory's files go to the same names plus `.bwarp` under `<outdir>`. The encoding options are those of `ratio`. `decompress` does the reverse, of a directory's `.bwarp` files. Both read the next file while the GPU works on the current one, into page-locked buffers (`decompress` also writes the previous one meanwhile), write files of 32 MiB and more past the Windows file cache, which is about twice as fast for them, refuse to overwrite files without `--force`, and print each file's speed and the totals with reading and writing; `--verbose` adds the phases.
- `ratio`: build the BWT on the GPU, encode and decode it with the CPU reference codec, check the round trip and print the compressed size, bits per byte, runs per byte, the share of tables and index, and the CPU coding speeds. `--segment` (default 15) sets the frequency tables per 2^LOG2 bytes, `--slice` (default 15, 10 to 16) one 32-lane rANS stream per 2^LOG2 bytes, `--contexts` (default 16, 1 to 16) the most symbol contexts a table may use, `--sample` (default automatic, 10 to 31) an inverse BWT chain per 2^LOG2 bytes. `--gpu` also encodes and decompresses on the GPU, checks both, and prints their speeds without transfers (the faster of two runs), that of the stream kernel and that of the inverse BWT. It also runs the whole compression on the GPU, BWT and encoder without a round trip through the host, checks it against the CPU encoder and prints its speed host to host (`gcmp`); `--verbose` adds each phase, and the time of building the BWT and encoding it as two calls.
- `selftest`: the checker on known BWTs and planted errors, codec round trips over several parameter sets with truncated and corrupted copies (the GPU decoder and decompressor must accept exactly what the CPU decoder accepts, with the same result), and the GPU construction on adversarial inputs, verified by the checker and then compressed and decompressed on the GPU (GPU cases skipped without a device).

## From sadcs

bwarp needs only sadcs's BWT, so it takes only that path:

- `src/cuda/samplerank.cuh`, `classsort.cuh`, `merge.cuh`, `devicecommon.cuh`, `differencecover.h`: the GPU construction. The merge writes only the BWT (and the splitter indices it needs internally), not the suffix array, plus the rows of sampled text positions for the inverse. Only the cover modulo 32 is kept.
- `src/cuda/gpubwarp.cu`: sadcs's host driver (staging, contexts, memory planning) with `gpuBwt()`, plus bwarp's `gpuDecode()`, `gpuDecompress()`, `gpuEncode()` and `gpuCompress()`, which builds the BWT and encodes it on the device. No `gpuSuffixArray()` and no `gpuRankSample()`.
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

- `src/cuda/`: the GPU variant; its interface is `gpubwarp.h`. Builds without CUDA use `gpubwarp_stub.cpp` instead. `decode.cuh` holds the decoder kernels, `inverse.cuh` the inverse BWT.
- `src/codec/`: the format (`format.h`), table coding (`table.h`) and container (`container.h`), shared by CPU and GPU, and the CPU reference codec.
- `src/common/`: the BWT checker and file I/O.
- `src/cli/`: the command line and the selftest.
- `docs/`: the design records, starting with [handoff.md](docs/handoff.md), and the [format](docs/format.md).

## License

MIT, see [LICENSE](LICENSE). Builds with the GPU variant include NVIDIA's CUB (BSD-3-Clause); [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) has its notice. `cmake --install` puts both files next to `bwarp.exe`.
