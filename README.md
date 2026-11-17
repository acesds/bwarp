# bwarp

bwarp is a BWT file compressor that does everything on the GPU: the Burrows–Wheeler transform, the entropy coding and, on the way back, the inverse transform. The data stays on the card the whole time.

How does it work? I took the GPU suffix sorting from my other project, [sadcs](https://github.com/acesds/sadcs):
1. It ranks a sample of the suffixes (7 of every 32 positions) by prefix doubling with radix sorts.
2. It sorts the other suffixes against that sample, in 7 classes.
3. A multiway merge writes out the BWT directly.

Then the BWT gets entropy coded, still on the GPU. Decompression runs it all backwards: it decodes the streams, then undoes the BWT along many independent chains, starting from rows the compressor wrote down.

So how fast is it? The whole thing started as an experiment: could a BWT-based codec decompress enwik9 (1 GB) in under a second? (Sure, some LZ codecs are faster, but where's the fun in that?) Turns out it can, in 0.91 s. Here are enwik8 (100 MB) and enwik9, compressed with `--segment 16 --slice 16`:

| | enwik8 | enwik9 |
|---|---|---|
| Compressed | 23,511,124 bytes | 181,939,260 bytes |
| Ratio | 4.25:1 (1.88 bits/byte) | 5.50:1 (1.46 bits/byte) |
| Compress | 0.26 s (2.57 ns/byte) | 1.63 s (1.63 ns/byte) |
| Decompress | 0.22 s (2.20 ns/byte) | 0.91 s (0.91 ns/byte) |

That's the wall clock for the whole process, CUDA start-up included. CUDA start-up and teardown take a bigger share of a smaller file, which is why enwik8 is slower per byte. I've only measured it on my own machine, an RTX 5070 Ti with a Ryzen 7 5800X and an NVMe SSD under Windows 11, so your times will differ ([docs/ltcb.md](docs/ltcb.md) has more).

This is about speed only, so the model is simple on purpose. If ratio is what you're after, context mixing is usually the better fit. Here's all the modeling bwarp does:
- The BWT is cut into slices (32 KiB by default), and each slice into 32 lanes, one per thread of a warp.
- The runs in each lane become (symbol, length) tokens.
- A symbol is coded as its rank in the segment's alphabet, most frequent first, with the previous symbol in the lane as context (up to 16 contexts).
- A length of 1 to 8 gets its own code. Longer ones get a power-of-two bucket plus extra bits, with the run's symbol as context (up to 4 contexts).
- Each segment (32 KiB by default) gets a semi-static model: its token counts, quantized and stored at the front of the block, and fixed while coding.
- Interleaved rANS codes the tokens, one state per lane, so a warp decodes a whole slice in lockstep.

The [docs/format.md](docs/format.md) has all the details.

## Command line

```
bwarp info
bwarp bwt <in> <out.bwt>
bwarp check <in> <in.bwt>
bwarp test <in|dir> [--recursive]
bwarp bench <in|dir> [--runs N] [--context] [--pinned] [--recursive]
bwarp compress <in> [<out>] | <dir> <outdir> [--segment LOG2] [--slice LOG2] [--contexts K] [--sample LOG2] [--block BYTES] [--recursive] [--force] [--quiet | --verbose]
bwarp decompress <in.bwarp> [<out>] | <dir> <outdir> [--recursive] [--force] [--quiet | --verbose]
bwarp ratio <in|dir> [--segment LOG2] [--slice LOG2] [--contexts K] [--sample LOG2] [--gpu [--verbose]] [--recursive]
bwarp selftest [--verbose]
```

The `.bwt` file is in divsufsort's `bwt` format as a single block: n and the primary index as little-endian int32 values, then the n bytes.

- `info`: print the GPU, or that bwarp was built without CUDA.
- `bwt`: build the BWT on the GPU.
- `check`: verify a BWT file against its text by inverting it on the CPU (sequential, about 12 MB/s on enwik8).
- `test`: build and check in memory, with the GPU's phases and memory. On a directory it runs every file in it (`--recursive` to include subdirectories), one row per file plus a total.
- `bench`: time the GPU build, the best of `--runs` (default 5) after a warmup, with each phase. `--context` reuses one GPU context per file, `--pinned` uses page-locked host buffers. `--context --pinned` measures the way libcubwt's benchmark does.
- `compress`: compress on the GPU, BWT and encoder without a round trip through the host, into `<in>.bwarp` unless `<out>` is given; a directory's files go to the same names plus `.bwarp` under `<outdir>`. The encoding options are those of `ratio`; `--block` (default and at most 2^30, 1 GiB, so enwik9 is one block) sets the block size. Each block stores the CRC-32C of its text, computed on the CPU while the GPU builds the BWT. `decompress` does the reverse, of a directory's `.bwarp` files, and checks each block's checksum while it writes the block, with one pool of device memory reserved for the largest block and reused by all (`gpuCreateDecompressContext()`). Both read the first block while they set up, then the next block while the GPU works on the current one, into page-locked buffers, and write the previous one meanwhile; they write blocks of 32 MiB and more past the Windows file cache, which is about twice as fast for them, refuse to overwrite files without `--force`, remove the output of a file that fails, and print each file's speed and the totals with setup, reading and writing, also in ns/byte as the Large Text Compression Benchmark reports it. `--quiet` prints only errors; `--verbose` adds each block's phases.
- `ratio`: build the BWT on the GPU, encode and decode it with the CPU reference codec, check the round trip and print the compressed size, bits per byte, runs per byte, the share of tables and index, and the CPU coding speeds. `--segment` (default 15) sets the frequency tables per 2^LOG2 bytes, `--slice` (default 15, 10 to 16) one 32-lane rANS stream per 2^LOG2 bytes, `--contexts` (default 16, 1 to 16) the most symbol contexts a table may use, `--sample` (default automatic, 10 to 31) an inverse BWT chain per 2^LOG2 bytes. `--gpu` also encodes and decompresses on the GPU, checks both, and prints their speeds without transfers (the faster of two runs), that of the stream kernel and that of the inverse BWT. It also runs the whole compression on the GPU, BWT and encoder without a round trip through the host, checks it against the CPU encoder and prints its speed host to host (`gcmp`); `--verbose` adds each phase, and the time of building the BWT and encoding it as two calls.
- `selftest`: the checker on known BWTs and planted errors, CRC-32C against its definition, codec round trips over several parameter sets with truncated and corrupted copies (the GPU decoder and decompressor must accept exactly what the CPU decoder accepts, with the same result), and the GPU construction on adversarial inputs, verified by the checker and then compressed and decompressed on the GPU, and `compress` and `decompress` on temporary files in one and in several blocks, with an existing output, a wrong checksum and a truncated file (GPU cases skipped without a device).

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
- `src/common/`: the BWT checker, CRC-32C (SSE4.2, on several threads for large inputs) and file I/O.
- `src/cli/`: the command line and the selftest.
- `docs/`: the [format](docs/format.md) and the [Large Text Compression Benchmark figures](docs/ltcb.md).
- `tools/`: `ltcb.ps1`, which measures those figures.

## License

MIT, see [LICENSE](LICENSE). Builds with the GPU variant include NVIDIA's CUB (BSD-3-Clause); [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) has its notice. `cmake --install` puts both files next to `bwarp.exe`.
