# bwarp

bwarp is a BWT file compressor that does everything on the GPU: the Burrows–Wheeler transform, the entropy coding and, on the way back, the inverse transform. The data stays on the card the whole time.

How does it work? I took the GPU suffix sorting from my other project, [sadcs](https://github.com/acesds/sadcs):
1. It ranks a sample of the suffixes (7 of every 32 positions) by prefix doubling with radix sorts.
2. It sorts the other suffixes against that sample, in 6 classes by their distance to the next sampled position.
3. A multiway merge of those 6 lists and the sample writes out the BWT directly.

Then the BWT gets entropy coded, still on the GPU. Decompression runs it all backwards: it decodes the streams, then undoes the BWT along many independent chains, starting from rows the compressor wrote down.

With this design, bwarp decompresses enwik9 (1 GB) in under a second on my test machine: 0.93 s. Here are enwik8 (100 MB) and enwik9, compressed with `--segment 16 --slice 16`:

| | enwik8 | enwik9 |
|---|---|---|
| Compressed | 23,511,124 bytes | 181,939,260 bytes |
| Ratio | 4.25:1 (1.88 bits/byte) | 5.50:1 (1.46 bits/byte) |
| Compress | 0.26 s (2.58 ns/byte) | 1.60 s (1.60 ns/byte) |
| Decompress | 0.20 s (2.04 ns/byte) | 0.93 s (0.93 ns/byte) |

That's the wall clock for the whole process, CUDA start-up included. CUDA start-up and teardown take a bigger share of a smaller file, which is why enwik8 is slower per byte. I've only measured it on my own machine, an RTX 5070 Ti with a Ryzen 7 5800X and an NVMe SSD under Windows 11, so your times will differ ([docs/ltcb.md](docs/ltcb.md) has more).

This is about speed only, so the model is simple on purpose. If ratio is what you're after, context mixing is usually the better fit. Here's all the modeling bwarp does:
- The BWT is cut into slices (32 KiB by default), and each slice into 32 lanes, one per thread of a warp.
- The runs in each lane become (symbol, length) tokens.
- A symbol is coded as its rank in the segment's alphabet, most frequent first, with the previous symbol in the lane as context (up to 16 contexts).
- A length of 1 to 8 gets its own code. Longer ones get a power-of-two bucket plus extra bits, with the run's symbol as context (up to 4 contexts).
- Each segment (32 KiB by default) gets a semi-static model: its token counts, quantized and stored at the front of the block, and fixed while coding.
- Interleaved rANS codes the tokens, one state per lane, so a warp decodes a whole slice in lockstep.

The [docs/format.md](docs/format.md) has all the details.

## Complexity

For a block of n bytes:

- **Building the BWT** takes O(n log ℓ) work, where ℓ is the length of the longest repeated substring, so O(n log n) at worst (think of a file of one repeated byte). Only the prefix doubling depends on ℓ. Radix sorts handle the first 36 characters. After that, each doubling round doubles the compared length, starting at 32, and keeps only the suffixes still tied. The class sorts are O(n), and so is the merge in practice: a comparison looks at no more than 31 characters, then at one rank of each suffix. Strictly, the merge's binary searches add O((n/128) log n), fewer than 2 comparisons per byte for blocks of up to 2^30 bytes. On enwik9 the doubling rounds take about 4% of the build.
- **Entropy coding** is O(n) both ways.
- **The inverse BWT** is O(n): one counting pass for the LF mapping, then n steps spread over many independent chains.
- **GPU memory** peaks at about 10n bytes to compress. Decompressing takes about 6.3n plus the compressed block, or 10.3n when a block has at most 2,048 chains, because those also walk forward. The CUDA runtime adds its own ~1.4 GiB on top.
- **Host memory** is about a block's input plus its output, twice over when a file has more than one block, so the next block can be read meanwhile.

## Requirements

- **An NVIDIA Ampere GPU (RTX 30 series) or newer.** The GPU code is built for sm_86 and sm_120, tested on an RTX 3090 and an RTX 5070 Ti; the RTX 40 series runs the sm_86 code but is untested. The v0.1.0 binaries contain sm_120 only, so they need an RTX 50 series GPU.
- **An NVIDIA driver for CUDA 13**, version 580 or newer. The CUDA runtime is built into bwarp.
- **Windows:** 64-bit Windows 10 or 11. Nothing else needs installing.
- **Linux:** x86-64 with glibc 2.34 and the C++ library of GCC 13.2 or newer, for example Ubuntu 24.04, Debian 13 or Fedora 39.

## Command line

```
bwarp compress enwik9                # writes enwik9.bwarp
bwarp decompress enwik9.bwarp out    # writes out
bwarp compress data packed           # every file in data, into packed
```

| Command | |
|---|---|
| `compress`, `decompress` | Compress or decompress a file, or every file in a directory |
| `info` | Show the GPU |
| `test` | Build the BWT on the GPU and check it |
| `bench` | Time the GPU's BWT construction |
| `ratio` | Show the compressed size, with a checked round trip |
| `bwt`, `check` | Write the BWT of a file, or check one against its text |
| `selftest` | Run the built-in tests |

`bwarp` without arguments lists the options. [docs/commands.md](docs/commands.md) describes every command and option in detail.

## Build

bwarp builds on Windows x64 and on Linux (tested under WSL2 with Ubuntu). Without the CUDA toolkit you get the CPU parts only: the reference codec, the checker and the selftest. The GPU variant targets sm_86 (Ampere) and sm_120 (Blackwell) plus PTX for newer GPUs; pass `-DBWARP_CUDA_ARCHITECTURES=...` to build for others, though I've only tested it on sm_86 and sm_120.

### Linux

You'll need CMake 3.23 or newer, Ninja, GCC or Clang with C++20, and for the GPU variant the CUDA toolkit (`nvcc` on the `PATH` or in `CUDACXX`). Then:

```
./build.sh
```

This builds the CPU and GPU parts into `build-cuda/bin/Release/` if it finds `nvcc`, otherwise the CPU parts only into `build/bin/Release/`. `./build.sh cpu` always builds the CPU parts only. By hand, that's `cmake --preset linux-cuda` or `cmake --preset linux`, then `cmake --build` on the build directory.

### Windows

You'll need:

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
- `docs/`: the [commands](docs/commands.md), the [format](docs/format.md) and the [Large Text Compression Benchmark figures](docs/ltcb.md).
- `tools/`: `ltcb.ps1`, which measures those figures.

## License

MIT, see [LICENSE](LICENSE). Builds with the GPU variant include NVIDIA's CUB (BSD-3-Clause); [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) has its notice. The release packages include both files, and `cmake --install` puts them in the install prefix, with the binary in `bin/`.
