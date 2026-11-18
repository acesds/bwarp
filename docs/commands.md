# bwarp commands

Every command and option in detail. `bwarp` without arguments prints a short summary, and `bwarp --version`
prints the version.

## compress and decompress

```
bwarp compress <in> [<out>] | <dir> <outdir> [--block BYTES] [--segment LOG2] [--slice LOG2]
               [--contexts K] [--sample LOG2] [--recursive] [--force] [--quiet | --verbose]
bwarp decompress <in.bwarp> [<out>] | <dir> <outdir> [--recursive] [--force] [--quiet | --verbose]
```

`compress` compresses a file on the GPU, BWT and encoder, without a round trip through the host. It writes `<in>.bwarp` unless `<out>` is given. Given a directory, it compresses each file in it to the same name plus `.bwarp` under `<outdir>`; `--recursive` includes subdirectories. `decompress` does the reverse, of a directory's `.bwarp` files.

A file is compressed in blocks of `--block` bytes, by default and at most 2^30 (1 GiB), so enwik9 is one block. Each block stores the CRC-32C of its text, which the CPU computes while the GPU builds the BWT. `decompress` checks it while it writes the block.

Both commands:
- read the next block while the GPU works on the current one, into page-locked buffers, and write the previous block meanwhile
- write blocks of 32 MiB and more past the Windows file cache, which is about twice as fast for them
- refuse to overwrite files unless `--force` is given
- remove the output of a file that fails
- print each file's speed and the totals, including setup, reading and writing, also in ns/byte as the Large Text Compression Benchmark reports it

`decompress` reserves one pool of device memory, for the largest block, and reuses it for all blocks.

| Option | Default | |
|---|---|---|
| `--block BYTES` | 2^30 | Block size, 1 to 2^30 |
| `--segment LOG2` | 15 | One frequency table per 2^LOG2 bytes |
| `--slice LOG2` | 15 | One 32-lane rANS stream per 2^LOG2 bytes, 10 to 16 |
| `--contexts K` | 16 | The most symbol contexts a table may use, 1 to 16 |
| `--sample LOG2` | automatic | One inverse BWT chain per 2^LOG2 bytes, 10 to 31 |
| `--quiet` | | Print only errors |
| `--verbose` | | Also print each block's phases |

The encoding options don't apply to `decompress`: the block headers record them.

## info

```
bwarp info
```

Prints the GPU, or that bwarp was built without CUDA.

## bwt and check

```
bwarp bwt <in> <out.bwt>
bwarp check <in> <in.bwt>
```

`bwt` builds the BWT of a file on the GPU. `check` verifies a BWT file against its text by inverting it on the CPU. That runs sequentially, at about 12 MB/s on enwik8.

The `.bwt` file is in divsufsort's `bwt` format, as a single block: n and the primary index as little-endian int32 values, then the n bytes.

## test

```
bwarp test <in|dir> [--recursive]
```

Builds the BWT on the GPU and checks it, in memory, and prints the GPU's phases and memory. On a directory it runs every file in it (`--recursive` to include subdirectories), with one row per file and a total.

## bench

```
bwarp bench <in|dir> [--runs N] [--context] [--pinned] [--recursive]
```

Times the GPU's BWT construction, host to host: a warmup, then the best of `--runs` runs (default 5), with each phase of the best one. It doesn't check the output; `test` does.

- `--context` reuses one GPU context per file, created before the runs and not timed.
- `--pinned` uses page-locked host buffers, allocated before the runs.

`--context --pinned` measures the way libcubwt's benchmark does.

## ratio

```
bwarp ratio <in|dir> [--segment LOG2] [--slice LOG2] [--contexts K] [--sample LOG2] [--gpu [--verbose]]
            [--recursive]
```

Builds the BWT on the GPU, then encodes and decodes it with the CPU reference codec and checks the round trip. Prints the compressed size, bits per byte, runs per byte, the share of tables and index, and the CPU coding speeds. The encoding options are those of `compress`.

`--gpu` also encodes and decompresses on the GPU and checks both. It prints their speeds without transfers (the faster of two runs), and those of the stream decoder and of the inverse BWT. It also runs the whole compression on the GPU, checks it against the CPU encoder and prints its speed host to host (`gcmp`). `--verbose` adds each phase, and the time of building the BWT and encoding it as two separate calls.

## selftest

```
bwarp selftest [--verbose]
```

Runs the built-in tests and prints how many passed. Without a GPU, the GPU cases are skipped.

- The checker on known BWTs and on planted errors.
- CRC-32C against its definition.
- Codec round trips over several parameter sets, with truncated and corrupted copies. The GPU decoder and decompressor must accept exactly what the CPU decoder accepts, with the same result.
- The GPU construction on adversarial inputs, verified by the checker, then compressed and decompressed on the GPU.
- `compress` and `decompress` on temporary files, in one block and in several, and with an existing output, a wrong checksum and a truncated file.

`--verbose` prints every case.
