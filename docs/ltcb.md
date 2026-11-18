# bwarp on the Large Text Compression Benchmark

The [Large Text Compression Benchmark](https://www.mattmahoney.net/dc/text.html) (LTCB) ranks
programs by the compressed size of enwik9 plus the size of the zipped decompresser, and also reports
enwik8, compression and decompression times in ns/byte, and memory. This page records bwarp's
figures, how they were measured, and a draft entry. `tools\ltcb.ps1` reproduces them.

## Results (2026-11-18)

Measured on v0.1.0, with `bwarp.exe` built by `build.bat` from the source zip below, as
`bwarp compress --segment 16 --slice 16` and `bwarp decompress`. Each time is the wall
clock of the whole process, CUDA start-up and teardown included, and is the best of 3 runs. Both
round trips were checked byte for byte. The times include writing the output to the drive, past the
file cache, but the input is read warm: after the first run it comes from the Windows file cache,
not the SSD.

| | enwik8 | enwik9 |
|---|---|---|
| Compressed bytes | 23,511,124 | 181,939,260 |
| Compression, ns/byte | 2.58 | 1.60 |
| Decompression, ns/byte | 2.04 | 0.93 |

| | Bytes |
|---|---|
| Decompresser: the zipped source (`git archive --format=zip -9`; `src`, CMake files, `build.bat`, `LICENSE`) | 104,799 |
| **enwik9 + decompresser** | **182,044,059** |

Peak memory on enwik9, from polling the running processes:

| | Host | GPU |
|---|---|---|
| Compression | 1.3 GiB | 10.7 GiB |
| Decompression | 1.4 GiB | 7.4 GiB |

The GPU needs about 10 bytes of memory per input byte to compress a block. Blocks are at most 1 GiB,
so a 16 GB card compresses enwik9 in one block.

## Settings

- **`--segment 16 --slice 16`:** frequency tables and streams per 64 KiB instead of the default
  32 KiB. This saves 0.54% on enwik9 at no cost in speed. It costs a little on smaller or mixed data
  (+0.20% on enwik8, +0.10% on Silesia), so it isn't the default. enwik8 is reported with the same
  options, as LTCB expects one setting for both.
- **`--sample 20`** would save another 57 KB on enwik9 but doubles the inverse BWT. It isn't used.

## Where the time goes (enwik9)

- **Compression** (~1.6 s):
  - ~0.35 s fixed cost: CUDA initialisation (~90 ms), the 1 GiB page-locked buffer (~60 ms),
    reserving the GPU memory pool (~65 ms), and teardown (~150 ms).
  - ~1.13 s on the GPU: the BWT build (ranking 383 ms, class sorts 117 ms, merge 452 ms) and the
    encoder.
  - Reading the input overlaps with setup.
- **Decompression** (~0.9 s): ~0.2 s setup, ~0.3 s on the GPU, and the 1 GB write. The write's
  speed depends on the drive's state, so this time varies by up to ±0.3 s.

## Test machine

- **GPU:** NVIDIA GeForce RTX 5070 Ti (16 GB, sm_120), driver 610.88, CUDA 13.5.
- **CPU and memory:** AMD Ryzen 7 5800X, 64 GB DDR4-3600.
- **Drive and OS:** Samsung 970 NVMe SSD, Windows 11 Pro.

## Draft entry

```
Program         bwarp 0.1.0 (GPU)
Options         compress --segment 16 --slice 16
enwik8          23,511,124
enwik9          181,939,260
Decompresser    104,799 (zipped source)
Total           182,044,059
Comp ns/byte    1.6
Decomp ns/byte  0.9
Mem (MB)        10,700 GPU + 1,300 host
Alg             BWT
Note            GPU (RTX 5070 Ti); BWT and entropy coding both on the GPU
Machine         Ryzen 7 5800X, 64 GB DDR4-3600, RTX 5070 Ti 16 GB, Windows 11 Pro
```

Before submitting:
1. Rerun `tools\ltcb.ps1` on the commit being submitted.
2. Check that the source zip of that commit builds with `build.bat` from a clean checkout.
