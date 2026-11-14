# bwarp: handoff

**bwarp** is a GPU-based BWT file compressor, built for very high throughput in both directions. The name combines BWT and warp, the GPU's unit of lockstep execution that the decoder is built around. The BWT construction is taken from sadcs (`C:\Users\Anwender\Code\repos\sadcs`, BWT path only); sadcs itself stays unchanged.

This document records the discussion of 2026-11-14: the goal, the reasoning behind the chosen design, the options that were set aside, and the first steps. Nothing has been built or measured yet. Every number below is either a sadcs measurement or an estimate, and each one says which.

## Goal

The only real advantage BWT compression has over PPM and CM is speed, not ratio. So bwarp optimizes throughput first:

- **Ultra-fast encoding and decoding, both on the GPU.**
- **Ratio:** as good as possible without giving up speed. It doesn't need to beat bsc/QLFC.
- **Data stays on the device.** The BWT is already in device memory after sadcs, so only the compressed bits cross PCIe. The decoder writes the reconstructed text to device memory.

## Where the time goes

sadcs builds the BWT at about 0.95–1.15 GB/s on a GeForce RTX 5070 Ti, measured (enwik8 1163 MB/s, enwik9 943 MB/s, transfers included; see the sadcs README).

- **Encoding is bounded by the BWT.** A static, table-driven entropy coder on the GPU should run at tens of GB/s (estimate).
- **Decoding is bounded by the inverse BWT.** It's a chain of dependent random reads (LF mapping), limited by memory latency, not bandwidth. Its GPU throughput is unknown; measuring it is the first task.

Design rule: the entropy stage needs to stay about 10× faster than the BWT and the inverse BWT (≥10–20 GB/s each way). Beyond that, extra codec speed doesn't show end to end. Within that budget, maximize ratio.

## Chosen design: RLE plus semi-static rANS per segment

### Format

- The BWT is split into **segments** of about 64 KB to 1 MB (to be tuned). Segments are independent, and an offset index makes each one addressable.
- Each segment is run-length encoded into **(symbol, run length) tokens**.
- **Symbols** are coded with a small context, e.g. the previous run's symbol, bucketed so the tables stay small.
- **Run lengths** are coded as log-buckets plus raw extra bits, like zstd's length codes.
- Each segment carries **normalized frequency tables** (12-bit precision, compactly coded), or a reference to earlier or shared tables.
- **Interleaved rANS:** 32 states per segment, one per warp lane. Lane i codes tokens i, i+32, i+64, and so on.

### Encoder (GPU)

Every step is a histogram, a scan or a table lookup. There are no sequential dependencies across segments.

1. Run detection: adjacent difference, then stream compaction.
2. Histogram the tokens per segment and context.
3. Normalize the frequencies and build the tables.
4. rANS encoding, one warp per segment, 32 lanes. rANS encodes backwards, so each lane walks its tokens in reverse.
5. Prefix-sum the segment sizes and pack the output.

### Decoder (GPU)

1. One warp per segment, tables in shared memory.
2. All lanes decode in lockstep: a table lookup plus a multiply-add per symbol.
3. Expand runs: a prefix sum over the run lengths gives each run's output position, then fill.
4. Fused with the inverse BWT, see below.

### Ratio gains that cost no decode speed

All of these cost only encoder time and table bytes:

- **Segment size:** smaller segments follow the BWT's local statistics better but carry more table overhead.
- **Adaptive segment boundaries:** split where the statistics change, not at fixed offsets.
- **Table sharing:** a segment refers to an earlier segment's tables or to a preset set.
- **Context choice at encode time:** the encoder picks one of a few context functions per segment and signals it in about 2 bits.

## Inverse BWT on the GPU

- **Sampled starting points.** Store an extra primary index every r positions, like libsais's auxiliary indexes. The LF chain then splits into n/r independent chains, one per thread. The cost is 4 bytes per r symbols, so r trades size for parallelism.
- **Fusion with entropy decoding.** LF mapping needs symbol counts at segment boundaries. The entropy decoder sees every symbol of its segment, so it can produce exact counts as a side effect (shared-memory atomics), and no separate rank pass is needed. The normalized rANS frequencies are not exact counts, so they can't serve for this.
- **Ceiling:** the random gather rate of the card. This decides the whole decoder's speed.

## Options set aside, and why

| option | why not (for now) |
|---|---|
| M99 (Yuta Mori, `Yuta Mori/m99coder`) | It codes per-symbol counts over a binary tree of positions. Encoding parallelizes, since all counts come from a bottom-up histogram reduction. But the bitstream is written depth-first with value-dependent code lengths, so decoding is serial unless the format is changed, and per-node work over up to 256 symbols is irregular. A modeled count coder (counts in BWT output are usually 0, everything, or near the proportional split) would beat its truncated binary code on ratio. |
| Wavelet tree with RLE bitvectors | Both directions are flat and parallel (each level is a 1-bit stable split; decoding is a rank-guided scatter per level). But it makes about 8 passes over the data where direct coding makes one. **This is the fallback** if the direct coder's ratio disappoints. |
| MTF + RLE0 + entropy coder (bzip2) | MTF is sequential and data-dependent on decode. Its state at chunk boundaries can be found with a scan, but RLE plus context modeling should recover most of what MTF gains. |
| Huffman | BWT output is very skewed, and Huffman needs at least 1 bit per symbol. rANS handles fractional bits at about the same speed. |
| Adaptive arithmetic coding (QLFC, CM-style) | Best ratio, but serial within each stream. Contrary to the goal. |

## A bigger lever: the sort transform

If speed decides everything, an order-k sort transform (ST4–ST8, as in libbsc) can replace the full BWT. The forward transform is a few radix passes over k-byte contexts, much faster than suffix sorting, and the inverse is more parallel than the LF chain. The ratio loss is usually small at k=6–8. This attacks the actual bottleneck. It's out of scope for the first version, but the format should allow a transform-type field.

## First steps

1. **Inverse-BWT microbenchmark (GPU).** A naive parallel inverse BWT with sampled starting points on enwik8 and enwik9, sweeping r. This sets the decode ceiling and the budget for the entropy stage.
2. **Ratio prototype (CPU, plain C).** RLE plus segmented semi-static rANS on sadcs BWTs of Silesia, enwik8 and enwik9, compared with bzip2 and bsc. Sweep segment size, context function and table coding. Include M99 and an RLE wavelet tree only as reference points.
3. **GPU rANS kernels.** Interleaved rANS is well-trodden, so this part is low risk. Then encoder and decoder throughput benchmarks.
4. **Fusion.** Entropy decoding produces the boundary counts, and the inverse BWT follows.

Open decisions:

- Segment size and table coding (from step 2).
- Sample spacing r for the inverse BWT (from step 1).
- Block size: one block for the whole input (sadcs handles up to about 1.547 GB on a 16 GB card at 10.02 bytes per input byte), or fixed-size blocks to cap the decoder's memory.
- Whether to call sadcs as a library or read its `.bwt` files. The file format is divsufsort's: n and the primary index as little-endian int32, then n bytes, all one block.

## Environment

- **GPU:** GeForce RTX 5070 Ti (Blackwell, sm_120, 70 SMs, 48 MiB L2, 16 GiB GDDR7, about 15 GiB free with the display attached). Host: Ryzen 7 5800X, PCIe 4.0 x16, Windows 11.
- **CUDA 13.5** at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5`, not on PATH. Build from a `.bat` that first calls `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`.
- **Nsight Systems:** `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.2\target-windows-x64\nsys.exe`.
- **Corpora:** `C:\Users\Anwender\Code\Compression\Testfiles\{Silesia,Gauntlet,enwik8,enwik9,Calgary}`. Make BWTs with `sadcs build --bwt <in> <out.bwt> --gpu` (exe: `C:\Users\Anwender\Code\repos\sadcs\build-cuda\bin\Release\sadcs.exe`).
- **Measuring:** only when the GPU is idle (other GPU jobs sometimes run in the background). Compare builds with interleaved runs, since runs drift by about 5%. Time the way sadcs and libcubwt do: context created beforehand, pinned host buffers, best of N.
- **References:** sadcs design record `C:\Users\Anwender\Code\repos\sadcs\docs\gpu-design.md` (what worked on this card, e.g. radix passes and tiled merges, and what didn't). libcubwt checkout: `C:\Users\Anwender\Code\repos\libcubwt`. M99 source: `C:\Users\Anwender\Code\Compression\Yuta Mori\m99coder\m99coder.c`.
