# bwarp format, version 0

A bwarp file holds the BWT of one block of up to 2^31 − 1 bytes. All integers are little-endian. `src/codec/format.h` has the constants and coding primitives; `src/codec/cpucodec.cpp` is the reference implementation.

## Layout

| Bytes | Field |
|---|---|
| 4 | magic `BWRP` (0x50525742) |
| 1 | version, 0 |
| 1 | transform, 0 = BWT |
| 1 | logSegment, logSlice ≤ logSegment ≤ 24 |
| 1 | logSlice, 10 to 17 |
| 4 | n, the text length |
| 4 | primary, the sentinel's row among the sorted rotations |
| 8 × (segments + 1) | segment offsets |
| 8 × (slices + 1) | slice offsets |
| | the segment tables, in order |
| | the slice streams, in order |

The BWT is that of the text plus a sentinel smaller than every byte, with the sentinel left out, so it has n bytes.

It is cut into slices of 2^logSlice bytes and segments of 2^logSegment bytes (the last of each may be shorter): `slices = ceil(n / 2^logSlice)`, `segments = ceil(n / 2^logSegment)`. Segment g's table models slices [g × 2^(logSegment − logSlice), …).

Offsets are absolute file positions. Table g occupies [segment offset g, g + 1), stream s [slice offset s, s + 1). The first segment offset is the end of the slice offsets, the last equals the first slice offset, and the last slice offset is the file size.

## Tokens

A slice of B bytes is split into 32 lanes of `L = ceil(B / 32)` contiguous bytes; lane j covers [j × L, min((j + 1) × L, B)) of the slice and may be empty. Each lane is cut into maximal runs, and each run becomes a token (symbol, length). Runs never cross a lane.

**Symbols** are coded as their rank in the segment's alphabet, which lists the bytes that start a token in the segment by token count, descending, ties by byte value.

**Lengths** 1 to 8 have codes 0 to 7. A longer length with `lg = floor(log2(length))` and `hi` the bit below its top bit has code `8 + 2 × (lg − 3) + hi` and `lg − 1` extra bits, the length's low `lg − 1` bits. With slices of at most 2^17 bytes a lane holds at most 4096 bytes, so a run has at most 11 extra bits. The number of length codes is that of the longest possible run, `code(L_max) + 1` with `L_max = ceil(2^logSlice / 32)`.

**Contexts.** With K contexts, `bucket(r) = min(r, K − 1)`. A symbol's context is the bucket of the previous token's rank in the same lane, or K − 1 for a lane's first token. A length's context is the bucket of its own symbol's rank.

## Tables

Bit-packed, least significant bit first, padded to a byte:

- alphabet size − 1, 8 bits;
- the alphabet in rank order, 8 bits per byte;
- K − 1, 4 bits;
- for each symbol context: a used bit, and if set, the alphabet's frequencies;
- for each length context: a used bit, and if set, the frequencies of all length codes.

Each frequency f is written as the Elias gamma code of f + 1 (`floor(log2(v))` zeros, a one, then the low `floor(log2(v))` bits of v). A used context's frequencies sum to 4096. A context whose bit is clear must not occur.

The encoder scales the counts by rounding, gives every nonzero count at least 1 and corrects the sum on the largest frequencies (`normalizeFrequencies`). Decoders only need the frequencies.

## Streams

Each slice is one stream of 32 interleaved rANS coders, one per lane:

- 32 states (u32), lane 0 first: the encoder's final states, the decoder's initial ones;
- 16-bit words, in the order the decoder reads them.

States lie in [2^16, 2^32), probabilities have 12 bits. Decoding a symbol with frequency f and cumulative frequency c from state x:

```
slot = x & 4095                 # selects the symbol with c <= slot < c + f
x = f × (x >> 12) + slot − c
if x < 2^16: x = (x << 16) | next word
```

Every token takes three ops: its symbol, its length code, and its extra bits as one uniform symbol of frequency `2^(12 − e)` and cumulative frequency `extra << (12 − e)` for e extra bits. With e = 0 that op leaves the state unchanged and reads nothing.

The decoder takes the tokens in steps: step t decodes token t of each lane that still has bytes left. Within a step it runs the symbol op for these lanes in ascending order, then the length op, then the extra op. A lane is done once its tokens fill its bytes exactly.

A valid stream ends with every word read and every state back at 2^16, the encoder's initial state.

The encoder runs the same sequence backwards (step, op and lane all descending), writing a word `x & 0xFFFF` and shifting x right by 16 before encoding whenever `x ≥ 2^4 × 2^16 × f`. It then reverses the words.
