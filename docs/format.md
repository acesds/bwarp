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

**Symbols** are coded as their rank in the segment's alphabet, which lists the bytes that start a token in the segment by quantized token count, descending, ties by byte value (see [Tables](#tables)).

**Lengths** 1 to 8 have codes 0 to 7. A longer length with `lg = floor(log2(length))` and `hi` the bit below its top bit has code `8 + 2 × (lg − 3) + hi` and `lg − 1` extra bits, the length's low `lg − 1` bits. With slices of at most 2^17 bytes a lane holds at most 4096 bytes, so a run has at most 11 extra bits. The number of length codes is that of the longest possible run, `code(L_max) + 1` with `L_max = ceil(2^logSlice / 32)`.

**Contexts.** With K contexts, `bucket(r) = min(r, K − 1)`. A symbol's context is the bucket of the previous token's rank in the same lane, or K − 1 for a lane's first token. A length's context is the bucket of its own symbol's rank.

## Tables

A table holds quantized token counts, from which the decoder derives the ranks and the frequencies. Each segment's table is coded on its own, so tables decode in parallel.

### Binary coder

A table is a sequence of binary decisions coded with one rANS state, the same primitives as the streams: the final encoder state (u32), then 16-bit words in reading order. A decision with probability p (of a 0, in units of 2^-12) decodes as

```
bit = (x & 4095) >= p
x = bit ? (4096 − p) × (x >> 12) + (x & 4095) − p : p × (x >> 12) + (x & 4095)
if x < 2^16: x = (x << 16) | next word
```

An adaptive probability starts at 2048 and after each decision becomes `p + ((4096 − p) >> 3)` after a 0, `p − (p >> 3)` after a 1. Raw bits use p = 2048 and come most significant first. A valid table ends with every word read and the state at 2^16.

**Integers** v ≥ 0 are coded as Exp-Golomb codes of u = v + 1: with b = floor(log2(u)), b decisions of 1 and a 0 (decision i uses `length[i]`; after 15 ones the 0 is left out), then u's b bits below its top one, most significant first; the first uses `top[b]`, the others are raw. Signed values s are coded as v = 2s for s ≥ 0, −2s − 1 otherwise.

### Quantized counts

With m mantissa bits, a count c is stored as q = c below 2^m, otherwise, with e = floor(log2(c)),

```
q = ((e − m + 1) << m) + ((c >> (e − m)) & (2^m − 1))
```

Below 2^m, q stands for itself. Otherwise q stands for the middle of its step: with s = (q >> m) − 1, `(((q & (2^m − 1)) | 2^m) << s) + ((1 << s) >> 1)`. A valid q is at most q(2^24).

### Contents

1. K − 1, 4 raw bits.
2. m, 2 raw bits.
3. The byte counts: tokens per byte value, for bytes 0 to 255.
4. The length-code counts: tokens per length code.
5. For each symbol context 0 to K − 1, its row of counts by rank.
6. For each length context 0 to K − 1, its row of counts by length code.

**Count lists** (3, 4) code each entry as a decision "nonzero", whose probability is chosen by whether the previous entry was zero (the first entry counts as following a zero), then for a nonzero entry the signed difference from the previous nonzero q in the list (0 at the start). Lists 3 and 4 each have their own probabilities.

The **ranks** are the bytes with a nonzero count, by q descending, ties by byte value.

**Rows** (5, 6) are predicted from the lists as if symbol and context were independent. With R[r] the dequantized count of rank r, N their sum, L[l] that of length code l, LN their sum, and T[c] the sum of R[r] over the ranks r with `bucket(r) = c`:

- symbol context c: total T[c], weights R, divisor N − R[c] for c < K − 1, otherwise N. For c < K − 1, entry c is left out: a lane's run never follows one of the same byte.
- length context c: total T[c], weights L, divisor LN.

An entry's prediction is `(total × weight + divisor / 2) / divisor` (0 if the divisor is 0), at most 2^24, and p its q.

Each row starts with a decision "used". An unused row is all zero and its context must not occur. In a used row each entry is a decision "nonzero", with the probability for min(p, 31), and for a nonzero entry the signed difference from max(p, 1), as an integer with the probabilities for min(floor(log2(p + 1)), 5). A used row has a nonzero entry. Symbol rows and length rows each have their own probabilities.

All probabilities start afresh for each table.

### Frequencies

A used row's dequantized counts are scaled to frequencies summing to 4096: each is rounded, at least 1 if nonzero, and the difference to 4096 is added to the largest or taken from the largest repeatedly, leaving each at least 1 (`normalizeFrequencies`).

The encoder picks K and m per table; the reference tries K ∈ {1, 2, 4, 8, 16} up to its limit and m ∈ {0, 1}, and keeps the pair with the fewest table bits plus estimated stream bits.

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
