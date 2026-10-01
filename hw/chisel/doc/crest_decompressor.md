# CREST decompressor: lossless weight decompression for every stored format

CREST (Centre-Relative Encoding with Split Tiers) is the format: keys counted from a centre, a plane of
fixed-width codes and two escape tiers. `CrestDecompressor` is the xDMA writer extension that decodes it.

Status: implemented as writer extension 1 of `snax_split_cluster` (`HasCrestDecompressor`), Tier-1
tested on published checkpoints and run on the cluster (section 6). It generalises W4, an INT4-only
design (a fixed 3-bit code c = v + 3 for -3...3, one escape tier, groups of 32 beats), to the formats
real checkpoints ship in.
The format is defined by the reference codec `crest_decompressor/crest_codec.py`; where this text and the
code disagree, the code wins.

The extension is lossless: it reproduces the stored weight bytes exactly. Quantisation is not its
business; it takes the codes a checkpoint already has.

## 1. Why the W4 decompressor is not enough

Measured on the stored payload of published checkpoints (3 sampled layers each, from the
checkpoint cache that `CREST_WEIGHT_CACHE` names, section 5):

| Format | Checkpoint | W4 format | CREST | Ceiling (entropy) |
| --- | --- | --- | --- | --- |
| BF16 | Qwen2.5-7B-Instruct | - | 1.414x | 1.497x |
| INT8 per channel (W8A8) | RedHatAI Qwen2.5-7B w8a8 | - | 1.165x | 1.261x |
| FP8 E4M3, 128x128 blocks | DeepSeek-V3 | - | 1.124x | 1.208x |
| FP8 E4M3, 128x128 blocks | Qwen3-8B-FP8 | - | 1.099x | 1.173x |
| INT4 GPTQ sym g128 | Qwen2.5-7B-GPTQ-Int4 | 0.970x | 1.096x | 1.178x |
| INT4 with per-group zero points | AWQ Qwen2.5-7B, RedHat Qwen3-8B w4a16 | 0.970x | 0.99-1.00x | 1.09-1.11x |
| MXFP4 / NVFP4 | gpt-oss-20b / nvidia Qwen3-8B-FP4 | 0.970x | 0.985x | 1.02-1.04x |

The W4 decompressor assumes two's-complement nibbles centred on 0. No published INT4 format is
stored that way: GPTQ stores unsigned nibbles around a zero point of 8, AWQ and compressed-tensors
use a zero point per group, MXFP4 and NVFP4 are sign-magnitude E2M1. On all of them every W4 group
goes raw and pays its header word (0.97x). The 8- and 16-bit formats it cannot take at all.

Not worth compressing: MXFP4, NVFP4 and per-group zero-point INT4. Their codes use the code space
almost fully; send them with the extension off. Their scale streams do compress (MXFP4 E8M0 0.25
-> 0.09 bit per weight, NVFP4 E4M3 0.50 -> 0.30), which is a separate transfer.

## 2. The format

A transfer is a stream of 512-bit words: one stream header word, then one record per group of
`G` = 64 output beats (the last group may be shorter). CSR 0 is N, the output beats.

**Modes** (stream header [1:0]). The output is the weights as stored, in beats of L lanes:

| Mode | Lanes | Lane | Coded symbol | Raw bits per lane |
| --- | --- | --- | --- | --- |
| 0 nib | 128 | 4 bit | the nibble | - |
| 1 byte | 64 | 8 bit | the byte | - |
| 2 bf16 | 32 | 16 bit | the exponent byte (bits 14:7) | 8: sign and mantissa |

**Keys, not a table** (stream header [2] keymap, [10:3] centre). Every symbol gets a key; small keys
are the frequent symbols:

- keymap 0, two's complement: v = (s - centre) mod 2^w as a signed value, key = zigzag(v) = 0, -1,
  +1, -2, ... GPTQ INT4 uses centre 8 (its zero point), INT8 centre 0, BF16 exponents ~120.
- keymap 1, sign-magnitude: key = 2 * zigzag(|m| - centre) + sign. FP8 E4M3 uses centre ~100 (the
  most common magnitude code).

The inverse is an adder per lane, so the decoder needs no per-tensor table. On the checkpoints
above a free per-tensor frequency table would gain nothing (INT4, INT8, BF16) or 0.7% (FP8).

**Plane code** (stream header [13:11] p, [16:14] e1). With K0 = 2^p - 2, a lane's p-bit plane
code is:

| Plane code | Meaning |
| --- | --- |
| 0 .. K0 - 1 | the key itself |
| K0 | tier-1 escape: an e1-bit entry, key = K0 + entry |
| K0 + 1 | tier-2 escape: the raw symbol, w bits |

Both escape tiers are flagged by the plane alone, so the decoder finds every entry with two prefix
popcounts and never parses an entry to find the next one. Supported p: 2-3 in mode nib, 2-7 in
byte and bf16; e1 1-3 in nib, 1-7 otherwise.

**Groups.** A coded group is E escape words, then the plane:

- escape words: a 32-bit header ([14:0] E, [15] 0, [31:16] n1 = tier-1 entries), from bit 32 the
  n1 tier-1 entries, then the tier-2 entries, both in lane order, zero padded. One region for both
  tiers, rounded to whole words once.
- plane: ceil(gb * L * lb / 512) words, lb = p (+ 8 raw bits after the code in bf16). Beat j, lane
  i at plane bit j * L * lb + i * lb.

A raw group is one header word with only bit 15 set, then the gb beats unchanged. The encoder makes
a group raw if it would need more than EMAX = 12 escape words or would not be shorter.

**Parameters.** The encoder searches keymap, centre, p and e1 per tensor (exhaustively; it is a
histogram calculation). What real checkpoints choose: GPTQ INT4 (int, 9, 3, 2), FP8 (sm, ~100, 6,
5), INT8 (int, 0, 6, 5), BF16 (int, 119, 3, 1).

## 3. Micro-architecture

`CrestDecompressor.scala`:

- **Router**: stream header, then HDR -> ESC -> PLANE (or RAW) per group. Two escape buffers of
  `escWords` = 12 words, so the next group's escape words load while this one decodes.
- **Plane window**: words A, B and a plane bit pointer that advances by L * lb per beat; crossing
  512 retires A. L * lb is a multiple of 32 in every mode, so the slice is a 16:1 funnel over the
  2-word window at 32-bit steps. `planeDepth` = 8 words of plane queue let the router read a
  group's up to 12 escape words while the previous group's last beats decode; at 4, FP8 and INT8
  lose ~4 cycles per group boundary.
- **Stage 1**: each lane's field for the configured (mode, p) (a mux over the supported field
  widths), its class, two 128-wide prefix popcounts, and two escape windows: the escape buffer
  shifted by the tier-1 and the tier-2 pointer. Both pointer updates depend only on counts from the
  plane.
- **Stage 2**: escaped lanes take their entries, `k1Lanes` = 32 tier-1 and `k2Lanes` = 16 tier-2
  per pass; keys go back to symbols (zigzag inverse plus the centre); the lanes are assembled at the
  mode's width. A beat with more entries takes another pass. On the checkpoints that is 1.4% of the
  GPTQ beats and none of the FP8, INT8 or BF16 ones.

Against W4: escape buffers 2 x 3 -> 2 x 12 words, plane queue 4 -> 8 words, one
more prefix popcount, escape window and expand, the 4-phase plane mux -> a 16:1 funnel, and an
adder per lane for the key. Area has not been synthesised.

## 4. Interface and software

Ports: `ext_data_i` the stream, `ext_data_o` the N beats, CSR 0 = N,
`ext_start_i` latches N and resets, `ext_busy_o` high until the N-th beat is taken, bypass when
disabled. `snax_xdma_crest_decompress(src, in_words, dst, out_beats)` in the snax xDMA library
configures one 1-D task (reader `in_words` words, writer N beats) and arms the extension.

## 5. Tests

`CrestDecompressorTester` takes its vectors from `crest_codec.py`: nine synthetic edge cases (dense
tier-1 and tier-2 beats, an 11-word escape region, sign-magnitude in both widths, entries crossing
word edges, raw groups between coded ones, a partial group) and, when the environment variable
`CREST_WEIGHT_CACHE` names the checkpoint cache, 1024-4096-beat chunks of the published checkpoints;
without it the suite runs the synthetic cases only. All of them back to back on one DUT under three mixes of
random input gaps and output back-pressure, then without stalls (throughput), plus bypass.

## 6. Results

Tier-1, input always valid, output always ready:

| Case | Beats | Words in | Cycles | Beats / cycle |
| --- | --- | --- | --- | --- |
| BF16 | 4096 | 2881 | 4104 | 0.998 |
| FP8, DeepSeek-V3 | 4096 | 3580 | 4111 | 0.996 |
| FP8, Qwen3 | 4096 | 3648 | 4112 | 0.996 |
| INT8 | 4096 | 3475 | 4125 | 0.993 |
| INT4 GPTQ | 4096 | 3701 | 4196 | 0.976 |
| MXFP4 (all groups raw) | 1024 | 1041 | 1045 | 0.980 |

On `snax_split_cluster` (QuestaSim, app `snax-xdma-crest-decompress`), xDMA task cycles, L1 -> L1:

| Case | Mode | Ratio | Decompress | Plain copy, N beats | Plain copy, W words |
| --- | --- | --- | --- | --- | --- |
| BF16 | bf16 | 1.421x | 2064 | 2057 | 1450 |
| FP8 DeepSeek-V3 | byte | 1.143x | 2070 | 2057 | 1800 |
| INT8 | byte | 1.187x | 2073 | 2057 | 1734 |
| INT4 GPTQ | nib | 1.100x | 2134 | 2057 | 1870 |
| MXFP4 (raw) | nib | 0.983x | 1052 | 1033 | 1050 |

## 7. Open points

- The remote-stream test (a transfer from another cluster) is not done.
- Area. The two 6144-bit escape windows are the largest new logic; a word-select plus fine shift
  would bound them.
- Per-group zero-point INT4 (AWQ) would compress to ~1.09x with the zero points as a side input,
  which this extension deliberately does not take (one stream in, one out).
