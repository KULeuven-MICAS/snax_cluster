// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// RMSNorm on the four-engine cluster, and the two things that make it cheap.
//
// WHAT RMSNORM IS. The normalisation an LLM layer puts in front of each of its two
// blocks. For a tile of T token rows of D features, every row is divided by its own
// root-mean-square:
//
//     out[t, f] = x[t, f] / sqrt( (1/D) * SUM_f x[t, f]^2 )
//
// No mean subtraction, no learnable gain (a layer that wants the usual per-channel
// weight applies it as a separate elementwise multiply). Reduce each row to one number,
// invert its square root, scale the row by it.
//
// Yet it is one of the most expensive SIMD operations a layer runs, and the arithmetic
// is not why -- that is one multiply per element, the same as a residual add, which
// measures 4x cheaper over the same tile. The cost is ALL IN THE ONE SCALAR PER ROW:
// where it has to be computed, and how it gets back to the data. Everything below is
// about that scalar.
//
// ======================================================================================
// THE DATAFLOW -- WHAT THE BLOCK ACTUALLY SEES
// ======================================================================================
//
// The SIMD block is a stream machine. One reader sweeps an affine iteration space, the
// armed operators transform the stream in a FIXED order, one writer drains it:
//
//     read -> EW0 -> Map -> Reduce -> EW1 -> Fp16ToInt8 -> write
//
// The unit of that stream is a BEAT: 512 bits = 64 B = 32 FP16 LANES. And the one fact
// that decides everything about this kernel is how the reduce is built --
//
//     StreamReduce carries ONE FP32 ACCUMULATOR PER LANE, acc[0..31], which persist
//     from beat to beat. Every beat, lane k folds into acc[k]. Nothing else moves.
//
// So a reduction ALONG BEATS is free: it is the accumulators doing what they already do,
// one beat per cycle, and the answer is sitting in them when the stream ends. A reduction
// ACROSS THE LANES of a beat is a completely different machine: a log-depth fold through
// treeBuf, serialised over treeLanes ALUs, holding the reader's input port low while it
// runs -- once per row -- and then the single result has to be SPLATTED back across all
// 32 lanes to be usable as a beat again.
//
// Worse, that splatted beat is then hard to CONSUME. A 2-operand elementwise reads both
// operands from ONE 3-D affine address stream {operand, beat, row}:
//
//     addr = base + o*operand_delta + b*64 + r*row_stride
//
// Operand 0 (the data) wants exactly that. Operand 1 wants the SAME scalar for every beat
// b of row r -- i.e. no b term. The three loops share one stride set, so b cannot be
// zeroed for one operand only. The scalar therefore has to be REPLICATED into a full
// [T, D] plane before the elementwise can read it: an extra pass to write it, and an
// elementwise that reads twice the beats it writes.
//
// Which of the two machines RMSNorm gets is decided entirely by the LAYOUT of its input.
// This is the same argument the FlashAttention apps make for softmax's rowmax; see
// snax-flashattn-decode.c, which stores its score tile transposed for exactly this reason.
//
// ======================================================================================
// A WORKED EXAMPLE -- 4 tokens, 8 features, 4 lanes to a beat
// ======================================================================================
//
// Real shapes are T=32, D=128, 32 lanes; shrink everything by 8 and it fits on a page.
// The tile, chosen so every number below comes out exact:
//
//            f0  f1  f2  f3  f4  f5  f6  f7      SUM x^2   mean=/8   1/sqrt
//     t0  [   4   2   2   2   1   1   1   1 ]        32         4      0.50
//     t1  [   8   4   4   4   2   2   2   2 ]       128        16      0.25
//     t2  [   4   2   2   2   1   1   1   1 ]        32         4      0.50
//     t3  [   8   4   4   4   2   2   2   2 ]       128        16      0.25
//
// ---- ROW-MAJOR x[T, D] : each TOKEN ROW contiguous -- what the layer hands us ---------
//
// A beat is 4 consecutive features of ONE token, so t0's row is two beats:
//
//     beat 0  [  4   2   2   2 ]        beat 1  [  1   1   1   1 ]
//
//   accumulate. SUMSQ squares each lane and adds it into THAT LANE'S OWN accumulator;
//   lane k only ever touches acc[k], and acc[k] persists from one beat to the next.
//   Nothing moves sideways here:
//
//                        lane 0    lane 1    lane 2    lane 3
//     beat 0 = f0..f3  [    4         2         2         2   ]   raw
//              squared [   16         4         4         4   ]
//     acc              [   16         4         4         4   ]   acc[k] += sq[k]
//
//     beat 1 = f4..f7  [    1         1         1         1   ]   raw
//              squared [    1         1         1         1   ]
//     acc              [ 16+1=17    4+1=5     4+1=5     4+1=5 ]   acc[k] += sq[k]
//
//   Now read it down the COLUMNS, which is where the problem shows up -- each lane
//   collected EVERY FOURTH FEATURE of the row, never the whole row:
//
//     acc[0] = f0^2 + f4^2 = 4^2 + 1^2 = 17
//     acc[1] = f1^2 + f5^2 = 2^2 + 1^2 =  5
//     acc[2] = f2^2 + f6^2 = 2^2 + 1^2 =  5
//     acc[3] = f3^2 + f7^2 = 2^2 + 1^2 =  5
//
//   The row's sum is 16+4+4+4+1+1+1+1 = 32, and 17+5+5+5 = 32 -- correct, but spread
//   over four separate registers. Collapsing them means ADDING ACROSS LANES, and no
//   path in the accumulate datapath does that:
//
//   FOLD ACROSS LANES -- the only step in the whole reduce that moves data sideways. It
//   is a balanced binary tree: each round pairs up the survivors and adds, halving the
//   count, and round r+1 cannot start until round r has finished. So the number of
//   ROUNDS is log2(lanes) -- here log2(4) = 2:
//
//     round 1   [ 17   5   5   5 ]  ->  [ 17+5 , 5+5 ]  =  [ 22  10 ]
//     round 2   [ 22  10 ]          ->  [ 22+10 ]       =  [ 32 ]
//
//   At the real width that is log2(32) = 5 rounds and 31 adds. Cycles are not the depth
//   alone: the adds of a round are spread over `treeLanes` fold ALUs (2 in this cfg),
//   each chunk held 2 cycles, so
//
//     round      1    2   3   4   5
//     pairs     16    8   4   2   1   = 31 adds
//     chunks     8    4   2   1   1   = 16 chunks / 2 ALUs, x2 cc = ~32 cc PER ROW
//
//   and the reader's input port is held low for all of it. That is the ~35 cc/row that
//   turns a 256 cc reduce into a 1,387 cc one. (`treeLanes` is a cfg knob: 16 would take
//   the fold to ~10 cc/row. It is set to 2 today.)
//
//   SPLAT -- the fold produced one NUMBER, but the datapath only moves whole 64 B BEATS;
//   there is no narrower write. So the result is copied into every lane:
//
//     32   ->   [ 32  32  32  32 ]
//
//   which is not waste in itself -- the elementwise further down has to multiply all 32
//   lanes of a data beat by the same scalar, so 32 copies is exactly the shape it wants.
//   The waste is one step later: that beat has to be REPEATED for every beat of the row.
//
//   the scalar now has to leave: mean = 32/8 = 4, and 1/sqrt(4) = 0.5 must be computed
//   somewhere, then written back into a beat, then REPLICATED into all 8 features of
//   t0's row so the elementwise can pair it up:
//
//     x     beat 0  [  4   2   2   2 ]   beat 1  [  1   1   1   1 ]
//     plane beat 0  [ 0.5 0.5 0.5 0.5]   beat 1  [ 0.5 0.5 0.5 0.5]   <- 8 copies of ONE
//     out   beat 0  [  2   1   1   1 ]   beat 1  [ 0.5 0.5 0.5 0.5]      number
//
//   Four times over the tile: four folds, four scalars, four rows of replication -- and
//   the elementwise reads 16 beats to write 8.
//
// ---- TRANSPOSED x^T[D, T] : each FEATURE contiguous, ONE TOKEN PER LANE ---------------
//
// A beat is now one feature across ALL FOUR tokens, so the tile is 8 beats:
//
//     beat f0 [ 4  8  4  8 ]   beat f1 [ 2  4  2  4 ]   ...   beat f7 [ 1  2  1  2 ]
//       ^  ^  ^  ^
//       t0 t1 t2 t3   -- and lane t is token t in EVERY beat, forever
//
//   accumulate -- exactly the same hardware, exactly the same rule, lane k into acc[k]:
//     after f0        acc = [ 16  64  16  64 ]
//     after f1..f3    acc = [ 28 112  28 112 ]
//     after f4..f7    acc = [ 32 128  32 128 ]   <- THE ANSWER. All four tokens. Done.
//
//   Read down the columns again and the difference is the whole story. Lane t now sees
//   token t in EVERY beat, so it collected that token's WHOLE row:
//
//     acc[0] = t0's f0^2 + f1^2 + ... + f7^2 =  32     one complete row sum, one register
//     acc[1] = t1's f0^2 + f1^2 + ... + f7^2 = 128
//     acc[2] =  32      acc[3] = 128
//
//   Same accumulators, same adds. The layout alone decided whether a row's terms landed
//   in ONE lane or got scattered across four.
//
//   No fold. No splat. SIMD_RED_LANEWISE just says "emit the accumulators", and ONE beat
//   comes out holding every token's sum of squares:  [ 32 128  32 128 ]
//
//   One StreamMap pass over that SINGLE BEAT, a = 1/8 and func = RSQRT:
//
//     [ 32 128  32 128 ]  ->  [ 1/sqrt(4)  1/sqrt(16)  1/sqrt(4)  1/sqrt(16) ]
//                          =  [   0.50        0.25        0.50       0.25    ]
//
//   And SIMD_EW_STICKY_B latches that one beat as operand B, then multiplies every data
//   beat against it -- each lane by its own token's scalar, because lane t never stops
//   being token t:
//
//     beat f0 [ 4  8  4  8 ] * [0.50 0.25 0.50 0.25] = [ 2  2  2  2 ]
//     beat f1 [ 2  4  2  4 ] * [0.50 0.25 0.50 0.25] = [ 1  1  1  1 ]
//     ...
//
//   Nothing is replicated, nothing is folded, and the read stream is D+1 beats, not 2*D.
//   (Transposing the row-major answer above gives the same values: t0's f0 is 2, t1's
//   f0 is 2, and out beat f0 is [2 2 2 2]. Same arithmetic, different order in memory.)
//
// ======================================================================================
// THE THREE PATHS THIS APP RUNS
// ======================================================================================
//
//   LEGACY -- four passes, and three of them exist only to move one number.
//     1  reduce(SUMSQ)                 x  -> bt   per-row scalar, splatted
//     2  SCALAR EPILOGUE on the core   no beats. Per row: mean by exponent subtract, then
//                                      sqrt_f16 and recip_f16 -- SIX SERIAL `divu`, because
//                                      the cluster cores are rv32ima with no FPU -- then
//                                      SIXTEEN volatile word stores to splat the result
//                                      across the row's whole 64 B beat.
//     3  bcast_map(a=1.0)              bt -> bc   the [T,D] replication above
//     4  ew2(MUL)                      (x,bc) -> y
//
//     Measured at [32,128]: 7,719 cc, of which the scalar epilogue alone is 4,584 -- 59%
//     of the kernel. Timing that epilogue's arithmetic and its stores apart (done
//     separately) puts 3,914 cc on the six divides and 778 on the 512 stores, so it is
//     the DIVIDES that cost, at ~122 cc a row.
//
//   RSQRT -- the current solution. No layout change, no extra pass, no new pass cost.
//     1  reduce(SUMSQ)                 x  -> bt        unchanged
//     2  bcast_map(a=1/D, RSQRT)       bt -> bc        passes 2 and 3 above, collapsed
//     3  ew2(MUL)                      (x,bc) -> y     unchanged
//
//     Pass 3 of the legacy path was ALREADY a StreamMap -- the replication, carrying an
//     identity multiply, a = 1.0. StreamMap computes out = func(a*x + b), so giving that
//     same pass a = 1/D and func = RSQRT makes it emit 1/sqrt(SUM/D) instead of SUM, and
//     the scalar never leaves the datapath. It measures 479 cc against the identity
//     multiply's 477 -- the SAME 296 cc of datapath -- and deletes 5,061 cc. It is free
//     because the pass it rides on had to happen anyway.
//
//   TRANSPOSED -- the worked example above, using both mechanisms.
//     0  xDMA 8x8 transpose             x   -> x^T    ON HART 2, not the SIMD block
//     1  reduce(SUMSQ|LANEWISE)         x^T -> ssq    one beat, all T tokens, no fold
//     2  map(a=1/D, RSQRT), ONE BEAT    ssq -> seed   21 cc of datapath
//     3  ew(MUL|STICKY_B)          seed+x^T -> y^T    no replication, D+1 beats read
//
//     Pass 0 is REAL here, not assumed: the app drives the xDMA's transposer on hart 2
//     and checks its output against the reference bit for bit, so the transposed numbers
//     below are paid for, not free.
//
// MEASURED, warm, Verilator, snax_split_cluster, T=32 D=128 (+-0.5% run to run):
//
//   hart 1, the SIMD block          hart 2, the xDMA (layout conversions)
//     legacy         7,717 cc         x   -> x^T   244 cc  (181 datapath)
//     rsqrt          3,135 cc         y^T -> y     385 cc  (327 datapath)
//     transposed     1,073 cc         y   -> A     299 cc  (240 datapath, NO extension)
//
//     row-major path:   3,135 on hart 1  +  299 on hart 2
//     transposed path:  1,073 on hart 1  +  928 on hart 2
//
// Two engines, so two honest bounds: SIMD-only the transposed path is 65% cheaper, fully
// serialised it is still 42% cheaper (2,001 vs 3,434). It always wins on hart 1, which is
// the busier engine, and the direction of that trade -- work off the SIMD and onto the
// xDMA -- is the one a layer wants.

//
// The row-major reduce is 1,387 cc of datapath for 128 beats, of which only ~256 is the
// arithmetic -- the other ~1,130 is 32 rows x ~35 cc of fold. Transposed it is 270 cc for
// the same 128 beats. That 5.1x is the fold, and nothing else.
//
// Accuracy moves the right way too. Both rsqrt paths land within 2 FP16 ULP of the true
// 1/sqrt on every output element; the core's integer sqrt+reciprocal is already 2 ULP off
// on inv_rms ALONE, before that error is multiplied through the row. So the ROM that
// removes the cycles is also the more accurate of the two -- there is no trade.
//
// ======================================================================================
// SO SHOULD THE INPUT BE TRANSPOSED? -- READ THIS BEFORE BUILDING ANYTHING
// ======================================================================================
//
// The transposed path needs x^T, and it emits y^T. Both conversions are xDMA work on
// hart 2, and this app runs and times BOTH of them rather than assuming them.
//
//   WHERE THE LAYER PUTS THIS KERNEL:
//       GEMM(D-layout) -> reshape -> RMSNorm -> reshape -> A-layout -> quantise -> GEMM
//
//   y^T -> A IS NOT ONE PASS, and that is a hardware fact, not an oversight. A conversion
//   into A-layout needs an 8-byte run contiguous on BOTH sides -- four consecutive
//   FEATURES of one token, which is exactly 8 B at fp16 and why the quantise has to come
//   AFTER the reshape. In y those four are adjacent. In y^T features are rows*2 bytes
//   apart and what is contiguous is four consecutive TOKENS, so no pair of strides makes
//   a common run: packed and A both run along rows, y^T runs down columns. HeMAiA's own
//   nest.py refuses exactly this, by name, and routes it to the transposer instead.
//
//   So the transposed path pays for TWO conversions, both measured below:
//       x -> x^T      before the kernel
//       y^T -> y      after it, so the existing row_major_to_a can run unchanged
//
//   row_major_to_a itself needs NO transposer -- it is a pure stride nest, and this app
//   programs and checks it (path 2 of HeMAiA's kernel: meshRow 16 is a multiple of 8, and
//   tileSize*2 = 8 B is exactly the atom). What the transposed path buys is SIMD time on
//   hart 1; what it costs is xDMA time on hart 2. The table above prices both.
//
//   THE INPUT-SIDE TRANSPOSE NEED NOT BE PAID AT ALL. Transposing both sides of the
//   PRODUCER's matmul rewrites it with its axes exchanged: (A.B)^T = B^T.A^T is the SAME
//   GEMM with its two operands swapped, which is M and N exchanged in the config and
//   nothing at run time. Not speculative -- the FlashAttention kernels in this tree
//   already do it, which is why their softmax gets a LANEWISE rowmax for free. Run the
//   producer that way and its D-layout output IS the [D, T] transposed result, so the
//   D -> packed reshape that already runs emits x^T directly. Probed against HeMAiA's own
//   convert_args on a (16, 4, 16) array at fp16:
//
//       D[32,128]      -> packed    SUPPORTED    the layer today
//       D[128,32]      -> packed    SUPPORTED    <- x^T from a swapped producer, free
//       packed[32,128] -> A         SUPPORTED    the layer today
//       packed[128,32] -> B         REFUSED      transpose -- and this is what a swapped
//                                                CONSUMER would want from y^T
//
//   So the input side is free and the output side is not, and the last line is why: a
//   consumer run in swapped form needs y^T as its B operand, and packed -> B is a
//   transpose no stride nest expresses. Either way y^T has to become y again. Net, the
//   transposed path costs ONE conversion more than the row-major one -- the 385 cc
//   y^T -> y -- to save 2,062 cc of SIMD time.
//
//   Check before banking it: swapping the producer propagates a layout requirement onto
//   ITS operands. A weight can be staged swapped offline for nothing; an activation
//   arriving from the previous layer is the thing to trace.
//
// ======================================================================================
//
// The app runs all three paths over the same tensor and checks each: legacy against the
// golden built from the core's own integer rsqrt (tight, it must reproduce it bit for
// bit), and the two hardware-rsqrt paths against the exact reference.

#include "data.h"
#include "snax-core-roles.h"
#include "snax-simd-lib.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(SIMD_EXT_STREAMREDUCE) || !defined(SIMD_EXT_STREAMMAP) || \
    !defined(SIMD_EXT_STREAMELEMENTWISE_1)
#error \
    "Regenerate the SIMD CSR map with StreamReduce, StreamMap and StreamElementwise."
#endif

#define BEAT SIMD_BEAT_BYTES
#define FP16_PER_BEAT 32

// ============================================================ the core's FPU-less rsqrt
//
// Only the LEGACY path uses these. They are here so the app can measure -- and check
// itself against -- the kernel the rsqrt ROM replaced, rather than quoting a number.
// The cluster cores are rv32ima: a float op traps, so this is integer bit manipulation on
// the FP16 pattern, and every Newton step is an rv32M `divu` on a serial divider.

// Integer FP16 reciprocal of a positive normal. One divu.
static inline uint16_t recip_f16(uint16_t s) {
    uint32_t E = (s >> 10) & 0x1Fu;
    uint32_t M = 1024u + (s & 0x3FFu);
    uint32_t q = ((1u << 21) + (M >> 1)) / M;
    if (q >= 2048u) return (uint16_t)((30u - E) << 10);
    return (uint16_t)(((29u - E) << 10) | ((q - 1024u) & 0x3FFu));
}

// Integer FP16 square root of a positive normal. FIVE divu -- this is the cost.
static inline uint16_t sqrt_f16(uint16_t v) {
    uint32_t E = (v >> 10) & 0x1Fu;
    if (E == 0u) return 0u;
    uint32_t M = 1024u + (v & 0x3FFu);
    int32_t e = (int32_t)E - 15;
    uint32_t sig = (e & 1) ? (2u * M) : M;
    int32_t oe = (e & 1) ? ((e - 1) >> 1) : (e >> 1);
    uint32_t n = sig << 10;
    uint32_t x = 1448u;
    x = (x + n / x) >> 1;
    x = (x + n / x) >> 1;
    x = (x + n / x) >> 1;
    x = (x + n / x) >> 1;
    x = (x + n / x) >> 1;
    return (uint16_t)(((uint32_t)(oe + 15) << 10) | ((x - 1024u) & 0x3FFu));
}

static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

// ============================================================ pass helpers

static inline uint32_t run_shapes(const snax_simd_shape_t *in,
                                  const snax_simd_shape_t *out) {
    snax_simd_program_fast(in, out);
    snax_simd_fire();
    snax_simd_wait_all();
    return snax_simd_last_task_cycle();
}

// Per-row reduction. 2-D reader {beat, row}, 1-D writer. `mode` carries SUMSQ plus any of
// TAP / FP32OUT / LANEWISE; with LANEWISE pass rows=1 and beats=D, so the whole tile is
// one "row" and the per-lane accumulators ARE the per-token sums.
static uint32_t pass_reduce(void *src, void *dst, uint32_t rows, uint32_t beats,
                            uint32_t mode, uint32_t dst_beats) {
    snax_simd_shape_t in, out;
    snax_simd_shape_rows(&in, src, rows, beats, beats * BEAT);
    snax_simd_shape_flat(&out, dst, dst_beats);
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, beats, mode);
    return run_shapes(&in, &out);
}

// out = func(a*x + b) over `beats` contiguous beats.
static uint32_t pass_map(void *src, void *dst, uint32_t beats, uint32_t a_bits,
                         uint32_t b_bits, uint32_t func) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, src, beats);
    snax_simd_shape_flat(&out, dst, beats);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, a_bits, b_bits, func);
    return run_shapes(&in, &out);
}

// Broadcast one beat per row into `beats` beats per row, with a StreamMap applied on the
// fly. THIS IS THE PASS THAT BECAME FREE: the reader's inner stride of 0 re-presents the
// row's scalar beat, and whatever `func` is armed runs on it at no extra cost -- a = 1.0 /
// LINEAR is the legacy identity, a = 1/D / RSQRT is the normalisation.
static uint32_t pass_bcast_map(void *src_beats, void *dst, uint32_t rows,
                               uint32_t beats, uint32_t dst_row_stride,
                               uint32_t a_bits, uint32_t func) {
    snax_simd_shape_t in, out;
    snax_simd_shape_2d(&in, src_beats, beats, 0u, rows, BEAT);
    snax_simd_shape_rows(&out, dst, rows, beats, dst_row_stride);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, a_bits, 0u, func);
    return run_shapes(&in, &out);
}

// The reader AGU strides FORWARD only, so the base must be the LOWER operand; MUL and ADD
// commute, so the swap costs nothing.
static inline void *ew2_base(void *a, void *b, uint32_t *operand_stride) {
    uint32_t al = (uint32_t)(uintptr_t)a, bl = (uint32_t)(uintptr_t)b;
    if (bl >= al) {
        *operand_stride = bl - al;
        return a;
    }
    *operand_stride = al - bl;
    return b;
}

// Two-operand elementwise. Reader is 3-D {operand, beat, row} -- the shape whose shared
// strides are exactly why a per-row scalar has to be replicated first.
static uint32_t pass_ew2(void *src_a, void *src_b, void *dst, uint32_t rows,
                         uint32_t beats, uint32_t src_row_stride, uint32_t op) {
    uint32_t operand_stride;
    void *base = ew2_base(src_a, src_b, &operand_stride);
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, base, 1);  // seeds the neutral dims, then overwritten
    in.dim = 3;
    in.bound[0] = 2u;
    in.stride[0] = operand_stride;
    in.bound[1] = beats;
    in.stride[1] = BEAT;
    in.bound[2] = rows;
    in.stride[2] = src_row_stride;
    snax_simd_shape_flat(&out, dst, rows * beats);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 2u, op);
    return run_shapes(&in, &out);
}

// One-operand elementwise with STICKY-B: beat 0 seeds operand B and emits NOTHING, beats
// 1..N emit op(B, beat). N+1 beats in, N out -- so the seed must sit immediately below the
// data and the writer covers exactly the data, not one beat early.
static uint32_t pass_ew_sticky(void *seed_then_data, void *dst, uint32_t data_beats,
                               uint32_t op) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, seed_then_data, data_beats + 1u);
    snax_simd_shape_flat(&out, dst, data_beats);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1u,
                   op | SIMD_EW_STICKY_B);
    return run_shapes(&in, &out);
}

// ==================================================== xDMA, with its CSRs written cheaply
//
// A csrw_ss whose address is a COMPILE-TIME CONSTANT folds to a single `csrw imm`. One
// whose address is computed does not: it lowers to a jump-table load from L2 plus an
// indirect jump, about 6 cycles instead of 1. snax_xdma_memcpy_nd() writes its AGU
// descriptor from loops indexed by a runtime variable, so every write pays that -- and on
// a transfer this small the descriptor, not the data, is most of the task. The SIMD side
// already carries the same fix (snax_simd_program_fast).
//
// So the descriptor is written here instead, unrolled, every address constant. Two library
// habits are dropped with it, both safe here and both expensive: the read-modify-write of
// the extension mask (a CSR READ is 5 cycles, and arming exactly one extension is just
// `1 << id`), and the zeroing of the 15 unused multicast destination pointers (30 writes;
// they are zero from reset and nothing here multicasts).

#define XD_W(addr, val) snax_write_xdma_cfg_reg((addr), (val))
#define XDMA_LANE_BYTES (XDMA_WIDTH / XDMA_SPATIAL_CHAN)

// Bound/stride vectors are the FULL five deep, neutral (bound 1, stride 0) where a task
// does not use a dimension. No runtime dimension count -- that is what put an index in the
// loop to begin with.
__attribute__((always_inline)) static inline void xdma_program_fast(
    void *src, void *dst, uint32_t sp_src, uint32_t sp_dst, const uint32_t *bs,
    const uint32_t *ss, const uint32_t *bd, const uint32_t *sd) {
    uint32_t hi = (uint32_t)snrt_cluster_base_addrh();
    XD_W(XDMA_SRC_ADDR_PTR_LSB, (uint32_t)(uintptr_t)src);
    XD_W(XDMA_SRC_ADDR_PTR_MSB, hi);
    XD_W(XDMA_DST_ADDR_PTR_LSB, (uint32_t)(uintptr_t)dst);
    XD_W(XDMA_DST_ADDR_PTR_MSB, hi);
    XD_W(XDMA_SRC_SPATIAL_STRIDE_PTR, sp_src);
    XD_W(XDMA_DST_SPATIAL_STRIDE_PTR, sp_dst);
    XD_W(XDMA_SRC_TEMP_BOUND_PTR + 0, bs[0]);
    XD_W(XDMA_SRC_TEMP_BOUND_PTR + 1, bs[1]);
    XD_W(XDMA_SRC_TEMP_BOUND_PTR + 2, bs[2]);
    XD_W(XDMA_SRC_TEMP_BOUND_PTR + 3, bs[3]);
    XD_W(XDMA_SRC_TEMP_BOUND_PTR + 4, bs[4]);
    XD_W(XDMA_SRC_TEMP_STRIDE_PTR + 0, ss[0]);
    XD_W(XDMA_SRC_TEMP_STRIDE_PTR + 1, ss[1]);
    XD_W(XDMA_SRC_TEMP_STRIDE_PTR + 2, ss[2]);
    XD_W(XDMA_SRC_TEMP_STRIDE_PTR + 3, ss[3]);
    XD_W(XDMA_SRC_TEMP_STRIDE_PTR + 4, ss[4]);
    XD_W(XDMA_DST_TEMP_BOUND_PTR + 0, bd[0]);
    XD_W(XDMA_DST_TEMP_BOUND_PTR + 1, bd[1]);
    XD_W(XDMA_DST_TEMP_BOUND_PTR + 2, bd[2]);
    XD_W(XDMA_DST_TEMP_BOUND_PTR + 3, bd[3]);
    XD_W(XDMA_DST_TEMP_BOUND_PTR + 4, bd[4]);
    XD_W(XDMA_DST_TEMP_STRIDE_PTR + 0, sd[0]);
    XD_W(XDMA_DST_TEMP_STRIDE_PTR + 1, sd[1]);
    XD_W(XDMA_DST_TEMP_STRIDE_PTR + 2, sd[2]);
    XD_W(XDMA_DST_TEMP_STRIDE_PTR + 3, sd[3]);
    XD_W(XDMA_DST_TEMP_STRIDE_PTR + 4, sd[4]);
    XD_W(XDMA_SRC_ENABLED_CHAN_PTR, 0xFFFFFFFFu);
    XD_W(XDMA_DST_ENABLED_CHAN_PTR, 0xFFFFFFFFu);
    XD_W(XDMA_DST_ENABLED_BYTE_PTR, 0xFFFFFFFFu);
}

// Arm exactly one reader extension, or none, in a single write each. The transposer is
// reader extension 0, so its one user CSR sits at XDMA_SRC_EXT_CSR_PTR: 0 selects the
// 8-bit block mode, 1 the 16-bit one.
__attribute__((always_inline)) static inline void xdma_arm(uint32_t src_mask,
                                                           uint32_t xpose_mode) {
    XD_W(XDMA_SRC_ENABLE_PTR, src_mask);
    XD_W(XDMA_SRC_EXT_CSR_PTR, xpose_mode);
    XD_W(XDMA_DST_ENABLE_PTR, 0u);
}

static uint32_t xdma_run(void) {
    uint32_t tid = snax_xdma_start();
    snax_xdma_local_wait(tid);
    return snax_xdma_last_task_cycle();
}

// Full [rows, cols] -> [cols, rows] fp16 transpose. Same descriptor the library helper
// derives -- 8x8 blocks, two beats each at 16 bit, the block grid walked so the writer
// lands each transposed block at its transposed position -- only programmed cheaply.
static uint32_t xdma_transpose_fp16(void *src, void *dst, uint32_t rows, uint32_t cols) {
    const uint32_t TILE = 8, EB = 2, TC = 2;  // TC: beats per 8x8 block at 16 bit
    uint32_t bs[5] = {TC, cols / TILE, rows / TILE, 1, 1};
    uint32_t ss[5] = {XDMA_LANE_BYTES, TILE * EB, cols * TILE * EB, 0, 0};
    uint32_t bd[5] = {TC, cols / TILE, rows / TILE, 1, 1};
    uint32_t sd[5] = {XDMA_LANE_BYTES, rows * TILE * EB, TILE * EB, 0, 0};
    xdma_arm(1u << READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16, 1u);
    xdma_program_fast(src, dst, cols * EB, rows * EB, bs, ss, bd, sd);
    return xdma_run();
}

// row-major -> A-layout, fp16. NO EXTENSION AT ALL: this is a pure stride nest, which is
// the whole reason an A-layout reshape is cheap and a B-layout one is not.
//
// A puts element (row, col) at ((m*K_T + k)*meshRow + r)*tileSize + s, with row = m*meshRow
// + r and col = k*tileSize + s. Read that as bytes and every axis is affine on both sides:
//
//   s   4 elements = 8 B, contiguous in BOTH layouts     <- the lane payload, the atom
//   r   +row_b_rm in the source, +8 B in A                      <- the 8 spatial channels
//   r/8 the other half of a 16-row tile                         <- dim 0
//   k   +8 B in the source, +tile_b in A                        <- dim 1
//   m   +meshRow*row_b_rm both sides                            <- dim 2
//
// meshRow = 16 is what makes dim 0 bound 2 rather than a refusal; tileSize*2 = 8 B is what
// makes the atom land. At int8 that run is 4 B and the conversion falls off the hardware
// path entirely -- hence quantise AFTER the reshape, never before.
static uint32_t xdma_row_major_to_a(void *src, void *dst, uint32_t rows, uint32_t cols,
                                    uint32_t mesh_row, uint32_t tile_size) {
    const uint32_t EB = 2;
    uint32_t M_T = rows / mesh_row, K_T = cols / tile_size;
    uint32_t row_b_blk = tile_size * EB;      // one A-tile row, and the 8 B atom
    uint32_t row_b_rm = cols * EB;            // one row-major row
    uint32_t tile_b = mesh_row * tile_size * EB;
    uint32_t bs[5] = {mesh_row / 8, K_T, M_T, 1, 1};
    uint32_t ss[5] = {8 * row_b_rm, row_b_blk, mesh_row * row_b_rm, 0, 0};
    uint32_t bd[5] = {mesh_row / 8, K_T, M_T, 1, 1};
    uint32_t sd[5] = {8 * row_b_blk, tile_b, K_T * tile_b, 0, 0};
    xdma_arm(0u, 0u);
    xdma_program_fast(src, dst, row_b_rm, row_b_blk, bs, ss, bd, sd);
    return xdma_run();
}

// Where element (row, col) lands in A-layout, in ELEMENTS. The independent statement of
// the same bijection the stride nest above encodes -- if they agree, the nest is right.
static inline uint32_t a_index(uint32_t row, uint32_t col, uint32_t cols,
                               uint32_t mesh_row, uint32_t tile_size) {
    uint32_t m = row / mesh_row, r = row % mesh_row;
    uint32_t k = col / tile_size, s = col % tile_size;
    return ((m * (cols / tile_size) + k) * mesh_row + r) * tile_size + s;
}

// ============================================================ main

int main() {
    int err = 0;
    // TCDM layout, derived on EVERY hart and not on the engine core alone: the staging
    // core below has to land the data where the engine core will read it, and
    // snrt_cluster_base_addrl() is the same on every hart, so both derive it.
    uint32_t base = snrt_cluster_base_addrl();
    uint32_t rows = rms_rows;
    uint32_t d = rms_d;
    uint32_t beats = rms_beats;
    uint32_t log2d = rms_log2d;
    uint32_t row_b = beats * BEAT;
    uint32_t tot_b = rows * row_b;
    uint32_t scal_b = rows * BEAT;

    uint8_t *x_in = (uint8_t *)base;
    uint8_t *bt = x_in + tot_b;              // rows scalar beats (the reduce's output)
    uint8_t *bc = bt + scal_b;               // [rows,D] replication plane
    uint8_t *out_l = bc + tot_b;             // legacy result
    uint8_t *out_r = out_l + tot_b;          // rsqrt result, row-major
    // The sticky seed beat MUST sit immediately below x^T: the reader is one flat sweep of
    // 1 + D beats and the seed is simply its first.
    uint8_t *xt_seed = out_r + tot_b;
    uint8_t *xt = xt_seed + BEAT;
    uint8_t *ssq_t = xt + (uint32_t)(d * BEAT);   // one beat: the LANEWISE reduce's output
    uint8_t *out_t = ssq_t + BEAT;                // transposed result
    uint8_t *xt_ref = out_t + (uint32_t)(d * BEAT);  // L3 copy of x^T, to check against
    uint8_t *y_p = xt_ref + (uint32_t)(d * BEAT);    // y^T transposed back to row-major
    uint8_t *y_a = y_p + tot_b;                      // ...and reshaped into A-layout
    // Six words hart 2 writes and hart 1 reads: every xDMA conversion runs on the xDMA
    // core, but the summary is printed by the SIMD core, and they share only TCDM.
    volatile uint32_t *xc = (volatile uint32_t *)(y_a + tot_b);

    if (snax_is_idma_core()) {
        snrt_dma_start_1d(x_in, rms_input, tot_b);
        // NOT the transposed input -- that is produced on the device below. This is only
        // the reference the device's own transpose is checked against.
        snrt_dma_start_1d(xt_ref, rms_input_t, (uint32_t)(d * BEAT));
        snrt_dma_wait_all();
    }
    // Unconditional: the hardware barrier counts every core in the cluster, so a hart that
    // skipped it would hang the ones that did not.
    snrt_cluster_hw_barrier();

    // ---- x -> x^T, on the engine that owns the Transposer ----
    // Run twice and report the second: the first call pays icache, and a layer runs this
    // warm. Same for every conversion below.
    if (snax_is_xdma_core()) {
        (void)xdma_transpose_fp16(x_in, xt, rows, d);
        uint32_t a0 = snrt_mcycle();
        uint32_t dp = xdma_transpose_fp16(x_in, xt, rows, d);
        uint32_t a1 = snrt_mcycle();
        xc[0] = a1 - a0;
        xc[1] = dp;
    }
    snrt_cluster_hw_barrier();

    if (snax_is_simd_core())
        printf("[Rmsnorm] T=%u D=%u beats/row=%u\n", rows, d, beats);

    // Declared OUT here, not inside: the kernel runs on the SIMD core but the report is
    // printed after the xDMA epilogue below, past the block's closing brace.
    uint32_t t_l1 = 0, t_l2 = 0, t_l3 = 0, t_l4 = 0;
    uint32_t t_r2 = 0, t_s1 = 0, t_s2 = 0, t_s3 = 0;
    uint32_t d_l1 = 0, d_l3 = 0, d_l4 = 0, d_r2 = 0, d_s1 = 0, d_s2 = 0, d_s3 = 0;
    uint32_t t0, t1;

    if (snax_is_simd_core()) {
    // The transposed path is only meaningful if the device actually produced x^T, so
    // check the xDMA's output against the reference before using it.
    uint32_t xbad = 0;
    for (uint32_t i = 0; i < rows * d; i++)
        if (((uint16_t *)xt)[i] != ((uint16_t *)xt_ref)[i]) xbad++;
    printf("[Rmsnorm] xdma transpose vs reference: %s (%u/%u differ)\n",
           xbad ? "FAIL" : "bit-exact", xbad, rows * d);
    if (xbad) err++;

    // 1/D as FP32 bits. D is a power of two, so this is an exponent subtract and exact --
    // the same identity the core's scalar epilogue used, moved into a CSR immediate.
    uint32_t inv_d_bits = 0x3F800000u - (log2d << 23);
    volatile uint16_t *bt_l = (volatile uint16_t *)bt;


    // Two iterations: 0 = cold (icache + the first AGU program), 1 = warm. Only warm is
    // reported, which is the regime a layer runs in after its first norm.
    for (int it = 0; it < 2; it++) {
        // ================= LEGACY: four passes =================
        t0 = snrt_mcycle();
        d_l1 = pass_reduce(x_in, bt, rows, beats, SIMD_RED_SUMSQ, rows);
        t1 = snrt_mcycle();
        t_l1 = t1 - t0;

        // The scalar epilogue, exactly as an FPU-less core has to write it: per row, the
        // mean by exponent subtract, the integer rsqrt, then sixteen volatile word stores
        // to fill the row's whole beat -- the broadcast below consumes all 32 lanes, so
        // writing lane 0 alone is not enough.
        t0 = snrt_mcycle();
        for (uint32_t r = 0; r < rows; r++) {
            uint16_t ssq = bt_l[r * 32u];
            uint32_t Es = (ssq >> 10) & 0x1Fu;
            uint16_t mean = (uint16_t)(((Es - log2d) << 10) | (ssq & 0x3FFu));
            uint16_t inv = recip_f16(sqrt_f16(mean));
            uint32_t inv2 = ((uint32_t)inv << 16) | inv;
            volatile uint32_t *row32 = (volatile uint32_t *)(bt_l + r * 32u);
            for (uint32_t l = 0; l < 16u; l++) row32[l] = inv2;
        }
        t1 = snrt_mcycle();
        t_l2 = t1 - t0;

        t0 = snrt_mcycle();
        d_l3 = pass_bcast_map(bt, bc, rows, beats, row_b, SIMD_F32_ONE, SIMD_FUNC_LINEAR);
        t1 = snrt_mcycle();
        t_l3 = t1 - t0;

        t0 = snrt_mcycle();
        d_l4 = pass_ew2(x_in, bc, out_l, rows, beats, row_b, SIMD_EW_MUL);
        t1 = snrt_mcycle();
        t_l4 = t1 - t0;

        // ================= RSQRT: the same three passes, minus the epilogue =============
        // Pass 1 and pass 3 are the legacy path's, unchanged and already timed. Only the
        // middle differs, so only the middle is re-run: the reduce's output is restored
        // first because the legacy epilogue above overwrote it in place.
        for (uint32_t r = 0; r < rows; r++) {
            volatile uint32_t *row32 = (volatile uint32_t *)(bt_l + r * 32u);
            uint32_t s2 = ((uint32_t)rms_ssq_golden[r] << 16) | rms_ssq_golden[r];
            for (uint32_t l = 0; l < 16u; l++) row32[l] = s2;
        }
        t0 = snrt_mcycle();
        d_r2 = pass_bcast_map(bt, bc, rows, beats, row_b, inv_d_bits, SIMD_FUNC_RSQRT);
        t1 = snrt_mcycle();
        t_r2 = t1 - t0;
        (void)pass_ew2(x_in, bc, out_r, rows, beats, row_b, SIMD_EW_MUL);

        // ================= TRANSPOSED: lanewise reduce + sticky-B =================
        t0 = snrt_mcycle();
        d_s1 = pass_reduce(xt, ssq_t, 1u, d, SIMD_RED_SUMSQ | SIMD_RED_LANEWISE, 1u);
        t1 = snrt_mcycle();
        t_s1 = t1 - t0;

        t0 = snrt_mcycle();
        d_s2 = pass_map(ssq_t, xt_seed, 1u, inv_d_bits, 0u, SIMD_FUNC_RSQRT);
        t1 = snrt_mcycle();
        t_s2 = t1 - t0;

        t0 = snrt_mcycle();
        d_s3 = pass_ew_sticky(xt_seed, out_t, d, SIMD_EW_MUL);
        t1 = snrt_mcycle();
        t_s3 = t1 - t0;
    }
    }  // snax_is_simd_core

    // ---- the OTHER end of the round trip, back on hart 2 ----
    // The transposed kernel produced y^T. A-layout wants four consecutive FEATURES in one
    // 8 B run and y^T has four consecutive TOKENS, so y^T -> A is a transpose and cannot
    // be a stride nest. Undo the transposition first, then reshape -- and reshape the
    // ROW-MAJOR path's own output too, since that one skips straight to this step. Both
    // conversions are the constant-address programming above.
    snrt_cluster_hw_barrier();
    if (snax_is_xdma_core()) {
        (void)xdma_transpose_fp16(out_t, y_p, d, rows);  // warm
        uint32_t c0 = snrt_mcycle();
        uint32_t dp = xdma_transpose_fp16(out_t, y_p, d, rows);
        uint32_t c1 = snrt_mcycle();
        xc[2] = c1 - c0;
        xc[3] = dp;

        (void)xdma_row_major_to_a(y_p, y_a, rows, d, 16, 4);  // warm
        uint32_t e0 = snrt_mcycle();
        dp = xdma_row_major_to_a(y_p, y_a, rows, d, 16, 4);
        uint32_t e1 = snrt_mcycle();
        xc[4] = e1 - e0;
        xc[5] = dp;
    }
    snrt_cluster_hw_barrier();
    if (!snax_is_simd_core()) return 0;

    // ============================================================ report
    uint32_t legacy = t_l1 + t_l2 + t_l3 + t_l4;
    uint32_t rsq = t_l1 + t_r2 + t_l4;
    uint32_t trans = t_s1 + t_s2 + t_s3;

    printf("[Rmsnorm] --- warm, wall = CSR cfg + fire + wait ---\n");
    printf("[Rmsnorm] LEGACY  reduce(SUMSQ)      wall=%u  datapath=%u\n", t_l1, d_l1);
    printf("[Rmsnorm] LEGACY  scalar epilogue    wall=%u  (%u rows x 6 divu + 16 stores)\n",
           t_l2, rows);
    printf("[Rmsnorm] LEGACY  bcast_map(a=1)     wall=%u  datapath=%u\n", t_l3, d_l3);
    printf("[Rmsnorm] LEGACY  ew2(MUL)           wall=%u  datapath=%u\n", t_l4, d_l4);
    printf("[Rmsnorm] RSQRT   bcast_map(1/D,RSQ) wall=%u  datapath=%u  (replaces %u)\n",
           t_r2, d_r2, t_l2 + t_l3);
    printf("[Rmsnorm] TRANS   reduce(SUMSQ|LANE) wall=%u  datapath=%u\n", t_s1, d_s1);
    printf("[Rmsnorm] TRANS   map(1/D,RSQ) 1beat wall=%u  datapath=%u\n", t_s2, d_s2);
    printf("[Rmsnorm] TRANS   ew(MUL|STICKY_B)   wall=%u  datapath=%u\n", t_s3, d_s3);
    printf("[Rmsnorm] --- hart 2 (xDMA): the layout conversions ---\n");
    printf("[Rmsnorm] XPOSE   x   -> x^T          wall=%u  datapath=%u\n", xc[0], xc[1]);
    printf("[Rmsnorm] XPOSE   y^T -> y            wall=%u  datapath=%u\n", xc[2], xc[3]);
    printf("[Rmsnorm] RESHAPE y   -> A            wall=%u  datapath=%u  (no extension)\n",
           xc[4], xc[5]);

    uint32_t xdma_trans = xc[0] + xc[2] + xc[4];  // both transposes + the reshape
    uint32_t xdma_row = xc[4];                    // row-major pays only the reshape
    printf("[Rmsnorm] === SIMD (hart 1):  legacy %u | rsqrt %u | transposed %u ===\n",
           legacy, rsq, trans);
    printf("[Rmsnorm] === xDMA (hart 2):  rsqrt %u | transposed %u ===\n", xdma_row,
           xdma_trans);
    printf("[Rmsnorm] === both serialised:  rsqrt %u | transposed %u ===\n",
           rsq + xdma_row, trans + xdma_trans);
    // Two bounds, because the two engines run concurrently: if the xDMA work hides behind
    // other SIMD work the comparison is the first line, if nothing overlaps it is the
    // third. The transposed path always wins on hart 1, which is the busier engine.
    printf("[Rmsnorm] === SIMD-only: rsqrt %u vs transposed %u (%u%% less) ===\n", rsq,
           trans, rsq ? (100u * (rsq - trans)) / rsq : 0u);

    // ============================================================ checks
    // 1) LEGACY must reproduce the core's own integer rsqrt -- the golden is built from a
    // bit-exact model of it, so anything but a tight match means the port drifted and the
    // cycles above are not this kernel's.
    uint16_t *ol = (uint16_t *)out_l;
    uint32_t bad = 0, worst = 0, checked = 0;
    for (uint32_t i = 0; i < rows * d; i++) {
        if ((rms_out_golden[i] & 0x7C00u) == 0) continue;  // skip the subnormal/zero tail
        checked++;
        uint32_t g = fp16_mono(rms_out_golden[i]), o = fp16_mono(ol[i]);
        uint32_t u = (o > g) ? (o - g) : (g - o);
        if (u > worst) worst = u;
        if (u > 2) {
            if (bad < 4)
                printf("[Rmsnorm] legacy mismatch[%u] (row %u): got %04x golden %04x\n", i,
                       i / d, ol[i], rms_out_golden[i]);
            bad++;
        }
    }
    printf("[Rmsnorm] legacy vs its golden: significant=%u/%u worst %u ULP (%s)\n", checked,
           rows * d, worst, bad ? "FAIL" : "ok");
    if (bad) err++;

    // 2) the two hardware-rsqrt paths, against the EXACT reference. They cannot match the
    // legacy golden -- that one carries the core's 3-ULP integer rsqrt -- and the point of
    // scoring them here is that they come out CLOSER to the truth than the path they
    // replace, not merely close enough.
    uint16_t *orr = (uint16_t *)out_r;
    uint16_t *ot = (uint16_t *)out_t;
    uint32_t rworst = 0, rbad = 0, sworst = 0, sbad = 0;
    for (uint32_t t = 0; t < rows; t++)
        for (uint32_t f = 0; f < d; f++) {
            uint16_t want = rms_out_exact[t * d + f];
            if ((want & 0x7C00u) == 0) continue;
            uint32_t g = fp16_mono(want);
            uint32_t o = fp16_mono(orr[t * d + f]);
            uint32_t u = (o > g) ? (o - g) : (g - o);
            if (u > rworst) rworst = u;
            if (u > 2) rbad++;
            // the transposed result is y^T: feature-major, one token per lane
            o = fp16_mono(ot[f * rows + t]);
            u = (o > g) ? (o - g) : (g - o);
            if (u > sworst) sworst = u;
            if (u > 2) sbad++;
        }
    printf("[Rmsnorm] rsqrt      vs exact: worst %u FP16 ULP (%s)\n", rworst,
           rbad ? "FAIL" : "ok");
    printf("[Rmsnorm] transposed vs exact: worst %u FP16 ULP (%s)\n", sworst,
           sbad ? "FAIL" : "ok");
    printf("[Rmsnorm] for reference, the core's integer rsqrt is %u ULP on inv_rms\n",
           rms_inv_rsqrt_ulp);
    if (rbad || sbad) err++;

    // 5) the round trip. y^T transposed back must equal the row-major path's own output
    // bit for bit -- same operands through the same FP units, only the lane order differed
    // -- and the A-layout reshape must put every element where a_index() says.
    uint32_t tripbad = 0;
    for (uint32_t i = 0; i < rows * d; i++)
        if (((uint16_t *)y_p)[i] != ((uint16_t *)out_r)[i]) tripbad++;
    printf("[Rmsnorm] y^T->y vs the row-major result: %s (%u/%u differ)\n",
           tripbad ? "FAIL" : "bit-exact", tripbad, rows * d);
    if (tripbad) err++;

    uint32_t abad = 0;
    for (uint32_t t = 0; t < rows; t++)
        for (uint32_t f = 0; f < d; f++) {
            uint16_t want = ((uint16_t *)y_p)[t * d + f];
            uint16_t got = ((uint16_t *)y_a)[a_index(t, f, d, 16, 4)];
            if (got != want) {
                if (abad < 4)
                    printf("[Rmsnorm] A mismatch (t=%u,f=%u): got %04x want %04x\n", t, f,
                           got, want);
                abad++;
            }
        }
    printf("[Rmsnorm] y->A layout vs the index map: %s (%u/%u differ)\n",
           abad ? "FAIL" : "exact", abad, rows * d);
    if (abad) err++;

    printf(err ? "[Rmsnorm] FAIL\n" : "[Rmsnorm] PASS\n");
    return err != 0;
}
