// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Softmax on the four-engine cluster, and the three things that make it cheap.
//
// WHAT SOFTMAX IS. The normalisation every attention head ends with, and the one
// non-linearity in a transformer that is not pointwise. For a tile of T rows of D scores,
// every row becomes a probability distribution:
//
//     out[t, f] = exp(x[t, f] - max_f x[t, f]) / SUM_f exp(x[t, f] - max_f x[t, f])
//
// The max subtraction is not part of the mathematics -- it cancels exactly -- it is there
// so the exponential cannot overflow. That makes softmax a kernel with TWO per-row
// scalars where RMSNorm has one, and both of them sit in the middle of the dependence
// chain: nothing can be exponentiated until the max is known, and nothing can be divided
// until every exponential has been summed. See snax-simd-rmsnorm.c, which is the same
// argument with one scalar instead of two.
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
// THERE ARE TWO ELEMENTWISE SLOTS, one on each side of the Map, and softmax is why. The
// chain is a fixed linear order, so with a single elementwise it can express reduce(f(x))
// but never f(x - s): the subtraction of the row max has to happen BEFORE the exponential
// and had nowhere to go. Every previous softmax on this block therefore paid a whole
// extra pass over the tile just to subtract. With EW0 in front of the Map, the subtract,
// the exponential and the row sum are ONE task.
//
// The unit of the stream is a BEAT: 512 bits = 64 B = 32 FP16 LANES. And the one fact
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
// 32 lanes to be usable as a beat again. Softmax pays that twice, once for the max and
// once for the sum.
//
// Worse, a splatted beat is then hard to CONSUME. A 2-operand elementwise reads both
// operands from ONE 3-D affine address stream {operand, beat, row}:
//
//     addr = base + o*operand_delta + b*64 + r*row_stride
//
// Operand 0 (the data) wants exactly that. Operand 1 wants the SAME scalar for every beat
// b of row r -- i.e. no b term. The three loops share one stride set, so b cannot be
// zeroed for one operand only. The scalar therefore has to be REPLICATED into a full
// [T, D] plane before the elementwise can read it: an extra pass to write it, and an
// elementwise that reads twice the beats it writes. Softmax needs two such planes.
//
// Which machine softmax gets is decided entirely by the LAYOUT of its input. This is the
// same argument the FlashAttention apps make; see snax-flashattn-decode.c, which stores
// its score tile transposed for exactly this reason.
//
// ======================================================================================
// A WORKED EXAMPLE -- 4 tokens, 8 features, 4 lanes to a beat
// ======================================================================================
//
// Real shapes are T=32, D=128, 32 lanes; shrink everything by 8 and it fits on a page.
// Write L = ln 2, so every exponential below is an exact power of two and nothing has to
// be rounded to be read:
//
//            f0   f1   f2   f3   f4   f5   f6   f7      max   Sexp   1/Sexp
//     t0  [   0   -L   -L   -L   -L   -L  -2L  -2L ]      0      4     0.25
//     t1  [   3    3    3    3    3    3    3    3 ]      3      8     0.125
//     t2  [  -L    0   -L   -L  -2L   -L  -2L   -L ]      0      4     0.25
//     t3  [   3    3    3    3    3    3    3    3 ]      3      8     0.125
//
// t1 and t3 sit three units higher than t0 and t2 purely so the max subtraction has
// something to do; t2 is t0 shuffled, so the two rows have the same sum reached in a
// different order.
//
// ---- ROW-MAJOR x[T, D] : each TOKEN ROW contiguous -- what attention hands us ---------
//
// A beat is 4 consecutive features of ONE token, so t0's row is two beats:
//
//     beat 0  [  0  -L  -L  -L ]        beat 1  [ -L  -L -2L -2L ]
//
//   THE MAX. Lane k compares into acc[k] and only ever touches acc[k]; acc persists from
//   one beat to the next. Nothing moves sideways here:
//
//                        lane 0    lane 1    lane 2    lane 3
//     beat 0 = f0..f3  [    0        -L        -L        -L   ]
//     acc              [    0        -L        -L        -L   ]   acc[k] = max(acc[k],..)
//     beat 1 = f4..f7  [   -L        -L       -2L       -2L   ]
//     acc              [    0        -L        -L        -L   ]
//
//   Now read it down the COLUMNS, which is where the problem shows up -- each lane
//   compared EVERY FOURTH FEATURE of the row, never the whole row:
//
//     acc[0] = max(f0, f4) = max(  0, -L) =   0
//     acc[1] = max(f1, f5) = max( -L, -L) =  -L
//     acc[2] = max(f2, f6) = max( -L,-2L) =  -L
//     acc[3] = max(f3, f7) = max( -L,-2L) =  -L
//
//   The row's max is 0, and max(0,-L,-L,-L) = 0 -- correct, but spread over four separate
//   registers. Collapsing them means COMPARING ACROSS LANES, and no path in the
//   accumulate datapath does that:
//
//   FOLD ACROSS LANES -- the only step in the whole reduce that moves data sideways. It
//   is a balanced binary tree: each round pairs up the survivors, halving the count, and
//   round r+1 cannot start until round r has finished. So the number of ROUNDS is
//   log2(lanes) -- here log2(4) = 2:
//
//     round 1   [ 0  -L  -L  -L ]  ->  [ max(0,-L) , max(-L,-L) ]  =  [ 0  -L ]
//     round 2   [ 0  -L ]          ->  [ max(0,-L) ]               =  [ 0 ]
//
//   At the real width that is log2(32) = 5 rounds and 31 compares. Cycles are not the
//   depth alone: the compares of a round are spread over `treeLanes` fold ALUs (2 in this
//   cfg), each chunk held 2 cycles, so
//
//     round      1    2   3   4   5
//     pairs     16    8   4   2   1   = 31 compares
//     chunks     8    4   2   1   1   = 16 chunks / 2 ALUs, x2 cc = ~32 cc PER ROW
//
//   and the reader's input port is held low for all of it. SOFTMAX PAYS THIS TWICE: once
//   here and once for the row sum below. (`treeLanes` is a cfg knob: 16 would take the
//   fold to ~10 cc/row. It is set to 2 today.)
//
//   SPLAT -- the fold produced one NUMBER, but the datapath only moves whole 64 B BEATS;
//   there is no narrower write. So the result is copied into every lane, 0 -> [0 0 0 0],
//   which is exactly the shape the elementwise below wants. The waste is one step later:
//   that beat has to be REPEATED for every beat of the row.
//
//   THE EXPONENTIAL AND THE SUM. -max has to be replicated across all 8 features of t0's
//   row before an elementwise can pair it up, then the exponential runs, then the sum
//   reduce repeats the whole fold-and-splat story:
//
//     x      beat 0  [  0  -L  -L  -L ]   beat 1  [ -L  -L -2L -2L ]
//     plane  beat 0  [  0   0   0   0 ]   beat 1  [  0   0   0   0 ]  <- 8 copies of -max
//     exp    beat 0  [  1  .5  .5  .5 ]   beat 1  [ .5  .5 .25 .25 ]
//     acc               [ 1.5  1  .75  .75 ]      acc[k] += exp[k]
//     fold              round 1 [ 2.5  1.5 ]  round 2 [ 4 ]          <- Sexp = 4
//     splat             [ 4  4  4  4 ]
//
//   And then 1/4 = 0.25 has to be replicated into a SECOND [T, D] plane so a second
//   elementwise can divide. Four times over the tile: two folds per row, two scalars per
//   row, two rows of replication -- and two elementwise passes that read 16 beats to
//   write 8.
//
// ---- TRANSPOSED x^T[D, T] : each FEATURE contiguous, ONE TOKEN PER LANE ---------------
//
// A beat is now one feature across ALL FOUR tokens, so the tile is 8 beats:
//
//     beat f0 [ 0  3 -L  3 ]   beat f1 [ -L  3  0  3 ]   ...   beat f7 [ -2L 3 -L 3 ]
//       ^  ^  ^  ^
//       t0 t1 t2 t3   -- and lane t is token t in EVERY beat, forever
//
//   THE MAX -- exactly the same hardware, exactly the same rule, lane k into acc[k]:
//     after f0        acc = [  0  3 -L  3 ]
//     after f1..f7    acc = [  0  3  0  3 ]   <- THE ANSWER. All four tokens. Done.
//
//   Read down the columns again and the difference is the whole story. Lane t now sees
//   token t in EVERY beat, so it compared that token's WHOLE row. No fold. No splat.
//   SIMD_RED_LANEWISE just says "emit the accumulators", and ONE beat comes out holding
//   every token's max:  [ 0  3  0  3 ]
//
//   ONE StreamMap pass over that SINGLE BEAT, a = -1 and func = LINEAR, negates it:
//
//     [ 0  3  0  3 ]  ->  [ 0  -3  0  -3 ]      <- the SEED beat
//
//   SIMD_EW_STICKY_B latches that one beat as operand B and adds it to every later beat,
//   each lane getting its own token's max because lane t never stops being token t. And
//   because EW0 sits IN FRONT of the Map, the exponential and the sum ride the same task:
//
//     read      f0 [  0   3  -L   3 ]     f1 [ -L   3   0   3 ]    ... 8 beats
//     EW0 +B       [  0   0  -L   0 ]        [ -L   0   0   0 ]
//     Map exp      [  1   1  .5   1 ]        [ .5   1   1   1 ]    <- passed through
//     Reduce acc   [  1   1  .5   1 ]        [1.5   2 1.5   2 ]    ... -> [ 4  8  4  8 ]
//
//   SIMD_RED_TAP writes the exponentials AND appends the accumulator beat, so one task
//   emits the whole exp tile plus every token's sum -- [ 4 8 4 8 ], one beat, no fold.
//
//   Nothing is replicated, nothing is folded, and the read stream is D+1 beats, not 2*D.
//   (Transposing the row-major answer above gives the same values: t0's Sexp is 4, t1's
//   is 8, and the tap beat is [4 8 4 8]. Same arithmetic, different order in memory.)
//
// ======================================================================================
// THE RECIPROCAL, WITHOUT A HOST -- rsqrt(s*s) = 1/s
// ======================================================================================
//
// Every softmax on this block used to end with a number that had to leave the machine.
// The division by Sexp is one reciprocal per row; the cluster cores are rv32ima with no
// FPU, so the deployed kernel PRECOMPUTED 1/Sexp offline and shipped it in data.h -- a
// benchmark cheating on the one operation it could not do. (sm_inv_sum below is that
// array, kept so the path it replaces can still be run and timed.)
//
// The block grew a reciprocal square root for RMSNorm: StreamMap's func = RSQRT computes
// 1/sqrt(a*x + b). It is not a reciprocal, but it is one square away from being one, and
// EW0 can do that square with no operand but the number itself -- present the beat TWICE
// at stride 0 and let the elementwise combine the pair:
//
//     read    [ s , s ]                one beat, the reader's inner stride 0
//     EW0     s * s
//     Map     1/sqrt(s*s) = 1/s        <- exact for any positive s
//     write   [ 1/s ]
//
// Transposed, that is ONE PASS -- two beats in, one out, 33 cc of datapath -- because all
// T sums live in the lanes of a single beat, and EW0 runs in STICKY-B form (beat 0 seeds
// the latch and emits nothing, beat 1 emits s*s). On the worked example:
// [ 4 8 4 8 ] -> [ 16 64 16 64 ] -> [ .25 .125 .25 .125 ], every value exact.
//
// Row-major there are T scalars rather than one, so the sticky latch cannot hold them all
// and EW0 runs in its ordinary 2-operand form over each row's tap beat paired with itself:
// 64 beats read, 32 written, against the plane's 128. The inversion then rides the
// broadcast pass that had to replicate the scalar anyway -- that pass was carrying an
// identity multiply, so giving it func = RSQRT costs the SAME 296 cc of datapath it already
// spent. Either way the scalar never leaves the block.
//
//   THE RANGE THIS RUNS ON. The transport between two chained operators is FP16, so s*s
//   is what has to fit, not s. Every term of Sexp is exp(x - max) <= 1, so s <= D and the
//   bound is D*D <= 65504 -- D <= 255 for a pathological all-equal row, and far beyond
//   that in practice (this tile's worst s*s is 460). The datagen checks it and refuses to
//   emit data that would break it, rather than leaving an inf to be discovered at run
//   time. Past that bound the two operators swap order -- Map(RSQRT) first, then EW1
//   squares 1/sqrt(s) <= 1, which cannot overflow at any D -- at the cost of rounding
//   twice on the way down instead of once on the way up (~3 FP16 ULP against ~1.5).
//
// ======================================================================================
// THE THREE PATHS THIS APP RUNS
// ======================================================================================
//
//   LEGACY -- seven passes, and two of them are core loops that exist to move one number.
//     1  reduce(MAX)                   x -> mx      per-row scalar, splatted, ONE FOLD/ROW
//     2  CORE LOOP on the SIMD hart    -max[r] replicated into a [T,D] plane: T*D halfword
//                                      stores, 4,096 of them at this shape
//     3  ew(ADD)                       (x, plane) -> xs
//     4  map(EXP)                      xs -> ex
//     5  reduce(ADD)                   ex -> sm     A SECOND read of the whole tile, and
//                                      a SECOND fold per row
//     6  CORE LOOP + HOST RECIPROCAL   1/sm[r] from data.h, replicated: T*D stores again
//     7  ew(MUL)                       (ex, plane) -> y
//
//   ROW-MAJOR -- the same layout, every one of those removed that can be.
//     1  reduce(MAX)                   x -> mx                            unchanged
//     2  bcast_map(a=-1, LINEAR)       mx -> plane        pass 2 above, on the datapath:
//                                      the broadcast is a stride-0 read and the negate is
//                                      the map's `a`, so the core loop disappears entirely
//     3  EW0(ADD) | Map(EXP) | Reduce(ADD,TAP)   (x, plane) -> ex + Sexp
//                                      passes 3, 4 and 5 collapsed into ONE task. The tile
//                                      is never re-read and the exp row is never re-swept
//     4a EW0(MUL)                      Sexp -> Sexp*Sexp   each row's tap beat presented
//                                      twice at stride 0; 64 beats read, not the plane's
//                                      256
//     4b bcast_map(a=1, RSQRT)         Sexp*Sexp -> 1/Sexp plane    pass 6 AND its host
//                                      reciprocal, riding the replication that had to
//                                      happen anyway, at the same cost it already had
//     5  ew(MUL) [| quantise]          (ex, plane) -> y        unchanged, and the INT8
//                                      narrow chains onto it for free
//
//   TRANSPOSED -- the worked example above, using all three mechanisms.
//     0  xDMA 8x8 transpose            x -> x^T    ON HART 2, not the SIMD block
//     1  reduce(MAX|LANEWISE)          x^T -> mx   one beat, all T tokens, NO FOLD
//     2  map(a=-1) ONE BEAT            mx -> seed  21 cc of datapath
//     3  EW0(ADD|STICKY_B) | Map(EXP) | Reduce(ADD|LANEWISE|TAP)
//                                      seed+x^T -> ex^T + Sexp   no plane, no fold,
//                                      D+1 beats read
//     4  EW0(MUL|STICKY_B) | Map(RSQRT) ONE BEAT   Sexp -> 1/Sexp
//     5  ew(MUL|STICKY_B)              seed+ex^T -> y^T          no plane, D+1 read
//
//     Pass 0 is REAL here, not assumed: the app drives the xDMA's transposer on hart 2
//     and checks its output against the reference bit for bit, so the transposed numbers
//     below are paid for, not free.
//
// MEASURED, warm, Verilator, snax_split_cluster, T=32 D=128 (+-0.5% run to run):
//
//   hart 1, the SIMD block            hart 2, the xDMA (layout conversions)
//     legacy       39,696 cc            x   -> x^T   236 cc  (179 datapath)
//     row-major     5,665 cc            y^T -> y     385 cc  (327 datapath)
//     transposed    1,748 cc            y   -> A     299 cc  (240 datapath, NO extension)
//
//     row-major path:   5,665 on hart 1  +  299 on hart 2
//     transposed path:  1,748 on hart 1  +  920 on hart 2
//
// Two engines, so two honest bounds: SIMD-only the transposed path is 69% cheaper, fully
// serialised it is still 55% cheaper (2,668 vs 5,964). It always wins on hart 1, which is
// the busier engine, and the direction of that trade -- work off the SIMD and onto the
// xDMA -- is the one a layer wants. (The conversions are short enough that their WALL time
// moves a few percent between runs; their datapath figures do not.)
//
// Of the legacy path's 39,696 cc, 33,944 are the two core broadcast loops -- 85% of the
// kernel spent replicating two numbers per row, at ~4.1 cc per halfword store. Neither
// loop survives, and one of them could not have: the reciprocal it broadcast came from
// data.h because this core cannot divide.
//
// Where the row-major path's remaining time goes: the fold. Its reduce(MAX) is 1,387 cc of
// datapath for 128 beats, of which only ~128 is the compare -- the rest is 32 rows x ~35
// cc of cross-lane fold, and the sum reduce pays it again. The transposed reduce over the
// same 128 beats is 270 cc. That 5.1x is the fold, and nothing else.
//
// ======================================================================================
// SO SHOULD THE INPUT BE TRANSPOSED? -- READ THIS BEFORE BUILDING ANYTHING
// ======================================================================================
//
// The transposed path needs x^T, and it emits y^T. Both conversions are xDMA work on
// hart 2, and this app runs and times BOTH of them rather than assuming them.
//
//   WHERE THE LAYER PUTS THIS KERNEL:
//       GEMM(D-layout) -> reshape -> Softmax -> reshape -> A-layout -> quantise -> GEMM
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
//   programs and checks it (meshRow 16 is a multiple of 8, and tileSize*2 = 8 B is exactly
//   the atom). The quantise then runs on the A-LAYOUT tile, and that is sound because
//   quantising is elementwise and order-preserving: permute then narrow is the same tile
//   as narrow then permute, and only the first of the two is a conversion the hardware can
//   express. This app measures that last pass too.
//
//   THE INPUT-SIDE TRANSPOSE NEED NOT BE PAID AT ALL. Transposing both sides of the
//   PRODUCER's matmul rewrites it with its axes exchanged: (A.B)^T = B^T.A^T is the SAME
//   GEMM with its two operands swapped, which is M and N exchanged in the config and
//   nothing at run time. Not speculative -- the FlashAttention kernels in this tree
//   already do it, which is why their softmax gets a LANEWISE rowmax for free. Run the
//   producer (here the Q.K^T matmul) that way and its D-layout output IS the [D, T]
//   transposed score tile, so the D -> packed reshape that already runs emits x^T
//   directly. Probed against HeMAiA's own convert_args on a (16, 4, 16) array at fp16:
//
//       D[32,128]      -> packed    SUPPORTED    the layer today
//       D[128,32]      -> packed    SUPPORTED    <- x^T from a swapped producer, free
//       packed[32,128] -> A         SUPPORTED    the layer today
//       packed[128,32] -> B         REFUSED      transpose -- and this is what a swapped
//                                                CONSUMER would want from y^T
//
//   So the input side is free and the output side is not, and the last line is why: the
//   P.V matmul run in swapped form needs y^T as its B operand, and packed -> B is a
//   transpose no stride nest expresses. Either way y^T has to become y again. Net, the
//   transposed path costs ONE conversion more than the row-major one -- the 385 cc
//   y^T -> y -- to save 3,917 cc of SIMD time.
//
//   Check before banking it: swapping the producer propagates a layout requirement onto
//   ITS operands. A weight can be staged swapped offline for nothing; an activation
//   arriving from the previous layer is the thing to trace.
//
// ======================================================================================
//
// The app runs all three paths over the same tensor and checks each against the exact
// softmax, cross-checks the on-device reciprocal against one the core computes itself
// from the device's OWN row sums, and closes the round trip through both layout
// conversions.

#include "data.h"
#include "snax-core-roles.h"
#include "snax-simd-lib.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

// The extensions this kernel needs, and WHAT IT NEEDS THEM TO DO. An op/func CSR is a
// runtime select over the set the cfg elaborated, and selecting outside that set does not
// fault -- it returns another op's answer. So the gate names capabilities, not extensions;
// each _HAS_ macro implies its extension exists. See the note in snax-simd-lib.h.
//
// Two of these are worth naming. RSQRT is what makes the reciprocal a datapath op rather
// than a host round trip, and without it StreamMap returns the LINEAR result -- so a build
// missing it would divide by Sexp*Sexp and still produce a plausible tensor. And the PRE-map
// elementwise (instance 0) is what lets the subtract, the exponential and the row sum share
// one task; on a cluster with only the post-map instance this kernel is not slower, it is
// wrong, because the max would be subtracted after the exponential.
//
// tailPassthrough is a BUILD-time feature, not a runtime select: without it the quantiser
// has one user CSR instead of two, and cfg_qnt() below writes two.
#if !defined(SIMD_EXT_STREAMREDUCE_HAS_MAX) ||          \
    !defined(SIMD_EXT_STREAMREDUCE_HAS_ADD) ||          \
    !defined(SIMD_EXT_STREAMMAP_HAS_EXP) ||             \
    !defined(SIMD_EXT_STREAMMAP_HAS_RSQRT) ||           \
    !defined(SIMD_EXT_STREAMELEMENTWISE_0_HAS_ADD) ||   \
    !defined(SIMD_EXT_STREAMELEMENTWISE_0_HAS_MUL) ||   \
    !defined(SIMD_EXT_STREAMELEMENTWISE_1_HAS_MUL) ||   \
    !defined(SIMD_EXT_FP16TOINT8_HAS_TAILPASSTHROUGH)
#error \
    "This cluster's SIMD block cannot run this softmax: it needs StreamReduce MAX+ADD, StreamMap EXP+RSQRT, a PRE-map StreamElementwise with MUL+ADD, a post-map one with MUL, and Fp16ToInt8 tailPassthrough."
#endif

#define BEAT SIMD_BEAT_BYTES
#define FP16_PER_BEAT 32

// ============================================================ the core's FPU-less recip
//
// Not on any path -- it is the CROSS-CHECK. The device now computes 1/Sexp for itself,
// and the only way to score that without trusting a golden built from a different
// exponential is to invert the device's OWN row sums here, on the core, the way an
// FPU-less machine has to: integer bit manipulation on the FP16 pattern with one rv32M
// `divu`. Accurate to ~1 ULP, which is enough to catch a wrong answer and not enough to
// be worth deploying (see the header: six of these per row is what RMSNorm's legacy
// epilogue cost).
static inline uint16_t recip_f16(uint16_t s) {
    uint32_t E = (s >> 10) & 0x1Fu;
    uint32_t M = 1024u + (s & 0x3FFu);
    uint32_t q = ((1u << 21) + (M >> 1)) / M;
    if (q >= 2048u) return (uint16_t)((30u - E) << 10);
    return (uint16_t)(((29u - E) << 10) | ((q - 1024u) & 0x3FFu));
}

static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

static inline uint32_t f16_ulp(uint16_t a, uint16_t b) {
    uint32_t x = fp16_mono(a), y = fp16_mono(b);
    return (x > y) ? (x - y) : (y - x);
}

// ============================================================ arming the chain
//
// One write to the enable mask arms the WHOLE chain -- a task with three operators is not
// three arming steps, it is one mask and each operator's CSRs at a compile-time-constant
// address. (snax_simd_enable_ext() walks the CSR map at run time to find each block, so
// every write it makes has a computed address and pays a jump-table load plus an indirect
// jump, about 6 cycles against 1.)

#define M_EW0 (1u << SIMD_EXT_STREAMELEMENTWISE_0)
#define M_MAP (1u << SIMD_EXT_STREAMMAP)
#define M_RED (1u << SIMD_EXT_STREAMREDUCE)
#define M_EW1 (1u << SIMD_EXT_STREAMELEMENTWISE_1)
#define M_QNT (1u << SIMD_EXT_FP16TOINT8)

#define ARM(mask) snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (mask))

__attribute__((always_inline)) static inline void cfg_ew0(uint32_t cnt, uint32_t op) {
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 0, cnt);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 1, op);
}
__attribute__((always_inline)) static inline void cfg_ew1(uint32_t cnt, uint32_t op) {
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_1_CSR + 0, cnt);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_1_CSR + 1, op);
}
__attribute__((always_inline)) static inline void cfg_map(uint32_t a, uint32_t b,
                                                          uint32_t func) {
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0, a);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 1, b);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 2, func);
}
__attribute__((always_inline)) static inline void cfg_red(uint32_t cnt, uint32_t mode) {
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 0, cnt);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1, mode);
}
// csr(1) is tailPeriod -- pass every (tailPeriod+1)'th beat through UNQUANTISED. 0 is off,
// which is right everywhere here: no trailing scalar beat reaches this quantiser.
__attribute__((always_inline)) static inline void cfg_qnt(uint32_t inv_scale) {
    snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 0, inv_scale);
    snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 1, 0u);
}

// ============================================================ pass helpers

// A task that never retires must not cost a whole simulation. run_shapes() bounds every
// wait and LATCHES the failure: once the block is wedged nothing after it can retire
// either, so the remaining passes short-circuit instead of paying the budget again each.
// `simd_hung` is folded into the error count at the end -- it cannot return early from
// here, because the barriers that follow are counted by every hart in the cluster.
static int simd_hung = 0;

static inline uint32_t run_shapes(const char *what, const snax_simd_shape_t *in,
                                  const snax_simd_shape_t *out) {
    if (simd_hung) return 0xFFFFFFFFu;
    snax_simd_program_fast(in, out);
    snax_simd_fire();
    if (snax_simd_wait_all_checked(what, SIMD_WAIT_BUDGET)) {
        simd_hung = 1;
        return 0xFFFFFFFFu;
    }
    return snax_simd_last_task_cycle();
}

// Per-row reduction. 2-D reader {beat, row}, 1-D writer. `mode` carries MAX or ADD plus
// any of TAP / LANEWISE; with LANEWISE pass rows=1 and beats=D, so the whole tile is one
// "row" and the per-lane accumulators ARE the per-token scalars.
static uint32_t pass_reduce(void *src, void *dst, uint32_t rows, uint32_t beats,
                            uint32_t mode, uint32_t dst_beats) {
    snax_simd_shape_t in, out;
    snax_simd_shape_rows(&in, src, rows, beats, beats * BEAT);
    snax_simd_shape_flat(&out, dst, dst_beats);
    ARM(M_RED);
    cfg_red(beats, mode);
    return run_shapes("reduce", &in, &out);
}

// out = func(a*x + b) over `beats` contiguous beats.
static uint32_t pass_map(void *src, void *dst, uint32_t beats, uint32_t a, uint32_t b,
                         uint32_t func) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, src, beats);
    snax_simd_shape_flat(&out, dst, beats);
    ARM(M_MAP);
    cfg_map(a, b, func);
    return run_shapes("map", &in, &out);
}

// Broadcast one beat per row into `beats` beats per row, with a StreamMap applied on the
// fly. The reader's inner stride of 0 re-presents the row's scalar beat, and whatever
// `func` is armed runs on it at no extra cost -- which is how -max gets negated without a
// pass of its own, and how 1/Sexp gets inverted without one either.
static uint32_t pass_bcast_map(void *src_beats, void *dst, uint32_t rows, uint32_t beats,
                               uint32_t dst_row_stride, uint32_t a, uint32_t b,
                               uint32_t func) {
    snax_simd_shape_t in, out;
    snax_simd_shape_2d(&in, src_beats, beats, 0u, rows, BEAT);
    snax_simd_shape_rows(&out, dst, rows, beats, dst_row_stride);
    ARM(M_MAP);
    cfg_map(a, b, func);
    return run_shapes("bcast_map", &in, &out);
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

static inline void shape_interleave(snax_simd_shape_t *s, void *base, uint32_t delta,
                                    uint32_t beats, uint32_t rows,
                                    uint32_t src_row_stride) {
    snax_simd_shape_flat(s, base, 1);  // seeds the neutral dims, then overwritten
    s->dim = 3;
    s->bound[0] = 2u;
    s->stride[0] = delta;
    s->bound[1] = beats;
    s->stride[1] = BEAT;
    s->bound[2] = rows;
    s->stride[2] = src_row_stride;
}

// Two-operand elementwise on the POST-map slot -- the shape whose shared strides are
// exactly why a per-row scalar has to be replicated first. `quant` non-zero chains the
// INT8 narrow onto the same pass, halving the beats the writer drains.
static uint32_t pass_ew2(void *src_a, void *src_b, void *dst, uint32_t rows,
                         uint32_t beats, uint32_t src_row_stride, uint32_t op,
                         uint32_t quant) {
    uint32_t delta;
    void *base = ew2_base(src_a, src_b, &delta);
    snax_simd_shape_t in, out;
    shape_interleave(&in, base, delta, beats, rows, src_row_stride);
    snax_simd_shape_flat(&out, dst, quant ? rows * beats / 2u : rows * beats);
    if (quant) {
        ARM(M_EW1 | M_QNT);
        cfg_qnt(quant);
    } else {
        ARM(M_EW1);
    }
    cfg_ew1(2u, op);
    return run_shapes("ew2", &in, &out);
}

// ROW-MAJOR, THREE PASSES IN ONE: EW0 subtracts the replicated max, Map exponentiates,
// Reduce sums -- and TAP makes the reduce emit the exp row as well as its sum, so the
// tile is written once and never re-read. Output rows are beats+1 beats: the row, then
// its Sexp. This is what the second elementwise slot bought.
static uint32_t pass_sub_exp_sum(void *x, void *plane, void *dst, uint32_t rows,
                                 uint32_t beats, uint32_t src_row_stride,
                                 uint32_t dst_row_stride) {
    uint32_t delta;
    void *base = ew2_base(x, plane, &delta);
    snax_simd_shape_t in, out;
    shape_interleave(&in, base, delta, beats, rows, src_row_stride);
    snax_simd_shape_rows(&out, dst, rows, beats + 1u, dst_row_stride);
    ARM(M_EW0 | M_MAP | M_RED);
    cfg_ew0(2u, SIMD_EW_ADD);
    cfg_map(SIMD_F32_ONE, 0u, SIMD_FUNC_EXP);
    cfg_red(beats, SIMD_RED_ADD | SIMD_RED_TAP);
    return run_shapes("EW0(ADD)|Map(EXP)|Reduce(TAP)", &in, &out);
}

// ROW-MAJOR RECIPROCAL, half of it. Square each row's Sexp: the reader presents that row's
// tap beat TWICE at stride 0 and EW0's 2-operand combine multiplies the pair. `rows` beats
// in each direction -- 32 of them at this shape, against the tile's 128.
//
// The other half is free. The inversion rides pass_bcast_map, which the row-major path has
// to run anyway to replicate the scalar, and which was carrying an identity multiply: give
// that same pass func = RSQRT and its output is 1/Sexp instead of Sexp*Sexp. (Squaring in
// the broadcast instead would work too, and cost the plane's whole width; doing it here
// costs a quarter of one row.)
static uint32_t pass_sq_beats(void *tap0, void *dst, uint32_t rows,
                              uint32_t src_row_stride) {
    snax_simd_shape_t in, out;
    snax_simd_shape_2d(&in, tap0, 2u, 0u, rows, src_row_stride);
    snax_simd_shape_flat(&out, dst, rows);
    ARM(M_EW0);
    cfg_ew0(2u, SIMD_EW_MUL);
    return run_shapes("EW0(MUL) square", &in, &out);
}

// TRANSPOSED, THREE PASSES IN ONE. Same fusion as pass_sub_exp_sum, but the max arrives
// as a STICKY seed beat instead of a replicated plane, and the reduce is LANEWISE, so
// there is no fold and no [T,D] operand anywhere. D+1 beats in (seed + tile), D+1 out
// (tile + the one sum beat).
static uint32_t pass_t_exp_sum(void *seed_then_xt, void *dst, uint32_t d) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, seed_then_xt, d + 1u);
    snax_simd_shape_flat(&out, dst, d + 1u);
    ARM(M_EW0 | M_MAP | M_RED);
    cfg_ew0(1u, SIMD_EW_ADD | SIMD_EW_STICKY_B);
    cfg_map(SIMD_F32_ONE, 0u, SIMD_FUNC_EXP);
    cfg_red(d, SIMD_RED_ADD | SIMD_RED_TAP | SIMD_RED_LANEWISE);
    return run_shapes("EW0(STICKY)|Map(EXP)|Reduce(LANE,TAP)", &in, &out);
}

// TRANSPOSED RECIPROCAL: 1/s for every token in the tile, from ONE beat. The sum beat is
// read twice at stride 0, EW0's sticky latch squares it, Map's RSQRT inverts it.
static uint32_t pass_recip_beat(void *sum_beat, void *dst) {
    snax_simd_shape_t in, out;
    snax_simd_shape_2d(&in, sum_beat, 2u, 0u, 1u, 0u);
    snax_simd_shape_flat(&out, dst, 1u);
    ARM(M_EW0 | M_MAP);
    cfg_ew0(1u, SIMD_EW_MUL | SIMD_EW_STICKY_B);
    cfg_map(SIMD_F32_ONE, 0u, SIMD_FUNC_RSQRT);
    return run_shapes("EW0(MUL,STICKY)|Map(RSQRT)", &in, &out);
}

// One-operand elementwise with STICKY-B: beat 0 seeds operand B and emits NOTHING, beats
// 1..N emit op(B, beat). N+1 beats in, N out -- so the seed must sit immediately below the
// data and the writer covers exactly the data, not one beat early.
static uint32_t pass_ew_sticky(void *seed_then_data, void *dst, uint32_t data_beats,
                               uint32_t op) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, seed_then_data, data_beats + 1u);
    snax_simd_shape_flat(&out, dst, data_beats);
    ARM(M_EW0);
    cfg_ew0(1u, op | SIMD_EW_STICKY_B);
    return run_shapes("ew(STICKY)", &in, &out);
}

// FP16 -> INT8 on its own, over a flat tile. Elementwise and order-preserving, which is
// what lets it run AFTER the A-layout reshape -- see the header.
static uint32_t pass_quant(void *src, void *dst, uint32_t beats, uint32_t inv_scale) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, src, beats);
    snax_simd_shape_flat(&out, dst, beats / 2u);
    ARM(M_QNT);
    cfg_qnt(inv_scale);
    return run_shapes("quantise", &in, &out);
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
    uint32_t row_b_blk = tile_size * EB;  // one A-tile row, and the 8 B atom
    uint32_t row_b_rm = cols * EB;        // one row-major row
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
    uint32_t rows = sm_rows;
    uint32_t d = sm_d;
    uint32_t beats = sm_beats;
    uint32_t row_b = beats * BEAT;
    uint32_t tot_b = rows * row_b;
    uint32_t scal_b = rows * BEAT;
    uint32_t pad_row_b = (beats + 1u) * BEAT;  // a row plus its trailing scalar beat
    uint32_t pad_b = rows * pad_row_b;
    uint32_t t_b = (uint32_t)(d * BEAT);  // the transposed tile: D beats of T lanes

    uint8_t *x_in = (uint8_t *)base;
    uint8_t *mx = x_in + tot_b;        // reduce(MAX) out, rows splatted beats
    uint8_t *plane = mx + scal_b;      // [T,D] operand plane: -max, then 1/Sexp
    uint8_t *xs = plane + tot_b;       // legacy x - max
    uint8_t *ex_a = xs + tot_b;        // legacy exp tile
    uint8_t *sm_l = ex_a + tot_b;      // legacy reduce(ADD) out
    uint8_t *sq = sm_l + scal_b;       // row-major Sexp*Sexp, rows beats
    uint8_t *out_l = sq + scal_b;      // legacy result
    uint8_t *ex_p = out_l + tot_b;     // row-major TAP out: [T, beats+1]
    uint8_t *rplane = ex_p + pad_b;    // 1/Sexp plane, written at the PADDED row stride so
                                       // it shares a row stride with ex_p
    uint8_t *out_r = rplane + pad_b;   // row-major result
    uint8_t *q_r = out_r + tot_b;      // row-major result, INT8, quant chained on
    // The sticky seed beat MUST sit immediately below the data it seeds: the reader is one
    // flat sweep of 1 + N beats and the seed is simply its first.
    uint8_t *xt_seed = q_r + tot_b / 2u;
    uint8_t *xt = xt_seed + BEAT;       // x^T, D beats
    uint8_t *mx_t = xt + t_b;           // one beat: the LANEWISE max
    uint8_t *ex_seed = mx_t + BEAT;     // one beat: 1/Sexp, and the sticky seed for pass 5
    uint8_t *ex_t = ex_seed + BEAT;     // exp^T, D beats, then its sum beat
    uint8_t *out_t = ex_t + (d + 1u) * BEAT;         // y^T
    uint8_t *xt_ref = out_t + t_b;                   // L3 copy of x^T, to check against
    uint8_t *y_p = xt_ref + t_b;                     // y^T transposed back to row-major
    uint8_t *y_a = y_p + tot_b;                      // ...and reshaped into A-layout
    uint8_t *q_a = y_a + tot_b;                      // ...and quantised, in A-layout
    // Six words hart 2 writes and hart 1 reads: every xDMA conversion runs on the xDMA
    // core, but the summary is printed by the SIMD core, and they share only TCDM.
    volatile uint32_t *xc = (volatile uint32_t *)(q_a + tot_b / 2u);

    if (snax_is_idma_core()) {
        snrt_dma_start_1d(x_in, sm_input, tot_b);
        // NOT the transposed input -- that is produced on the device below. This is only
        // the reference the device's own transpose is checked against.
        snrt_dma_start_1d(xt_ref, sm_input_t, t_b);
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

    if (snax_is_simd_core()) printf("[Softmax] T=%u D=%u beats/row=%u\n", rows, d, beats);

    // Declared OUT here, not inside: the kernel runs on the SIMD core but the report is
    // printed after the xDMA epilogue below, past the block's closing brace.
    uint32_t t_l1 = 0, t_l2 = 0, t_l3 = 0, t_l4 = 0, t_l5 = 0, t_l6 = 0, t_l7 = 0;
    uint32_t t_r2 = 0, t_r3 = 0, t_r4 = 0, t_r4b = 0, t_r5 = 0, t_rq = 0;
    uint32_t t_s1 = 0, t_s2 = 0, t_s3 = 0, t_s4 = 0, t_s5 = 0;
    uint32_t d_l1 = 0, d_l3 = 0, d_l4 = 0, d_l5 = 0, d_l7 = 0;
    uint32_t d_r2 = 0, d_r3 = 0, d_r4 = 0, d_r4b = 0, d_r5 = 0, d_rq = 0;
    uint32_t d_s1 = 0, d_s2 = 0, d_s3 = 0, d_s4 = 0, d_s5 = 0;
    uint32_t t_qa = 0, d_qa = 0;
    uint32_t t0, t1;

    if (snax_is_simd_core()) {
        // The transposed path is only meaningful if the device actually produced x^T, so
        // check the xDMA's output against the reference before using it.
        uint32_t xbad = 0;
        for (uint32_t i = 0; i < rows * d; i++)
            if (((uint16_t *)xt)[i] != ((uint16_t *)xt_ref)[i]) xbad++;
        printf("[Softmax] xdma transpose vs reference: %s (%u/%u differ)\n",
               xbad ? "FAIL" : "bit-exact", xbad, rows * d);
        if (xbad) err++;

        uint16_t *mxh = (uint16_t *)mx;
        uint16_t *pl = (uint16_t *)plane;

        // Two iterations: 0 = cold (icache + the first AGU program), 1 = warm. Only warm
        // is reported, which is the regime a layer runs in after its first tile.
        for (int it = 0; it < 2; it++) {
            // ================= LEGACY: seven passes, two of them core loops ============
            t0 = snrt_mcycle();
            d_l1 = pass_reduce(x_in, mx, rows, beats, SIMD_RED_MAX, rows);
            t1 = snrt_mcycle();
            t_l1 = t1 - t0;

            // The core broadcast, exactly as an FPU-less core has to write it: read the
            // splatted max beat's low lane, flip its sign (an integer XOR on the FP16
            // pattern -- the one FP operation this core CAN do), and store it into every
            // one of the row's D features.
            t0 = snrt_mcycle();
            for (uint32_t r = 0; r < rows; r++) {
                uint16_t neg = (uint16_t)(mxh[r * FP16_PER_BEAT] ^ 0x8000u);
                for (uint32_t c = 0; c < d; c++) pl[r * d + c] = neg;
            }
            t1 = snrt_mcycle();
            t_l2 = t1 - t0;

            t0 = snrt_mcycle();
            d_l3 = pass_ew2(x_in, plane, xs, rows, beats, row_b, SIMD_EW_ADD, 0u);
            t1 = snrt_mcycle();
            t_l3 = t1 - t0;

            t0 = snrt_mcycle();
            d_l4 = pass_map(xs, ex_a, rows * beats, SIMD_F32_ONE, 0u, SIMD_FUNC_EXP);
            t1 = snrt_mcycle();
            t_l4 = t1 - t0;

            t0 = snrt_mcycle();
            d_l5 = pass_reduce(ex_a, sm_l, rows, beats, SIMD_RED_ADD, rows);
            t1 = snrt_mcycle();
            t_l5 = t1 - t0;

            // The second core loop, and the one this machine could not do for itself: the
            // reciprocal comes from data.h because an rv32ima core has no FPU.
            t0 = snrt_mcycle();
            for (uint32_t r = 0; r < rows; r++)
                for (uint32_t c = 0; c < d; c++) pl[r * d + c] = sm_inv_sum[r];
            t1 = snrt_mcycle();
            t_l6 = t1 - t0;

            t0 = snrt_mcycle();
            d_l7 = pass_ew2(ex_a, plane, out_l, rows, beats, row_b, SIMD_EW_MUL, 0u);
            t1 = snrt_mcycle();
            t_l7 = t1 - t0;

            // ================= ROW-MAJOR: same layout, both core loops gone ============
            // Pass 1 is the legacy path's reduce(MAX), unchanged and already timed.
            t0 = snrt_mcycle();
            d_r2 = pass_bcast_map(mx, plane, rows, beats, row_b, SIMD_F32_NEG_ONE, 0u,
                                  SIMD_FUNC_LINEAR);
            t1 = snrt_mcycle();
            t_r2 = t1 - t0;

            t0 = snrt_mcycle();
            d_r3 = pass_sub_exp_sum(x_in, plane, ex_p, rows, beats, row_b, pad_row_b);
            t1 = snrt_mcycle();
            t_r3 = t1 - t0;

            t0 = snrt_mcycle();
            d_r4 = pass_sq_beats(ex_p + beats * BEAT, sq, rows, pad_row_b);
            t1 = snrt_mcycle();
            t_r4 = t1 - t0;

            t0 = snrt_mcycle();
            d_r4b = pass_bcast_map(sq, rplane, rows, beats, pad_row_b, SIMD_F32_ONE, 0u,
                                   SIMD_FUNC_RSQRT);
            t1 = snrt_mcycle();
            t_r4b = t1 - t0;

            t0 = snrt_mcycle();
            d_r5 = pass_ew2(ex_p, rplane, out_r, rows, beats, pad_row_b, SIMD_EW_MUL, 0u);
            t1 = snrt_mcycle();
            t_r5 = t1 - t0;

            // The SAME pass with the INT8 narrow chained on: no re-read of the FP16 tile,
            // and the writer drains half the beats.
            t0 = snrt_mcycle();
            d_rq = pass_ew2(ex_p, rplane, q_r, rows, beats, pad_row_b, SIMD_EW_MUL,
                            sm_inv_scale);
            t1 = snrt_mcycle();
            t_rq = t1 - t0;

            // ================= TRANSPOSED: no fold, no plane, no host ==================
            t0 = snrt_mcycle();
            d_s1 = pass_reduce(xt, mx_t, 1u, d, SIMD_RED_MAX | SIMD_RED_LANEWISE, 1u);
            t1 = snrt_mcycle();
            t_s1 = t1 - t0;

            t0 = snrt_mcycle();
            d_s2 = pass_map(mx_t, xt_seed, 1u, SIMD_F32_NEG_ONE, 0u,
                            SIMD_FUNC_LINEAR);
            t1 = snrt_mcycle();
            t_s2 = t1 - t0;

            t0 = snrt_mcycle();
            d_s3 = pass_t_exp_sum(xt_seed, ex_t, d);
            t1 = snrt_mcycle();
            t_s3 = t1 - t0;

            t0 = snrt_mcycle();
            d_s4 = pass_recip_beat(ex_t + (uint32_t)(d * BEAT), ex_seed);
            t1 = snrt_mcycle();
            t_s4 = t1 - t0;

            t0 = snrt_mcycle();
            d_s5 = pass_ew_sticky(ex_seed, out_t, d, SIMD_EW_MUL);
            t1 = snrt_mcycle();
            t_s5 = t1 - t0;
        }
    }  // snax_is_simd_core

    // ---- the OTHER end of the round trip, back on hart 2 ----
    // The transposed kernel produced y^T. A-layout wants four consecutive FEATURES in one
    // 8 B run and y^T has four consecutive TOKENS, so y^T -> A is a transpose and cannot
    // be a stride nest. Undo the transposition first, then reshape -- and the reshape is
    // the step the ROW-MAJOR path pays too, since that one skips straight to it.
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

    // The last pass of the layer's epilogue, and the reason it is last: quantising is
    // elementwise, so it commutes with the A-layout permutation, and at INT8 the reshape's
    // 8 B atom would have become 4 B and fallen off the hardware path.
    (void)pass_quant(y_a, q_a, rows * beats, sm_inv_scale);  // warm
    t0 = snrt_mcycle();
    d_qa = pass_quant(y_a, q_a, rows * beats, sm_inv_scale);
    t1 = snrt_mcycle();
    t_qa = t1 - t0;

    // ============================================================ report
    uint32_t legacy = t_l1 + t_l2 + t_l3 + t_l4 + t_l5 + t_l6 + t_l7;
    uint32_t rowmaj = t_l1 + t_r2 + t_r3 + t_r4 + t_r4b + t_r5;
    uint32_t trans = t_s1 + t_s2 + t_s3 + t_s4 + t_s5;

    printf("[Softmax] --- warm, wall = CSR cfg + fire + wait ---\n");
    printf("[Softmax] LEGACY  reduce(MAX)        wall=%u  datapath=%u\n", t_l1, d_l1);
    printf("[Softmax] LEGACY  core bcast -max    wall=%u  (%u stores)\n", t_l2, rows * d);
    printf("[Softmax] LEGACY  ew(ADD)            wall=%u  datapath=%u\n", t_l3, d_l3);
    printf("[Softmax] LEGACY  map(EXP)           wall=%u  datapath=%u\n", t_l4, d_l4);
    printf("[Softmax] LEGACY  reduce(ADD)        wall=%u  datapath=%u\n", t_l5, d_l5);
    printf("[Softmax] LEGACY  core bcast 1/sum   wall=%u  (%u stores, HOST reciprocal)\n",
           t_l6, rows * d);
    printf("[Softmax] LEGACY  ew(MUL)            wall=%u  datapath=%u\n", t_l7, d_l7);
    printf("[Softmax] ROWMAJ  bcast_map(a=-1)    wall=%u  datapath=%u  (replaces %u)\n",
           t_r2, d_r2, t_l2);
    printf("[Softmax] ROWMAJ  EW0|MAP|RED(TAP)   wall=%u  datapath=%u  (replaces %u)\n",
           t_r3, d_r3, t_l3 + t_l4 + t_l5);
    printf("[Softmax] ROWMAJ  EW0(MUL) square     wall=%u  datapath=%u  (%u beats, not "
           "%u)\n",
           t_r4, d_r4, 2u * rows, 2u * rows * beats);
    printf("[Softmax] ROWMAJ  bcast_map(RSQRT)   wall=%u  datapath=%u  (replaces %u + the "
           "host)\n",
           t_r4b, d_r4b, t_l6);
    printf("[Softmax] ROWMAJ  ew(MUL)            wall=%u  datapath=%u\n", t_r5, d_r5);
    printf("[Softmax] ROWMAJ  ew(MUL)+quant      wall=%u  datapath=%u  (quant marginal "
           "%u)\n",
           t_rq, d_rq, t_rq - t_r5);
    printf("[Softmax] TRANS   reduce(MAX|LANE)   wall=%u  datapath=%u\n", t_s1, d_s1);
    printf("[Softmax] TRANS   map(a=-1) 1 beat   wall=%u  datapath=%u\n", t_s2, d_s2);
    printf("[Softmax] TRANS   EW0(STICKY)|MAP|RED(LANE,TAP) wall=%u  datapath=%u\n", t_s3,
           d_s3);
    printf("[Softmax] TRANS   EW0(MUL,STICKY)|MAP(RSQ) 1 beat wall=%u  datapath=%u\n",
           t_s4, d_s4);
    printf("[Softmax] TRANS   ew(MUL|STICKY)     wall=%u  datapath=%u\n", t_s5, d_s5);
    printf("[Softmax] --- hart 2 (xDMA): the layout conversions ---\n");
    printf("[Softmax] XPOSE   x   -> x^T          wall=%u  datapath=%u\n", xc[0], xc[1]);
    printf("[Softmax] XPOSE   y^T -> y            wall=%u  datapath=%u\n", xc[2], xc[3]);
    printf("[Softmax] RESHAPE y   -> A            wall=%u  datapath=%u  (no extension)\n",
           xc[4], xc[5]);
    printf("[Softmax] --- hart 1 again, after the reshape ---\n");
    printf("[Softmax] QUANT   A(fp16) -> A(int8)  wall=%u  datapath=%u\n", t_qa, d_qa);

    uint32_t xdma_trans = xc[0] + xc[2] + xc[4];  // both transposes + the reshape
    uint32_t xdma_row = xc[4];                    // row-major pays only the reshape
    printf("[Softmax] === SIMD (hart 1):  legacy %u | row-major %u | transposed %u ===\n",
           legacy, rowmaj, trans);
    printf("[Softmax] === xDMA (hart 2):  row-major %u | transposed %u ===\n", xdma_row,
           xdma_trans);
    printf("[Softmax] === both serialised:  row-major %u | transposed %u ===\n",
           rowmaj + xdma_row, trans + xdma_trans);
    // Two bounds, because the two engines run concurrently: if the xDMA work hides behind
    // other SIMD work the comparison is the first line, if nothing overlaps it is the
    // third. The transposed path always wins on hart 1, which is the busier engine.
    printf("[Softmax] === SIMD-only: row-major %u vs transposed %u (%u%% less) ===\n",
           rowmaj, trans, rowmaj ? (100u * (rowmaj - trans)) / rowmaj : 0u);

    // ============================================================ checks
    // The tolerance is on the loose side on purpose: the FP16 chain alone is already
    // sm_chain_ulp off the exact softmax before the device's exp LUT adds anything, and a
    // broken lane map or a wrong scalar is off by hundreds of ULP, not tens.
    const uint32_t TOL = 16u;

    // 1) all three paths against the EXACT softmax. Same tensor, same tolerance, so the
    // three numbers are directly comparable -- and the two that compute their own
    // reciprocal should not be the worse of the three.
    uint16_t *ol = (uint16_t *)out_l;
    uint16_t *orr = (uint16_t *)out_r;
    uint16_t *ot = (uint16_t *)out_t;
    uint32_t lw = 0, lb = 0, rw = 0, rb = 0, sw = 0, sb = 0, checked = 0;
    for (uint32_t t = 0; t < rows; t++)
        for (uint32_t f = 0; f < d; f++) {
            uint16_t want = sm_out_exact[t * d + f];
            if ((want & 0x7C00u) == 0) continue;  // skip the subnormal/zero tail
            checked++;
            uint32_t u = f16_ulp(ol[t * d + f], want);
            if (u > lw) lw = u;
            if (u > TOL) lb++;
            u = f16_ulp(orr[t * d + f], want);
            if (u > rw) rw = u;
            if (u > TOL) rb++;
            // the transposed result is y^T: feature-major, one token per lane
            u = f16_ulp(ot[f * rows + t], want);
            if (u > sw) sw = u;
            if (u > TOL) sb++;
        }
    printf("[Softmax] vs exact (%u significant of %u, tol %u ULP):\n", checked, rows * d,
           TOL);
    printf("[Softmax]   legacy     worst %u ULP (%s)\n", lw, lb ? "FAIL" : "ok");
    printf("[Softmax]   row-major  worst %u ULP (%s)\n", rw, rb ? "FAIL" : "ok");
    printf("[Softmax]   transposed worst %u ULP (%s)\n", sw, sb ? "FAIL" : "ok");
    printf("[Softmax]   for reference, the FP16 chain alone is %u ULP off exact\n",
           sm_chain_ulp);
    if (lb || rb || sb) err++;

    // 2) THE RECIPROCAL THE DEVICE COMPUTED FOR ITSELF. Scored against one the core
    // derives from the device's OWN row sums -- not against a golden, which would fold in
    // the difference between the hardware exp LUT and np.exp and measure the wrong thing.
    // Both paths' sums come out of a TAP beat; the row-major one folds across lanes and
    // the transposed one does not, so this also checks the two agree.
    uint32_t iworst_r = 0, iworst_s = 0, ibad = 0;
    uint16_t *st = (uint16_t *)(ex_t + (uint32_t)(d * BEAT));  // [T] sums, one per lane
    uint16_t *is = (uint16_t *)ex_seed;                        // [T] 1/sum, one per lane
    for (uint32_t r = 0; r < rows; r++) {
        uint16_t sum_row = *(uint16_t *)(ex_p + r * pad_row_b + beats * BEAT);
        uint32_t u = f16_ulp(*(uint16_t *)(rplane + r * pad_row_b), recip_f16(sum_row));
        if (u > iworst_r) iworst_r = u;
        if (u > 3u) ibad++;
        u = f16_ulp(is[r], recip_f16(st[r]));
        if (u > iworst_s) iworst_s = u;
        if (u > 3u) ibad++;
    }
    printf("[Softmax] 1/sum on device vs the core's own divu: row-major %u ULP, "
           "transposed %u ULP (%s)\n",
           iworst_r, iworst_s, ibad ? "FAIL" : "ok");
    if (ibad) err++;

    // 3) the two INT8 outputs. q_r rode the row-major normalise; q_a was narrowed AFTER
    // the A-layout reshape, which is the order the layer actually needs and only works
    // because quantising commutes with a permutation.
    int8_t *qr = (int8_t *)q_r;
    int8_t *qa = (int8_t *)q_a;
    uint32_t qrw = 0, qrb = 0, qaw = 0, qab = 0;
    for (uint32_t t = 0; t < rows; t++)
        for (uint32_t f = 0; f < d; f++) {
            int g = (int)sm_golden_i8[t * d + f];
            int dv = (int)qr[t * d + f] - g;
            uint32_t a = (dv < 0) ? (uint32_t)(-dv) : (uint32_t)dv;
            if (a > qrw) qrw = a;
            if (a > 2u) qrb++;
            dv = (int)qa[a_index(t, f, d, 16, 4)] - g;
            a = (dv < 0) ? (uint32_t)(-dv) : (uint32_t)dv;
            if (a > qaw) qaw = a;
            if (a > 2u) qab++;
        }
    printf("[Softmax] int8 row-major (quant chained)  worst |delta|=%u (%s)\n", qrw,
           qrb ? "FAIL" : "ok");
    printf("[Softmax] int8 A-layout  (quant after reshape) worst |delta|=%u (%s)\n", qaw,
           qab ? "FAIL" : "ok");
    if (qrb || qab) err++;

    // 4) the round trip. y^T transposed back must match the row-major path's own output:
    // same operands through the same FP units, so the only thing that can differ is the
    // ORDER the row sum was accumulated in -- a lane-partial chain transposed against a
    // tree fold row-major -- which moves the last bit of Sexp and nothing else.
    uint32_t tworst = 0, tdiff = 0;
    for (uint32_t i = 0; i < rows * d; i++) {
        uint32_t u = f16_ulp(((uint16_t *)y_p)[i], ((uint16_t *)out_r)[i]);
        if (u > tworst) tworst = u;
        if (u) tdiff++;
    }
    printf("[Softmax] y^T->y vs the row-major result: worst %u ULP (%u/%u differ at all) "
           "(%s)\n",
           tworst, tdiff, rows * d, (tworst > 2u) ? "FAIL" : "ok");
    if (tworst > 2u) err++;

    // 5) the reshape, against an independent statement of the same bijection.
    uint32_t abad = 0;
    for (uint32_t t = 0; t < rows; t++)
        for (uint32_t f = 0; f < d; f++) {
            uint16_t want = ((uint16_t *)y_p)[t * d + f];
            uint16_t got = ((uint16_t *)y_a)[a_index(t, f, d, 16, 4)];
            if (got != want) {
                if (abad < 4)
                    printf("[Softmax] A mismatch (t=%u,f=%u): got %04x want %04x\n", t, f,
                           got, want);
                abad++;
            }
        }
    printf("[Softmax] y->A layout vs the index map: %s (%u/%u differ)\n",
           abad ? "FAIL" : "exact", abad, rows * d);
    if (abad) err++;

    // A hung task cannot fail a check by itself -- every buffer downstream of it simply
    // keeps whatever was there before, which can look like anything -- so it is counted
    // here explicitly rather than left to be inferred from the numbers.
    if (simd_hung) {
        printf("[Softmax] FAIL: a task never retired; every figure above it is meaningless\n");
        err++;
    }
    printf(err ? "[Softmax] FAIL\n" : "[Softmax] PASS\n");
    return err != 0;
}
