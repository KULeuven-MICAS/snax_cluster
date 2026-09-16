// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// FlashAttention inner loop on the four-engine cluster, instrumented for
// hardware utilisation.
//
// WHAT FLASHATTENTION IS. Attention is softmax(Q.K^T / sqrt(d)) . V. Computed literally,
// the whole N x N score matrix has to exist before the softmax can normalise it, and at a
// real sequence length it is far larger than any scratchpad. FlashAttention never
// materialises it. It walks the keys in TILES of Bc, and carries three running quantities
// per query row:
//
//   m   the largest score seen so far       [Br]
//   l   the sum of exp(score - m) so far    [Br]
//   O   the weighted sum of values so far   [Br, d]
//
// One step of the loop, for a fixed query tile and the j-th key/value tile:
//
//     S    = Q . K_j^T              [Br, Bc]   GEMM
//     m_j  = rowmax(S)              [Br]       SIMD
//     m'   = max(m, m_j)            [Br]       the new running maximum
//     corr = exp(m - m')            [Br]       how much the PAST has to shrink
//     P    = exp(S - m')            [Br, Bc]   SIMD
//     l    = corr * l + rowsum(P)   [Br]       rescale, then fold this tile in
//     O    = corr * O               [Br, d]    SIMD
//     O   += P . V_j                [Br, d]    GEMM
//
//   and once, after the last tile:  O = O / l
//
// corr is the whole trick. Every term already in l and O was divided by exp(m_old); if a
// tile raises the maximum, multiplying by exp(m_old - m_new) re-bases all of them onto the
// new one in a single pass. Nothing is approximated, so the tiled answer IS the
// full-matrix answer -- and the N x N score matrix never exists, only [Br, Bc] of it at a
// time.
//
// HOW Q, K AND V ARE TILED. All three are [N, d] for a sequence of length N. Q is cut by
// rows into query tiles of Br; K and V are cut by rows into tiles of Bc. This kernel runs
// ONE query tile against NKV key/value tiles -- the inner loop of the full algorithm, and
// at Bc = 512, NKV = 4 that is 2048 keys attended by 32 queries.
//
//         Q [N, d]              K [N, d]              V [N, d]
//       <-- d = 128 ->        <-- d = 128 ->        <-- d = 128 ->
//       +-------------+       +-------------+       +-------------+  -+-
//  Q_0  |  this tile  | Br=32 |     K_0     |       |     V_0     |  Bc = 512
//       +-------------+       +-------------+       +-------------+  -+-
//  Q_1  |             |       |     K_1     |       |     V_1     |
//       +-------------+       +-------------+       +-------------+
//  ...  |             |  N    |     K_2     |  N    |     V_2     |  N
//       +-------------+       +-------------+       +-------------+
//       |             |       |     K_3     |       |     V_3     |
//       +-------------+       +-------------+       +-------------+
//                               NKV = 4 tiles, streamed one at a time
//
// and the two matmuls that make up one step, with the shape of every intermediate:
//
//         Q_i               K_j^T                  S = Q_i . K_j^T
//       +--------+      +----------------+        +----------------+
//    Br |        |  x   |                | d  =   |                | Br
//       +--------+      +----------------+        +----------------+
//           d                  Bc                        Bc
//
//       P = softmax(S)         V_j                 O_i += P . V_j
//       +----------------+   +--------+           +--------+
//    Br |                | x |        | Bc    +=  |        | Br
//       +----------------+   +--------+           +--------+
//              Bc                d                     d
//
// Everything to this point is the algorithm as it is normally written, and as the figures
// above draw it: query-major, one query row per matrix row. This kernel does not store it
// that way.
//
// THE TRANSPOSED FORM. Transposing both sides of each product rewrites the same two
// matmuls with their axes exchanged. Using (A.B)^T = B^T.A^T:
//
//     S^T = (Q_i . K_j^T)^T = K_j . Q_i^T      [Bc, Br]   was [Br, Bc]
//     P^T = exp(S^T - m)                       [Bc, Br]   m is still per query row
//     O^T = (P . V_j)^T     = V_j^T . P^T      [d,  Br]   was [Br, d]
//
// Same arithmetic on the same operands, every matrix laid down the other way round:
//
//         K_j                Q_i^T               S^T = K_j . Q_i^T
//       +--------+          +-----+               +-----+
//    Bc |        |    x   d |     |    =       Bc |     |
//       +--------+          +-----+               +-----+
//           d                 Br                    Br
//
//        V_j^T                P^T               O^T += V_j^T . P^T
//      +------------+        +-----+              +-----+
//    d |            |   x Bc |     |    +=      d |     |
//      +------------+        +-----+              +-----+
//            Bc                Br                   Br
//
// The rewrite costs NOTHING to obtain: B^T.A^T is the same GEMM with its two operands
// swapped, which is M and N exchanged in params.hjson. No transposer, no extra pass,
// nothing at run time.
//
// WHY IT IS WORTH DOING. The SIMD engine sees memory as a stream of BEATS -- 512 bits =
// 32 FP16 lanes -- and its reductions are LANEWISE: lane k accumulates across successive
// beats, in the accumulator register it already carries. With 4 lanes instead of 32, a
// MAX reduction over three beats looks like this:
//
//         lane     0   1   2   3         lane     0   1   2   3
//     beat 0   [   3   9   1   4 ]   ->   acc [   3   9   1   4 ]
//     beat 1   [   7   2   1   8 ]   ->   acc [   7   9   1   8 ]
//     beat 2   [   5   6   0   2 ]   ->   acc [   7   9   1   8 ]
//                                                 |   |   |   |
//                                            one result per lane, and no lane
//                                            ever looks at its neighbour
//
// One accumulator per lane, all of them updated every cycle, so the reduction takes
// exactly as long as the stream does: three beats in, one beat out, nothing at the end.
//
// Reducing ACROSS the 32 lanes of one beat -- max(3, 9, 1, 4) in the picture above -- is a
// different and far more expensive thing: a log-depth fold through treeBuf, serialised,
// then a scalar drain. Softmax reduces per QUERY ROW, so the layout alone decides which of
// the two the rowmax and the rowsum get:
//
// Take a tiny tile -- Br = 4 query rows, Bc = 8 keys, 4 lanes to a beat. S(q,k) is the
// score of query q against key k, so the tile is 4 x 8 = 32 scores. What the softmax
// needs from it is ONE MAXIMUM PER QUERY ROW:
//
//     rowmax(q0) = max( S(q0,k0), S(q0,k1), ... , S(q0,k7) )     one number
//     rowmax(q1) = max( S(q1,k0), S(q1,k1), ... , S(q1,k7) )     one number
//     rowmax(q2), rowmax(q3)                                     Br = 4 in all
//
// Those Br numbers are the m of the recurrence above -- what gets subtracted before the
// exponential. rowsum works identically, summing instead of maximising. The 32 scores
// are the same in both layouts below; only the order they lie in memory changes, and
// with it how much that reduction costs.
//
//   S stored [Br, Bc] -- each QUERY ROW contiguous
//
//     beat 0 [ S(q0,k0) S(q0,k1) S(q0,k2) S(q0,k3) ]   ] q0's row is spread over two
//     beat 1 [ S(q0,k4) S(q0,k5) S(q0,k6) S(q0,k7) ]   ]   beats AND across the lanes
//     beat 2 [ S(q1,k0) S(q1,k1) S(q1,k2) S(q1,k3) ]
//     ...                                            8 beats for the tile
//
//     rowmax(q0) = max( beat 0 lanes 0-3 , beat 1 lanes 0-3 )
//
//                  Two beats, and the four values inside each of them must be combined
//                  with one another. Combining lanes WITHIN a beat is the horizontal
//                  FOLD: a log-depth tree, then a scalar drain -- and it has to run once
//                  per query row, so Br times over the tile.
//
//   S^T stored [Bc, Br] -- each KEY contiguous                        <- this kernel
//
//     beat 0 [ S(q0,k0) S(q1,k0) S(q2,k0) S(q3,k0) ]   key 0
//     beat 1 [ S(q0,k1) S(q1,k1) S(q2,k1) S(q3,k1) ]   key 1
//     ...                                            one beat per key, Bc of them
//     beat 7 [ S(q0,k7) S(q1,k7) S(q2,k7) S(q3,k7) ]   key 7
//              lane 0    lane 1    lane 2    lane 3
//              = q0      = q1      = q2      = q3
//
//     rowmax(q0) = max( lane 0 of beat 0, lane 0 of beat 1, ... , lane 0 of beat 7 )
//
//                  which is just lane 0's accumulator once the last beat has gone by --
//                  exactly the MAX example above. No lane is ever combined with another,
//                  so there is no fold, and rowmax(q1), rowmax(q2), rowmax(q3) come out
//                  of lanes 1, 2, 3 in the very same pass.
//
// The second layout reduces in the accumulator the engine carries anyway, so the
// rowmax costs nothing beyond reading the tile. The first costs Br folds, each
// serialised behind its own drain. A free rewrite buys the whole difference.
//
// Everything downstream inherits the orientation. m is ONE beat (32 lanes = 32 query
// rows), not Br scalars, so S - m is a single broadcast beat rather than Br separately
// armed scalar broadcasts; corr, l and the O rescale are all the same shape.
//
// This kernel also stops before the final O / l.
//
// That is also why it is the workload for this cluster. The per-tile epilogue is nothing
// but reductions and pointwise transcendentals, which is exactly what the SIMD datapath
// extensions do, and it runs while the GEMM is already busy with the next tile.
//
// (The 1/sqrt(d) score scaling is not applied here. The operands are bounded instead, so
// the scores stay inside FP16 without a scaling pass; see QSHIFT in data.h.)
//
// THE TAP, the reduce's other trick. A reduce normally SWALLOWS its input: N beats go
// in, one comes out. Softmax needs both halves, though -- the maximum AND the tile it
// came from, because the very next step subtracts one from the other. Reducing the
// plain way throws the tile away and the next task has to read it all over again:
//
//     beat 0 [   3   9   1   4 ]
//     beat 1 [   7   2   1   8 ]   ->   [   7   9   1   8 ]   the tile is gone
//     beat 2 [   5   6   0   2 ]
//
// TAP mode passes the input straight through and APPENDS the result, so the same
// single pass delivers both -- 3 beats in, 4 out:
//
//     beat 0 [   3   9   1   4 ]   ->   [   3   9   1   4 ]
//     beat 1 [   7   2   1   8 ]   ->   [   7   2   1   8 ]
//     beat 2 [   5   6   0   2 ]   ->   [   5   6   0   2 ]
//                                       [   7   9   1   8 ]  <- the reduction, appended
//
// At Bc = 512 that is the difference between reading the score tile once and reading
// it twice. The epilogue below uses it: that pass must emit the quantised tile AND the
// row sum of it, and the tap is what lets one sweep produce both.
//
// The rowmax pass does NOT tap. It only needs the statistic, and the tile it read is
// still sitting in the GEMM's own output buffer where the epilogue can read it again --
// so it reduces 512 beats down to one and writes nothing else. Each S^T buffer carries
// a spare beat in front for exactly that reason: the epilogue is a sticky-B pass, its
// broadcast operand has to sit immediately before the tile, and that beat is where
// -m16 goes. A tap would have had to COPY the whole tile to put it behind a latch.
//
// ONE KV TILE. Br = 32 query rows, Bc = 512 keys, d = 128, NKV = 4 tiles. Bc and d are
// INDEPENDENT, so the two matmuls have different shapes; gemm_set_shape() switches between
// them per dispatch, rewriting only the ten CSRs that depend on M and K. A beat is
// 512 bits = 64 B = 32 FP16 lanes = one value per query row.
//
// That last equality is not free, and it is worth knowing where it comes from. The array
// is 16 columns wide, so it emits the tile as [M][N][meshRow][meshCol] blocks -- 16
// queries at a time, with the two N halves of a query row in blocks 256 B apart. Written
// contiguously, a 64 B beat would be 16 queries x 2 KEYS, and a LANEWISE reduce down it
// would mix two keys and (since n alternates every 8 beats) two different queries per
// lane. The D port's spatial map is what fixes this: its 32 channels are grouped [4, 8],
// so four channels lay down one key's 16 scores and the groups step by a whole 32-query
// key row, interleaving the two N halves in memory at no cost. A beat is then exactly one
// key, all Br queries -- which is the premise everything below rests on.
//
//   Every name carries its PRECISION: 8 = INT8, 16 = FP16, 32 = INT32. Beat counts are
//   in -> out, and a beat is one task's worth of 32 lanes.
//
//   GEMM (hart 0)  S16^T = K8 . Q8^T        K8 [512,128]  64 KiB  ]  2.10 M MAC
//                                         Q8^T [128, 32]   4 KiB  ]  2048 cycles
//                                     -> S16^T [512, 32]  32 KiB     at 1024 MAC/cycle
//                     the mesh computes INT32, and Int32ToFp16 on the D32 writer port
//                     converts it FREE as the tile drains -- which also HALVES the beats
//                     written, so S16^T reaches TCDM as 512 FP16 beats, not 1024
//   SIMD (hart 1), in the order the loop fires them:
//
//                  rmax16 = rowmax(S16^T)        512 ->   1  StreamReduce MAX|LANEWISE
//                  m16    = max(m16_old, rmax16)     2 -> 1  StreamReduce MAX|LANEWISE
//                  -m16   = negate m16               2 -> 2  StreamMap LINEAR, a = -1
//
//                       [ -m16 ][ S16^T  x512 ]        [ -m16 ][ m16_old ]
//                         latch   the GEMM's buffer       rmax    mrun
//                         \___ the fused pass latches     \___ reduced as a pair to
//                              -m16, then adds it to            delta below
//                              all 512 beats of the tile
//
//                     ONE value at TWO addresses. Operands here are paired by ADJACENCY
//                     -- a task reads one flat stream and pairs whatever is next to each
//                     other in it -- and -m16 has two different partners, so one copy
//                     cannot serve both. The task reads mnew TWICE at stride 0 and
//                     writes both: 2 in, 2 out, one fill+drain instead of two.
//
//                  delta  = m16_old - m16            2 -> 1  StreamReduce ADD|LANEWISE
//                  corr16 = exp(delta)               2 -> 2  StreamMap EXP
//
//                       [ corr16 ][ l16_old ]        [ corr16 ][ O16^T  x128 ]
//                         corrL     lrun               corrO     oacc
//                         \___ sticky MUL gives         \___ sticky MUL rescales
//                              lsc16, below                  all 128 beats of O
//
//                     The same fan-out for the same reason: corr16 pairs with l16_old
//                     in one task and with the O tile in another.
//
//                  P8^T   = int8(exp(S16^T-m16)) 513 -> 257  ) FUSED: EW0 (sticky ADD)
//                  sum16  = sum of P16 over keys             ) -> Map (EXP) -> Reduce TAP
//                                                            ) -> Fp16ToInt8 (tail passthrough)
//
//                     THE WHOLE EPILOGUE IS ONE SWEEP. sum16 has no task of its own: it is
//                     the TAP beat, written after the tile. And the quantiser runs in the
//                     same pass because its tail passthrough leaves that trailing beat
//                     alone -- without it the scalar would be narrowed along with the data,
//                     which is why P had to be written in FP16 and read back by a separate
//                     quantise task. That third trip over the tile is gone, and so is the
//                     FP16 P buffer: 512 beats written + 512 read per tile.
//
//                       read   [ -m16 ][ S16^T  x512 ]     the latch, then the tile
//                       write  [ P8^T  x256 ][ sum16 ]     into THIS tile's p8 buffer
//                                              \___ lsc16 is written beside it later,
//                                                   so the l16 row reads the adjacent
//                                                   pair [ sum16 ][ lsc16 ]. Both follow
//                                                   the p8 ping-pong now.
//
//                  O16^T *= corr16               129 -> 128  StreamElementwise sticky MUL
//
//                     ---- P8^T and the rescaled O16^T are published to the GEMM here.
//                          Everything below only prepares the NEXT tile, so the GEMM
//                          never waits on it. ----
//
//                  lsc16  = corr16 * l16_old         2 -> 1  StreamElementwise sticky MUL
//                  l16    = sum16 + lsc16            2 -> 1  StreamReduce ADD|LANEWISE
//                  m16_old, l16_old <- m16, l16      2 -> 2  StreamMap LINEAR, a = 1
//
//                     The "commit". Old and new live in SEPARATE beats all tile long,
//                     because delta needs m16_old after m16 already exists, and lsc16
//                     needs l16_old after l16 is being formed -- so nothing may
//                     overwrite the old pair until both readers are done. The last task
//                     of the tile copies the new pair over the old one, which is all
//                     that carries state into tile j+1:
//
//                       mnew -> mrun        m16 becomes the next tile's m16_old
//                       lnew -> lrun        l16 becomes the next tile's l16_old
//
//                     An identity StreamMap (a = 1, b = 0) is how a copy is expressed
//                     here -- the core has no cheaper way to move TCDM beats -- and one
//                     strided 2-beat task does both, the same trick as the fan-outs
//                     above run in reverse: two sources, two destinations, one pass.
//
//   GEMM (hart 0)  O32^T += V8^T . P8^T    V8^T [128,512]  64 KiB  ]  2.10 M MAC
//                                          P8^T [512, 32]  16 KiB  ]  2048 cycles
//                                     -> O32^T [128, 32]  16 KiB     accumulated in place
//
//   O16^T and O32^T are the two representations of O described below: the SIMD rescales
//   the FP16 copy, the GEMM accumulates the INT32 one, and they are never joined.
//
// Per KV tile that is 4.19 M MAC and 4096 GEMM cycles against ~1170 beats read and ~400
// written on the SIMD side, so the arithmetic floor is GEMM-bound and the softmax has to
// fit inside its shadow. Neither engine is short of time; TCDM BANDWIDTH is what the two
// of them contend for, which is why the beats written matter as much as the cycles.
// The whole run is NKV of these. Every buffer above lives in TCDM at once; the footprint
// guard in main() reports the total against the 512 kB budget.
//
// TWO IDEAS CARRY THIS KERNEL.
//
// 1. THE OVERLAP. Two engines on two harts, and the reason FlashAttention is the
//    driving workload is that its softmax can run on the SIMD engine while the
//    GEMM works on the next KV tile. The loop is software-pipelined over NKV
//    key/value tiles:
//
//      GEMM core:  S(0) ; then per j:  S(j+1)  ||  <SIMD does tile j>  ; O(j)
//      SIMD core:  per j: wait for S(j), softmax it, publish P8(j)
//
//    S and P8 are double-buffered so producer and consumer never touch the same
//    memory, and the handoff is two monotonically increasing counters in TCDM
//    rather than snrt_cluster_hw_barrier(), which would re-serialise the two
//    cores and drag harts 2 and 3 in. Within a tile the SIMD tasks are fired
//    back to back with no wait between them; see the note in the loop.
//
// 2. THE TRANSPOSE. Computing in [Bc, Br] puts one query row in every lane, which
//    turns the softmax reductions into the lanewise accumulation StreamReduce does
//    for free, and costs nothing to obtain. Argued in full above.
//
// WHAT IS AND IS NOT VALIDATED. The softmax is checked twice over, by two kinds of
// test that fail in different ways.
//
// Exact invariants, which hold whatever the data is: P contains a bit-exact 1.0 and
// P8^T a bit-exact 1 in every query row (the row's own maximum subtracts to zero, and
// exp(0) = 1), and l carries NKV identical row sums. Between them they pin the reduce,
// the subtract, the exponential, the quantiser, the tap and the l recurrence.
//
// A numerical golden, which pins the VALUES those invariants cannot see: m, the whole
// row sum, and P away from its fixed point, against a float model of the same tile in
// datagen.py, compared as a ULP distance on the FP16 bit patterns. The invariants say
// the pipeline is wired correctly; the golden says the arithmetic is accurate.
//
// Every KV tile is fed the same K, so the final tile stands for all of them.
//
// O is the exception. The GEMM accumulates P.V into oacc32 in INT32 in place, and the
// SIMD rescales an FP16 copy by corr each tile -- both halves of the online update are
// present and correctly sized, but they are never joined, because closing the loop
// needs an FP16 -> INT32 conversion the datapath does not have. So O is shape- and
// dataflow-accurate and nothing checks its value: treat it as a cycle measurement, one
// P.V matmul per KV tile, which is the right load -- not as a result.

#include <stdint.h>
#include "data.h"
#include "snax-core-roles.h"
#include "snax-simd-lib.h"
#include "snax-versacore-to-lib.h"
#include "snrt.h"

// The matmul engine on hart 0 is VersaCore: a 1024-MAC INT8 array whose single spatial
// unrolling (Mu, Ku, Nu) is what this kernel calls meshRow/tileSize/meshCol = 16/4/16. It
// has ONE output port, carrying INT32 with an optional convert to FP16 on the way out, and
// no rescale unit -- this kernel needs neither a quantised output nor a rescale.
//
// meshRow, tileSize, meshCol, BR, BC, DHEAD, NKV and QSHIFT all arrive from data.h. They
// are derived there from the same M/N/K and array shape that produced the streamer
// descriptors, so the kernel's idea of the tile and the descriptors' idea of it cannot
// drift apart. Change them in data/params.hjson.

// BC and DHEAD are independent. Bc is the tiling knob, free to grow until TCDM is
// full; d is a property of the model. The two matmuls therefore have different
// shapes, which gemm_set_shape() switches between per dispatch.

#define SBEATS BC        // S^T / P beats: one per KEY, 32 query lanes each
#define DBEATS DHEAD     // O^T beats:     one per HEAD element, 32 query lanes each
#define PBEATS (BC / 2)  // after Fp16ToInt8 halves them

// Sources for the iDMA's initialisation pass. Clearing what a query tile accumulates into
// is per-query-tile work, so it belongs on the DMA engine and inside the measured window --
// as scalar loops it was ~40,000 stores that appeared in no figure at all.
static const int32_t fa_zero[BR * DHEAD] = {0};

// WHERE THE RESULT LANDS. Its own array, not a data.h buffer: writing (O, m, l) past the
// end of C put the state block straight on top of o32_golden, and the check then compared
// against a golden the kernel had just overwritten. The O check caught it; nothing else
// would have.
static int32_t fa_out[BR * DHEAD + 128];
static const uint16_t fa_minf[SIMD_BEAT_BYTES / 2] = {
    0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu,
    0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu,
    0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu,
    0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu, 0xFBFFu
};

// A spin that cannot hang the simulation for ever. The two cores are hand-synchronised,
// so a deadlock is possible, and an infinite loop in Verilator burns wall-clock with no
// diagnosis.
#define SPIN_LIMIT 2000000u
#define SNAX_SPIN_UNTIL(cond, timeout_flag)          \
    do {                                             \
        uint32_t spins__ = 0;                        \
        while (!(cond)) {                            \
            if (++spins__ > SPIN_LIMIT) {            \
                (timeout_flag)++;                    \
                break;                               \
            }                                        \
        }                                            \
    } while (0)

// The task geometries. In TCDM, not on the stack and not in .bss: program_fast() reads
// every field of two of these per task, and .bss maps to DRAM on this target, so each
// field would cost an L3 round trip. Only the SIMD core touches them.
SNRT_L1_DATA static snax_simd_shape_t shapes[32];

// SNRT_L1_DATA lives in .l1, which is NOLOAD -- it is NOT zero-initialised. A shape that
// is declared but never filled programs the AGU from whatever was in TCDM, and the task
// still completes: it reads and writes the wrong addresses, silently wrong rather than
// faulted. Clear them once, so an unfilled shape is an empty task the library REFUSES.
static inline void snax_simd_shapes_clear(snax_simd_shape_t *sh, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint8_t *b = (uint8_t *)&sh[i];
        for (uint32_t k = 0; k < sizeof(snax_simd_shape_t); k++) b[k] = 0;
    }
}

static inline uint16_t fp16_at(volatile uint8_t *p, uint32_t i) {
    return ((volatile uint16_t *)p)[i];
}

// How far apart two FP16 values are, in representable steps. The bit pattern of a
// non-negative float is already monotone as an integer; reflecting the negatives about
// zero extends that across the whole range, and the difference is then a ULP distance.
// Integer-only, which matters because the core running the check has no FPU, and it
// degrades gracefully: a value one step off scores 1 rather than just "not equal".
#define GOLD_ULP_M 1    // m is a max of converted integers: expected exact
#define GOLD_ULP_P 2    // P is a LUT exponential, ~1 ULP
#define GOLD_ULP_SUM 2  // rowsum accumulates in FP32 and narrows once

static inline int32_t fp16_order(uint16_t h) {
    return (h & 0x8000u) ? -(int32_t)(h & 0x7FFFu) : (int32_t)h;
}

static inline uint32_t fp16_ulp(uint16_t a, uint16_t b) {
    int32_t d = fp16_order(a) - fp16_order(b);
    return (uint32_t)(d < 0 ? -d : d);
}

// Re-point an already-configured streamer at new buffers without touching its shape:
// four csrw against the ~60 of gemm_configure_once(). Pass -1 to leave a pointer alone.
//
// always_inline so the compile-time-constant CSR address propagates into the csrw_ss
// switch and folds to a single direct `csrw <imm>`. Out of line the address is opaque and
// every access pays a jump-table load from L2 plus an indirect jump -- measured, on this
// core, to be the dominant cost of accelerator configuration.
__attribute__((always_inline)) static inline void gemm_set_bases(
    uint32_t a, uint32_t b, uint32_t c, uint32_t d32) {
    csrw_ss(BASE_PTR_READER_0_LOW, a);
    csrw_ss(BASE_PTR_READER_1_LOW, b);
    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, c);
    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, d32);
}

__attribute__((always_inline)) static inline void gemm_launch(void) {
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(GEMMX_START, 1);
}

// Wait for the ACCELERATOR TO START before waiting for it to finish.
//
// snax-versacore-to-lib's wait_versacore_and_streamer() polls busy straight after two
// STREAMER_START writes. That is only safe while those writes are SLOW: out of line each
// goes through the csrw_ss jump table and costs ~20-30 cycles, which is just enough for
// busy to rise before the first poll. Inlined they collapse to 2 cycles, the first poll
// reads busy = 0 on a task that has not started, and the wait returns at once -- so the
// caller reads a partial performance counter and reconfigures the engine out from under a
// running matmul. It is invisible on short tasks and shows up only as a large tile
// reporting FEWER cycles than its own arithmetic floor.
//
// The rise-wait is bounded so a task that completes before we look cannot hang us: if
// busy never rises, either it already finished (the fall-waits exit at once, which is
// correct) or the engine was never started, which the caller's own timeout catches.
//
// Returns how many rise-polls it burned. A csrr is 5.1 cycles, so two per iteration is
// ~10 cycles of pure latency per iteration, charged to "config" in the report -- this is
// the number that says how much of that line a combinational busy would reclaim.
// Deassert the start pulse. Separate from the wait below so the caller can stage
// the NEXT dispatch's config in between: once STREAMER_START has fired, the streamer
// has snapshotted the whole CSR bank into csrCfgReg (Streamer.scala:377-386) and
// VersaCore into csrReg (VersaCore.scala:168-183), so the running task no longer
// reads the bank and the bank is free to be overwritten. ReqRspManager only throttles
// writes to the START address itself (:193-205); every other CSR write is accepted at
// one per cycle regardless of busy.
__attribute__((always_inline)) static inline void gemm_ack(void) {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
}

__attribute__((always_inline)) static inline uint32_t gemm_wait(void) {
    uint32_t g = 0;
    for (; g < 64u; g++) {
        if (csrr_ss(GEMMX_BUSY) || csrr_ss(STREAMER_BUSY_CSR)) break;
    }
    while (csrr_ss(GEMMX_BUSY)) {
    }
    while (csrr_ss(STREAMER_BUSY_CSR)) {
    }
    csrw_ss(GEMMX_START, 0);
    return g;
}

// The FULL streamer programming: bounds, strides, remap indices and the accelerator CSRs.
// Called ONCE, for shape 1. Everything that differs between the two matmul shapes is then
// patched per dispatch by gemm_set_shape() -- ten registers instead of these ~60.
//
// Written out here rather than handed to snax-versacore-to-lib's
// set_versacore_streamer_csr(), because this app patches a subset of the descriptor per
// dispatch (see gemm_set_shape() and gemm_d32_emit_fp16()) and needs the halved D32 shape
// the converter implies. Note the extension window: the D write path here carries the
// INT32->FP16 converter ALONE, so READER_WRITER_EXTENSION_1_CSR_NUM is 2 -- the enable
// bitmask and the extra-loop policy -- not the 7 of a cluster that also stacks a dynamic
// rescale unit, and the converter is enable bit 0 rather than bit 1. The
// generated streamer_csr_addr_map.h is the authority on what exists, and the writes
// below follow it.
//
// Not for per-dispatch use: every CSR address here is a run-time value to the csrw_ss
// switch, so each access pays a jump-table load plus an indirect jump.
//
// The descriptor arrays live in THIS frame rather than main's: SNRT_LOG2_STACK_SIZE is 10,
// so a hart has 1 KiB, and carrying these alongside main's shape structs and a printf
// frame overflows into the neighbouring hart's stack.
static void gemm_configure_once(void) {
    int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2, Atlbound3, Atlbound4, Atlbound5};
    int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2, Atlstride3, Atlstride4, Atlstride5};
    int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
    int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};
    // TWO spatial strides: the C and D32 ports declare spatial_bounds [[8, 4]] and the
    // streamer reads S_STRIDE_NUM_READER_WRITER_* = 2 of them. A 1-element array would
    // feed the second from off the end of the stack.
    int32_t Cslstride[] = {Cslstride0, Cslstride1};
    int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2};
    int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2};
    int32_t D32slstride[] = {D32slstride0, D32slstride1};
    int32_t D32tlbound[] = {D32tlbound0, D32tlbound1, D32tlbound2};
    int32_t D32tlstride[] = {D32tlstride0, D32tlstride1, D32tlstride2};

    {
        uint32_t l1b = (uint32_t)snrt_l1_next();
        gemm_set_bases(l1b + (uint32_t)delta_local_a, l1b + (uint32_t)delta_local_b,
                       l1b + (uint32_t)delta_local_c, l1b + (uint32_t)delta_local_d32);
    }

    // A -- reader 0. Six temporal dimensions; the descriptor uses three and pads the rest
    // with bound 1, stride 0.
    csrw_ss(S_STRIDE_READER_0_0, Aslstride0);
    for (int i = 0; i < T_BOUND_NUM_READER_0; i++) {
        csrw_ss(T_BOUND_BASE_READER_0 + i, Atlbound[i]);
        csrw_ss(T_STRIDE_BASE_READER_0 + i, Atlstride[i]);
    }
    csrw_ss(ADDR_REMAP_INDEX_READER_0, set_addr_remap_index_A);

    // B -- reader 1.
    csrw_ss(S_STRIDE_READER_1_0, Bslstride0);
    for (int i = 0; i < T_BOUND_NUM_READER_1; i++) {
        csrw_ss(T_BOUND_BASE_READER_1 + i, Btlbound[i]);
        csrw_ss(T_STRIDE_BASE_READER_1 + i, Btlstride[i]);
    }
    csrw_ss(ADDR_REMAP_INDEX_READER_1, set_addr_remap_index_B);

    // C -- the READ half of the one bidirectional port.
    for (int i = 0; i < S_STRIDE_NUM_READER_WRITER_0; i++)
        csrw_ss(S_STRIDE_BASE_READER_WRITER_0 + i, Cslstride[i]);
    for (int i = 0; i < T_BOUND_NUM_READER_WRITER_0; i++) {
        csrw_ss(T_BOUND_BASE_READER_WRITER_0 + i, Ctlbound[i]);
        csrw_ss(T_STRIDE_BASE_READER_WRITER_0 + i, Ctlstride[i]);
    }
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, set_addr_remap_index_C);
    // C is the only port with a channel mask (configurable_channel = 1). All 32 channels
    // on: attention reads a full C, never a broadcast one.
    for (int i = 0; i < ENABLED_CHANNEL_READER_WRITER_0_CSR_NUM; i++)
        csrw_ss(ENABLED_CHANNEL_READER_WRITER_0 + i, channel_en_C[i]);

    // D32 -- the WRITE half of that same port.
    for (int i = 0; i < S_STRIDE_NUM_READER_WRITER_1; i++)
        csrw_ss(S_STRIDE_BASE_READER_WRITER_1 + i, D32slstride[i]);
    for (int i = 0; i < T_BOUND_NUM_READER_WRITER_1; i++) {
        csrw_ss(T_BOUND_BASE_READER_WRITER_1 + i, D32tlbound[i]);
        csrw_ss(T_STRIDE_BASE_READER_WRITER_1 + i, D32tlstride[i]);
    }
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, set_addr_remap_index_D32);

    // The accelerator itself. take_in_new_c = 1: every output block starts from C, which
    // is what makes the second matmul's O += P.V accumulation free (see the loop).
    // The array-shape and data-type CSRs select among VersaCore's runtime-selectable
    // unrollings and operand formats; this cluster declares one of each.
    set_versacore_csr(1, K, M * N, gen_subtraction_config(0, 0), array_shape,
                      data_type);
}

// Switch the GEMM between the two matmul shapes.
//
//   S^T = K.Q^T    M1 = Bc/meshRow, K1 = d/tileSize
//   O^T = V^T.P^T  M2 = d/meshRow,  K2 = Bc/tileSize
//
// Only M and K move, so only the CSRs that depend on them are rewritten: eight stream
// registers plus the two VersaCore bounds, against the ~60 gemm_configure_once() writes.
// Every address here is a compile-time constant, so each store folds through the
// always_inline csrw_ss to a single `csrw <imm>`.
//
// VersaCore states the matmul as two counts rather than as M/N/K. Output-stationary, one
// dispatch is: hold an Mu x Nu block in the accumulator, stream ACCUM_BOUND pairs of A
// and B tiles through it, retire, repeat OUTPUT_BOUND times. So ACCUM_BOUND is the
// contraction depth K and OUTPUT_BOUND is the M*N output blocks. N never moves between
// the two shapes, so neither does anything derived from it alone.
#if ENABLED_CHANNEL_READER_WRITER_0_CSR_NUM != 1
#error "C channel mask spans more than one CSR -- unroll the write in gemm_set_shape()"
#endif

//
// EVERY VALUE ARRIVES AS AN ARGUMENT, none is read from data.h. M, N, K, Atlstride2,
// Btlstride1 and delta_local_* are plain `int32_t` globals -- not const, so the compiler
// cannot fold them -- and .bss maps to DRAM on this target. Reading them here cost an L3
// round trip per field per dispatch: 134 cycles for 20 csrw, against the 1.0 cycle a
// folded `csrw imm` takes. The caller hoists them into locals once, outside the loop.
// gemm_d32_emit_fp16() below already worked this way; this is the same rule applied to
// the rest of the descriptor.
__attribute__((always_inline)) static inline void gemm_set_shape(
    int32_t k, int32_t m, int32_t blocks, int32_t as2, int32_t bs1, uint32_t c_chan) {
    // QK READS NO C. S^T = K.Q^T is a fresh product: the DM core zeroed C and nothing
    // accumulates into it, so the 64 KiB of INT32 zeros it fetches per tile are pure
    // traffic on the one port A and B are also being fed from.
    //
    // Switching the C reader's channels off is what removes it. A disabled channel still
    // pops its address and still presents a beat -- it just presents ZERO and issues no
    // TCDM request (see DataResponser). That is exactly the C this matmul wants, so the
    // array keeps taking C normally and the accumulator is still reset by it; only the
    // memory traffic disappears. take_in_new_c must therefore stay 1: at 0 the array
    // would stop draining the reader, whose beats keep coming, and the depth-1 C FIFO
    // would fill and never report empty.
    //
    // O^T = V^T.P^T is the opposite. C and D32 both point at oacc32, and reading C back
    // is exactly how O accumulates across KV tiles, so it needs every channel.
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_0, c_chan);

    csrw_ss(ACCUM_BOUND, k);               // A,B tile pairs folded into one output block
    csrw_ss(OUTPUT_BOUND, blocks);         // output blocks this dispatch retires

    csrw_ss(T_BOUND_READER_0_0, k);        // A stream
    csrw_ss(T_BOUND_READER_0_2, m);
    csrw_ss(T_STRIDE_READER_0_2, as2);
    csrw_ss(T_BOUND_READER_1_0, k);        // B stream
    csrw_ss(T_BOUND_READER_1_2, m);
    csrw_ss(T_STRIDE_READER_1_1, bs1);
    csrw_ss(T_BOUND_READER_WRITER_0_2, m); // C stream
    csrw_ss(T_BOUND_READER_WRITER_1_2, m); // D32 stream
}

// The D32 output port carries an Int32ToFp16Converter. Arm it for a matmul whose
// result is CONSUMED as floating point; disarm it for one that ACCUMULATES in place, since
// such a matmul reads its own previous output back through C as INT32.
//
// Arming also halves the beats the writer emits -- two INT32 beats merge into one FP16 beat
// -- so the D32 descriptor must halve with it, or the writer waits for beats that never
// arrive. Which parts halve is NOT uniform, because the two N blocks interleave (see the
// port's [4, 8] spatial map in the cluster cfg):
//
//   bound0   halves   a transaction still spans 512 B, but covers twice the key rows
//   stride2  halves   an M block is 16 key rows, and a key row is half as many bytes
//   stride1  does NOT halve
//
// stride1 is the offset of the OTHER N block inside a key row, which is meshCol elements
// of the type being written -- and the interleave is laid out on the FP16 row, so it is
// the same 32 B in both modes. Halving it would drop the second N block on top of the
// first.
//
// The descriptor values come in as ARGUMENTS rather than read from the D32tl* globals.
// Those globals live in .bss, which this target maps to DRAM, so reading them here would
// cost an L3 round trip per field per dispatch. The caller hoists them into locals once,
// where they stay in registers.
static inline void gemm_d32_emit_fp16(int on, uint32_t bound0, uint32_t stride1,
                                      uint32_t stride2) {
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 0, on ? 1u : 0u);  // enable bitmask
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 0u);            // extra_loops index 0 => 2:1
    csrw_ss(T_BOUND_READER_WRITER_1_0, bound0);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, stride1);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, stride2);
}

int main() {
    uint8_t *l1 = (uint8_t *)snrt_l1_next();
    int8_t *local_a = (int8_t *)(l1 + delta_local_a);
    int8_t *local_b = (int8_t *)(l1 + delta_local_b);
    int32_t *local_c = (int32_t *)(l1 + delta_local_c);

    // Scratch, laid out at run time from where data.h ends. ADJACENCY IS THE LAYOUT
    // RULE: a sticky-B task reads one flat stream whose FIRST beat is the latched
    // operand, so every broadcast operand must sit immediately before its data, and
    // a LANEWISE 2-beat reduce needs its two operands adjacent. Each pair below is
    // annotated with the task it serves; moving one silently feeds a task the wrong
    // beat rather than faulting.
    uint32_t top = ((uint32_t)delta_local_d32 + BR * BC * 2 + 63u) & ~63u;
    const uint32_t BEAT = SIMD_BEAT_BYTES;

    // THE SCORE TILE IS READ WHERE THE GEMM WROTE IT. Each S^T buffer carries a spare
    // beat in FRONT of the tile, and that beat is the sticky-B latch the epilogue needs
    // immediately before its data -- so the tile never has to be copied to sit behind
    // one, and the rowmax is a plain lanewise reduce: SBEATS in, ONE out.
    //
    // rmax is allocated ABOVE both buffers on purpose. The -m_new fan-out below writes
    // the latch and the rmax slot as one 2-beat strided task, and the AGU's stride is
    // unsigned -- a destination below its partner wraps the address space.
    uint32_t s_a   = top;       top += (1 + SBEATS) * BEAT;  // [ -m_new ][ S^T x Bc ]
    uint32_t s_b   = top;       top += (1 + SBEATS) * BEAT;  // the other half of the ping-pong
    uint8_t *rmax  = l1 + top;  top += BEAT;          // rowmax, and later -m_new
    uint8_t *mrun  = l1 + top;  top += BEAT;          // running m   ] pair for max(m_old,rowmax)
    uint8_t *mnew  = l1 + top;  top += BEAT;
    uint8_t *delta = l1 + top;  top += BEAT;          // m_old - m_new
    uint8_t *corrL = l1 + top;  top += BEAT;          // exp(delta)  ] latch for corr*l_old
    uint8_t *lrun  = l1 + top;  top += BEAT;          // running l   ]
    uint8_t *lnew  = l1 + top;  top += BEAT;
    // No FP16 P buffer: the epilogue quantises the tile in the sweep that produces it,
    // so P exists only as INT8, inside the P8 buffers below. The tapped rowsum and
    // corr*l_old live there too -- see the P8 allocation.
    uint8_t *corrO = l1 + top;  top += BEAT;          // exp(delta)  ] latch for O *= corr
    uint8_t *oacc  = l1 + top;  top += DBEATS * BEAT; // O^T = [d, Br] ]
    uint32_t oacc32 = top;      top += BR * DHEAD * 4; // O^T as the GEMM sees it: INT32
    // P8 CARRIES ITS OWN TAIL: [ P8 x PBEATS ][ rowsum ][ corr*l_old ].
    //
    // The epilogue pass narrows the tile and emits the tapped rowsum in ONE sweep
    // (Fp16ToInt8 tailPassthrough lets the reduce's trailing FP16 beat past the
    // quantiser), so the rowsum lands wherever the writer's flat stream puts it --
    // immediately after the 256 INT8 beats. corr*l_old is allocated right behind it so
    // task 13 still reads the adjacent pair a LANEWISE 2-beat reduce needs. Both are
    // per-tile now, because the P8 buffer they sit in ping-pongs.
    uint32_t p8_0  = top;       top += (PBEATS + 2) * BEAT;
    uint32_t p8_1  = top;       top += (PBEATS + 2) * BEAT;
    // O^T lives in TWO representations, and they are deliberately not joined.
    //
    // The GEMM accumulates P.V in INT32 in `oacc32`, in place (C and D32 both point
    // there, so the matmul computes O += P.V natively). The SIMD rescales an FP16 `oacc`
    // by corr each tile, which is the other half of the online update.
    //
    // Combining them would need FP16 <-> INT32 conversions in both directions each tile
    // -- two more tile-sized passes -- and the datapath has Int32ToFp16 but no
    // Fp16ToInt32. So the two halves are cycle-accurate and structurally correct but NOT
    // numerically joined: O remains, as the header says, a cycle measurement. The
    // softmax statistics (m, l, P) ARE numerically checked, and they are what the
    // invariants below pin down.
    // K AND V STREAM. Every KV tile pulls a fresh 64 KiB K and a fresh 64 KiB V from main
    // memory, double-buffered so the load of tile j+1 overlaps the compute of tile j.
    //
    // The two have DIFFERENT lifetimes, and that is what makes two buffers each enough:
    // K(j) dies at the end of QK(j), V(j) only at the end of PV(j). Freeing them on
    // separate counters gives the iDMA three dispatches of slack on each; freeing both on
    // the later one would put every load straight onto the critical path.
    //
    // The bytes are identical each tile -- the same K replayed -- so the goldens do not
    // move. Only the traffic becomes real.
#define KVBYTES ((uint32_t)(BC * DHEAD))
    uint32_t k_buf[2], v_buf[2];
    k_buf[0] = (uint32_t)delta_local_a;   // the staged buffer becomes buffer 0
    k_buf[1] = top; top += KVBYTES;
    v_buf[0] = top; top += KVBYTES;
    v_buf[1] = top; top += KVBYTES;

    volatile uint32_t *pub  = (volatile uint32_t *)(l1 + top); top += 512;
    volatile uint32_t *sync = (volatile uint32_t *)(l1 + top); top += 64;

    // s_hdr is the latch beat the SIMD reads from; d32_delta is the tile itself, one
    // beat past it, which is where the GEMM's D32 port writes.
    const int32_t s_hdr[2]     = {(int32_t)s_a, (int32_t)s_b};
    const int32_t d32_delta[2] = {(int32_t)(s_a + BEAT), (int32_t)(s_b + BEAT)};
    const int32_t p8_delta[2] = {(int32_t)p8_0, (int32_t)p8_1};
    // The two tail slots of the P8 buffer tile j writes.
#define RSUM_OF(j) (l1 + p8_delta[(j) & 1] + PBEATS * BEAT)
#define LSC_OF(j)  (l1 + p8_delta[(j) & 1] + (PBEATS + 1) * BEAT)

    // The handoff. sync[0] counts S tiles the GEMM has finished producing;
    // sync[1] counts softmax tiles the SIMD core has finished consuming and
    // whose P8 is ready. Both only ever increase, so a reader never needs a
    // lock -- it just waits for the count to pass a threshold.

    uint32_t gemm_cycles = 0, simd_cycles = 0, gemm_stream_cycles = 0;
    uint32_t gemm_cyc_s1 = 0, gemm_cyc_s2 = 0;  // GEMM cycles, split by matmul shape
    // Stall census per shape: A and B are operand feed, D is drain backpressure.
    uint32_t gemm_sa_s1 = 0, gemm_sb_s1 = 0, gemm_sd_s1 = 0;
    uint32_t gemm_sa_s2 = 0, gemm_sb_s2 = 0, gemm_sd_s2 = 0;
    uint32_t gemm_wall = 0, simd_wall = 0;
    uint32_t gemm_rise = 0;  // rise-poll iterations, summed over all dispatches
    uint32_t dma_busy = 0, dma_wall = 0, dma_block = 0, dma_store = 0;
    uint32_t c_conv = 0, c_max = 0, c_exp = 0, c_quant = 0;
    uint32_t gemm_stall = 0, simd_stall = 0;  // time each core spent waiting
    int err = 0;      // invariant failures
    uint32_t o_bad = 0;  // O elements differing from the golden
    int cfg_err = 0;  // tasks the library refused to configure
    int timeouts = 0;

    // ---- stage Q, K, V and the zero bias ------------------------------------
    if (snrt_is_dm_core()) {
        // No operand staging here. Q, K and V all arrive inside the measured window, so the
        // load side and the store side are symmetric: Q once per QUERY tile, K and V once per
        // KV tile, (O, m, l) out once per query tile.

        // NOTHING IS ZEROED HERE ANY MORE. local_c was cleared twice, 32,768 scalar stores,
        // and it is never read: QK's C channels are masked off and PV points C at oacc32. The
        // accumulators that ARE read are cleared by the iDMA inside the window, below.
        //
        // A and B arrive ALREADY bounded by >>QSHIFT, so the scores stay inside FP16.
        // datagen.py applies the shift when it writes the data.
        sync[0] = 0;  // S tiles produced by the GEMM
        sync[1] = 0;  // softmax tiles consumed by the SIMD
        sync[2] = 0;  // K tiles landed
        sync[3] = 0;  // PV dispatches retired -- frees a V buffer
        sync[4] = 0;  // QK dispatches retired -- frees a K buffer
        sync[5] = 0;  // V tiles landed
        sync[6] = 0;  // the SIMD has drained its last task
        sync[7] = 0;  // the accumulators have been cleared
    }
    if (snax_is_gemm_core()) {
        // TCDM footprint guard. An overrun corrupts whatever follows rather than
        // faulting, so check the layout rather than trust the arithmetic.
        printf("  TCDM footprint  %lu bytes of %u  (Bc=%d, d=%d, QSHIFT=%d)\n",
               (unsigned long)top, 512u * 1024u, BC, DHEAD, QSHIFT);
        if (top > 512u * 1024u) {
            printf("  TCDM OVERFLOW: tile does not fit -- reduce M\n");
            cfg_err++;
        }
    }
    snrt_cluster_hw_barrier();

    // Each descriptor set must match the dimension it walks, on BOTH shapes. data.h carries
    // the geometry and the descriptors together, so what this catches is a stale data.h
    // against a rebuilt cluster cfg: the array shape comes from the cfg and moves
    // independently. A mismatch does not fault -- the second matmul quietly uses the
    // first one's bounds.
    if (snax_is_gemm_core()) {
        if (M * meshRow != BC || N * meshCol != BR || K * tileSize != DHEAD) {
            printf("shape 1 mismatch: M*meshRow=%ld (want Bc=%d)  N*meshCol=%ld (want Br=%d)"
                   "  K*tileSize=%ld (want d=%d)\n",
                   (long)(M * meshRow), BC, (long)(N * meshCol), BR,
                   (long)(K * tileSize), DHEAD);
            cfg_err++;
        }
        if (S2_M * meshRow != DHEAD || S2_N * meshCol != BR || S2_K * tileSize != BC) {
            printf("shape 2 mismatch: M*meshRow=%ld (want d=%d)  N*meshCol=%ld (want Br=%d)"
                   "  K*tileSize=%ld (want Bc=%d)\n",
                   (long)(S2_M * meshRow), DHEAD, (long)(S2_N * meshCol), BR,
                   (long)(S2_K * tileSize), BC);
            cfg_err++;
        }
    }

    if (snax_is_gemm_core()) gemm_configure_once();




    // Every task geometry is fixed, so build them all up front: the loop body is then
    // arm + program_fast + launch and nothing else.
    snax_simd_shape_t *sh = shapes;
    if (snax_is_simd_core()) {
        snax_simd_shapes_clear(shapes, 32);
        // The GEMM emits FP16 (its D32 port carries the Int32ToFp16Converter), so the score
        // tile arrives in SBEATS beats rather than the 2x an INT32 tile needed.

        // 1  rowmax: a lanewise reduce over the tile in the GEMM's own buffer, one beat out.
        snax_simd_shape_flat(&sh[0], l1 + d32_delta[0], SBEATS);
        snax_simd_shape_flat(&sh[1], rmax, 1);
        // 2  m_new = max(m_old, rowmax): LANEWISE over the adjacent pair [rmax][mrun].
        snax_simd_shape_flat(&sh[2], rmax, 2);
        snax_simd_shape_flat(&sh[3], mnew, 1);
        // 3+4  -m_new into both latches in one task: read m_new twice (stride 0) and fan
        //      the negated value out, once into the latch beat in front of THIS tile's
        //      S^T buffer and once into the rmax slot, which task 2 has consumed so it
        //      now pairs with mrun. One 2-beat task pays one reader/extension/writer
        //      fill+drain where two 1-beat tasks pay two. Base AND stride follow the
        //      ping-pong, and program_1d writes both.
        snax_simd_shape_broadcast(&sh[4], mnew, 2);
        snax_simd_shape_2d(&sh[5], l1 + s_hdr[0], 2,
                           (uint32_t)(rmax - (l1 + s_hdr[0])), 1, 0);
        // 5  delta = m_old - m_new: LANEWISE ADD over [-m_new][m_old].
        snax_simd_shape_flat(&sh[8], rmax, 2);
        snax_simd_shape_flat(&sh[9], delta, 1);
        // 6+7  corr = exp(delta), fanned out to both latches (corrL before l_old, corrO
        //      before O) in one task, the same way as 3+4.
        snax_simd_shape_broadcast(&sh[10], delta, 2);
        snax_simd_shape_2d(&sh[11], corrL, 2, (uint32_t)(corrO - corrL), 1, 0);
        // 8+9 FUSED into one pass over the tile.
        //
        // The chain is [EW0, Map, Reduce, EW1, Fp16ToInt8] in that fixed order, so a combine
        // that must happen BEFORE the pointwise transform uses EW0. Sticky-B suppresses its
        // seed beat, so EW0 emits exactly the SBEATS data beats and the epilogue is one pass:
        //
        //     read [ -m_new ][ S^T x Bc ]   -- the latch beat and the tile, in place
        //       -> EW0: sticky ADD -> Map: EXP -> Reduce: ADD|LANEWISE|TAP -> Fp16ToInt8
        //     write [P8 x Bc/2][rowsum]
        //
        // Fusing EW0 and Map keeps S^T - m_new inside the chain, so it is never written out.
        // Fp16ToInt8 joins the SAME pass because its tail passthrough leaves the reduce's
        // trailing FP16 beat alone: the quantise-only task that used to read the whole tile
        // back out of TCDM is gone, and with it the FP16 P buffer.
        snax_simd_shape_flat(&sh[14], l1 + s_hdr[0], 1 + SBEATS);
        snax_simd_shape_flat(&sh[17], l1 + p8_delta[0], PBEATS + 1);
        // 11 corr * l_old: sticky MUL over [corrL][lrun].
        snax_simd_shape_flat(&sh[20], corrL, 2);
        snax_simd_shape_flat(&sh[21], LSC_OF(0), 1);  // seed emits nothing, so just the result
        // 13 l_new = corr*l_old + rowsum: LANEWISE ADD over the adjacent pair [rsum][lsc].
        snax_simd_shape_flat(&sh[24], RSUM_OF(0), 2);
        snax_simd_shape_flat(&sh[25], lnew, 1);
        // 14 O *= corr: sticky MUL, latch corrO then the O tile, into the GEMM's C buffer.
        snax_simd_shape_flat(&sh[26], corrO, 1 + DBEATS);
        snax_simd_shape_flat(&sh[27], oacc, DBEATS);  // in place, past the consumed latch
        // 15,16 commit the running state for the next KV tile.
        // 15+16 fused: two different sources and two different destinations,
        // but both pairs are a constant stride apart, so one strided 2-beat
        // copy commits m and l together.
        snax_simd_shape_2d(&sh[28], mnew, 2, (uint32_t)(lnew - mnew), 1, 0);
        snax_simd_shape_2d(&sh[29], mrun, 2, (uint32_t)(lrun - mrun), 1, 0);
    }
    snrt_cluster_hw_barrier();

    // One full program before the loop establishes every CSR that no task
    // below varies: address high word, spatial stride, channel/byte masks and
    // temporal dims 1-2. From then on each task writes only its six 1-D CSRs
    // via snax_simd_program_1d -- 6 writes instead of 21.
    if (snax_is_simd_core()) snax_simd_program_fast(&sh[0], &sh[1]);
    snrt_cluster_hw_barrier();

    // ONE ORIGIN FOR ALL THREE LANES. mcycle counts the same clock on every hart, but each
    // core stamping its own after the barrier left them ~530 cycles apart -- enough that a
    // cross-lane reading of the trace ("QK0 starts before K(0) lands") was an artefact of
    // the offset rather than a fact about the machine. One core publishes the origin and
    // everyone subtracts that same value, so the lanes are comparable to the cycle.
    if (snax_is_gemm_core()) sync[8] = snrt_mcycle();
    snrt_cluster_hw_barrier();
    const uint32_t t_org = sync[8];

    // ---- hart 3: the K/V stream ------------------------------------------
    // Spans are recorded against each core's own mcycle at the barrier below, so the three
    // lanes share an origin to within the barrier's release skew.
    if (snrt_is_dm_core()) {
        uint32_t t0 = t_org;
        // THE CLEAR GOES FIRST, and that is not the obvious choice. Issuing it behind Q and
        // K(0) shortens the fill by ~390 cycles, because QK(0) needs neither O nor m. It was
        // measured and it LOST: the 24 KiB of clearing then lands inside QK(0)'s execution
        // window and costs it 562 cycles of contention. Overlapping DMA with compute is not
        // free on this machine -- it costs more than the serialisation it saves.
        uint32_t i0 = snrt_mcycle() - t0;
        snrt_dma_start_1d((void *)oacc, (void *)fa_zero, (uint32_t)DBEATS * BEAT);
        snrt_dma_start_1d((void *)mrun, (void *)fa_minf, BEAT);
        snrt_dma_start_1d((void *)lrun, (void *)fa_zero, BEAT);
        snrt_dma_wait_all();
        uint32_t i1 = snrt_mcycle() - t0;
        pub[58] = i0; pub[59] = i1;
        dma_busy += i1 - i0;
        sync[7] = 1;

        uint32_t q0 = snrt_mcycle() - t0;
        snrt_dma_start_1d(local_b, B, N * K * tileSize * meshCol * sizeof(int8_t));
        snrt_dma_wait_all();
        uint32_t q1 = snrt_mcycle() - t0;
        pub[56] = q0; pub[57] = q1;
        dma_busy += q1 - q0;
        for (uint32_t j = 0; j < NKV; j++) {
            // Four marks, not two: the blocked time between them is the point. Bracketing
            // K and V together would charge the wait for a free buffer to the transfer and
            // make a port running at 95% look like one running at 25%.
            if (j >= 2) SNAX_SPIN_UNTIL(sync[4] >= j - 1, timeouts);  // K buffer free
            uint32_t k0 = snrt_mcycle() - t0;
            snrt_dma_start_1d(l1 + k_buf[j & 1], A, KVBYTES);
            snrt_dma_wait_all();
            uint32_t k1 = snrt_mcycle() - t0;
            sync[2] = j + 1;
            if (j >= 2) SNAX_SPIN_UNTIL(sync[3] >= j - 1, timeouts);  // V buffer free
            uint32_t v0 = snrt_mcycle() - t0;
            snrt_dma_start_1d(l1 + v_buf[j & 1], A, KVBYTES);
            snrt_dma_wait_all();
            uint32_t v1 = snrt_mcycle() - t0;
            sync[5] = j + 1;
            pub[34 + 4 * j] = k0; pub[35 + 4 * j] = k1;
            pub[36 + 4 * j] = v0; pub[37 + 4 * j] = v1;
            dma_busy += (k1 - k0) + (v1 - v0);
            dma_block += v0 - k1;
        }
        // THE STORE. A query tile's result leaving the cluster is (O, m, l): the INT32
        // accumulator plus the eight beats of running state. On one cluster it is a drain
        // tail; KV-sharded across clusters it is the partial each shard contributes to the
        // cross-cluster fold, so it belongs in the model now rather than after the split.
        // SPLIT. O is final the instant the last PV retires; only m and l wait on the SIMD's
        // trailing commit. Shipping them together charged O's 16 KiB with the SIMD's drain.
        SNAX_SPIN_UNTIL(sync[3] >= NKV, timeouts);   // every PV retired -- O is final
        uint32_t s0d = snrt_mcycle() - t0;
        snrt_dma_start_1d((void *)fa_out, (void *)(l1 + oacc32), (uint32_t)(BR * DHEAD) * 4u);
        snrt_dma_wait_all();
        uint32_t s1d = snrt_mcycle() - t0;            // O is away
        SNAX_SPIN_UNTIL(sync[6] >= 1, timeouts);     // the SIMD drained -- m, l are final
        uint32_t s2d = snrt_mcycle() - t0;
        snrt_dma_start_1d((void *)((uint8_t *)fa_out + (uint32_t)(BR * DHEAD) * 4u),
                          (void *)rmax, 8u * BEAT);
        snrt_dma_wait_all();
        uint32_t s3d = snrt_mcycle() - t0;
        pub[60] = s2d; pub[61] = s3d;
        pub[53] = s0d; pub[54] = s1d;
        dma_store = (s1d - s0d) + (s3d - s2d);
        dma_busy += dma_store;

        dma_wall = snrt_mcycle() - t0;
        pub[55] = dma_store;
        pub[50] = dma_busy;
        pub[51] = dma_wall;
        pub[52] = dma_block;
    }

    // ======================= the pipelined loop ==============================
    if (snax_is_gemm_core()) {
        // Both D32 descriptors, read out of DRAM ONCE. FP16 halves the beats the writer
        // emits, so the innermost bound and the OUTER stride halve with it. The N-block
        // interleave offset does not -- see gemm_d32_emit_fp16().
        // The descriptor, hoisted out of DRAM once. Everything below is a local from
        // here on, so a dispatch is pure `csrw imm` and never touches .bss again.
        const uint32_t l1u = (uint32_t)l1;
        const int32_t s1_k = K, s1_m = M, s1_blk = M * N;
        const int32_t s1_as2 = Atlstride2, s1_bs1 = Btlstride1;
        const int32_t s2_k = S2_K, s2_m = S2_M, s2_blk = S2_M * N;
        const int32_t s2_as2 = S2_Atlstride2, s2_bs1 = S2_Btlstride1;
        const uint32_t base_b = l1u + (uint32_t)delta_local_b;
        const uint32_t base_c = l1u + (uint32_t)delta_local_c;
        const uint32_t base_o = l1u + oacc32;
        const uint32_t d32_b0_f = (uint32_t)D32tlbound0 / 2,  d32_b0_i = (uint32_t)D32tlbound0;
        const uint32_t d32_s1_f = (uint32_t)D32tlstride1,     d32_s1_i = (uint32_t)D32tlstride1;
        const uint32_t d32_s2_f = (uint32_t)D32tlstride2 / 2, d32_s2_i = (uint32_t)D32tlstride2;
        uint32_t g0 = t_org;

        // ---- staged dispatch -------------------------------------------------
        // A dispatch's 21 CSR writes happen during its PREDECESSOR's execution,
        // not its own. STREAMER_START latches the whole bank into csrCfgReg, so a
        // running task stops reading the bank the instant it fires and the next
        // descriptor can be written into it for nothing. Only the START write is
        // ordered against the data dependency; the config writes never are,
        // because a descriptor depends on loop indices alone.
        //
        // QK's C MUST be restated, not left at "keep current": the O matmul points
        // C at oacc32 for its in-place accumulation, so without this S would be
        // K.Q^T + O_accumulated from the second tile onward.
#define STAGE_QK(t)                                                            \
        do {                                                                   \
            gemm_set_shape(s1_k, s1_m, s1_blk, s1_as2, s1_bs1, 0u);            \
            gemm_set_bases(l1u + k_buf[(t) & 1], base_b, base_c,               \
                           l1u + (uint32_t)d32_delta[(t) & 1]);                \
            gemm_d32_emit_fp16(1, d32_b0_f, d32_s1_f, d32_s2_f);               \
        } while (0)
        // Tile 0 has nothing to accumulate onto and masks the C reader off rather
        // than clearing oacc32: a disabled channel presents ZERO and issues no
        // TCDM request, which is exactly the seed the accumulator needs.
#define STAGE_PV(t)                                                            \
        do {                                                                   \
            gemm_set_shape(s2_k, s2_m, s2_blk, s2_as2, s2_bs1,                 \
                           (t) == 0u ? 0u : 0xFFFFFFFFu);                      \
            gemm_set_bases(l1u + v_buf[(t) & 1],                               \
                           l1u + (uint32_t)p8_delta[(t) & 1], base_o, base_o); \
            gemm_d32_emit_fp16(0, d32_b0_i, d32_s1_i, d32_s2_i);               \
        } while (0)

        // Dispatch 0 has no predecessor to hide behind. Writing it here puts its
        // cold instruction fetches inside the wait for K(0), which is dead time.
        STAGE_QK(0u);

        for (uint32_t j = 0; j <= NKV; j++) {
            if (j < NKV) {
                // S(j) into the buffer the SIMD core is not reading. It last
                // held tile j-2, which is free once tile j-1 has been consumed.
                {
                    uint32_t w0 = snrt_mcycle();
                    if (j >= 2) SNAX_SPIN_UNTIL(sync[1] >= j - 1, timeouts);
                    SNAX_SPIN_UNTIL(sync[2] >= j + 1, timeouts);  // K(j) has landed
                    gemm_stall += snrt_mcycle() - w0;
                }
                pub[10 + 4 * j] = snrt_mcycle() - g0;
                gemm_launch();   // S^T = K.Q^T, staged one dispatch ago
                gemm_ack();
                // The successor is PV(j-1) in this same iteration, except at j == 0
                // where nothing precedes it and the next dispatch is QK(1).
                if (j == 0u) STAGE_QK(1u); else STAGE_PV(j - 1u);
                pub[64 + 2 * j] = snrt_mcycle() - g0;  // successor staged, array running
                gemm_rise += gemm_wait();
                { uint32_t c = csrr_ss(GEMMX_PERFORMANCE_COUNTER);
                  gemm_cycles += c; gemm_cyc_s1 += c;
                  gemm_sa_s1 += csrr_ss(GEMMX_STALL_A);
                  gemm_sb_s1 += csrr_ss(GEMMX_STALL_B);
                  gemm_sd_s1 += csrr_ss(GEMMX_STALL_D); }
                gemm_stream_cycles += csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
                pub[11 + 4 * j] = snrt_mcycle() - g0;
                sync[0] = j + 1;  // publish AFTER the streamer has drained
                sync[4] = j + 1;  // K(j) is dead -- the iDMA may reuse its buffer
            }
            if (j > 0) {
                // O(j-1) = P(j-1).V. Waiting here is what lets S(j) above run
                // concurrently with the SIMD core's tile j-1.
                uint32_t w0 = snrt_mcycle();
                SNAX_SPIN_UNTIL(sync[1] >= j, timeouts);
                SNAX_SPIN_UNTIL(sync[5] >= j, timeouts);   // V(j-1) has landed
                gemm_stall += snrt_mcycle() - w0;
                pub[12 + 4 * (j - 1)] = snrt_mcycle() - g0;
                // Transposed: O^T = V^T.P^T, so P^T is the B operand (it has
                // exactly B's shape) and V^T stays in A.
                //
                // C AND D32 BOTH POINT AT oacc32, so the matmul computes
                // O += P.V in place -- the accumulation across KV tiles is the
                // GEMM's own C input, costing nothing extra.
                //
                // Tile 0 has nothing to accumulate onto, and instead of clearing
                // oacc32 it masks the C reader off. A disabled channel presents
                // ZERO and issues no TCDM request, which is exactly the seed the
                // accumulator needs, so the 16 KiB clear disappears from the DMA
                // head that gates this loop's first dispatch.
                gemm_launch();   // O^T = V^T.P^T, staged one dispatch ago
                gemm_ack();
                // The successor is QK(j+1) while KV tiles remain, otherwise the
                // next iteration's PV, and nothing at all after the last.
                if (j + 1u < (uint32_t)NKV)      STAGE_QK(j + 1u);
                else if (j < (uint32_t)NKV)      STAGE_PV(j);
                pub[65 + 2 * (j - 1)] = snrt_mcycle() - g0;
                gemm_rise += gemm_wait();
                { uint32_t c = csrr_ss(GEMMX_PERFORMANCE_COUNTER);
                  gemm_cycles += c; gemm_cyc_s2 += c;
                  gemm_sa_s2 += csrr_ss(GEMMX_STALL_A);
                  gemm_sb_s2 += csrr_ss(GEMMX_STALL_B);
                  gemm_sd_s2 += csrr_ss(GEMMX_STALL_D); }
                gemm_stream_cycles += csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
                pub[13 + 4 * (j - 1)] = snrt_mcycle() - g0;
                sync[3] = j;  // V(j-1) is dead -- the iDMA may reuse its buffer
            }
        }
        gemm_wall = snrt_mcycle() - g0;
#undef STAGE_QK
#undef STAGE_PV
    }

    if (snax_is_simd_core()) {
        uint32_t s0 = t_org;
        uint32_t simd_busy0 = snax_simd_busy_cycles();

        for (uint32_t j = 0; j < NKV; j++) {
            uint32_t w0 = snrt_mcycle();
            SNAX_SPIN_UNTIL(sync[0] >= j + 1, timeouts);
            if (j == 0) SNAX_SPIN_UNTIL(sync[7] >= 1, timeouts);
            simd_stall += snrt_mcycle() - w0;

            pub[26 + 2 * j] = snrt_mcycle() - s0;
            pub[72 + j] = snax_simd_busy_cycles() - simd_busy0;
            sh[0].base = l1 + d32_delta[j & 1];
            // The latch beat and the tile behind it both live in the buffer the GEMM
            // just wrote, so the epilogue's read and the -m_new fan-out follow it too.
            sh[5].base      = l1 + s_hdr[j & 1];
            sh[5].stride[0] = (uint32_t)(rmax - (l1 + s_hdr[j & 1]));
            sh[14].base     = l1 + s_hdr[j & 1];
            // The epilogue writes [P8][rowsum] into this tile's P8 buffer, and corr*l_old is
            // the slot behind the rowsum, so all three follow the ping-pong.
            sh[17].base = l1 + p8_delta[j & 1];
            sh[21].base = LSC_OF(j);
            sh[24].base = RSUM_OF(j);

            // ---- the online softmax, in full -------------------------------
            // Every task below is one beat of arithmetic except 1, 8, 9 and 14.
            // That ratio is the point: the STATE UPDATE is dominated by per-task
            // start/drain, not by compute.

            // 1  rowmax over the tile the GEMM has already written as FP16. The tile is
            //    read where it lies and swallowed: one beat of per-lane maxima comes out.
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMREDUCE));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR, SBEATS);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1,
                                    SIMD_RED_MAX | SIMD_RED_LANEWISE);
            snax_simd_program_1d(&sh[0], &sh[1]);
            snax_simd_fire();

            // 2  m_new = max(m_old, rowmax)
            snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                           SIMD_RED_MAX | SIMD_RED_LANEWISE);
            snax_simd_program_1d(&sh[2], &sh[3]);
            snax_simd_fire();

            // 3+4  -m_new into both latches, one task
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR,
                           snax_simd_f32_neg(SIMD_F32_ONE), 0, SIMD_FUNC_LINEAR);
            snax_simd_program_1d(&sh[4], &sh[5]);
            snax_simd_fire();

            // 5  delta = m_old - m_new
            snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                           SIMD_RED_ADD | SIMD_RED_LANEWISE);
            snax_simd_program_1d(&sh[8], &sh[9]);
            snax_simd_fire();

            // 6,7  corr = exp(delta), to both places a latch is needed
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_ONE,
                           0, SIMD_FUNC_EXP);
            snax_simd_program_1d(&sh[10], &sh[11]);
            snax_simd_fire();

            // 8+9 P = exp(S - m_new) AND rowsum, in ONE pass over the tile.
            // EW0 is upstream of Map, so the per-lane subtract happens before the exponential
            // and the tile is never written out in between.
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                        (1u << SIMD_EXT_STREAMMAP) |
                                        (1u << SIMD_EXT_STREAMREDUCE) |
                                        (1u << SIMD_EXT_FP16TOINT8));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 0, 1);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 1,
                                    SIMD_EW_ADD | SIMD_EW_STICKY_B);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0, SIMD_F32_ONE);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 1, 0);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 2, SIMD_FUNC_EXP);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR, SBEATS);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1,
                                    SIMD_RED_ADD | SIMD_RED_LANEWISE | SIMD_RED_TAP);
            // The quantiser narrows the SBEATS data beats and passes the reduce's trailing
            // rowsum beat through untouched, so this one task writes [P8 x PBEATS][rowsum].
            snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 0, SIMD_F32_ONE);
            snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 1, SIMD_QUANT_TAIL(SBEATS));
            snax_simd_program_1d(&sh[14], &sh[17]);
            snax_simd_fire();

            // PUBLISH HERE, not at the end of the tile. Everything below updates the running
            // l and m for the NEXT tile and the GEMM needs none of it; the task queue already
            // orders those ahead of tile j+1's first task, so correctness costs nothing.
            {
                uint32_t want = snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR);
                uint32_t spins = 0;
                while (snax_read_simd_cfg_reg(SIMD_FINISHED_TASK_PTR) < want) {
                    if (++spins > 40000u) {
                        printf("SIMD STALL tile %u (publish): submitted=%u finished=%u status=%08x\n",
                               j, want, snax_read_simd_cfg_reg(SIMD_FINISHED_TASK_PTR),
                               snax_read_simd_cfg_reg(SIMD_STATUS));
                        timeouts++;
                        break;
                    }
                }
            }
            if (timeouts) break;
            sync[1] = j + 1;

            // 14 O *= corr, AFTER the publish. The GEMM does not read this: its C and D32
            // both point at oacc32, the INT32 accumulator, while this rescales the FP16
            // `oacc`, which has no reader at all. Sitting ahead of the publish put 269 cc
            // per tile on the GEMM's critical path for nothing.
            snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1,
                           SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1,
                           SIMD_EW_MUL | SIMD_EW_STICKY_B);
            snax_simd_program_1d(&sh[26], &sh[27]);
            snax_simd_fire();
            pub[27 + 2 * j] = snrt_mcycle() - s0;

            // 11 corr * l_old
            snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1,
                           SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1,
                           SIMD_EW_MUL | SIMD_EW_STICKY_B);
            snax_simd_program_1d(&sh[20], &sh[21]);
            snax_simd_fire();

            // 13 l_new = corr*l_old + rowsum
            snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                           SIMD_RED_ADD | SIMD_RED_LANEWISE);
            snax_simd_program_1d(&sh[24], &sh[25]);
            snax_simd_fire();

            // 15,16 commit the running state
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_ONE,
                           0, SIMD_FUNC_LINEAR);
            snax_simd_program_1d(&sh[28], &sh[29]);
            snax_simd_fire();
            // The tile's ISSUE ends here, not at the publish above: tasks 14, 11, 13 and
            // 15+16 are all fired afterwards. Bracketing to the publish would attribute
            // their engine time to an interval that does not contain them.
            pub[80 + j] = snrt_mcycle() - s0;


        }
        // One drain for the whole loop rather than one per tile: the last tile's trailing
        // state updates must land before the invariants read l and m. Bounded, so a stuck
        // engine reports itself instead of hanging the simulator.
        {
            uint32_t want = snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR);
            uint32_t spins = 0;
            while (snax_read_simd_cfg_reg(SIMD_FINISHED_TASK_PTR) < want) {
                if (++spins > 40000u) {
                    printf("SIMD STALL at drain: submitted=%u finished=%u status=%08x\n",
                           want, snax_read_simd_cfg_reg(SIMD_FINISHED_TASK_PTR),
                           snax_read_simd_cfg_reg(SIMD_STATUS));
                    timeouts++;
                    break;
                }
            }
        }
        pub[76] = snax_simd_busy_cycles() - simd_busy0;
        sync[6] = 1;  // m, l and O are final -- the store may go
        simd_cycles = snax_simd_busy_cycles() - simd_busy0;
        simd_wall = snrt_mcycle() - s0;
    }
    snrt_cluster_hw_barrier();

    // ---- invariants for the FULL algorithm ----------------------------------
    //
    // Three independent checks, each pinning a different part of the online softmax:
    //
    //   P^T  contains exactly 1     per query row   the max, the subtract, the exp AND the
    //                                                quantiser -- the maximal key gives
    //                                                exp(0) = 1.0, which at inv_scale 1.0
    //                                                is the only value that rounds to 1
    //   l    == NKV * rowsum(one tile)              the TAPPED rowsum and the l recurrence
    //
    // The third is the sharpest. Every KV tile is fed the same K, so after the first tile
    // the running max stops changing, corr = exp(0) = 1, and l must accumulate exactly NKV
    // identical row sums; a wrong tap, correction or l update makes it drift.
    if (snax_is_simd_core()) {
        volatile int8_t *p8t = (volatile int8_t *)(l1 + p8_delta[(NKV - 1) & 1]);
        volatile uint8_t *rsum = RSUM_OF(NKV - 1);
        for (uint32_t i = 0; i < BR; i++) {   // i = query row = lane
            // exp(S - m) == 1.0 for the maximal key pins the running max, the per-lane
            // subtract in EW0 and the exponential in Map -- and now the quantiser too,
            // since P is only ever INT8 in memory.
            // The count, not just the presence: a dropped beat, a tail written into the
            // wrong slot or a shifted write all move it, including where every sampled
            // beat below still reads the 0 it expects.
            int ones = 0;
            for (uint32_t j = 0; j < SBEATS; j++)
                if (p8t[j * BR + i] == 1) ones++;
            int16_t want_ones = p8ones_golden[i];
            if (ones == 0 || (want_ones >= 0 && ones != want_ones)) {
                printf("query %2u: P^T has %d ones, expected %d  m=%04x rowsum=%04x l=%04x\n",
                       i, ones, (int)want_ones, fp16_at(mrun, i), fp16_at(rsum, i),
                       fp16_at(lrun, i));
                err++;
            }
        }
        // l must have accumulated NKV identical row sums. With every KV tile fed the
        // same K, the running max stops changing after tile 0, so corr = exp(0) = 1 and
        // the recurrence degenerates to l = rowsum added NKV times -- i.e. NKV * rowsum,
        // which in binary floating point is the same EXPONENT shifted by log2(NKV) with
        // the mantissa within a rounding step. Checking the exponent needs no FPU (this
        // core has none) and still catches every failure that matters: a dead tap gives
        // l = 0, a dropped update gives l = rowsum (shift 0), a doubled one gives 2x.
        if ((NKV & (NKV - 1)) == 0) {
            uint32_t shift = 0u;  // log2(NKV)
            for (uint32_t t = NKV; t > 1u; t >>= 1) shift++;
            for (uint32_t i = 0; i < BR; i++) {
                uint16_t r = fp16_at(rsum, i), lv = fp16_at(lrun, i);
                uint32_t er = (r >> 10) & 0x1Fu, el = (lv >> 10) & 0x1Fu;
                if (er == 0 || er == 0x1F) continue;   // denormal/inf: not a fair compare
                if (el != er + shift) {
                    printf("query %2u: l exponent %lu, expected %lu  (rowsum=%04x l=%04x)"
                           " -- the tap or the l recurrence is wrong\n",
                           i, (unsigned long)el, (unsigned long)(er + shift), r, lv);
                    err++;
                }
            }
        }

        // ---- against the numerical golden -----------------------------------
        // The invariants above pin one element per query row -- the row's own maximum,
        // which subtracts to zero. The golden pins what they cannot see: m against the
        // true maximum over all Bc keys, the row sum over every key, and P itself on one
        // beat in PGOLD_STRIDE. datagen.py computes all three in float from the same
        // operands, at the precision each stage of the hardware works in.
        //
        // P is compared as INT8 because that is the only form it exists in: the epilogue
        // quantises the tile in the sweep that produces it. The exponential is still
        // covered in FULL FP16 precision by rowsum, which integrates every key -- the
        // INT8 compare adds the quantiser's rounding threshold on top of it.
        uint32_t worst_m = 0, worst_sum = 0;
        for (uint32_t i = 0; i < BR; i++) {
            uint32_t u = fp16_ulp(fp16_at(mrun, i), m_golden[i]);
            if (u > worst_m) worst_m = u;
            u = fp16_ulp(fp16_at(rsum, i), rowsum_golden[i]);
            if (u > worst_sum) worst_sum = u;
        }
        uint32_t p8_bad = 0;
        for (uint32_t b = 0; b < PGOLD_NBEATS; b++) {
            uint32_t key = (uint32_t)b * PGOLD_STRIDE;
            for (uint32_t i = 0; i < BR; i++) {
                int16_t want = p8_golden[b * BR + i];
                if (want < 0) continue;  // within a rounding step of the .5 threshold
                if ((int16_t)p8t[key * BR + i] != want) p8_bad++;
            }
        }
        // ---- O, the only check that reaches the SECOND matmul --------------
        // m, P8 and rowsum all stop at the score tile: they say nothing about
        // O^T = V^T.P^T. This compares the INT32 accumulator element by element, so a
        // wrong shape-2 bound or stride, a broken accumulate-in-place through C, or a D32
        // map that lands blocks in the wrong order shows up here and nowhere else.
        //
        // The coverage is exact but narrow in ONE direction: at inv_scale = 1.0 the
        // quantiser leaves a single 1 per query row and zeros elsewhere, so each output
        // element selects one V column rather than mixing 512. Every one of the BR*DHEAD
        // outputs is still checked exactly, and picking the wrong column -- which is what
        // a bad descriptor does -- changes the value.
        if (O_GOLDEN_VALID) {
            volatile int32_t *o32 = (volatile int32_t *)(l1 + oacc32);
            for (uint32_t i = 0; i < (uint32_t)(BR * DHEAD); i++) {
                if (o32[i] != o32_golden[i]) {
                    if (o_bad < 4)
                        printf("O[%4u] = %ld, expected %ld\n", i, (long)o32[i],
                               (long)o32_golden[i]);
                    o_bad++;
                }
            }
            err += (int)o_bad;
        } else {
            printf("  golden           O compare SKIPPED (a P lane sits on the rounding"
                   " boundary)\n");
        }

        printf("  golden           m %u ULP  P8 %u wrong  rowsum %u ULP  O %u wrong"
               "  (limits %u/0/%u/0)\n",
               worst_m, p8_bad, worst_sum, o_bad, GOLD_ULP_M, GOLD_ULP_SUM);
        if (worst_m > GOLD_ULP_M || p8_bad != 0 || worst_sum > GOLD_ULP_SUM || o_bad) {
            printf("  golden MISMATCH: the chain is wired correctly but the arithmetic "
                   "disagrees with the float model\n");
            err++;
        }
    }

    // ---- report -------------------------------------------------------------
    // Each core owns its own totals; publish through TCDM so one core prints.
    if (snax_is_simd_core()) {
        pub[0] = simd_cycles;
        pub[1] = (uint32_t)err;
        pub[2] = (uint32_t)cfg_err;
        pub[3] = simd_wall;
        pub[4] = c_conv;
        pub[5] = c_max;
        pub[6] = c_exp;
        pub[7] = c_quant;
        pub[8] = simd_stall;
        pub[9] = (uint32_t)timeouts;
    }
    snrt_cluster_hw_barrier();

    if (snax_is_gemm_core()) {
        uint32_t simd_total = pub[0];
        uint32_t simd_wall_r = pub[3];
        // Both cores enter the loop from the same barrier, so the pipeline is
        // however long the slower of them stayed in it.
        uint32_t pipeline = gemm_wall > simd_wall_r ? gemm_wall : simd_wall_r;
        if (pub[51] > pipeline) pipeline = pub[51];  // the store tail is part of the window
        uint32_t busy = gemm_cycles + simd_total;

        printf("\n=== FlashAttention on the four-engine cluster ===\n");
        printf("  tile             Br=%d Bc=%d d=%d, %d KV tiles, mesh %dx%dx%d\n",
               BR, BC, DHEAD, NKV, meshRow, tileSize, meshCol);
        printf("  GEMM streamer    %5u cycles (operand feed + D32 drain)\n",
               gemm_stream_cycles);
        // Cycles per ARRAY PASS, against the arithmetic floor of 1. One pass is
        // Mu*Ku*Nu = 1024 MAC, and a dispatch is M*N*K of them, so this is the engine's
        // own efficiency with the tile shape factored out.
        //
        // The two shapes differ 4x in blocks per pass (256 x 32 against 64 x 128), so
        // printing both separates a per-OUTPUT-BLOCK cost -- C deserialise, D serialise,
        // block turnaround -- from a per-PASS one. The stall census below attributes
        // whatever is above 1.00: A and B are cycles the streamer did not have an operand
        // ready, D is cycles the array refused the pass because the drain was busy. The
        // array itself sustains 1.02 cyc/pass in these shapes when fed ideally
        // (VersaCoreThroughputTest), so A or B dominating means the feed is the limit.
        printf("  S^T=K.Q^T        %5u cycles for %5u array passes (%u.%02u cyc/pass,"
               " %d blocks x %d accum)\n",
               gemm_cyc_s1, (unsigned)(NKV * M * N * K),
               gemm_cyc_s1 / (unsigned)(NKV * M * N * K),
               (gemm_cyc_s1 * 100u / (unsigned)(NKV * M * N * K)) % 100u,
               (int)(NKV * M * N), (int)K);
        printf("    stalls         A %5u  B %5u  D %5u  (feed %u%%, drain %u%%)\n",
               gemm_sa_s1, gemm_sb_s1, gemm_sd_s1,
               100u * (gemm_sa_s1 + gemm_sb_s1) / (gemm_cyc_s1 ? gemm_cyc_s1 : 1u),
               100u * gemm_sd_s1 / (gemm_cyc_s1 ? gemm_cyc_s1 : 1u));
        printf("  O^T=V^T.P^T      %5u cycles for %5u array passes (%u.%02u cyc/pass,"
               " %d blocks x %d accum)\n",
               gemm_cyc_s2, (unsigned)(NKV * S2_M * S2_N * S2_K),
               gemm_cyc_s2 / (unsigned)(NKV * S2_M * S2_N * S2_K),
               (gemm_cyc_s2 * 100u / (unsigned)(NKV * S2_M * S2_N * S2_K)) % 100u,
               (int)(NKV * S2_M * S2_N), (int)S2_K);
        printf("    stalls         A %5u  B %5u  D %5u  (feed %u%%, drain %u%%)\n",
               gemm_sa_s2, gemm_sb_s2, gemm_sd_s2,
               100u * (gemm_sa_s2 + gemm_sb_s2) / (gemm_cyc_s2 ? gemm_cyc_s2 : 1u),
               100u * gemm_sd_s2 / (gemm_cyc_s2 ? gemm_cyc_s2 : 1u));
        printf("  pipeline         %5u cycles, %u per KV tile\n", pipeline,
               pipeline / NKV);
        // Per core: what it spent running its engine, blocked on the other
        // core, and writing CSRs. The three must sum to that core's wall.
        printf("  GEMM core        busy %5u (%2u%%)  peer-wait %5u  config %5u\n",
               gemm_cycles, 100u * gemm_cycles / pipeline, gemm_stall,
               gemm_wall > gemm_cycles + gemm_stall
                   ? gemm_wall - gemm_cycles - gemm_stall : 0);
        // config = wall - busy - peer-wait: CSR writes, the 5 RO counter reads per
        // dispatch (5.1 cycles each, instrumentation only), and the busy-rise poll.
        printf("  iDMA core        moving %5u (%2u%%)  blocked on a buffer %5u  wall %5u\n",
               pub[50], 100u * pub[50] / pipeline, pub[52], pub[51]);
        printf("    init           %u B cleared in %u cc  [%u -> %u]\n",
               (uint32_t)DBEATS * SIMD_BEAT_BYTES + 2u * SIMD_BEAT_BYTES,
               pub[59] - pub[58], pub[58], pub[59]);
        printf("    Q load         %u B in %u cc  [%u -> %u]\n",
               (uint32_t)(N * K * tileSize * meshCol), pub[57] - pub[56], pub[56], pub[57]);
        printf("    store O        %u B in %u cc  [%u -> %u]   then m,l in %u cc  [%u -> %u]\n",
               (uint32_t)(BR * DHEAD) * 4u, pub[54] - pub[53], pub[53], pub[54],
               pub[61] - pub[60], pub[60], pub[61]);
        printf("    K+V per tile   %u B in %u cc = %u.%u B/cc of a 64 B/cc port\n",
               2u * KVBYTES, pub[50] / NKV,
               (2u * KVBYTES * NKV) / pub[50],
               ((2u * KVBYTES * NKV * 10u) / pub[50]) % 10u);
        // Raw spans, for the trace: every engine's own start/end per dispatch or tile,
        // measured against that core's mcycle at the loop entry.
        printf("  TRACE gemm");
        for (uint32_t i = 0; i < 4u * NKV; i++) printf(" %u", pub[10 + i]);
        printf("\n  TRACE simd");
        for (uint32_t i = 0; i < 2u * NKV; i++) printf(" %u", pub[26 + i]);
        printf("\n  TRACE gcfg");
        for (uint32_t i = 0; i < 2u * NKV; i++) printf(" %u", pub[64 + i]);
        printf("\n  TRACE send");
        for (uint32_t i = 0; i < NKV; i++) printf(" %u", pub[80 + i]);
        printf("\n  TRACE sbusy");
        for (uint32_t i = 0; i <= NKV; i++) printf(" %u", pub[72 + i]);
        printf("\n  TRACE idma");
        for (uint32_t i = 0; i < 4u * NKV; i++) printf(" %u", pub[34 + i]);
        printf("\n");
        printf("  SIMD core        busy %5u (%2u%%)  peer-wait %5u  config %5u\n",
               simd_total, 100u * simd_total / pipeline, pub[8],
               simd_wall_r > simd_total + pub[8]
                   ? simd_wall_r - simd_total - pub[8] : 0);
        // Two engines run concurrently, so this is out of 200%, not 100%:
        // above 100% means both were busy at once for part of the run. It is
        // engine-seconds delivered per second of wall clock.
        printf("  occupancy        %u%% of the pipeline (200%% = both engines saturated)\n",
               100u * busy / pipeline);
        if (pub[9]) printf("  WARNING: %u spin timeouts -- the handoff deadlocked\n", pub[9]);
        printf("  %s (%u invariant, %u config)\n",
               (pub[1] == 0 && pub[2] == 0 && pub[9] == 0) ? "PASS" : "FAIL",
               pub[1], pub[2]);
    }
    snrt_cluster_hw_barrier();
    return 0;
}
