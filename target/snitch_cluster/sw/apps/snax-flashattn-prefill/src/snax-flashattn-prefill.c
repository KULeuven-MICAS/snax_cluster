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
//                         \___ the fused pass latches     \___ combined as a pair by
//                              -m16, then adds it to            the corr task below
//                              all 512 beats of the tile
//
//                     ONE value at TWO addresses. Operands here are paired by ADJACENCY
//                     -- a task reads one flat stream and pairs whatever is next to each
//                     other in it -- and -m16 has two different partners, so one copy
//                     cannot serve both. The task reads mnew TWICE at stride 0 and
//                     writes both: 2 in, 2 out, one fill+drain instead of two.
//
//                  corr16 = exp(m16_old - m16)       2 -> 1  FUSED: EW0 (ADD) -> Map (EXP)
//
//                       [ corr16 ][ l16_old ]
//                         corrL     lrun
//                         \___ sticky MUL gives lsc16, below
//
//                     corr16 has ONE reader, so it is written to one latch. O is
//                     never rescaled by corr -- see the note on O's representation below.
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
//                     ---- P8^T is published to the GEMM here. Everything below only
//                          prepares the NEXT tile, so the GEMM never waits on it. ----
//
//                  lsc16  = corr16 * l16_old         2 -> 1  StreamElementwise sticky MUL
//                  l16    = sum16 + lsc16            2 -> 1  StreamReduce ADD|LANEWISE
//                  m16_old, l16_old <- m16, l16      2 -> 2  StreamMap LINEAR, a = 1
//
//                     The "commit". Old and new live in SEPARATE beats all tile long,
//                     because corr16 needs m16_old after m16 already exists, and lsc16
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
//   O32^T is the ONLY representation of O. An FP16 copy rescaled by corr each tile would
//   have no reader at all -- see "O^T PER QUERY TILE, in ONE representation" in the
//   allocation block.
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
// O is checked too, but it pins a WEAKER property than the rest, and the difference
// matters. The GEMM accumulates P.V into oacc32 in INT32 and that is the only copy of O;
// the online rescale is NOT applied to it, because corr is FP16 and scaling an INT32
// accumulator by it needs an FP16 -> INT32 conversion the datapath does not have. For
// THIS data that costs nothing: every KV tile is fed the same K, so the running max stops
// moving after tile 0 and corr is 1 from then on, which is why o32 matches o32_golden on
// all 4,096 elements. With genuinely varying K it would not. So the O check pins the two
// matmuls and the D32 layout; it does NOT exercise the online update.
//
// ============================ SHARDING THE KEYS ACROSS CLUSTERS ============================
//
// Everything above is ONE cluster walking NKV key/value tiles in sequence. The multi-cluster
// form cuts the SAME sequence into P slices and gives one to each cluster, which runs this
// whole file on it unchanged. Q is replicated -- every cluster attends the same query tile.
//
//      K, V [N, d]  INT8                  what cluster c holds when its loop ends
//       +---------------+  -+-           m_c [Br] FP16   l_c [Br] FP16   O_c [d, Br] INT32
//   c=0 |   keys 0..511 |   |    ->        mrun            lrun            oacc32
//       +---------------+   |            (1 beat)        (1 beat)        (16 KiB)
//   c=1 | keys 512..1023|   |    ->      m_1             l_1             O_1
//       +---------------+   N
//   c=2 |keys 1024..1535|   |    ->      m_2             l_2             O_2
//       +---------------+   |
//   c=3 |keys 1536..2047|   |    ->      m_3             l_3             O_3
//       +---------------+  -+-
//                                      four PARTIAL answers to the same question
//
// PRECISIONS, because they are not uniform and the mismatch is most of the work below:
//
//   quantity   lives in                  precision   role in the fold
//   ---------------------------------------------------------------------------------
//   Q, K, V    DRAM -> TCDM              INT8        operands, never folded
//   S^T        s_a / s_b                 FP16        GEMM output via Int32ToFp16
//   P8^T       p8_0 / p8_1               INT8        quantised by Fp16ToInt8
//   m_c        mrun    [32 lanes]        FP16        the KEY field of a partial
//   l_c        lrun    [32 lanes]        FP16        a twisted VALUE field
//   O_c        oacc32  [d, Br]           INT32  <--  Int32ToFp32, on the xDMA reader
//   ---------------------------------------------------------------------------------
//   alpha_c    inside the junction       FP32        derived from the keys, never stored
//   the fold   inside the junction       FP32        the ARITHMETIC is FP32, always
//   transport  on the wire               MIXED       FP16 for (m, l), FP32 for O
//
// The last two lines are what to hold on to. `fmt` selects how a beat is SLICED and
// nothing else -- every compare, exponential and FMA inside the operator is FP32 whatever
// it says. And the two folds need not agree: (m, l) rides FP16 because mrun and lrun
// already ARE FP16, so that fold converts nothing, while O rides FP32 because its raw
// numerator fits nothing narrower. That split is measured, not assumed -- the table is
// under THE O FOLD below.
//
// THESE DO NOT ADD. Each shard's l and O are expressed relative to ITS OWN maximum: if
// shard 0 saw a top score of 10 and shard 1 saw 2, then shard 1's numbers are all scaled by
// e^2 where shard 0's are scaled by e^10. Adding them adds quantities in different units.
// Shard 1 must first be re-based onto the common maximum, by exp(2 - 10) -- which is the
// same `corr` the loop above applies when a TILE raises the running max, with the tile index
// replaced by the shard index:
//
//     m* = max_c m_c            alpha_c = exp(m_c - m*)      <- one FP32 scalar per shard,
//     l* = sum_c alpha_c . l_c                                    per query row
//     O* = sum_c alpha_c . O_c                  and, once, at the very end:  O*/l*
//
//   operands in  FP16 ---> widen ---> FP32 compare / exp / FMA ---> narrow ---> FP16 out
//
// A max on the key, coupled to a rescaled sum on the values, with ONE alpha shared by every
// value coordinate of a partial. That is precisely the xDMA MonoidJunction's operator, and in
// ONE PASS nothing else computes it: an ADD collective returns the wrong answer and a MAX
// collective loses the sum.
//
// THE ALTERNATIVE, AND WHY IT IS NOT TAKEN. In TWO passes the twist disappears entirely. Fold
// (m, l) first -- 4 beats -- broadcast m* and l* back, let every shard pre-scale its own
// numerator by alpha_c / l*, and the numerators are then already on a common basis, so the
// collective is a plain ElementwiseJunction ADD: no key, no chunking, no key replication, and
// the beats carry nothing but payload. Measured on the same bench
// (FaGatherElementwiseTester), against the same golden:
//
//     E16  FP16 pre-scaled ADD    worst/term 1.6e-03    132 beats/shard
//     E32  FP32 pre-scaled ADD    worst/term 2.4e-07    260 beats/shard
//
// E32 is the most accurate row in this file AND the cheapest FP32 one, and it is still not
// the choice -- for two reasons the bench cannot show:
//
//   - its alpha is computed in float64 BY THE BENCH. Real software would get alpha from the
//     same exp LUT, and that LUT is exactly where method A's 3.1e-06 comes from (the junction
//     tester independently measures 2.2e-06 on the same lookup). So E32 in practice lands
//     beside A, not above it: the bench is flattering it.
//   - the l* broadcast SERIALISES. No shard can pre-scale until the first fold has completed
//     and come back to it, which is a full round trip added to the critical path.
//
// Paying a round trip, and a second dispatch, to save 10% of the beats is the wrong trade.
// One pass, no broadcast, no sequencing between the two folds.
//
// ---- THE ROUTE: the fold happens IN the data movement, not after it ----
//
// Arming a junction is what turns a transfer into a GATHER (`collectiveMode :=
// junctionEnabled`). Each hop's WRITER folds the beats arriving from the previous cluster
// against the beats it reads from its own TCDM, and forwards the result. No cluster ever
// holds more than two partials, and the wire carries one partial's worth of traffic per hop:
//
//     cluster 0          cluster 1          cluster 2          cluster 3 (root)
//    +-----------+      +-----------+      +-----------+      +-----------+
//    | m0 l0 O0  |      | m1 l1 O1  |      | m2 l2 O2  |      | m3 l3 O3  |   <- local (B)
//    +-----+-----+      +-----+-----+      +-----+-----+      +-----+-----+
//          |                  | B                | B                | B
//          |  A         +-----v-----+  A   +-----v-----+  A   +-----v-----+
//          +----------->|  A (+) B  |----->|  A (+) B  |----->|  A (+) B  |
//                       +-----------+      +-----------+      +-----------+
//                                                                   |
//                                                          (m*, l*, O*) in TCDM
//
//   A = the stream arriving from the previous hop      B = this cluster's own partial
//
// The HEAD is special: nothing arrives at it, so it gets a reader frame only and simply
// SENDS. Every other node gets both frames -- read your own partial, arm your junction,
// forward. The chain is the writerPtr list in data order; bit 1 of XDMA_DST_JCT_ENABLE_PTR
// selects the monoid (WRITER_JCT_MONOIDJUNCTION) and its geometry word is
// XDMA_DST_JCT_CSR_PTR + 1.
//
// Contrast with the obvious alternative -- every shard ships to a root that folds P streams.
// That costs the root P-1 receives and P-1 buffers; this costs every node one of each, and
// the fold is hidden inside a transfer that had to happen anyway.
//
// ---- THE BEAT: what "field-major" actually means ----
//
// This is the part that bites, because it is not how the kernel stores anything today. The
// junction reads a 512-bit beat as a flat grid of FP16 lanes and addresses them
//
//     lane = field * S + slot          S = 1 << sigma = partials per beat
//
// so a partial's coordinates are STRIDED BY S, and consecutive lanes are DIFFERENT QUERY
// ROWS. For the (m, l) fold -- 2 fields, sigma = 3, so S = 8 partials per beat:
//
//   lane    0    1    2    3    4    5    6    7    8    9   10   11  ..  15   16..31
//   field   0    0    0    0    0    0    0    0    1    1    1    1      1    >= F,
//   slot    0    1    2    3    4    5    6    7    0    1    2    3      7    dead
//   holds  m_0  m_1  m_2  m_3  m_4  m_5  m_6  m_7  l_0  l_1  l_2  l_3    l_7
//   FP16    ^ every lane is one FP16 element; 32 of them in a 512-bit beat
//          '------- the key of 8 query rows -------''---- their l ----'
//
// Eight query rows per beat, so Br = 32 is FOUR beats -- and only 16 of the 32 lanes are
// live in each, because sigma is a 2-bit field capped at 3 and S cannot reach 16. The upper
// half carries the identity. That is a known, accepted cost: filling it would need a third
// sigma bit to buy 128 bytes on a transfer whose cost is dominated by arming it.
//
// So the repack is:
//
//     mrun  [32 FP16, 1 beat,  64 B]  --+
//                                        +-->  4 junction beats, 16/32 lanes live, 256 B
//     lrun  [32 FP16, 1 beat,  64 B]  --+      (128 B of payload, 128 B of identity pad)
//
// mrun and lrun are already one FP16 value per query row, which IS the slot axis -- but they
// are two separate 32-row beats today, and the junction wants 8 rows of (m, l) per beat.
// That is a strided 2-beat xDMA task, the same shape as the -m_new fan-out this kernel
// already issues.
//
// ---- THE O FOLD: an INT32 accumulator onto a floating-point collective ----
//
// `oacc32` is INT32 and the monoid is not: its twist is `exp(m_lose - m*)`, so every operand
// it touches is a float. Something has to convert, and WHERE it converts and INTO WHAT are
// both decided rather than open.
//
// WHERE: ON THE xDMA READER, in the shard's own read-for-the-gather. The conversion is a
// property of the TRANSFER, not of the matmul, so it rides the chain that performs the
// transfer and costs no extra local pass and no extra buffer:
//
//        oacc32 in TCDM (INT32)
//             |
//             v
//     +-------------------------------------------------------+
//     | xDMA reader chain                                      |
//     |   HasTransposer      (bypassed for this transfer)      |
//     |   HasInt32ToFp32     INT32 -> FP32, one PE per lane    |  <-- the new block
//     +-------------------------------------------------------+
//             |  FP32
//             v
//        data switch --> MonoidJunction --> link --> next hop
//
// The obvious question is why this is not the converter the cluster ALREADY has. There is an
// Int32ToFp16Converter on the GEMM's own streamer (reader_writer slot 1, the D write path),
// and two things rule it out. It sits in the GEMM's WRITE path, so arming it changes what the
// matmul writes -- which is why PV runs with it disarmed, since PV reads its own previous
// output back through C as INT32; reusing it would entangle the collective with the matmul's
// descriptor. And it converts to FP16, which is the wrong target for O -- see below.
//
// INTO WHAT: FP32, and this is MEASURED. `FaGatherTester` folds four shards hop-by-hop
// through the real junction against a float64 online-softmax golden, scoring each candidate
// relative to max|term| -- the standard bound for a floating-point summation, and the right
// one here because `O*` is a signed sum that cancels (largest term typically 1.0x the sum,
// but 140x in the worst lanes):
//
//     encoding                                 worst/term   p99 rel   beats/shard
//     A   FP32 raw   key=m, vals=(l,O)          3.1e-06     1.1e-05      288
//     B   FP16 pow2  key=m+p.ln2, (l,O)/2^p     5.1e-03     4.9e-02      144
//     B32 FP32 pow2  (isolates the embedding)   3.3e-06     1.2e-05      288
//     C   FP16 O/l   key=m        (control)     3.0e+00     2.6e+01      144
//
// Three readings, and the third is why this file carries the table and not a conclusion:
//
//   - A vs B32: the clever FP16 re-embedding costs NOTHING in accuracy. Not the problem.
//   - B32 vs B: FP16 costs 1600x, and the tail is worse than the median -- p99 4.9e-02, so
//     one output element in a hundred is off by 5%, in the lanes where the shards cancel.
//   - C is the encoding a reasonable person writes first (convert, divide by l, key stays m)
//     and it is wrong by 3x max|term|. Finite, format-legal, plausible in a spot check. That
//     is the failure this whole note exists to prevent.
//
// So in FP32 the partial is the one the algebra already writes down, with NO re-embedding:
//
//     key      m_c         the running max, as it stands
//     field 1  l_c         the running normalizer, as it stands
//     fields.. O_c[j]      the RAW numerator, straight out of oacc32
//
// and `Int32ToFp32` is the simplest converter in the tree because of where INT32 sits on the
// number line. It CANNOT overflow -- the largest INT32 has FP32 exponent 158 against a 255
// ceiling -- so there is no saturation path to get wrong; and below 2^24 it is EXACT, which
// covers the whole flash numerator range (`sum P8.V8` ~ 127.l.127 ~ 1e7). The same value
// converted to FP16 saturates to +-Inf, and an Inf in a value field propagates to Inf or NaN
// through the fold -- MonoidCombine's key front end turns two equal infinite keys into a NaN
// by itself. The row is destroyed.
//
// The price is the payload column: 288 beats per shard against 144, about +576 cycles for a
// four-cluster chain, ~3% of this pipeline. That is the trade -- 3% of wall clock for 1600x
// accuracy and the deletion of an entire software apparatus.
//
// ---- THE GEOMETRY: `n` IS 4 BITS, SO THE PARTIAL IS SLICED ----
//
// Independently of the format, a partial is at most F = 16 fields while a d = 128 row needs
// 1 + 128 = 129 value coordinates. It is CHUNKED, with THE KEY REPLICATED INTO EVERY CHUNK:
//
//   one query row's partial              sliced into 9 chunks of 15 values
//   [ m ][ l ][ O_0 ] .. [ O_127 ]  ->   chunk 0: [ m ][ l ][ O_0   .. O_13  ]
//        '---- 129 values ----'          chunk 1: [ m ][ O_14  .. O_28  ]
//                                        ...
//                                        chunk 8: [ m ][ O_119 .. O_127 ]   n = 9
//
// Replicating the key is EXACT, not an approximation, and the whole layout rests on it: the
// twist is a function of the keys ALONE, and every chunk of a row carries the same key, so
// every chunk receives the identical alpha. Chunks are therefore independent beats needing
// no ordering and no shared state -- the junction's O4 obligation ("no state between beat
// pairs") spent on purpose.
//
// AND THE ONE TRAP. An FP32 beat carries 16 elements, not the 32 an elemWidth = 16 instance
// was elaborated for, so the geometry must satisfy F * S <= 16: at F = 16 that forces
// sigma = 0, S = 1, one partial per beat. The saturation and the O5 check are both computed
// from the ELABORATION lane count, so a word with F * S = 32 -- which is the correct FP16
// word -- is ACCEPTED at FP32 and silently reads slot 0's fields as slot 1, with
// `jct_cfgerr_o` low. Measured, and pinned by MonoidFp32GeomTester.
//
//   lane    0    1    2    3   ..  15      one partial, 16 FP32 fields, S = 1
//   field   0    1    2    3       15
//   holds   m    l   O_0  O_1     O_13     (chunk 0; later chunks carry O_14.. instead)
//
// Br = 32 rows x 9 chunks x 1 partial/beat = 288 beats = 18432 B per shard.
//
// ---- WHAT THIS KERNEL WOULD HAVE TO CHANGE ----
//
// Nothing in the pipelined loop. The shards are independent until the store, and the store
// is where all of it lands.
//
//   1. THE STORE BECOMES THE COLLECTIVE. Today it is a plain snrt_dma_start_1d of oacc32 and
//      the running state into fa_out. Sharded, it is an xDMA task with READER_EXT_INT32TOFP32
//      armed on the reader side and the monoid on the writer side, plus a hop chain. The
//      existing split survives unchanged -- O is final when the last PV retires, m and l wait
//      on the SIMD's trailing commit -- because (m, l) and O are two separate folds with two
//      different geometry words anyway.
//
//   2. TWO GEOMETRY WORDS, and they do NOT share a transport:
//        (m, l)   F=2,  n=1,  nExp=1,  sigma=3, fmt=FP16 -> 0x0C040108    4 beats,   256 B
//        O chunk  F=16, n=15, nExp=15, sigma=0, fmt=FP32 -> 0x003C3F01  288 beats, 18432 B
//      (m, l) stays FP16 because mrun and lrun ARE FP16 already -- that fold needs no
//      conversion at all, which is what makes the FP16 transport work worth having. O goes
//      FP32 for the reasons above. Mixing is free: separate transfers, separate CSR words.
//      NOTE THE TRAP: `fmt` is CSR(0)[14:12] and ZERO NOW MEANS FP16, so a word written
//      before that field existed is a legal FP16 word and is accepted silently. Name FP32.
//
//   3. `m` HAS TO BE WIDENED FP16 -> FP32 for the O fold's key -- 32 scalars per shard per
//      query tile, integer bit-work on the exponent (`fp16_bits_to_fp32_bits` already exists
//      as a static inline duplicated in two SIMD apps; lift it rather than copy it a third
//      time). That is the ONLY scalar conversion left: FP32 transport deletes the exponent
//      extraction, the key arithmetic and the scaling pass the FP16 route needed.
//
//   4. BEAT COUNTS MUST MATCH ON BOTH OPERAND STREAMS. The join is two independently
//      dispatched cfgs with no hardware backstop; a mismatch stalls for ever. The junction
//      raises XDMA_JCT_STATUS bit 1 (starved) rather than hanging silently, and bit 0 if the
//      armed operator cannot honour its word. Check both AFTER the transfer -- arming
//      successfully is not evidence that the fold ran.
//
// If the inter-cluster link ever turns out to be the contended resource rather than the
// engines, row B is the fallback and is already validated: the power-of-two embedding is
// correct, it just costs accuracy. That measurement cannot be made in this tree, which has
// one cluster.
//
// NONE OF THIS IS EXERCISABLE HERE. On a single cluster the junction sits on the crossing and
// localLoopback bypasses it, so no path reaches the operator: this app behaves identically
// whether the junction is built or not. The operator is covered by the Tier-1 junction suite;
// the above is what the SOFTWARE side still owes.

#include <stdint.h>
#include "data.h"
#include "snax-core-roles.h"
#include "snax-simd-lib.h"

// The extensions this kernel needs, and WHAT IT NEEDS THEM TO DO. An op/func CSR is a
// runtime select over the set the cfg elaborated, and selecting outside that set does not
// fault -- it returns another op's answer. So the gate names capabilities, not extensions;
// each _HAS_ macro implies its extension exists. See the note in snax-simd-lib.h.
// The softmax chain: a LANEWISE rowmax (MAX), the shifted exponential (EXP) fed by the
// PRE-map elementwise (ADD, sticky), the row sum (ADD), and the tail passthrough that lets
// the same pass quantise the tile while the reduce appends its scalar. tailPassthrough is a
// BUILD-time feature, not a runtime select: without it csr(1) is not there at all and the
// SIMD_QUANT_TAIL write below lands on the next extension's CSR block.
#if !defined(SIMD_EXT_STREAMREDUCE_HAS_MAX) ||          \
    !defined(SIMD_EXT_STREAMREDUCE_HAS_ADD) ||          \
    !defined(SIMD_EXT_STREAMMAP_HAS_EXP) ||             \
    !defined(SIMD_EXT_STREAMELEMENTWISE_0_HAS_ADD) ||   \
    !defined(SIMD_EXT_FP16TOINT8_HAS_TAILPASSTHROUGH)
#error \
    "This cluster's SIMD block cannot run the FlashAttention softmax: it needs StreamReduce MAX+ADD, StreamMap EXP, a PRE-map StreamElementwise ADD, and Fp16ToInt8 tailPassthrough."
#endif
#include "snax-versacore-to-lib.h"
#include "snax-xdma-lib.h"
#include "snrt.h"
#include "snax-perf-census.h"

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

// STEPS is the flattened (KV tile, query tile) sequence the pipeline walks. Decode is this
// kernel at NQ = 1; prefill is the same kernel with more query tiles sharing each K/V pass.
#define STEPS  (NKV * NQ)
#define SBEATS BC        // S^T / P beats: one per KEY, 32 query lanes each
#define DBEATS DHEAD     // O^T beats:     one per HEAD element, 32 query lanes each
#define PBEATS (BC / 2)  // after Fp16ToInt8 halves them

// The cluster's TCDM, which is both the footprint guard's limit and the ceiling on the
// arena zero-fill below.
#define TCDM_BYTES (512u * 1024u)

// ---- the running-state fill, armed with CONSTANT CSR addresses ----------------
//
// snax_xdma_memcpy_nd() cannot be used here. csrw_ss() is a switch over every CSR
// address (snRuntime/src/csr.h): with a compile-time-constant address it folds to a
// one-cycle `csrw imm`, but with a computed one it becomes a jump-table load out of
// .rodata -- which this target maps to main memory -- plus an indirect jump, per
// write. The library builds its addresses in loops (`PTR + i * 2`), so on this cfg
// its 59 dynamic writes -- 30 of them zeroing multicast slots we never use, 16
// padding temporal dimensions we never use -- measured 6,638 cycles for a 128-byte
// fill against 46 on the iDMA.
//
// Unrolled with constant addresses the same descriptor is ~31 writes of 1 cycle.
// The 15 unused multicast destination slots are left at their reset value of zero:
// this hart issues nothing but unicast fills, so nothing ever sets them.
__attribute__((always_inline)) static inline void xdma_fill_arm(
    uint32_t dst, uint32_t stride, uint32_t bound, uint32_t pattern) {
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, dst);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_SPATIAL_STRIDE_PTR, 8);
    snax_write_xdma_cfg_reg(XDMA_DST_SPATIAL_STRIDE_PTR, 8);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 0, bound);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 1, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 2, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 0, stride);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 1, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 2, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 4, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 0, bound);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 1, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 2, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 0, stride);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 1, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 2, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 4, 0);
    // Reader channels OFF: a disabled channel issues no TCDM request at all, so the
    // beat the writer sees comes entirely from the memset. No read, and no fetch of a
    // constant from main memory -- which is the whole point of generating it here.
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, 0xFFFFFFFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, 0xFFFFFFFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLE_PTR, 1u << WRITER_EXT_VERILOGMEMSET);
    snax_write_xdma_cfg_reg(XDMA_DST_EXT_CSR_PTR, pattern);
}

// A REAL transfer on the xDMA: main memory -> L1, one contiguous tile. The fill above
// is a generated pattern with the reader OFF; this turns the reader ON and the writer
// extension OFF, so what the reader fetched is what lands. A source address in main
// memory is what makes it a REMOTE transfer -- the peer endpoint serves it.
//
// Hand-armed with CONSTANT CSR addresses: a csrw_ss with a computed address degrades
// to a jump-table load from main memory at ~92 cycles a write.
__attribute__((always_inline)) static inline void xdma_tile_arm(
    uint32_t dst, uint32_t src, uint32_t stride, uint32_t bound) {
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, src);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_SPATIAL_STRIDE_PTR, 8);
    snax_write_xdma_cfg_reg(XDMA_DST_SPATIAL_STRIDE_PTR, 8);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 0, bound);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 1, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 2, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 0, stride);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 1, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 2, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 4, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 0, bound);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 1, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 2, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 0, stride);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 1, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 2, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 4, 0);
    // EXACTLY 8 channels each; the engine has eight, and a reader told it owns 32
    // waits on twenty-four that never answer.
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, 0xFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, 0xFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, 0xFFFFFFFFu);
    // BOTH enables: the memset above leaves its writer extension armed.
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLE_PTR, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLE_PTR, 0);
}

// Re-point an armed descriptor at the next tile. BOTH base addresses, because the
// engine consumes its address registers as the transfer walks -- a second task issued
// without restoring them reads from wherever the first one finished, at the right
// rate and the right size, with the wrong data.
__attribute__((always_inline)) static inline void xdma_tile_retask(
    uint32_t dst, uint32_t src, uint32_t bound) {
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, src);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 0, bound);
}

// WHERE THE RESULT LANDS. Its own array, not a data.h buffer. NQ query tiles of O fill C
// exactly, so the state block that follows them ran off the end and landed on o32_golden --
// the check was comparing against a golden the kernel had just overwritten. The O check
// caught it; every other golden passed.
static int32_t fa_out[NQ * (BR * DHEAD) + NQ * 128];

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

// Submit the configured dispatch. The id is kept in SOFTWARE rather than read back from a
// counter: every launch writes exactly one start to each engine and so retires exactly one
// array task, which makes a plain increment identical to the hardware's count and saves a
// five-cycle read on the critical path.
__attribute__((always_inline)) static inline void gemm_launch(void) {
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(GEMMX_START, 1);
}

// Deassert the start pulse. Separate from the wait below so the caller can stage the NEXT
// dispatch's config in between: once STREAMER_START has fired, the streamer has snapshotted
// the whole CSR bank into `csrCfgReg` (Streamer.scala) and VersaCore into `csrReg`
// (VersaCore.scala), so the running task no longer reads the bank and the bank is
// free to be overwritten. ReqRspManager only throttles writes to the START address itself;
// every other CSR write is accepted at one per cycle regardless of busy.
__attribute__((always_inline)) static inline void gemm_ack(void) {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
}

// Wait for ONE named dispatch, by id. TWO conditions, and BOTH are required.
//
// The streamer's finished counter says its data movers retired. That is NOT the same as the
// matmul being over: a writer's address generator finishes issuing before the array has
// finished producing into it, so the counter alone returns early. Measured, that reports
// 0.14 cycles per array pass against a floor of 1.0 -- a tile taking less time than its own
// arithmetic -- because the performance counter is read mid-flight.
//
// The busy flag says the array is done, but alone it cannot tell "not started yet" from
// "already finished": polling straight after an inlined start write reads zero on a task
// that has not begun.
//
// Composing them removes both problems and the rise-poll with them. Once the counter has
// reached this id the task provably started, so the busy poll below can only be observing
// its end. Subtracting before the compare keeps the counter test correct across a wrap.
//
// Bounded, so a stuck engine reports itself instead of hanging the simulation. Returns
// non-zero on timeout, which the caller folds into its own timeout count.
__attribute__((always_inline)) static inline uint32_t gemm_wait(uint32_t task_id) {
    uint32_t spins = 0;
    while ((int32_t)(csrr_ss(GEMMX_FINISHED_TASK) - task_id) < 0) {
        if (++spins > 200000u) return 1;
    }
    csrw_ss(GEMMX_START, 0);
    return 0;
}

// Drain the array itself. Separate from the per-dispatch wait ABOVE, and that separation is
// the whole point once two configurations can be queued: the busy flag does not fall between
// back-to-back dispatches, so polling it per dispatch waits for the NEXT one too. Measured,
// that put 2,260 cycles on the softmax of the last tile, because the publish that releases
// the SIMD sat behind a dependency it does not have.
//
// Per dispatch the streamer's counter is the right signal and it is sufficient: what the
// publish claims is that the score tile is IN THE SCRATCHPAD, which is exactly the writer
// having drained. The array's own tail matters only before the result is read back, so it is
// waited for once, here, at the end of the loop.
__attribute__((always_inline)) static inline uint32_t gemm_drain(void) {
    uint32_t spins = 0;
    while (csrr_ss(GEMMX_BUSY)) {
        if (++spins > 200000u) return 1;
    }
    return 0;
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
    // Scratch starts at delta_local_c, not past delta_local_d32: both of those data.h
    // regions are dead. C is never read (QK's C channels are masked off, PV points C at its
    // own accumulator) and the D32 region lost its purpose when the score tile moved into
    // the ping-pong buffers below. Reclaiming them is what makes room for NQ query tiles.
    uint32_t top = ((uint32_t)delta_local_c + 63u) & ~63u;
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
    // ONE STATE BLOCK PER QUERY TILE, eight beats each and always in this order, so a shape
    // built for query tile 0 reaches tile q by adding q*8 beats to its base and nothing else.
    // The strides between the beats -- which is what pairs operands by adjacency -- do not move.
    uint8_t *stateq = l1 + top;  top += (uint32_t)NQ * 8u * BEAT;
#define ST(q)     (stateq + (uint32_t)(q) * 8u * BEAT)
#define ST_OFF(q) ((uint32_t)(q) * 8u * BEAT)
    uint8_t *rmax  = ST(0) + 0 * BEAT;   // rowmax, and later -m_new
    uint8_t *mrun  = ST(0) + 1 * BEAT;   // running m   ] pair for max(m_old, rowmax)
    uint8_t *mnew  = ST(0) + 2 * BEAT;
    // No beat for m_old - m_new: the difference is consumed inside the extension chain
    // by the fused corr task below and never reaches memory.
    uint8_t *corrL = ST(0) + 3 * BEAT;   // exp(m_old - m_new) ] latch for corr*l_old
    uint8_t *lrun  = ST(0) + 4 * BEAT;   // running l          ]
    uint8_t *lnew  = ST(0) + 5 * BEAT;
    // No FP16 P buffer: the epilogue quantises the tile in the sweep that produces it,
    // so P exists only as INT8, inside the P8 buffers below. The tapped rowsum and
    // corr*l_old live there too -- see the P8 allocation.
    // O^T PER QUERY TILE, in ONE representation. The GEMM accumulates P.V into it in
    // INT32 (C and D32 both point here), and that is the O that is stored and checked.
    //
    // There is no FP16 copy of O. One per query tile, with its own corr latch and a
    // 257-beat sticky-B rescale every step, would cost NQ x 8 KiB of scratchpad, NQ
    // clears on the critical head and a pass per step, and nothing would read it.
    // Applying corr to the INT32
    // accumulator instead needs an INT32 x FP16 multiply the SIMD chain does not have.
    // Two converters exist and NEITHER is this one: Int32ToFp16 on the GEMM's output port
    // and Int32ToFp32 on the xDMA reader (for the cross-cluster fold). Both go
    // INT32 -> float; this needs Fp16 -> Int32, the other direction.
    uint32_t oacc32 = top;      top += (uint32_t)NQ * BR * DHEAD * 4; // INT32 O^T, per tile
#define OACC32_OF(q) (oacc32 + (uint32_t)(q) * (uint32_t)(BR * DHEAD) * 4u)
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
    // One Q tile per query tile. They hold the same bytes -- so the goldens are identical for
    // every q -- but each is loaded separately, which is what a real prefill pays.
    uint32_t q_buf[NQ];
    for (uint32_t qi = 0; qi < (uint32_t)NQ; qi++) {
        q_buf[qi] = top; top += (uint32_t)(N * K * tileSize * meshCol);
    }
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

    // Room for STEPS spans per lane rather than NKV: prefill walks NKV*NQ dispatches.
    volatile uint32_t *pub  = (volatile uint32_t *)(l1 + top); top += 2048;
    volatile uint32_t *sync = (volatile uint32_t *)(l1 + top); top += 64;

    // THE TASK GEOMETRIES LIVE IN THE ARENA, not in a .l1 static. TCDM is SRAM with no
    // reset: an address that has never been written reads X in RTL, and .l1 is NOLOAD, so
    // a linker-placed array there starts UNDEFINED rather than zero. Allocating them here
    // puts them inside the block the xDMA zeroes below, which is what makes them defined.
    // Still TCDM and not .bss: program_fast() reads every field of two of these per task,
    // and .bss maps to DRAM on this target, so each field would cost an L3 round trip.
    snax_simd_shape_t *shapes = (snax_simd_shape_t *)(l1 + top);
    top += (32u * sizeof(snax_simd_shape_t) + 63u) & ~63u;

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
    uint32_t gemm_seq  = 0;  // dispatches issued so far; the id to wait on
    uint32_t dma_busy = 0, dma_wall = 0, dma_block = 0, dma_store = 0;
    uint32_t c_conv = 0, c_max = 0, c_exp = 0, c_quant = 0;
    uint32_t gemm_stall = 0, simd_stall = 0;  // time each core spent waiting
    int err = 0;      // invariant failures
    uint32_t o_bad = 0;  // O elements differing from the golden
    int cfg_err = 0;  // tasks the library refused to configure
    int timeouts = 0;

    // ---- hart 2 defines the whole arena, before anyone reads it -------------
    // TCDM is SRAM with no reset. A location that has never been written reads X on RTL,
    // and X propagates: one undefined beat entering the SIMD chain poisons a whole tile,
    // and the failure surfaces far from its cause. Nothing may be left to a
    // write-before-read argument -- every byte the kernel touches is written once here.
    //
    // The xDMA does it with its writer-side memset: the reader channels are off, so the
    // pattern is generated inside the cluster and NOTHING is fetched from main memory for
    // it. One task, one 64-byte beat per cycle, before the barrier that releases the
    // other harts -- so it costs wall clock but no traffic and no measured pipeline time.
    //
    // This is also what zeroes `sync` (the counters below) and `shapes`; neither is
    // cleared anywhere else.
    //
    //   sync[0] S tiles produced by the GEMM      sync[4] QK retired -- frees a K buffer
    //   sync[1] softmax tiles the SIMD consumed   sync[5] V tiles landed
    //   sync[2] K tiles landed                    sync[6] the SIMD drained its last task
    //   sync[3] PV retired -- frees a V buffer    sync[7] m is live
    if (snax_is_xdma_core()) {
        uint32_t zbytes = top > TCDM_BYTES ? TCDM_BYTES : top;
        uint32_t z0 = snrt_mcycle();
        xdma_fill_arm((uint32_t)l1, BEAT, (zbytes + BEAT - 1u) / BEAT, 0u);
        snax_xdma_local_wait(snax_xdma_start());
        printf("  arena preinit   %lu bytes in %lu cc  (xDMA writer memset, outside the "
               "measured window)\n",
               (unsigned long)zbytes, (unsigned long)(snrt_mcycle() - z0));
    }
    snrt_cluster_hw_barrier();

    // ---- stage Q, K, V and the zero bias ------------------------------------
    // No operand staging here. Q, K and V all arrive inside the measured window, so the
    // load side and the store side are symmetric: Q once per QUERY tile, K and V once per
    // KV tile, (O, m, l) out once per query tile.
    //
    // A and B arrive ALREADY bounded by >>QSHIFT, so the scores stay inside FP16.
    // datagen.py applies the shift when it writes the data.
    if (snax_is_gemm_core()) {
        // TCDM footprint guard. An overrun corrupts whatever follows rather than
        // faulting, so check the layout rather than trust the arithmetic.
        printf("  TCDM footprint  %lu bytes of %u  (Bc=%d, d=%d, QSHIFT=%d)\n",
               (unsigned long)top, TCDM_BYTES, BC, DHEAD, QSHIFT);
        if (top > TCDM_BYTES) {
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
        // 5  the pair the fused corr task combines: [-m_new][m_old].
        snax_simd_shape_flat(&sh[8], rmax, 2);
        // 6  corr lands in the one latch that has a reader: corrL, which sits
        //    immediately before l_old for the sticky MUL of task 11.
        snax_simd_shape_flat(&sh[11], corrL, 1);
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
        // trailing FP16 beat alone. Without it, narrowing would need its own task reading
        // the whole tile back out of TCDM, and an FP16 P buffer to read it from.
        snax_simd_shape_flat(&sh[14], l1 + s_hdr[0], 1 + SBEATS);
        snax_simd_shape_flat(&sh[17], l1 + p8_delta[0], PBEATS + 1);
        // 11 corr * l_old: sticky MUL over [corrL][lrun].
        snax_simd_shape_flat(&sh[20], corrL, 2);
        snax_simd_shape_flat(&sh[21], LSC_OF(0), 1);  // seed emits nothing, so just the result
        // 13 l_new = corr*l_old + rowsum: LANEWISE ADD over the adjacent pair [rsum][lsc].
        snax_simd_shape_flat(&sh[24], RSUM_OF(0), 2);
        snax_simd_shape_flat(&sh[25], lnew, 1);
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
    // The cluster contention counters are armed HERE, by one hart, so that the
    // window they cover is the same barrier-to-barrier span the three lanes
    // measure with mcycle. They are cluster-global, so arming them on the GEMM
    // hart covers the engines whose cores are asleep as well.
    snax_perf_snapshot_t perf;
    if (snax_is_gemm_core()) snax_perf_arm();
    if (snax_is_gemm_core()) sync[8] = snrt_mcycle();
    snrt_cluster_hw_barrier();
    const uint32_t t_org = sync[8];

    // ---- hart 3: the K/V stream ------------------------------------------
    // Spans are recorded against each core's own mcycle at the barrier below, so the three
    // lanes share an origin to within the barrier's release skew.
    // ---- hart 2: the running state, generated locally ---------------------
    // m starts at -inf, one beat per query tile. It is a constant, and the xDMA's writer
    // generates a 32-bit pattern itself, so NOTHING is fetched from main memory for it and
    // the iDMA is free from cycle zero for K(0) -- which is what the GEMM's first dispatch
    // waits on. The state blocks are a regular stride apart, so one task covers every
    // query tile whatever NQ is.
    //
    // l is NOT filled here. It starts at zero, and the arena fill before the barrier
    // already wrote zero over every byte of the state blocks, so a second task would
    // rewrite what is already there.
    //
    // 0xFBFF is the FP16 nearest -infinity, and no single byte repeats into it. That is
    // why the writer's memset takes a 32-bit PATTERN rather than a byte.
    if (snax_is_xdma_core()) {
        uint32_t t0 = t_org;

        // ---- THE HALF OF K(0) THIS ENGINE CARRIES --------------------------
        // The head of the pipeline is one transfer long: nothing computes until the
        // first key tile lands, and one DMA port moves it at ~46 B/cc. Both engines
        // are idle at cycle zero -- the iDMA has only Q to fetch and the xDMA's first
        // V is not wanted for thousands of cycles -- so K(0) is split down the middle
        // and carried by both at once. The two halves are contiguous in the same
        // buffer, so the GEMM's A stream never learns that two engines wrote it.
        //
        // Before the -inf fill, not after: `sync[7]` is not read until the softmax's
        // first tile, while `sync[2]` gates the very first dispatch.
        xdma_tile_arm((uint32_t)(l1 + k_buf[0] + KVBYTES / 2u),
                      (uint32_t)(uintptr_t)A + KVBYTES / 2u, BEAT,
                      (KVBYTES / 2u) / BEAT);
        {
            snax_xdma_task_t tk = snax_xdma_start_task();
            uint32_t fp = tk.remote ? XDMA_FINISH_REMOTE_TASK_PTR
                                    : XDMA_FINISH_LOCAL_TASK_PTR;
            uint32_t spins = 0;
            while (snax_read_xdma_cfg_reg(fp) < tk.task_id) {
                if (++spins > SPIN_LIMIT) { timeouts++; break; }
            }
        }
        sync[13] = 1;   // the xDMA's half of K(0) has landed

        uint32_t x0 = snrt_mcycle() - t0;
        xdma_fill_arm((uint32_t)mrun, 8u * BEAT, (uint32_t)NQ, 0xFBFFFBFFu);
        snax_xdma_local_wait(snax_xdma_start());
        pub[226] = x0; pub[227] = snrt_mcycle() - t0;
        sync[7] = 1;   // m is live -- the SIMD may start tile 0

        // ---- the V stream, concurrent with hart 3's K stream ---------------
        // K and V are independent tiles the pipeline needs at the same rate, so one
        // engine doing both serialises two transfers that need no ordering. V(j-2) is
        // dead only once EVERY query tile has used it -- NQ steps per KV tile.
        uint32_t xdma_busy = 0, xdma_block = 0;
        xdma_tile_arm((uint32_t)(l1 + v_buf[0]), (uint32_t)(uintptr_t)A, BEAT,
                      KVBYTES / BEAT);
        for (uint32_t j = 0; j < NKV; j++) {
            uint32_t w0 = snrt_mcycle() - t0;
            if (j >= 2) SNAX_SPIN_UNTIL(sync[3] >= (j - 1) * (uint32_t)NQ, timeouts);
            uint32_t v0 = snrt_mcycle() - t0;
            xdma_block += v0 - w0;
            xdma_tile_retask((uint32_t)(l1 + v_buf[j & 1]),
                             (uint32_t)(uintptr_t)A, KVBYTES / BEAT);
            // A main-memory read is served by the PEER, so the hardware runs it as a
            // REMOTE task; waiting on the local counter spins for ever. BOUNDED, and
            // it prints the whole engine state on timeout: an unbounded wait here
            // turns a one-line bug report into a run that never ends.
            {
                snax_xdma_task_t tk = snax_xdma_start_task();
                uint32_t ptr = tk.remote ? XDMA_FINISH_REMOTE_TASK_PTR
                                         : XDMA_FINISH_LOCAL_TASK_PTR;
                uint32_t spins = 0;
                while (snax_read_xdma_cfg_reg(ptr) < tk.task_id) {
                    if (++spins > 200000u) {
                        printf("  V(%lu) TIMEOUT  task %lu on the %s counter\n"
                               "    dst=%08lx (align%%64=%lu)  src=%08lx (align%%64=%lu)"
                               "  bound=%lu\n"
                               "    commit l/r %lu/%lu  finish l/r %lu/%lu\n",
                               (unsigned long)j, (unsigned long)tk.task_id,
                               tk.remote ? "REMOTE" : "local",
                               (unsigned long)(uintptr_t)(l1 + v_buf[j & 1]),
                               (unsigned long)((uintptr_t)(l1 + v_buf[j & 1]) % 64u),
                               (unsigned long)(uintptr_t)A,
                               (unsigned long)((uintptr_t)A % 64u),
                               (unsigned long)(KVBYTES / BEAT),
                               (unsigned long)snax_read_xdma_cfg_reg(XDMA_COMMIT_LOCAL_TASK_PTR),
                               (unsigned long)snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR),
                               (unsigned long)snax_read_xdma_cfg_reg(XDMA_FINISH_LOCAL_TASK_PTR),
                               (unsigned long)snax_read_xdma_cfg_reg(XDMA_FINISH_REMOTE_TASK_PTR));
                        timeouts++;
                        break;
                    }
                }
            }
            uint32_t v1 = snrt_mcycle() - t0;
            sync[5] = j + 1;
            pub[202 + 4 * j] = v0; pub[203 + 4 * j] = v1;
            xdma_busy += v1 - v0;
            if (timeouts) break;
        }
        pub[232] = xdma_busy;
        pub[233] = snrt_mcycle() - t0;
        pub[234] = xdma_block;
    }

    if (snrt_is_dm_core()) {
        uint32_t t0 = t_org;
        uint32_t q0 = snrt_mcycle() - t0;
        for (uint32_t qi = 0; qi < (uint32_t)NQ; qi++)
            snrt_dma_start_1d((void *)(l1 + q_buf[qi]), B,
                              N * K * tileSize * meshCol * sizeof(int8_t));
        snrt_dma_wait_all();
        uint32_t q1 = snrt_mcycle() - t0;
        pub[224] = q0; pub[225] = q1;
        dma_busy += q1 - q0;
        for (uint32_t j = 0; j < NKV; j++) {
            // Four marks, not two: the blocked time between them is the point. Bracketing
            // K and V together would charge the wait for a free buffer to the transfer and
            // make a port running at 95% look like one running at 25%.
            // K(j-2) is dead only once EVERY query tile has used it, which is NQ steps per
            // KV tile rather than one. Freeing on the step count, not the tile count.
            // V has moved to the xDMA on hart 2; this lane carries K only.
            uint32_t w0 = snrt_mcycle() - t0;
            if (j >= 2) SNAX_SPIN_UNTIL(sync[4] >= (j - 1) * (uint32_t)NQ, timeouts);
            uint32_t k0 = snrt_mcycle() - t0;
            // K(0) is split with the xDMA -- see its lane. Every later tile is
            // already hidden behind the previous tile's compute, so only the first
            // one is worth two engines.
            snrt_dma_start_1d(l1 + k_buf[j & 1], A,
                              (j == 0u) ? (uint32_t)(KVBYTES / 2u) : (uint32_t)KVBYTES);
            snrt_dma_wait_all();
            if (j == 0u) SNAX_SPIN_UNTIL(sync[13] >= 1, timeouts);
            uint32_t k1 = snrt_mcycle() - t0;
            // K(0) is two engines and every other tile is one, so they are counted
            // apart -- an average over both describes neither.
            if (j == 0u) pub[97] = k1 - k0; else pub[96] += k1 - k0;
            sync[2] = j + 1;
            pub[200 + 4 * j] = k0; pub[201 + 4 * j] = k1;
            dma_busy += k1 - k0;
            dma_block += k0 - w0;
        }
        // THE STORE. A query tile's result leaving the cluster is (O, m, l): the INT32
        // accumulator plus the eight beats of running state. On one cluster it is a drain
        // tail; KV-sharded across clusters it is the partial each shard contributes to the
        // cross-cluster fold, so it belongs in the model now rather than after the split.
        // That fold is the xDMA MonoidJunction, armed on this very transfer -- the DMA
        // below becomes the collective, and nothing above it moves. See "SHARDING THE KEYS
        // ACROSS CLUSTERS" at the top of this file for the geometry words, the `fmt`
        // trap, and why the numerator crosses in FP32 via Int32ToFp32 rather than FP16.
        // SPLIT. O is final the instant the last PV retires; only m and l wait on the SIMD's
        // trailing commit. Shipping them together charged O's 16 KiB with the SIMD's drain.
        // ONE QUERY TILE AT A TIME, AS EACH BECOMES FINAL. The step order is
        // j = kv * NQ + q, so query tile q retires at step (NKV-1)*NQ + q -- the last
        // NQ steps of the run, one per tile. Waiting for ALL of them puts every byte of
        // O in the tail; waiting per tile lets all but the last hide behind the steps
        // that follow it.
        // The span from the first store to the last now contains three WAITS, so it is
        // not a transfer time: each store is timed on its own and only those are summed.
        uint32_t s0d = 0, o_busy = 0;
        for (uint32_t qi = 0; qi < (uint32_t)NQ; qi++) {
            SNAX_SPIN_UNTIL(sync[3] >= ((uint32_t)NKV - 1u) * (uint32_t)NQ + qi + 1u,
                            timeouts);
            uint32_t q0 = snrt_mcycle() - t0;
            if (qi == 0u) s0d = q0;
            snrt_dma_start_1d((void *)((uint8_t *)fa_out + qi * (uint32_t)(BR * DHEAD) * 4u),
                              (void *)(l1 + OACC32_OF(qi)), (uint32_t)(BR * DHEAD) * 4u);
            snrt_dma_wait_all();
            o_busy += (snrt_mcycle() - t0) - q0;
        }
        pub[98] = o_busy;
        uint32_t s1d = snrt_mcycle() - t0;            // every O is away
        SNAX_SPIN_UNTIL(sync[6] >= 1, timeouts);     // the SIMD drained -- m, l are final
        uint32_t s2d = snrt_mcycle() - t0;
        snrt_dma_start_1d((void *)((uint8_t *)fa_out + (uint32_t)NQ * (uint32_t)(BR * DHEAD) * 4u),
                          (void *)stateq, (uint32_t)NQ * 8u * BEAT);
        snrt_dma_wait_all();
        uint32_t s3d = snrt_mcycle() - t0;
        pub[230] = s2d; pub[231] = s3d;
        pub[228] = s0d; pub[229] = s1d;
        dma_store = pub[98] + (s3d - s2d);   // transfer only, not the waits between
        dma_busy += dma_store;

        dma_wall = snrt_mcycle() - t0;
        pub[223] = dma_store;
        pub[220] = dma_busy;
        pub[221] = dma_wall;
        pub[222] = dma_block;
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
#define STAGE_QK(t)                                                              \
        do {                                                                     \
            const uint32_t sk = (t) / (uint32_t)NQ, sq = (t) % (uint32_t)NQ;     \
            gemm_set_shape(s1_k, s1_m, s1_blk, s1_as2, s1_bs1, 0u);              \
            gemm_set_bases(l1u + k_buf[sk & 1], l1u + q_buf[sq], base_c,         \
                           l1u + (uint32_t)d32_delta[(t) & 1]);                  \
            gemm_d32_emit_fp16(1, d32_b0_f, d32_s1_f, d32_s2_f);                 \
        } while (0)
        // Every query tile keeps its own accumulator, so each one's first KV tile
        // masks the C reader off rather than clearing oacc32: a disabled channel
        // presents ZERO and issues no TCDM request, which is the seed it needs.
#define STAGE_PV(t)                                                              \
        do {                                                                     \
            const uint32_t sk = (t) / (uint32_t)NQ, sq = (t) % (uint32_t)NQ;     \
            const uint32_t ob = l1u + OACC32_OF(sq);                             \
            gemm_set_shape(s2_k, s2_m, s2_blk, s2_as2, s2_bs1,                   \
                           sk == 0u ? 0u : 0xFFFFFFFFu);                         \
            gemm_set_bases(l1u + v_buf[sk & 1],                                  \
                           l1u + (uint32_t)p8_delta[(t) & 1], ob, ob);           \
            gemm_d32_emit_fp16(0, d32_b0_i, d32_s1_i, d32_s2_i);                 \
        } while (0)

        // Dispatch 0 has no predecessor to hide behind. Writing it here puts its
        // cold instruction fetches inside the wait for K(0), which is dead time.
#ifdef GEMM_QUEUE
        // ---- QUEUED DISPATCH --------------------------------------------------
        // See the decode kernel for the full rationale. In short: with cfgQueueDepth = 2
        // in both the streamer's and the array's CSR managers, a start write snapshots
        // the bank and retires, so dispatch d is armed AND started while d-1 still runs
        // and the array takes it the cycle it retires. Prefill is where this should pay:
        // it has 31 dispatch boundaries, each measured at ~43 cycles of dead array time.
        //
        // The per-dispatch counters are dropped here on purpose -- they restart on every
        // start pulse, so once d+1 has begun they do not describe d. Build without
        // GEMM_QUEUE for the stall decomposition.
        //
        // The order is QK(0), then QK(t) and PV(t-1) per step, then a trailing PV.
        uint8_t sk[2 * STEPS], si[2 * STEPS];
        {
            uint32_t d = 0;
            sk[d] = 0; si[d++] = 0;
            for (uint32_t j = 1; j <= (uint32_t)STEPS; j++) {
                if (j < (uint32_t)STEPS) { sk[d] = 0; si[d++] = (uint8_t)j; }
                sk[d] = 1; si[d++] = (uint8_t)(j - 1);
            }
        }
        uint32_t tid_q[2] = {0u, 0u};
        uint32_t have_prev = 0u, prev_d = 0u;
        for (uint32_t d = 0; d <= 2u * (uint32_t)STEPS; d++) {
            if (d < 2u * (uint32_t)STEPS) {
                const uint32_t kind = sk[d], t = si[d];
                // Arm first: the predecessor snapshotted the bank on its own start write,
                // so these 21 writes land inside the dependency wait instead of after it.
                if (kind == 0u) STAGE_QK(t); else STAGE_PV(t);
                {
                    uint32_t w0 = snrt_mcycle();
                    if (kind == 0u) {
                        if (t >= 2u) SNAX_SPIN_UNTIL(sync[1] >= t - 1, timeouts);
                        SNAX_SPIN_UNTIL(sync[2] >= t / (uint32_t)NQ + 1, timeouts);
                    } else {
                        SNAX_SPIN_UNTIL(sync[1] >= t + 1, timeouts);
                        SNAX_SPIN_UNTIL(sync[5] >= t / (uint32_t)NQ + 1, timeouts);
                    }
                    gemm_stall += snrt_mcycle() - w0;
                }
                pub[(kind == 0u ? 10u : 12u) + 4u * t] = snrt_mcycle() - g0;
                tid_q[d & 1u] = ++gemm_seq;
                gemm_launch();
                gemm_ack();
            }
            if (have_prev) {
                const uint32_t kind = sk[prev_d], t = si[prev_d];
                timeouts += gemm_wait(tid_q[prev_d & 1u]);
                pub[(kind == 0u ? 11u : 13u) + 4u * t] = snrt_mcycle() - g0;
                if (kind == 0u) { sync[0] = t + 1; sync[4] = t + 1; }
                else            { sync[3] = t + 1; }
            }
            have_prev = (d < 2u * (uint32_t)STEPS);
            prev_d = d;
        }
#else
        STAGE_QK(0u);

        for (uint32_t j = 0; j <= STEPS; j++) {
            uint32_t tid = 0;
            if (j < STEPS) {
                // S(j) into the buffer the SIMD core is not reading. It last
                // held tile j-2, which is free once tile j-1 has been consumed.
                {
                    uint32_t w0 = snrt_mcycle();
                    if (j >= 2) SNAX_SPIN_UNTIL(sync[1] >= j - 1, timeouts);
                    SNAX_SPIN_UNTIL(sync[2] >= j / (uint32_t)NQ + 1, timeouts);  // K(kv) landed
                    gemm_stall += snrt_mcycle() - w0;
                }
                pub[10 + 4 * j] = snrt_mcycle() - g0;
                tid = ++gemm_seq; gemm_launch();   // S^T = K.Q^T, staged one dispatch ago
                gemm_ack();
                // The successor is PV(j-1) in this same iteration, except at j == 0
                // where nothing precedes it and the next dispatch is QK(1).
                if (j == 0u) STAGE_QK(1u); else STAGE_PV(j - 1u);
                pub[80 + 2 * j] = snrt_mcycle() - g0;  // successor staged, array running
                timeouts += gemm_wait(tid);
                { uint32_t c = csrr_ss(GEMMX_PERFORMANCE_COUNTER);
                  gemm_cycles += c; gemm_cyc_s1 += c;
                  gemm_sa_s1 += csrr_ss(GEMMX_STALL_A);
                  gemm_sb_s1 += csrr_ss(GEMMX_STALL_B);
                  gemm_sd_s1 += csrr_ss(GEMMX_STALL_D); }
                gemm_stream_cycles += csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
                pub[11 + 4 * j] = snrt_mcycle() - g0;
                sync[0] = j + 1;  // publish AFTER the streamer has drained
                sync[4] = j + 1;  // one more QK step done -- see the iDMA's K-free test
            }
            if (j > 0) {
                // O(j-1) = P(j-1).V. Waiting here is what lets S(j) above run
                // concurrently with the SIMD core's tile j-1.
                uint32_t w0 = snrt_mcycle();
                SNAX_SPIN_UNTIL(sync[1] >= j, timeouts);
                SNAX_SPIN_UNTIL(sync[5] >= (j - 1) / (uint32_t)NQ + 1, timeouts);  // V landed
                gemm_stall += snrt_mcycle() - w0;
                pub[12 + 4 * (j - 1)] = snrt_mcycle() - g0;
                tid = ++gemm_seq; gemm_launch();   // O^T = V^T.P^T, staged one dispatch ago
                gemm_ack();
                // The successor is QK(j+1) while KV tiles remain, otherwise the
                // next iteration's PV, and nothing at all after the last.
                if (j + 1u < (uint32_t)STEPS)      STAGE_QK(j + 1u);
                else if (j < (uint32_t)STEPS)      STAGE_PV(j);
                pub[81 + 2 * (j - 1)] = snrt_mcycle() - g0;
                timeouts += gemm_wait(tid);
                { uint32_t c = csrr_ss(GEMMX_PERFORMANCE_COUNTER);
                  gemm_cycles += c; gemm_cyc_s2 += c;
                  gemm_sa_s2 += csrr_ss(GEMMX_STALL_A);
                  gemm_sb_s2 += csrr_ss(GEMMX_STALL_B);
                  gemm_sd_s2 += csrr_ss(GEMMX_STALL_D); }
                gemm_stream_cycles += csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
                pub[13 + 4 * (j - 1)] = snrt_mcycle() - g0;
                sync[3] = j;  // one more PV step done -- see the iDMA's V-free test
            }
        }
#endif  /* GEMM_QUEUE */
        gemm_wall = snrt_mcycle() - g0;
#undef STAGE_QK
#undef STAGE_PV
    }

    if (snax_is_simd_core()) {
        uint32_t s0 = t_org;
        uint32_t simd_busy0 = snax_simd_busy_cycles();

        for (uint32_t i = 0; i < (uint32_t)STEPS; i++) {
            const uint32_t q = i % (uint32_t)NQ;
            const uint32_t so = ST_OFF(q);
            uint32_t w0 = snrt_mcycle();
            SNAX_SPIN_UNTIL(sync[0] >= i + 1, timeouts);
            if (i == 0) SNAX_SPIN_UNTIL(sync[7] >= 1, timeouts);
            simd_stall += snrt_mcycle() - w0;

            pub[120 + 2 * i] = snrt_mcycle() - s0;
            pub[180 + i] = snax_simd_busy_cycles() - simd_busy0;
            sh[0].base = l1 + d32_delta[i & 1];
            // The latch beat and the tile behind it both live in the buffer the GEMM just
            // wrote, so the epilogue's read and the -m_new fan-out follow it too.
            sh[5].base      = l1 + s_hdr[i & 1];
            sh[5].stride[0] = (uint32_t)(rmax + so - (l1 + s_hdr[i & 1]));
            sh[14].base     = l1 + s_hdr[i & 1];
            sh[17].base = l1 + p8_delta[i & 1];
            sh[21].base = LSC_OF(i);
            sh[24].base = RSUM_OF(i);
            // EVERY shape that touches running state moves to this step's query tile. Only
            // the base moves: the eight beats keep their order, so the strides that pair
            // operands by adjacency are unchanged.
            sh[1].base  = rmax  + so;   // the rowmax DESTINATION -- missing this sends every
            sh[2].base  = rmax  + so;   // tile's maximum to tile 0's slot
            sh[3].base  = mnew  + so;
            sh[4].base  = mnew  + so;
            sh[8].base  = rmax  + so;
            sh[11].base = corrL + so;
            sh[20].base = corrL + so;  sh[25].base = lnew  + so;
            sh[28].base = mnew  + so;  sh[29].base = mrun  + so;

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

            // 5+6 FUSED: corr = exp(m_old - m_new), one pass.
            //
            // The chain's fixed order [EW0, Map, Reduce, EW1, Fp16ToInt8] puts the
            // combine BEFORE the transform, so EW0 with operandCount 2 folds the
            // adjacent pair [-m_new][m_old] into one beat and Map exponentiates it in
            // flight: the difference is never written to memory and never read back.
            //
            // The l update's corr*l_old followed by + rowsum cannot share a task the
            // same way: its combine-after-combine would need a second operand fetched
            // at EW1, and the chain feeds EW1 only from the stage above it.
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                        (1u << SIMD_EXT_STREAMMAP));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 0, 2);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 1, SIMD_EW_ADD);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0, SIMD_F32_ONE);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 1, 0);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 2, SIMD_FUNC_EXP);
            snax_simd_program_1d(&sh[8], &sh[11]);
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
                        printf("SIMD STALL step %u (publish): submitted=%u finished=%u status=%08x\n",
                               i, want, snax_read_simd_cfg_reg(SIMD_FINISHED_TASK_PTR),
                               snax_read_simd_cfg_reg(SIMD_STATUS));
                        timeouts++;
                        break;
                    }
                }
            }
            if (timeouts) break;
            sync[1] = i + 1;

            pub[121 + 2 * i] = snrt_mcycle() - s0;

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
            pub[160 + i] = snrt_mcycle() - s0;


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
        pub[180 + (uint32_t)STEPS] = snax_simd_busy_cycles() - simd_busy0;
        sync[6] = 1;  // m, l and O are final -- the store may go
        simd_cycles = snax_simd_busy_cycles() - simd_busy0;
        simd_wall = snrt_mcycle() - s0;
    }
    snrt_cluster_hw_barrier();

    // Read the contention counters before anything below it runs: the golden
    // compare and the report walk TCDM and print, which is traffic the kernel
    // does not do and which would otherwise land inside the census.
    if (snax_is_gemm_core()) snax_perf_read(&perf);

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
        // The LAST STEP belongs to query tile NQ-1, and every query tile sees the same K,
        // so checking that one checks the recurrence for all of them.
        const uint32_t ql = (uint32_t)(STEPS - 1) % (uint32_t)NQ;
        const uint32_t sl = ST_OFF(ql);
        volatile int8_t *p8t = (volatile int8_t *)(l1 + p8_delta[(STEPS - 1) & 1]);
        volatile uint8_t *rsum = RSUM_OF(STEPS - 1);
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
                       i, ones, (int)want_ones, fp16_at(mrun + sl, i), fp16_at(rsum, i),
                       fp16_at(lrun + sl, i));
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
                uint16_t r = fp16_at(rsum, i), lv = fp16_at(lrun + sl, i);
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
            uint32_t u = fp16_ulp(fp16_at(mrun + sl, i), m_golden[i]);
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
            volatile int32_t *o32 = (volatile int32_t *)(l1 + OACC32_OF(ql));
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
        if (pub[221] > pipeline) pipeline = pub[221];  // the store tail is part of the window
        if (pub[233] > pipeline) pipeline = pub[233];  // and so is the V stream's tail
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
               gemm_cyc_s1, (unsigned)(STEPS * M * N * K),
               gemm_cyc_s1 / (unsigned)(STEPS * M * N * K),
               (gemm_cyc_s1 * 100u / (unsigned)(STEPS * M * N * K)) % 100u,
               (int)(STEPS * M * N), (int)K);
        printf("    stalls         A %5u  B %5u  D %5u  (feed %u%%, drain %u%%)\n",
               gemm_sa_s1, gemm_sb_s1, gemm_sd_s1,
               100u * (gemm_sa_s1 + gemm_sb_s1) / (gemm_cyc_s1 ? gemm_cyc_s1 : 1u),
               100u * gemm_sd_s1 / (gemm_cyc_s1 ? gemm_cyc_s1 : 1u));
        printf("  O^T=V^T.P^T      %5u cycles for %5u array passes (%u.%02u cyc/pass,"
               " %d blocks x %d accum)\n",
               gemm_cyc_s2, (unsigned)(STEPS * S2_M * S2_N * S2_K),
               gemm_cyc_s2 / (unsigned)(STEPS * S2_M * S2_N * S2_K),
               (gemm_cyc_s2 * 100u / (unsigned)(STEPS * S2_M * S2_N * S2_K)) % 100u,
               (int)(STEPS * S2_M * S2_N), (int)S2_K);
        printf("    stalls         A %5u  B %5u  D %5u  (feed %u%%, drain %u%%)\n",
               gemm_sa_s2, gemm_sb_s2, gemm_sd_s2,
               100u * (gemm_sa_s2 + gemm_sb_s2) / (gemm_cyc_s2 ? gemm_cyc_s2 : 1u),
               100u * gemm_sd_s2 / (gemm_cyc_s2 ? gemm_cyc_s2 : 1u));
        printf("  pipeline         %5u cycles, %u per step (NKV=%d x NQ=%d)\n", pipeline,
               pipeline / STEPS, NKV, NQ);
        // Per core: what it spent running its engine, blocked on the other
        // core, and writing CSRs. The three must sum to that core's wall.
        printf("  GEMM core        busy %5u (%2u%%)  peer-wait %5u  config %5u\n",
               gemm_cycles, 100u * gemm_cycles / pipeline, gemm_stall,
               gemm_wall > gemm_cycles + gemm_stall
                   ? gemm_wall - gemm_cycles - gemm_stall : 0);
        // config = wall - busy - peer-wait: CSR writes, the 5 RO counter reads per
        // dispatch (5.1 cycles each, instrumentation only), and the busy-rise poll.
        printf("  iDMA core        moving %5u (%2u%%)  blocked on a buffer %5u  wall %5u\n",
               pub[220], 100u * pub[220] / pipeline, pub[222], pub[221]);
        // K and V are on SEPARATE engines. Their transfers overlap, so the two
        // "moving" figures are concurrent and must not be added into one port's load.
        printf("  xDMA core        moving %5u (%2u%%)  blocked on a buffer %5u  wall %5u\n",
               pub[232], 100u * pub[232] / pipeline, pub[234], pub[233]);
        printf("    init           %u B cleared in %u cc  [%u -> %u]\n",
               (uint32_t)NQ * 2u * SIMD_BEAT_BYTES,
               pub[227] - pub[226], pub[226], pub[227]);
        printf("    Q load         %u B in %u cc  [%u -> %u]\n",
               (uint32_t)NQ * (uint32_t)(N * K * tileSize * meshCol), pub[225] - pub[224], pub[224], pub[225]);
        printf("    store O        %u B in %u cc of transfer, staged over %u cc  [%u -> %u]"
               "   then m,l in %u cc  [%u -> %u]\n",
               (uint32_t)NQ * (uint32_t)(BR * DHEAD) * 4u, pub[98],
               pub[229] - pub[228], pub[228], pub[229],
               pub[231] - pub[230], pub[230], pub[231]);
        printf("    K(0) split     %u B in %u cc = %u.%u B/cc across TWO engines\n",
               KVBYTES, pub[97], KVBYTES / pub[97], ((KVBYTES * 10u) / pub[97]) % 10u);
        printf("    K per tile     %u B in %u cc = %u.%u B/cc of a 64 B/cc port  (iDMA, tiles 1+)\n",
               KVBYTES, pub[96] / (NKV - 1),
               (KVBYTES * (NKV - 1)) / pub[96],
               ((KVBYTES * (NKV - 1) * 10u) / pub[96]) % 10u);
        printf("    V per tile     %u B in %u cc = %u.%u B/cc of a 64 B/cc port  (xDMA)\n",
               KVBYTES, pub[232] / NKV,
               (KVBYTES * NKV) / (pub[232] ? pub[232] : 1u),
               ((KVBYTES * NKV * 10u) / (pub[232] ? pub[232] : 1u)) % 10u);
        // Raw spans, for the trace: every engine's own start/end per dispatch or tile,
        // measured against that core's mcycle at the loop entry.
        printf("  TRACE gemm");
        for (uint32_t i = 0; i < 4u * STEPS; i++) printf(" %u", pub[10 + i]);
        printf("\n  TRACE gcfg");
        for (uint32_t i = 0; i < 2u * STEPS; i++) printf(" %u", pub[80 + i]);
        printf("\n  TRACE simd");
        for (uint32_t i = 0; i < 2u * STEPS; i++) printf(" %u", pub[120 + i]);
        printf("\n  TRACE send");
        for (uint32_t i = 0; i < STEPS; i++) printf(" %u", pub[160 + i]);
        printf("\n  TRACE sbusy");
        for (uint32_t i = 0; i <= STEPS; i++) printf(" %u", pub[180 + i]);
        printf("\n  TRACE idma");
        for (uint32_t i = 0; i < 4u * NKV; i++) printf(" %u", pub[200 + i]);
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
        snax_perf_report(&perf, pipeline);
    }
    snrt_cluster_hw_barrier();
    return 0;
}
