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
//     S^T = (Q_i . K_j^T)^T = K_j . Q_i^T      [Bc, Br]   (S is [Br, Bc])
//     P^T = exp(S^T - m)                       [Bc, Br]   m is still per query row
//     O^T = (P . V_j)^T     = V_j^T . P^T      [d,  Br]   (O is [Br, d])
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
// WHY IT IS WORTH DOING. The SIMD engine sees memory as a stream of BEATS -- 256 bits =
// 16 FP16 lanes, one beat per cycle -- and its reductions are LANEWISE: lane k accumulates
// across successive beats, in the accumulator registers it already carries. With 4 lanes
// instead of 16, a MAX reduction over three beats looks like this:
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
// Reducing ACROSS the lanes of one beat -- max(3, 9, 1, 4) in the picture above -- is a
// different and far more expensive thing: a log-depth fold across the lanes, then a scalar
// splatted over a whole output beat. Softmax reduces per QUERY ROW, so the layout alone
// decides which of the two the rowmax and the rowsum get:
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
// rowmax costs nothing beyond reading the tile. The first costs Br folds, and Br scalar
// beats to use them from. A free rewrite buys the whole difference.
//
// Everything downstream inherits the orientation. m is one beat per 16 query rows (two
// beats for Br = 32), not Br scalars, so S - m is a broadcast beat per half rather than Br
// separately armed scalar broadcasts; corr, l and the O rescale are all the same shape.
//
// This kernel also stops before the final O / l.
//
// That is also why it is the workload for this cluster. The per-tile epilogue is nothing
// but reductions and pointwise transcendentals, which is exactly what the SIMD datapath
// extensions do, and it runs while the GEMM is already busy with the next tile.
//
// (The operands are full-range INT8, so a score reaches 127^2 * d = 2,064,512, past FP16's
// 65,504. The temperature a is split: 2^-D32_FP16_SHIFT on the D-port converter while it
// rounds to FP16, and the remainder a' in StreamMap's exp. Neither costs a pass.)
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
// so it reduces each 512-beat plane down to one beat and writes nothing else. Each plane
// carries a spare beat in front for exactly that reason: the epilogue is a sticky-B pass,
// its broadcast operand has to sit immediately before the plane, and that beat is where
// the rowmax writes m16. A tap would have had to COPY the whole tile to put it behind a
// latch.
//
// ONE KV TILE. Br = 32 query rows, Bc = 512 keys, d = 128, NKV = 4 tiles. Bc and d are
// INDEPENDENT, so the two matmuls have different shapes; gemm_set_shape() switches between
// them per dispatch, rewriting only the ten CSRs that depend on M and K. A SIMD beat is
// 256 bits = 32 B = 16 FP16 lanes, and Br = 32 queries are TWO beats per key.
//
// Where those 16 lanes come from is what the layout rests on. The array is meshCol = 16
// columns wide, so it emits the tile as [M][N][meshRow][meshCol] blocks: one block row is
// one key's scores for 16 queries -- exactly one beat (the kernel requires
// SIMD_FP16_LANES == meshCol). The D port therefore lays S^T down as NH = Br / meshCol
// PLANES, plane n holding queries 16n..16n+15, one beat per key: its 16 channels write
// 4 key rows contiguously, and its N loop steps a whole plane. A LANEWISE reduce down a
// plane is then one query per lane -- the premise everything below rests on -- and each
// plane carries a spare beat on either side for the softmax (see THE SCORE PLANES in main()):
//
//     [ latch_n ][ S^T plane n: key 0 .. key Bc-1 ][ mold_n ]        n = 0, 1
//
//   Every name carries its PRECISION: 8 = INT8, 16 = FP16, 32 = INT32. Beat counts are
//   in -> out, and a beat is 16 lanes. A task over both halves names the halves h.
//
//   GEMM (hart 0)  S16^T = K8 . Q8^T        K8 [512,128]  64 KiB  ]  2.10 M MAC
//                                         Q8^T [128, 32]   4 KiB  ]  2048 cycles
//                                     -> S16^T [512, 32]  32 KiB     at 1024 MAC/cycle
//                     the mesh computes INT32, and Int32ToFp16 on the D32 writer port
//                     converts it FREE as the tile drains -- which also HALVES the bytes
//                     written, so S16^T reaches TCDM as 2 x 512 FP16 beats
//   SIMD (hart 1), in the order the loop fires them:
//
//                  m16   = max(mold_h, rowmax(S16^T_h))  2 x 513 -> 2  Reduce MAX|LANEWISE
//                          written into latch_h, the beat in front of each plane
//
//                       [ latch_h ][ S16^T plane h  x512 ][ mold_h ]
//                         \___ m16, the         \______________/
//                              epilogue's        ONE row of the rowmax: the running
//                              sticky operand    maximum is the row's last beat, so the
//                                                same lanewise MAX that finds rowmax
//                                                also folds in m16_old
//
//                  corr16 = exp(a*(m16_old - m16))   4 -> 2  FUSED: EW0 (SUB, pairs) -> Map (EXP)
//
//                       the pairs [ latch_h ][ mold_h ] = [ m16 ][ m16_old ], read at one
//                       stride, and SUB is x - B = m16_old - m16; corr16 lands in this
//                       step's corr slot, where the l update and PV of the same step --
//                       which loads it into the C column scaler to rescale that query
//                       tile's O as it streams back into the array -- both find it
//
//                  P8^T_h = int8(127*exp(a*(S16^T_h - m16)))  ) FUSED: EW0 (sticky SUB)
//                                         513 -> 257, per h   ) -> Map (EXP) -> Reduce TAP
//                  sum16_h = sum of P16_h over keys           ) -> Fp16ToInt8 (interleave,
//                                                             )    tail passthrough)
//
//                     THE WHOLE EPILOGUE IS ONE SWEEP PER HALF. One task per half, because
//                     a sticky latch is one beat. sum16 has no task of its own: it is the
//                     TAP beat, written after the half. And the quantiser runs in the same
//                     pass because its tail passthrough leaves that trailing beat alone --
//                     without it the scalar would be narrowed along with the data, and P
//                     would have to be written in FP16 and read back by a separate
//                     quantise task: a third trip over the tile.
//
//                       read   [ latch_h ][ S16^T plane h  x512 ]    the latch, then the half
//                       write  [ P8^T B blocks (k, h)  x256 ][ sum16_h ]   half h of P8
//                                                                \___ lsc16_h gets the beat
//                                                                     behind it later
//
//                     ---- P8^T is published to the GEMM here. Everything below only
//                          prepares the NEXT tile, so the GEMM never waits on it. ----
//
//                  lsc16  = corr16 * l16_old         4 -> 2  StreamElementwise MUL, pairs
//                  l16    = sum16 + lsc16            4 -> 2  StreamReduce ADD|LANEWISE
//                  mold_h <- m16                     2 -> 2  a copy, no operator
//
//                     Old and new never need a "commit". m16_old lives in the planes' mold
//                     beats and m16 in their latch beats, so the rowmax writes the new value
//                     without touching the old one that corr still needs; l16_old is
//                     consumed into lsc16 before the l update overwrites it in place. Each
//                     query tile has a score buffer of its own, so the one copy carries m16
//                     from the latches into the mold beats of the SAME buffer, which is all
//                     the state the tile's next step needs; after its last step it carries
//                     m16 into the state vector m instead.
//
//   GEMM (hart 0)  O32^T += V8^T . P8^T    V8^T [128,512]  64 KiB  ]  2.10 M MAC
//                                          P8^T [512, 32]  16 KiB  ]  2048 cycles
//                                     -> O32^T [128, 32]  16 KiB     accumulated in place
//
//   O32^T is the ONLY representation of O. An FP16 copy rescaled by corr each tile would
//   have no reader at all -- see "O^T PER QUERY TILE, in ONE representation" in the
//   allocation block.
//
// Per KV tile that is 4.19 M MAC and 4096 GEMM cycles against ~2,070 beats read and ~525
// written on the SIMD side (66 KiB in, 17 KiB out), so the arithmetic floor is GEMM-bound
// and the softmax has to fit inside its shadow. Neither engine is short of time; TCDM
// BANDWIDTH is what they contend for, which is why the beats written matter as much as the
// cycles. The whole run is NKV of these. Every buffer above lives in TCDM at once; the
// footprint guard in main() reports the total against the cluster's TCDM size.
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
// WHAT IS VALIDATED. Every KV tile has its own K and V and every query tile its own Q, so
// the running max moves and corr is not 1. The checks: P8 == 127 counts and sampled P8 of
// the last step (read through the interleaved layout), m and l of every query tile, the last
// row sum, and O of every query tile against the attention math -- V^T.P^T summed over the
// tiles WITH the online rescale -- within a per-element tolerance, since the SIMD
// exponential is a LUT.
//
// ============================ SHARDING THE KEYS ACROSS CLUSTERS ============================
//
// Everything above is ONE cluster walking NKV key/value tiles in sequence. The multi-cluster
// form cuts the SAME sequence into P slices and gives one to each cluster, which runs this
// whole file on it unchanged. Q is replicated -- every cluster attends the same query tile.
//
//      K, V [N, d]  INT8                  what cluster c holds when its loop ends
//       +---------------+  -+-           m_c [Br] FP16   l_c [Br] FP16   O_c [d, Br] INT32
//   c=0 |   keys 0..511 |   |    ->        m_st            l_st            oacc32
//       +---------------+   |            (64 B)          (64 B)          (16 KiB)
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
//   S^T        s_buf[] planes            FP16        GEMM output via Int32ToFp16
//   P8^T       p8_buf[]                  INT8        quantised by Fp16ToInt8
//   m_c        m_st    [32 lanes]        FP16        the KEY field of a partial
//   l_c        l_st    [32 lanes]        FP16        a twisted VALUE field
//   O_c        oacc32  [d, Br]           INT32  <--  Int32ToFp32, on the xDMA reader
//   ---------------------------------------------------------------------------------
//   alpha_c    inside the junction       FP32        derived from the keys, never stored
//   the fold   inside the junction       FP32        the ARITHMETIC is FP32, always
//   transport  on the wire               MIXED       FP16 for (m, l), FP32 for O
//
// The last two lines are what to hold on to. `fmt` selects how a beat is SLICED and
// nothing else -- every compare, exponential and FMA inside the operator is FP32 whatever
// it says. And the two folds need not agree: (m, l) rides FP16 because m_st and l_st
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
// This is the part that bites, because it is not how the kernel stores anything. The
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
//     m_st  [32 FP16, 64 B = 1 xDMA beat]  --+
//                                           +-->  4 junction beats, 16/32 lanes live, 256 B
//     l_st  [32 FP16, 64 B = 1 xDMA beat]  --+      (128 B of payload, 128 B of identity pad)
//
// m_st and l_st are already one FP16 value per query row, which IS the slot axis -- but they
// are two separate 32-row vectors, and the junction wants 8 rows of (m, l) per beat. That
// is a strided 2-beat xDMA task.
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
//     |   HasInt32ToFp32     INT32 -> FP32, one PE per lane    |  <-- the conversion
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
// accuracy and no re-embedding code at all.
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
//   1. THE STORE BECOMES THE COLLECTIVE. Here it is a plain snrt_dma_start_1d of oacc32 and
//      the running state into fa_out. Sharded, it is an xDMA task with READER_EXT_INT32TOFP32
//      armed on the reader side and the monoid on the writer side, plus a hop chain. The
//      existing split survives unchanged -- O is final when the last PV retires, m and l wait
//      on the SIMD's trailing commit -- because (m, l) and O are two separate folds with two
//      different geometry words anyway.
//
//   2. TWO GEOMETRY WORDS, and they do NOT share a transport:
//        (m, l)   F=2,  n=1,  nExp=1,  sigma=3, fmt=FP16 -> 0x0C040108    4 beats,   256 B
//        O chunk  F=16, n=15, nExp=15, sigma=0, fmt=FP32 -> 0x003C3F01  288 beats, 18432 B
//      (m, l) stays FP16 because m_st and l_st ARE FP16 already -- that fold needs no
//      conversion at all, which is what makes the FP16 transport work worth having. O goes
//      FP32 for the reasons above. Mixing is free: separate transfers, separate CSR words.
//      NOTE THE TRAP: `fmt` is CSR(0)[14:12] and ZERO MEANS FP16, so a word that leaves
//      the field clear is a legal FP16 word and is accepted silently. Name FP32.
//
//   3. `m` HAS TO BE WIDENED FP16 -> FP32 for the O fold's key -- 32 scalars per shard per
//      query tile, integer bit-work on the exponent (`fp16_bits_to_fp32_bits` already exists
//      as a static inline duplicated in two SIMD apps; lift it rather than copy it a third
//      time). That is the ONLY scalar conversion: FP32 transport needs no exponent
//      extraction, key arithmetic or scaling pass, all of which the FP16 route would.
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
// PRE-map elementwise (SUB, sticky and pairs), the row sum (ADD), and the tail passthrough
// that lets the same pass quantise the tile while the reduce appends its scalar.
// tailPassthrough is a BUILD-time feature, not a runtime select: without it csr(1) is not
// there at all and the SIMD_QUANT_TAIL write below lands on the next extension's CSR block.
#if !defined(SIMD_EXT_STREAMREDUCE_HAS_MAX) ||          \
    !defined(SIMD_EXT_STREAMREDUCE_HAS_ADD) ||          \
    !defined(SIMD_EXT_STREAMMAP_HAS_EXP) ||             \
    !defined(SIMD_EXT_STREAMELEMENTWISE_0_HAS_SUB) ||   \
    !defined(SIMD_EXT_FP16TOINT8_HAS_TAILPASSTHROUGH) || \
    !defined(SIMD_EXT_FP16TOINT8_HAS_INTERLEAVE4)
#error \
    "This cluster's SIMD block cannot run the FlashAttention softmax: it needs StreamReduce MAX+ADD, StreamMap EXP, a PRE-map StreamElementwise SUB, and Fp16ToInt8 tailPassthrough + interleave."
#endif
// ONE SIMD BEAT IS ONE ARRAY OUTPUT ROW: meshCol queries of one key. The score planes, the
// per-half epilogue and the P8 layout all rest on it, so the SIMD core's lane count (its
// num_channel in the cluster cfg) must equal the array's column count -- a design point,
// not a parameter this kernel adapts to.
#if SIMD_FP16_LANES != meshCol
#error "FlashAttention needs one SIMD beat per array output row: SIMD_FP16_LANES must equal meshCol."
#endif
// datagen.py derives the descriptors and the goldens from these layouts; the kernel
// allocates its buffers from the same two pitches, so check they describe the same thing.
#if S_PITCH != (BC + 2) * SIMD_BEAT_BYTES || P_PITCH != BC * meshCol + 2 * SIMD_BEAT_BYTES
#error "data.h's S_PITCH / P_PITCH do not match this kernel's plane layout -- regenerate data.h"
#endif
#include "snax-versacore-to-lib.h"
// The online rescale of O runs on the GEMM's C read path (Int32ColumnScale, reader_writer
// slot 0: enable + N + 16 factor words = 18 CSRs), and the D-port converter must be the
// shift build (enable + extra-loop policy + k = 3 CSRs). See snax-flashattn-decode.c.
#if !defined(READER_WRITER_EXTENSION_0_CSR_BASE) || READER_WRITER_EXTENSION_0_CSR_NUM != 18
#error "This cluster's GEMM streamer has no Int32ColumnScale on the C read path."
#endif
#if !defined(READER_WRITER_EXTENSION_1_CSR_NUM) || READER_WRITER_EXTENSION_1_CSR_NUM != 3
#error "The GEMM's D-port Int32ToFp16Converter has no power-of-two shift (build it with shift: 1)."
#endif
#define COLSCALE_CSR READER_WRITER_EXTENSION_0_CSR_BASE
#include "snax-xdma-lib.h"
#include "snrt.h"
#include "snax-perf-census.h"
#include "snax-tcdm-priority.h"

// The TCDM arbitration policy for the whole run (snax-tcdm-priority.h, SNAX_TCDM_POLICY_*):
// 3, every GEMM port pinned above the hardware's grading. Build with
// EXTRA_CFLAGS=-DFA_TCDM_PRIO=n to try another.
//
// WHY THE GEMM GOES FIRST. Prefill is GEMM-bound: the array has a step's QK and PV to run
// while the SIMD core needs about four fifths of that for its softmax, so every cycle the
// array's A or B reader loses a bank to the SIMD is a cycle of the pipeline, and a cycle the
// SIMD loses comes out of its slack. Under the hardware's own grading the SIMD wins too
// often: it consumes one 32 B beat EVERY cycle, so its reader FIFO runs near empty and bids
// high urgency on every request, against A and B readers that are usually ahead.
// Measured (vsim, NQ = NKV = 4, P8_V_PHASE = 192):
//
//   policy           pipeline   QK cyc/pass   PV cyc/pass   SIMD busy
//   hardware (0)     76,556     1.07          1.06          48,799 (63%)
//   GEMM first (3)   72,749     1.01          1.01          60,502 (83%)
//
// The SIMD pays for it in bank denials (44% of its requests) and still publishes every step
// before the PV that needs it, because the GEMM runs QK two steps ahead (THE DISPATCH ORDER).
#ifndef FA_TCDM_PRIO
#define FA_TCDM_PRIO 3
#endif
// The starvation guard's wait in cycles, 0 for the hardware's reset value (-DFA_TCDM_GUARD=n).
#ifndef FA_TCDM_GUARD
#define FA_TCDM_GUARD 0
#endif

// The matmul engine on hart 0 is VersaCore: a 1024-MAC INT8 array whose single spatial
// unrolling (Mu, Ku, Nu) is what this kernel calls meshRow/tileSize/meshCol = 16/4/16. It
// has ONE output port, carrying INT32 with an optional convert to FP16 on the way out, and
// no rescale unit -- this kernel needs neither a quantised output nor a rescale.
//
// meshRow, tileSize, meshCol, BR, BC, DHEAD, NKV, NQ and D32_FP16_SHIFT all arrive from data.h. They
// are derived there from the same M/N/K and array shape that produced the streamer
// descriptors, so the kernel's idea of the tile and the descriptors' idea of it cannot
// drift apart. Change them in data/params.hjson.

// BC and DHEAD are independent. Bc is the tiling knob, free to grow until TCDM is
// full; d is a property of the model. The two matmuls therefore have different
// shapes, which gemm_set_shape() switches between per dispatch.

// STEPS is the flattened (KV tile, query tile) sequence the pipeline walks. Decode is this
// kernel at NQ = 1; prefill is the same kernel with more query tiles sharing each K/V pass.
#define STEPS  (NKV * NQ)
// The query tile is NH SIMD beats wide: Br = NH * meshCol queries, one N column block of
// the array -- one beat -- per half. Every per-query vector (m, corr, l) is NH contiguous
// beats, and the score tile and P8 are NH planes (see THE SCORE PLANES in main()).
#define NH (BR / meshCol)

// ONE SCORE BUFFER PER QUERY TILE. Step i scores query tile i % NQ into buffer i % NSCORE,
// so with NSCORE = NQ a buffer always holds the same query tile, and the running maximum
// that tile's next step starts from can wait in the buffer's own mold beats between its
// steps (see THE SCORE PLANES). From NQ = 3 on it also covers the GEMM running QK two steps
// ahead of PV (see THE DISPATCH ORDER): S(i+2) is written while the softmax still holds
// S(i) and S(i+1); at NQ = 2, QK(i+2) waits for the softmax of step i to publish. NH*S_PITCH
// = 32,896 B of scratchpad each. P8 and corr rotate through three slots, because step i+3
// is the first to reuse step i's.
#define NSCORE NQ
#define NP8 3
#if NQ < 2
#error "prefill keeps one score buffer per query tile and needs at least two: NQ >= 2 (decode is NQ = 1)"
#endif

// The cluster's TCDM, which is both the footprint guard's limit and the ceiling on the
// arena zero-fill below.
#define TCDM_BYTES ((unsigned)SNRT_TCDM_SIZE)
// The most one xDMA transfer from main memory may span: the testbench's endpoint walks it
// with the cluster's TCDM address, placed relative to where the transfer starts, and that
// address covers the TCDM size rounded up to a power of two.
#define XDMA_TASK_SPAN TCDM_BYTES

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

// Every hart's stack pointer, published before the arena is touched.
//
// The stacks live at the TOP of the scratchpad, one per hart, and the arena grows up
// from the bottom -- so the real limit on the arena is the LOWEST stack, not the
// scratchpad size and not this core's own sp. Checking either of those reports a
// comfortable margin while the arena is already inside hart 3's stack, which the
// preinit memset then zeroes. The symptom is not a fault: the lower harts lose their
// locals, so `l1` reads 0, the array shape reads 0, a core mistakes its own role and
// the run deadlocks somewhere unrelated.
//
// In .bss (DRAM) deliberately: anything in the arena cannot be trusted to survive the
// very overrun it is meant to detect.
static volatile uint32_t hart_sp[SNAX_CLUSTER_NUM_CORES];

// WHERE THE RESULT LANDS. Its own array, not a data.h buffer. NQ query tiles of O fill C
// exactly, so a state block placed after them would run off the end onto o32_golden, and
// the check would compare against a golden the kernel had just overwritten.
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
#define GOLD_ULP_SUM 4  // rowsum: 512 LUT exps (~1 ULP each) summed in FP32, narrowed once
#define GOLD_ULP_L   4  // l: NKV rowsums through NKV corr MULs, each ~1 ULP

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
// back-to-back dispatches, so polling it per dispatch waits for the NEXT one too, and puts
// ~2,260 cycles on the softmax of the last tile: the publish that releases the SIMD would
// sit behind a dependency it does not have.
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
// INT32->FP16 converter ALONE, so READER_WRITER_EXTENSION_1_CSR_NUM is 3 -- the enable
// bitmask, the extra-loop policy and the shift k -- and the converter is enable bit 0,
// not the bit 1 it has on a cluster that also stacks a dynamic rescale unit. The
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
    int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2, 1, 1, 1};
    int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2, 0, 0, 0};
    // TWO spatial strides: the C and D32 ports declare spatial_bounds [[4, 4]] and the
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
    // Every A channel: shape 0 reads the whole 16 x 4 block. The masks reset to 0, and a
    // disabled channel presents zero.
    csrw_ss(ENABLED_CHANNEL_READER_0, 0xFFFFFFFFu);

    // B -- reader 1. The port is 16 channels, for the one-row GEMV shape that takes two
    // weight blocks a pass; shape 0 takes one, channels 0..7.
    csrw_ss(S_STRIDE_READER_1_0, Bslstride0);
    csrw_ss(S_STRIDE_READER_1_1, 0);
    csrw_ss(ENABLED_CHANNEL_READER_1, (1u << (tileSize * meshCol * 8 / 64)) - 1u);
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
    // Every C channel on: attention reads a full C, never a broadcast one.
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
    // Every D channel: shape 0's output beats are full.
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_1, 0xFFFFFFFFu);

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
// cannot fold them -- and .bss maps to DRAM on this target. Reading them here would cost
// an L3 round trip per field per dispatch: 134 cycles for 20 csrw, against the 1.0 cycle a
// folded `csrw imm` takes. So the caller hoists them into locals once, outside the loop,
// the same rule gemm_d32_emit_fp16() below follows.
//
// B's two strides differ per shape: QK reads Q^T blocks n-major {64, K*64}; PV reads P^T,
// which the per-half epilogue writes n-major too, block (k, n) at n*P_PITCH + k*64:
// {64, P_PITCH}.
__attribute__((always_inline)) static inline void gemm_set_shape(
    int32_t k, int32_t m, int32_t blocks, int32_t as2, int32_t bs0, int32_t bs1,
    uint32_t c_chan) {
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
    csrw_ss(T_STRIDE_READER_1_0, bs0);
    csrw_ss(T_STRIDE_READER_1_1, bs1);
    csrw_ss(T_BOUND_READER_WRITER_0_2, m); // C stream
    csrw_ss(T_BOUND_READER_WRITER_1_2, m); // D32 stream
}

// The D32 output port carries an Int32ToFp16Converter. Arm it for a matmul whose
// result is CONSUMED as floating point; disarm it for one that ACCUMULATES in place, since
// such a matmul reads its own previous output back through C as INT32.
//
// Arming also halves the beats the writer emits -- two INT32 beats merge into one FP16 beat
// -- so the D32 descriptor changes with it, or the writer waits for beats that never
// arrive. The two walks do not share their strides either (datagen.py cd_map()):
//
//   INT32  O^T, block (m, n) at m*2 KiB + n*1 KiB      D32tlbound0, D32tlstride1/2
//   FP16   S^T planes, block (m, n) at n*S_PITCH + m*512   D16_TLBOUND0, D16_TLSTRIDE1/2
//
// bound0 halves (a block is 4 FP16 chunks of 128 B instead of 8 INT32 ones); stride 0, the
// 128 B chunk, is the same in both, so it is left as gemm_configure_once() wrote it. The
// FP16 n stride is a whole score plane: that is what puts each N half of a key row in its
// own SIMD beat.
//
// The INT32 values come in as ARGUMENTS rather than read from the D32tl* globals. Those
// globals live in .bss, which this target maps to DRAM, so reading them here would cost an
// L3 round trip per field per dispatch. The caller hoists them into locals once, where
// they stay in registers.
static inline void gemm_d32_emit_fp16(int on, uint32_t bound0, uint32_t stride1,
                                      uint32_t stride2) {
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 0, on ? 1u : 0u);  // enable bitmask
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 0u);            // extra_loops index 0 => 2:1
    // Power-of-two output scale, S16 = RNE(S_int * 2^-k): full-range INT8 scores reach
    // 2,064,512 and would overflow FP16 without it.
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 2, on ? (uint32_t)D32_FP16_SHIFT : 0u);
    csrw_ss(T_BOUND_READER_WRITER_1_0, bound0);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, stride1);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, stride2);
}

// The C read path's column scaler: PV(t) computes O^T = corr (.) O^T + V^T.P^T for the
// query tile of step t. See snax-flashattn-decode.c for the full contract.
__attribute__((always_inline)) static inline void gemm_colscale_arm(uint32_t on,
                                                                    uint32_t nblk) {
    csrw_ss(COLSCALE_CSR + 1, nblk);
    csrw_ss(COLSCALE_CSR + 0, on);
}
__attribute__((always_inline)) static inline void gemm_colscale_factors(
    const volatile uint32_t *f) {
    csrw_ss(COLSCALE_CSR + 2, f[0]);    csrw_ss(COLSCALE_CSR + 3, f[1]);
    csrw_ss(COLSCALE_CSR + 4, f[2]);    csrw_ss(COLSCALE_CSR + 5, f[3]);
    csrw_ss(COLSCALE_CSR + 6, f[4]);    csrw_ss(COLSCALE_CSR + 7, f[5]);
    csrw_ss(COLSCALE_CSR + 8, f[6]);    csrw_ss(COLSCALE_CSR + 9, f[7]);
    csrw_ss(COLSCALE_CSR + 10, f[8]);   csrw_ss(COLSCALE_CSR + 11, f[9]);
    csrw_ss(COLSCALE_CSR + 12, f[10]);  csrw_ss(COLSCALE_CSR + 13, f[11]);
    csrw_ss(COLSCALE_CSR + 14, f[12]);  csrw_ss(COLSCALE_CSR + 15, f[13]);
    csrw_ss(COLSCALE_CSR + 16, f[14]);  csrw_ss(COLSCALE_CSR + 17, f[15]);
}
// ---- P8: where the epilogue writes it, and PV's bank phase --------------------------
//
// One epilogue task quantises one N half of the score tile -- one plane, meshCol queries
// -- and its INTERLEAVE4 turns every 4 key beats into two 32 B output beats that, laid
// down one after the other, are exactly one 64 B B block (k = key/4, n). So the writer's
// flat stream IS the half's B blocks in k order, followed by the tapped rowsum beat: P8 is
// n-major, half n at n*P_PITCH, block k at k*64, then [rowsum][corr*l_old].
//
// PV reads one A beat (V^T) and one B beat (P^T) per array pass, 64 B each, and TCDM is
// 32 banks x 8 B: a 256 B rotation of four 64 B groups. A walks k in 64 B steps through
// V; B walks k in 64 B steps through one half -- the SAME rate around the banks, so once
// the two streams have settled on different banks they stay there. The rate is what
// matters: the streams are decoupled by their FIFOs, a collision delays one of them, and
// at equal rates they then stay apart, so where they START moves PV by little (see
// P8_V_PHASE).
//
// Byte offset of P8(key, query) in a P8 buffer. datagen.py's p8_interleave_offset() is the
// same formula, asserted there against the quantiser's output order and PV's B walk.
#define P8_OFF(key, q)                                                          \
    (((q) / meshCol) * P_PITCH + ((key) / tileSize) * (tileSize * meshCol) +   \
     ((q) % meshCol) * tileSize + (key) % tileSize)
// A P8 buffer is NH halves, rounded up to whole 256 B rotations so that every P8 buffer
// sits at the same bank phase against V.
#define P8_BUF ((NH * P_PITCH + 255u) & ~255u)
// Where V's buffers start on the banks relative to P8's, in bytes mod 256. P8's halves are
// P_PITCH = 64 mod 256 apart, so half 0 and half 1 meet V at P8_V_PHASE and P8_V_PHASE - 64:
// any multiple of 64 but 0 and 64 keeps both off V's banks. Measured with the GEMM first
// (vsim, NQ = NKV = 4): pipeline 73,057 / 72,982 / 72,749 cycles at 64 / 128 / 192.
#ifndef P8_V_PHASE
#define P8_V_PHASE 192u
#endif

// One SIMD task: the reader's base and its two inner loops, the writer's base and its inner
// loop, then the start. Every task writes all eight, so none inherits a loop from the task
// before it; the writer's loop 1 and both sides' loop 2 stay at the neutral bound 1 /
// stride 0 that snax_simd_program_fast() installs once before the loop. Every CSR address
// is a constant, so a task is nine `csrw imm`.
__attribute__((always_inline)) static inline void simd_task(uint32_t src, uint32_t b0,
                                                            uint32_t s0, uint32_t b1,
                                                            uint32_t s1, uint32_t dst,
                                                            uint32_t wb0, uint32_t ws0) {
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_LSB, src);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 0, b0);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 0, s0);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 1, b1);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 1, s1);
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_LSB, dst);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 0, wb0);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 0, ws0);
    snax_simd_fire();
}

int main() {
    uint8_t *l1 = (uint8_t *)snrt_l1_next();
    int8_t *local_a = (int8_t *)(l1 + delta_local_a);
    int8_t *local_b = (int8_t *)(l1 + delta_local_b);
    int32_t *local_c = (int32_t *)(l1 + delta_local_c);

    // Scratch, laid out at run time from where data.h ends. ADJACENCY IS THE LAYOUT
    // RULE: a sticky-B task reads one flat stream whose FIRST beat is the latched
    // operand, so every broadcast operand must sit immediately before its data, and
    // the two operands of a pair are read at one fixed stride for both halves. Each
    // group below is annotated with the task it serves; moving one silently feeds a task
    // the wrong beat rather than faulting.
    // Scratch starts at delta_local_c, not past delta_local_d32: both of those data.h
    // regions are dead. C is never read (QK's C channels are masked off, PV points C at its
    // own accumulator) and the D32 region is unused, because the score tile goes into the
    // score buffers below. Reclaiming them is what makes room for NQ query tiles.
    uint32_t top = ((uint32_t)delta_local_c + 63u) & ~63u;
    const uint32_t BEAT = SIMD_BEAT_BYTES;
    const uint32_t l1u = (uint32_t)l1;

    // THE SCORE PLANES. A SIMD beat is meshCol queries of one key, so S^T is NH planes,
    // one per N half, and every plane carries a spare beat on either side:
    //
    //   [ latch_0 ][ S^T keys 0..Bc-1, queries 0..15  ][ mold_0 ]      S_PITCH
    //   [ latch_1 ][ S^T keys 0..Bc-1, queries 16..31 ][ mold_1 ]      S_PITCH
    //
    // THE TILE IS READ WHERE THE GEMM WROTE IT. The latch is the sticky-B operand of the
    // epilogue, which must sit immediately before its data: the rowmax writes m_new of that
    // half there, so the tile never has to be copied to sit behind one. mold is the running
    // maximum the step starts from; the rowmax reads it as each plane's last beat, so ONE
    // lanewise MAX of 1 + Bc beats per plane is max(m_old, rowmax) -- the new running
    // maximum, with no separate combine.
    //
    // Buffer q always holds query tile q (NSCORE = NQ), so the running maximum of a query
    // tile never leaves its buffer between steps: each step's last task copies the latches
    // into the mold beats of the SAME buffer, for that tile's next step.
    //
    // The planes start at 32 mod 64, so the GEMM's key rows start on a 64 B boundary.
    uint32_t s_buf[NSCORE];                       // plane 0's latch, per buffer
    top += BEAT;
    for (uint32_t i = 0; i < NSCORE; i++) { s_buf[i] = top; top += NH * S_PITCH; }
    top = (top + 63u) & ~63u;
#define S_MOLD ((uint32_t)(1 + BC) * SIMD_BEAT_BYTES)   // plane's latch -> its mold beat

    // THE RUNNING STATE: one FP16 lane per query, NH contiguous beats (Br * 2 = 64 B) per
    // vector, in this order -- corr below l, because the corr * l_old task reads the pair
    // (corr, l) of each half at one positive stride (the AGU's stride is unsigned).
    //
    // corr = exp(a*(m_old - m_new)) has TWO readers: the MUL corr*l_old on the SIMD, and
    // PV(i) on the GEMM core, which copies it into the C column scaler to rescale that
    // query tile's O as it streams back into the array. The GEMM reads it only after the
    // softmax publishes step i -- by which time the SIMD may already be on steps i+1 and
    // i+2. So corr rotates through NP8 slots like P8 does: step i writes slot i % NP8, and
    // step i+3 cannot start until QK(i+3), which the GEMM issues only after launching PV(i)
    // (see THE DISPATCH ORDER).
    //
    // m lives in the planes' latch and mold beats and l is rewritten in place by the l
    // update, so nothing has to copy a "new" value over an "old" one. m_st holds the -inf
    // seed of every query tile, then the final m; l_st one l per query tile.
    uint8_t *m_st  = l1 + top;  top += (uint32_t)NQ * NH * BEAT;  // m [NQ]: -inf seed, final m
    uint8_t *corrS = l1 + top;  top += NP8 * NH * BEAT;           // corr, slot i % NP8
    uint8_t *l_st  = l1 + top;  top += (uint32_t)NQ * NH * BEAT;  // l [NQ], seeded 0
#define M_OF(q)    ((uint32_t)(uintptr_t)m_st + (uint32_t)(q) * NH * SIMD_BEAT_BYTES)
#define L_OF(q)    ((uint32_t)(uintptr_t)l_st + (uint32_t)(q) * NH * SIMD_BEAT_BYTES)
#define CORR_OF(i) ((uint32_t)(uintptr_t)corrS + ((i) % NP8) * NH * SIMD_BEAT_BYTES)
    // O^T PER QUERY TILE, in ONE representation. The GEMM accumulates P.V into it in
    // INT32 (C and D32 both point here), and that is the O that is stored and checked.
    //
    // The online rescale is applied to it on the way into the next PV: the C read path's
    // Int32ColumnScale multiplies every column (= query) of the INT32 accumulator by corr
    // as it streams. So there is no FP16 copy of O and no rescale pass -- an FP16 copy per
    // query tile, with its own sticky-B rescale every step, would cost NQ x 8 KiB of
    // scratchpad and a pass per step, and nothing would read it.
    uint32_t oacc32 = top;      top += (uint32_t)NQ * BR * DHEAD * 4; // INT32 O^T, per tile
#define OACC32_OF(q) (oacc32 + (uint32_t)(q) * (uint32_t)(BR * DHEAD) * 4u)
    // P8 CARRIES ITS OWN TAIL. Each epilogue task writes one half's B blocks and then its
    // tapped rowsum beat in one flat stream (Fp16ToInt8 tailPassthrough lets the reduce's
    // trailing FP16 beat past the quantiser), and corr*l_old gets the beat after it, so
    // the l update reads each half's pair [rowsum][corr*l_old] at a one-beat stride:
    //
    //   half n:  [ B block (0, n) ] .. [ B block (Bc/4-1, n) ][ rowsum_n ][ lsc_n ]
    //
    // NP8 P8 buffers rotate, and their tails with them: step i writes buffer i % NP8, which
    // PV(i) reads, and step i+3 is the first to write it again -- after QK(i+3), which the
    // GEMM dispatches only after PV(i) (see THE DISPATCH ORDER). See "P8" above for the
    // bank phase this layout gives PV.
    top = (top + 255u) & ~255u;
    uint32_t p8_buf[NP8];
    for (uint32_t i = 0; i < NP8; i++) { p8_buf[i] = top; top += P8_BUF; }
#define RSUM_OFF ((uint32_t)BC * meshCol)          // a half's rowsum beat
    // One Q tile per query tile. They hold the same bytes -- so the goldens are identical for
    // every q -- but each is loaded separately, which is what a real prefill pays. A whole
    // number of 256 B rotations, so the V buffers below still meet P8 at P8_V_PHASE.
    uint32_t q_buf[NQ];
    for (uint32_t qi = 0; qi < (uint32_t)NQ; qi++) {
        q_buf[qi] = top; top += (uint32_t)(N * K * tileSize * meshCol);
    }
    top = (top + 255u) & ~255u;
    top += P8_V_PHASE;
    // K AND V STREAM. Every KV tile pulls a fresh 64 KiB K and a fresh 64 KiB V from main
    // memory, double-buffered so the load of tile j+1 overlaps the compute of tile j.
    //
    // The two have DIFFERENT lifetimes, and that is what makes two buffers each enough:
    // K(j) dies at the end of QK(j), V(j) only at the end of PV(j). Freeing them on
    // separate counters gives the iDMA three dispatches of slack on each; freeing both on
    // the later one would put every load straight onto the critical path.
    //
    // Every tile is DISTINCT: K(j) is A + j*Bc*d, V(j) is V + j*Bc*d, Q(q) is B + q*Br*d.
#define KVBYTES ((uint32_t)(BC * DHEAD))
    uint32_t k_buf[2], v_buf[2];
    k_buf[0] = (uint32_t)delta_local_a;   // the staged buffer becomes buffer 0
    // The K and V tiles are whole multiples of 256 B -- one full 32-bank rotation -- so
    // allocating one never shifts the bank the next one starts on, and both V buffers meet
    // every P8 buffer at P8_V_PHASE.
    k_buf[1] = top; top += KVBYTES;
    v_buf[0] = top; top += KVBYTES;
    v_buf[1] = top; top += KVBYTES;

    // Room for STEPS spans per lane rather than NKV: prefill walks NKV*NQ dispatches.
    volatile uint32_t *pub  = (volatile uint32_t *)(l1 + top); top += 2048;
    volatile uint32_t *sync = (volatile uint32_t *)(l1 + top); top += 64;

    // THE TWO SHAPES THAT SEED THE SIMD's AGU LIVE IN THE ARENA, not in a .l1 static. TCDM
    // is SRAM with no reset: an address that has never been written reads X in RTL, and .l1
    // is NOLOAD, so a linker-placed array there starts UNDEFINED rather than zero.
    // Allocating them here puts them inside the block the xDMA zeroes below.
    snax_simd_shape_t *shapes = (snax_simd_shape_t *)(l1 + top);
    top += (2u * sizeof(snax_simd_shape_t) + 63u) & ~63u;

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
    // BEFORE the preinit, because the preinit is what does the damage.
    {
        uint32_t sp; asm volatile("mv %0, sp" : "=r"(sp));
        hart_sp[snrt_cluster_core_idx()] = sp;
    }
    snrt_cluster_hw_barrier();
    if (snax_is_gemm_core()) {
        // Waiting for each one to be non-zero: the barrier does not wait for the other
        // harts' stores to land (see t_org).
        uint32_t lowest = 0xFFFFFFFFu;
        for (uint32_t c = 0; c < SNAX_CLUSTER_NUM_CORES; c++) {
            uint32_t sp_c = hart_sp[c];
            while (sp_c == 0u) sp_c = hart_sp[c];
            if (sp_c < lowest) lowest = sp_c;
        }
        uint32_t arena_end = (uint32_t)(uintptr_t)l1 + top;
        printf("  TCDM map        arena=[%08lx,%08lx)  lowest stack=%08lx"
               "  headroom=%ld B\n",
               (unsigned long)(uintptr_t)l1, (unsigned long)arena_end,
               (unsigned long)lowest, (long)((long)lowest - (long)arena_end));
        printf("  score planes    %u x %u B, P8 halves %u x %u B, V at bank phase %u vs P8\n",
               (unsigned)NH, (unsigned)S_PITCH, (unsigned)NH, (unsigned)P_PITCH,
               (unsigned)P8_V_PHASE);
        snax_tcdm_policy_install(FA_TCDM_PRIO, FA_TCDM_GUARD);
        printf("  TCDM arbitration %s (control 0x%x)\n", snax_tcdm_policy_name(FA_TCDM_PRIO),
               snax_tcdm_arb_ctrl_read());
        if (arena_end > lowest) {
            printf("  TCDM OVERFLOW: the arena runs %ld B into the stacks -- the "
                   "preinit below would zero them. Reduce Bc.\n",
                   (long)(arena_end - lowest));
            cfg_err++;
        }
    }
    snrt_cluster_hw_barrier();
    if (cfg_err) { snrt_cluster_hw_barrier(); return cfg_err; }

    if (snax_is_xdma_core()) {
        uint32_t zbytes = top > TCDM_BYTES ? TCDM_BYTES : top;
        uint32_t z0 = snrt_mcycle();
        xdma_fill_arm((uint32_t)l1, XDMA_WIDTH, (zbytes + XDMA_WIDTH - 1u) / XDMA_WIDTH, 0u);
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
    // Q, K and V are full-range INT8; the converter's 2^-D32_FP16_SHIFT keeps the scores
    // inside FP16.
    if (snax_is_gemm_core()) {
        // TCDM footprint guard. An overrun corrupts whatever follows rather than
        // faulting, so check the layout rather than trust the arithmetic.
        printf("  TCDM footprint  %lu bytes of %u  (Bc=%d, d=%d, S16 = S_int * 2^-%d)\n",
               (unsigned long)top, TCDM_BYTES, BC, DHEAD, D32_FP16_SHIFT);
        // ONE xDMA TRANSFER SPANS LESS THAN XDMA_TASK_SPAN of main memory: past it the
        // endpoint's address wraps onto the transfer's own first bytes, at full rate and with
        // no error. The xDMA moves a V tile, or half of K(0), per transfer.
        if (KVBYTES >= XDMA_TASK_SPAN) {
            printf("  xDMA SPAN: a V tile is %lu bytes, one transfer spans under %lu\n",
                   (unsigned long)KVBYTES, (unsigned long)XDMA_TASK_SPAN);
            cfg_err++;
        }
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




    // One full program before the loop establishes every CSR that no task below varies:
    // address high words, spatial strides, channel/byte masks, the writer's loop 1 and both
    // sides' loop 2. From then on each task writes only its eight per-task CSRs
    // (simd_task) -- 9 writes instead of 23.
    if (snax_is_simd_core()) {
        snax_simd_shape_flat(&shapes[0], (void *)m_st, 1);
        snax_simd_shape_flat(&shapes[1], (void *)m_st, 1);
        snax_simd_program_fast(&shapes[0], &shapes[1]);
    }
    snrt_cluster_hw_barrier();

    // ONE ORIGIN FOR ALL THREE LANES. mcycle counts the same clock on every hart, but each
    // core stamping its own after the barrier would leave them ~530 cycles apart -- enough
    // to make a cross-lane reading of the trace ("QK0 starts before K(0) lands") an
    // artefact of the offset rather than a fact about the machine. One core publishes the origin and
    // everyone subtracts that same value, so the lanes are comparable to the cycle.
    // The cluster contention counters are armed HERE, by one hart, so that the
    // window they cover is the same barrier-to-barrier span the three lanes
    // measure with mcycle. They are cluster-global, so arming them on the GEMM
    // hart covers the engines whose cores are asleep as well.
    snax_perf_snapshot_t perf;
    if (snax_is_gemm_core()) snax_perf_arm();
    if (snax_is_gemm_core()) sync[8] = snrt_mcycle();
    snrt_cluster_hw_barrier();
    // THE BARRIER DOES NOT ORDER THE STORE. The origin is a posted store, and the
    // barrier releases the other harts without waiting for it to land, so a read
    // straight after it can still return the preinit's zero -- which one does depends
    // on how the code happens to be laid out. Such a hart then reports every time as an
    // absolute mcycle, ~72,000 too late. mcycle is never zero here, so wait for it.
    uint32_t t_org_v = sync[8];
    while (t_org_v == 0u) t_org_v = sync[8];
    const uint32_t t_org = t_org_v;

    // ---- hart 3: the K/V stream ------------------------------------------
    // Spans are recorded against t_org, the one origin every lane shares (see above).
    // ---- hart 2: the running state, generated locally ---------------------
    // m starts at -inf, one 64 B vector -- exactly one xDMA beat -- per query tile. It is a
    // constant, and the xDMA's writer
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
                      (uint32_t)(uintptr_t)A + KVBYTES / 2u, XDMA_WIDTH,
                      (KVBYTES / 2u) / XDMA_WIDTH);
        {
            snax_xdma_task_t tk = snax_xdma_start_task();
            uint32_t fp = tk.remote ? XDMA_FINISH_REMOTE_TASK_PTR
                                    : XDMA_FINISH_LOCAL_TASK_PTR;
            uint32_t spins = 0;
            while (!snax_xdma_task_done(snax_read_xdma_cfg_reg(fp), tk.task_id)) {
                if (++spins > SPIN_LIMIT) { timeouts++; break; }
            }
        }
        sync[13] = 1;   // the xDMA's half of K(0) has landed

        uint32_t x0 = snrt_mcycle() - t0;
        xdma_fill_arm((uint32_t)m_st, XDMA_WIDTH, (uint32_t)NQ * (BR * 2u) / XDMA_WIDTH,
                      0xFBFFFBFFu);
        snax_xdma_local_wait(snax_xdma_start());
        pub[226] = x0; pub[227] = snrt_mcycle() - t0;
        sync[7] = 1;   // m is live -- the SIMD may start tile 0

        // ---- the V stream, concurrent with hart 3's K stream ---------------
        // K and V are independent tiles the pipeline needs at the same rate, so one
        // engine doing both serialises two transfers that need no ordering. V(j-2) is
        // dead only once EVERY query tile has used it -- NQ steps per KV tile.
        uint32_t xdma_busy = 0, xdma_block = 0;
        xdma_tile_arm((uint32_t)(l1 + v_buf[0]), (uint32_t)(uintptr_t)V, XDMA_WIDTH,
                      KVBYTES / XDMA_WIDTH);
        for (uint32_t j = 0; j < NKV; j++) {
            uint32_t w0 = snrt_mcycle() - t0;
            if (j >= 2) SNAX_SPIN_UNTIL(sync[3] >= (j - 1) * (uint32_t)NQ, timeouts);
            uint32_t v0 = snrt_mcycle() - t0;
            xdma_block += v0 - w0;
            xdma_tile_retask((uint32_t)(l1 + v_buf[j & 1]),
                             (uint32_t)(uintptr_t)V + j * KVBYTES, KVBYTES / XDMA_WIDTH);
            // A main-memory read is served by the PEER, so the hardware runs it as a
            // REMOTE task; waiting on the local counter spins for ever. BOUNDED, and
            // it prints the whole engine state on timeout: an unbounded wait here
            // turns a one-line bug report into a run that never ends.
            {
                snax_xdma_task_t tk = snax_xdma_start_task();
                uint32_t ptr = tk.remote ? XDMA_FINISH_REMOTE_TASK_PTR
                                         : XDMA_FINISH_LOCAL_TASK_PTR;
                uint32_t spins = 0;
                while (!snax_xdma_task_done(snax_read_xdma_cfg_reg(ptr), tk.task_id)) {
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
                               (unsigned long)(KVBYTES / XDMA_WIDTH),
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
            snrt_dma_start_1d((void *)(l1 + q_buf[qi]), B + qi * (N * K * tileSize * meshCol),
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
            // V is on the xDMA (hart 2); this lane carries K only.
            uint32_t w0 = snrt_mcycle() - t0;
            if (j >= 2) SNAX_SPIN_UNTIL(sync[4] >= (j - 1) * (uint32_t)NQ, timeouts);
            uint32_t k0 = snrt_mcycle() - t0;
            // K(0) is split with the xDMA -- see its lane. Every later tile is
            // already hidden behind the previous tile's compute, so only the first
            // one is worth two engines.
            snrt_dma_start_1d(l1 + k_buf[j & 1], A + j * KVBYTES,
                              (j == 0u) ? (uint32_t)(KVBYTES / 2u) : (uint32_t)KVBYTES);
            snrt_dma_wait_all();
            if (j == 0u) SNAX_SPIN_UNTIL(sync[13] >= 1, timeouts);
            uint32_t k1 = snrt_mcycle() - t0;
            // K(0) is two engines and every other tile is one, so they are counted
            // apart -- an average over both describes neither.
            if (j == 0u) pub[237] = k1 - k0; else pub[236] += k1 - k0;
            sync[2] = j + 1;
            pub[200 + 4 * j] = k0; pub[201 + 4 * j] = k1;
            dma_busy += k1 - k0;
            dma_block += k0 - w0;
        }
        // THE STORE. A query tile's result leaving the cluster is (O, m, l): the INT32
        // accumulator plus its running state. On one cluster it is a drain
        // tail; KV-sharded across clusters it is the partial each shard contributes to the
        // cross-cluster fold, so it is part of the measured pipeline.
        // That fold is the xDMA MonoidJunction, armed on this very transfer -- the DMA
        // below becomes the collective, and nothing above it moves. See "SHARDING THE KEYS
        // ACROSS CLUSTERS" at the top of this file for the geometry words, the `fmt`
        // trap, and why the numerator crosses in FP32 via Int32ToFp32 rather than FP16.
        // SPLIT. O is final the instant the last PV retires; only m and l wait on the SIMD's
        // trailing commit. Shipping them together would charge O's 16 KiB with the SIMD's
        // drain.
        // ONE QUERY TILE AT A TIME, AS EACH BECOMES FINAL. The step order is
        // j = kv * NQ + q, so query tile q retires at step (NKV-1)*NQ + q -- the last
        // NQ steps of the run, one per tile. Waiting for ALL of them puts every byte of
        // O in the tail; waiting per tile lets all but the last hide behind the steps
        // that follow it.
        // The span from the first store to the last contains three WAITS, so it is
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
        pub[238] = o_busy;
        uint32_t s1d = snrt_mcycle() - t0;            // every O is away
        SNAX_SPIN_UNTIL(sync[6] >= 1, timeouts);     // the SIMD drained -- m, l are final
        uint32_t s2d = snrt_mcycle() - t0;
        snrt_dma_start_1d((void *)((uint8_t *)fa_out + (uint32_t)NQ * (uint32_t)(BR * DHEAD) * 4u),
                          (void *)m_st, (uint32_t)(l_st - m_st) + (uint32_t)NQ * NH * BEAT);
        snrt_dma_wait_all();
        uint32_t s3d = snrt_mcycle() - t0;
        pub[230] = s2d; pub[231] = s3d;
        pub[228] = s0d; pub[229] = s1d;
        dma_store = pub[238] + (s3d - s2d);   // transfer only, not the waits between
        dma_busy += dma_store;

        dma_wall = snrt_mcycle() - t0;
        pub[223] = dma_store;
        pub[220] = dma_busy;
        pub[221] = dma_wall;
        pub[222] = dma_block;
    }

    // ======================= the pipelined loop ==============================
    if (snax_is_gemm_core()) {
        // The descriptor, hoisted out of DRAM once. Everything below is a local from
        // here on, so a dispatch is pure `csrw imm` and never touches .bss again.
        const int32_t s1_k = K, s1_m = M, s1_blk = M * N;
        const int32_t s1_as2 = Atlstride2, s1_bs0 = Btlstride0, s1_bs1 = Btlstride1;
        const int32_t s2_k = S2_K, s2_m = S2_M, s2_blk = S2_M * N;
        // PV walks P^T n-major, k one 64 B block, n one P8 half (P_PITCH) -- the order the
        // per-half epilogue writes it in.
        const int32_t s2_as2 = S2_Atlstride2, s2_bs0 = S2_Btlstride0, s2_bs1 = (int32_t)P_PITCH;
        const uint32_t g_n = (uint32_t)N;
        // QK masks C, so this address is never dereferenced -- point it somewhere real
        // anyway rather than at memory that belongs to a score buffer.
        const uint32_t base_c = l1u + oacc32;
        // D's two walks (gemm_d32_emit_fp16): INT32 O^T as data.h emits it, FP16 S^T planes
        // from data.h's compile-time D16_* constants.
        const uint32_t d32_b0_i = (uint32_t)D32tlbound0, d32_s1_i = (uint32_t)D32tlstride1,
                       d32_s2_i = (uint32_t)D32tlstride2;
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
        // K.Q^T + O_accumulated from the second tile onward. QK's D32 base is plane 0's
        // first key row, one beat past its latch.
#define STAGE_QK(t)                                                              \
        do {                                                                     \
            const uint32_t sk = (t) / (uint32_t)NQ, sq = (t) % (uint32_t)NQ;     \
            gemm_set_shape(s1_k, s1_m, s1_blk, s1_as2, s1_bs0, s1_bs1, 0u);      \
            gemm_set_bases(l1u + k_buf[sk & 1], l1u + q_buf[sq], base_c,         \
                           l1u + s_buf[(t) % NSCORE] + BEAT);                    \
            gemm_d32_emit_fp16(1, D16_TLBOUND0, D16_TLSTRIDE1, D16_TLSTRIDE2);   \
            gemm_colscale_arm(0u, g_n);                                          \
        } while (0)
        // Every query tile keeps its own accumulator, so each one's first KV tile
        // masks the C reader off rather than clearing oacc32: a disabled channel
        // presents ZERO and issues no TCDM request, which is the seed it needs.
#define STAGE_PV(t)                                                              \
        do {                                                                     \
            const uint32_t sk = (t) / (uint32_t)NQ, sq = (t) % (uint32_t)NQ;     \
            const uint32_t ob = l1u + OACC32_OF(sq);                             \
            gemm_set_shape(s2_k, s2_m, s2_blk, s2_as2, s2_bs0, s2_bs1,           \
                           sk == 0u ? 0u : 0xFFFFFFFFu);                         \
            gemm_set_bases(l1u + v_buf[sk & 1], l1u + p8_buf[(t) % NP8], ob, ob); \
            gemm_d32_emit_fp16(0, d32_b0_i, d32_s1_i, d32_s2_i);                 \
            gemm_colscale_arm(sk != 0u, g_n);                                    \
        } while (0)
        // corr of step t is final only once the softmax has published it: the one part of a
        // PV's configuration that sits between the publish and START. `f` is its slot,
        // CORR_OF(t), worked out while the previous dispatch ran.
#define PV_FACTORS(t, f)                                                         \
        do {                                                                     \
            if ((t) >= (uint32_t)NQ)                                             \
                gemm_colscale_factors((const volatile uint32_t *)(f));           \
        } while (0)

        // ---- THE DISPATCH ORDER: QK RUNS TWO STEPS AHEAD OF PV ------------------
        //
        //     QK(0) QK(1) | QK(2) PV(0) | QK(3) PV(1) | ... | PV(STEPS-2) PV(STEPS-1)
        //
        // PV(i) waits for the softmax of step i, which starts only once QK(i) has written
        // S(i). Issued right after QK(i+1), PV(i) would wait out whatever of SM(i) is longer
        // than QK(i+1) -- the head of the pipeline, where nothing else can fill the gap,
        // and the tail, where SM(STEPS-1) overlaps only PV(STEPS-2). Putting QK(i+2) in
        // between gives every softmax two dispatches of cover instead of one, so the array
        // runs back to back from QK(0) on. What it costs is buffering: QK(i+2) writes a
        // score buffer while SM(i) and SM(i+1) still hold theirs (NSCORE = NQ >= 3), and P8
        // and corr keep three slots each, because SM(i+3) is the first to reuse step i's,
        // and QK(i+3) -- which SM(i+3) waits for -- is dispatched only after PV(i).
        //
        // The successor of each dispatch (kind 0 = QK, 1 = PV, of step t) follows from the
        // dispatch alone, so the loop carries it in two registers rather than reading a
        // schedule table out of TCDM, where every load from this core competes with the
        // streams it is feeding: QK(t) is followed by PV(t-2) -- or by QK(t+1) at the head
        // -- and PV(t) by QK(t+3), or by PV(t+1) at the tail.
#define NEXT_DISPATCH(kind, t, nk, nt)                                           \
        do {                                                                     \
            if ((kind) == 0u) {                                                  \
                if ((t) >= 2u)                       { nk = 1u; nt = (t) - 2u; } \
                else if ((t) + 1u < (uint32_t)STEPS) { nk = 0u; nt = (t) + 1u; } \
                else                                 { nk = 1u; nt = 0u; }       \
            } else if ((t) + 3u < (uint32_t)STEPS)   { nk = 0u; nt = (t) + 3u; } \
            else                                     { nk = 1u; nt = (t) + 1u; } \
        } while (0)
        // What dispatch d waits for. A QK needs its K tile and a free score buffer (the one
        // step t - NSCORE used, released when that step's softmax publishes); a PV needs
        // its step's softmax published and its V tile.
#define DISPATCH_WAIT(kind, t)                                                   \
        do {                                                                     \
            if ((kind) == 0u) {                                                  \
                if ((t) >= NSCORE)                                               \
                    SNAX_SPIN_UNTIL(sync[1] >= (t) - (NSCORE - 1), timeouts);    \
                SNAX_SPIN_UNTIL(sync[2] >= (t) / (uint32_t)NQ + 1, timeouts);    \
            } else {                                                             \
                SNAX_SPIN_UNTIL(sync[1] >= (t) + 1, timeouts);                   \
                SNAX_SPIN_UNTIL(sync[5] >= (t) / (uint32_t)NQ + 1, timeouts);    \
            }                                                                    \
        } while (0)
        // What the retirement of dispatch d releases: a published S(t) and one more QK step
        // of its K tile (see the iDMA's K-free test), or one more PV step of its V tile.
#define DISPATCH_DONE(kind, t)                                                   \
        do {                                                                     \
            if ((kind) == 0u) { sync[0] = (t) + 1; sync[4] = (t) + 1; }          \
            else              { sync[3] = (t) + 1; }                             \
        } while (0)

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
        uint32_t tid_q[2] = {0u, 0u};
        uint32_t kind = 0u, t = 0u, pkind = 0u, pt = 0u;
        for (uint32_t d = 0; d <= 2u * (uint32_t)STEPS; d++) {
            if (d < 2u * (uint32_t)STEPS) {
                // Arm first: the predecessor snapshotted the bank on its own start write,
                // so these 21 writes land inside the dependency wait instead of after it.
                if (kind == 0u) STAGE_QK(t); else STAGE_PV(t);
                {
                    uint32_t w0 = snrt_mcycle();
                    DISPATCH_WAIT(kind, t);
                    gemm_stall += snrt_mcycle() - w0;
                }
                pub[(kind == 0u ? 10u : 12u) + 4u * t] = snrt_mcycle() - g0;
                if (kind == 1u) PV_FACTORS(t, CORR_OF(t));
                tid_q[d & 1u] = ++gemm_seq;
                gemm_launch();
                gemm_ack();
            }
            if (d > 0u) {
                timeouts += gemm_wait(tid_q[(d - 1u) & 1u]);
                pub[(pkind == 0u ? 11u : 13u) + 4u * pt] = snrt_mcycle() - g0;
                DISPATCH_DONE(pkind, pt);
            }
            pkind = kind; pt = t;
            NEXT_DISPATCH(pkind, pt, kind, t);
        }
#else
        // Dispatch 0 has no predecessor to hide behind. Writing it here puts its
        // cold instruction fetches inside the wait for K(0), which is dead time.
        STAGE_QK(0u);

        // Everything between one dispatch's retirement and the next one's START is on the
        // array's critical path, so the successor, its staging and its corr slot are all
        // worked out while the dispatch before it runs.
        uint32_t kind = 0u, t = 0u, corr_f = 0u;
        for (uint32_t d = 0; d < 2u * (uint32_t)STEPS; d++) {
            {
                uint32_t w0 = snrt_mcycle();
                DISPATCH_WAIT(kind, t);
                gemm_stall += snrt_mcycle() - w0;
            }
            pub[(kind == 0u ? 10u : 12u) + 4u * t] = snrt_mcycle() - g0;
            if (kind == 1u) PV_FACTORS(t, corr_f);
            const uint32_t tid = ++gemm_seq;
            gemm_launch();   // staged one dispatch ago
            gemm_ack();
            // Stage the successor while this one runs: the bank was snapshotted at START.
            uint32_t nkind, nt;
            NEXT_DISPATCH(kind, t, nkind, nt);
            if (d + 1u < 2u * (uint32_t)STEPS) {
                if (nkind == 0u) {
                    STAGE_QK(nt);
                } else {
                    STAGE_PV(nt);
                    corr_f = CORR_OF(nt);
                }
            }
            pub[80 + 2 * t + kind] = snrt_mcycle() - g0;  // successor staged, array running
            timeouts += gemm_wait(tid);
            {
                uint32_t c = csrr_ss(GEMMX_PERFORMANCE_COUNTER);
                gemm_cycles += c;
                if (kind == 0u) {
                    gemm_cyc_s1 += c;
                    gemm_sa_s1 += csrr_ss(GEMMX_STALL_A);
                    gemm_sb_s1 += csrr_ss(GEMMX_STALL_B);
                    gemm_sd_s1 += csrr_ss(GEMMX_STALL_D);
                } else {
                    gemm_cyc_s2 += c;
                    gemm_sa_s2 += csrr_ss(GEMMX_STALL_A);
                    gemm_sb_s2 += csrr_ss(GEMMX_STALL_B);
                    gemm_sd_s2 += csrr_ss(GEMMX_STALL_D);
                }
            }
            gemm_stream_cycles += csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
            pub[(kind == 0u ? 11u : 13u) + 4u * t] = snrt_mcycle() - g0;
            DISPATCH_DONE(kind, t);   // AFTER the streamer has drained
            kind = nkind; t = nt;
        }
#endif  /* GEMM_QUEUE */
#undef NEXT_DISPATCH
#undef DISPATCH_WAIT
#undef DISPATCH_DONE
        gemm_colscale_arm(0u, g_n);   // no later dispatch may inherit the enable
        gemm_wall = snrt_mcycle() - g0;
#undef STAGE_QK
#undef STAGE_PV
    }

    if (snax_is_simd_core()) {
        uint32_t s0 = t_org;
        uint32_t simd_busy0 = snax_simd_busy_cycles();
        // The geometry every step shares: the key beats per plane, the plane and P8-half
        // pitches, and where a half's rowsum sits.
        const uint32_t nb = (uint32_t)BC, sp = S_PITCH, pp = P_PITCH, ro = RSUM_OFF;

        // EVERY QUERY TILE STARTS FROM m = -inf: the xDMA's fill of m_st, copied into the
        // mold beats of every score buffer -- buffer q holds query tile q -- by one plain
        // copy, no operator armed. m_st is [q][h] beats, and so are the mold beats in
        // S_PITCH steps, because the buffers are contiguous: one 1-D sweep on each side.
        SNAX_SPIN_UNTIL(sync[7] >= 1, timeouts);
        snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0u);
        simd_task(M_OF(0), (uint32_t)NQ * NH, BEAT, 1u, 0u, l1u + s_buf[0] + S_MOLD,
                  (uint32_t)NQ * NH, S_PITCH);

        for (uint32_t i = 0; i < (uint32_t)STEPS; i++) {
            const uint32_t q = i % (uint32_t)NQ;
            uint32_t w0 = snrt_mcycle();
            SNAX_SPIN_UNTIL(sync[0] >= i + 1, timeouts);
            simd_stall += snrt_mcycle() - w0;

            pub[120 + 2 * i] = snrt_mcycle() - s0;
            pub[180 + i] = snax_simd_busy_cycles() - simd_busy0;
            // This step's buffers: the score buffer of its query tile, its P8 and corr
            // slots, and the tile's l.
            const uint32_t sb = l1u + s_buf[i % NSCORE];   // plane 0's latch
            const uint32_t pb = l1u + p8_buf[i % NP8];
            const uint32_t cq = CORR_OF(i);
            const uint32_t lq = L_OF(q);

            // ---- the online softmax, in full -------------------------------
            // Every task below is NH beats of arithmetic except 1 and 3. That ratio is
            // the point: the STATE UPDATE is dominated by per-task start/drain, not by
            // compute -- which is why each one covers both halves of the query tile.

            // 1  m_new = max(m_old, rowmax(S^T)), both halves in ONE task: a lanewise MAX
            //    over two rows of 1 + Bc beats, each row a plane's key beats and then its
            //    mold beat, the rows one plane apart. Its two beats go into the planes'
            //    latches, the beats in front of the key rows, where tasks 2, 3 and 6 read m.
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMREDUCE));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 0, nb + 1u);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1,
                                    SIMD_RED_MAX | SIMD_RED_LANEWISE);
            simd_task(sb + BEAT, nb + 1u, BEAT, NH, sp, sb, NH, sp);

            // 2  corr = exp(a*(m_old - m_new)), FUSED into one pass over the pairs
            //    [latch_h = m_new][mold_h = m_old], one pair per plane.
            //
            // The chain's fixed order [EW0, Map, Reduce, EW1, Fp16ToInt8] puts the combine
            // BEFORE the transform, so EW0 in PAIRS mode takes each [B][x] pair to x - B in
            // one beat and Map exponentiates it in flight: the difference is never written
            // to memory and never read back. a' goes here as well as into P: corr is the
            // ratio of the two P scales it bridges. corr lands in this step's corr slot,
            // where the MUL of task 4 and the GEMM's PV(i) both find it.
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                        (1u << SIMD_EXT_STREAMMAP));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 0, 2);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 1, SIMD_EW_SUB);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0, SCORE_SCALE_BITS);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 1, 0);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 2, SIMD_FUNC_EXP);
            simd_task(sb, 2u, (nb + 1u) * BEAT, NH, sp, cq, NH, BEAT);

            // 3  P = exp(a*(S - m_new)) AND rowsum, in ONE pass over each plane.
            //
            //     read  [ m_new_h ][ S^T plane h x Bc ]       the latch, then the tile
            //       -> EW0: sticky SUB -> Map: EXP -> Reduce: ADD|LANEWISE|TAP -> Fp16ToInt8
            //     write [ B blocks (k, h) x Bc/4 ][ rowsum_h ]  into half h of P8
            //
            // ONE TASK PER HALF, because a sticky latch is one beat: it holds m_new for the
            // meshCol queries of one plane. The quantiser narrows the Bc data beats and
            // passes the reduce's trailing rowsum beat through untouched. P8 = rne(127 * P),
            // written INTERLEAVED as PV's B operand: each group of 4 key beats comes out as
            // the two halves of B block (k = g, n = h), so the writer's flat stream is half
            // h's blocks in k order (see the decode kernel). The map keeps task 2's a' and
            // EXP.
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                        (1u << SIMD_EXT_STREAMMAP) |
                                        (1u << SIMD_EXT_STREAMREDUCE) |
                                        (1u << SIMD_EXT_FP16TOINT8));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 0, 1);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 1,
                                    SIMD_EW_SUB | SIMD_EW_STICKY_B);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 0, nb);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1,
                                    SIMD_RED_ADD | SIMD_RED_LANEWISE | SIMD_RED_TAP);
            snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 0, 0x42FE0000u);  // 127.0f
            snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 1,
                                    SIMD_QUANT_TAIL(nb) | SIMD_QUANT_ILV4);
            for (uint32_t h = 0; h < NH; h++)
                simd_task(sb + h * sp, nb + 1u, BEAT, 1u, 0u, pb + h * pp, nb / 2u + 1u, BEAT);

            // PUBLISH HERE, not at the end of the step. Everything below updates the running
            // l and m for the NEXT step of this query tile and the GEMM needs none of it; the
            // task queue already orders those ahead of step i+1's first task, so correctness
            // costs nothing.
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

            // 4  corr * l_old, both halves: the post-map elementwise in PAIRS mode over
            //    [corr_h][l_h], into each half's slot behind its rowsum.
            snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1,
                           SIMD_EXT_STREAMELEMENTWISE_1_CSR, 2, SIMD_EW_MUL);
            simd_task(cq, 2u, lq - cq, NH, BEAT, pb + ro + BEAT, NH, pp);

            // 5  l = rowsum + corr*l_old: a lanewise ADD over each half's adjacent pair
            //    [rowsum_h][lsc_h], written over l -- whose old value task 4 has consumed.
            //
            // The two cannot share a task: corr*l_old followed by + rowsum is a
            // combine-after-combine, whose second operand would have to be fetched at EW1,
            // and the chain feeds EW1 only from the stage above it.
            snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                           SIMD_RED_ADD | SIMD_RED_LANEWISE);
            simd_task(pb + ro, 2u, BEAT, NH, pp, lq, NH, BEAT);

            // 6  this query tile's next m_old: its latches (m_new) into the mold beats of
            //    the SAME buffer, which holds this query tile's next step -- a plain copy, no
            //    operator armed. Tasks 1 to 3 have read both, and QK(i + NQ) writes only the
            //    key rows, so the copy may land while the GEMM fills the planes around it.
            //    The tile's last step copies into m_st instead: the final m.
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0u);
            if (i + (uint32_t)NQ < (uint32_t)STEPS)
                simd_task(sb, NH, sp, 1u, 0u, sb + S_MOLD, NH, sp);
            else
                simd_task(sb, NH, sp, 1u, 0u, M_OF(q), NH, BEAT);
            // The step's ISSUE ends here, not at the publish above: tasks 4 to 6 are fired
            // afterwards. Bracketing to the publish would attribute their engine time to an
            // interval that does not contain them.
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

    // ---- invariants and goldens ---------------------------------------------
    if (snax_is_simd_core()) {
        // Every KV tile and every query tile is DISTINCT, so the running max moves
        // (GOLD_MAX_MOVES) and each query tile's O is rescaled on its own C read path.
        //
        // The LAST STEP's buffers survive: P8 and the row sum of query tile NQ-1, KV tile NKV-1.
        // The running state (m, l) and O survive for every query tile.
        const uint32_t ql = (uint32_t)(STEPS - 1) % (uint32_t)NQ;
        volatile uint8_t *m_l = (volatile uint8_t *)M_OF(ql);
        volatile uint8_t *l_l = (volatile uint8_t *)L_OF(ql);
        volatile int8_t *p8t = (volatile int8_t *)(l1 + p8_buf[(STEPS - 1) % NP8]);
        // The last step's rowsum: one beat behind each half's B blocks.
        volatile uint8_t *rsum0 = l1 + p8_buf[(STEPS - 1) % NP8] + RSUM_OFF;
#define RSUM_AT(i) fp16_at(rsum0 + ((i) / meshCol) * P_PITCH, (i) % meshCol)
        // P8 == 127 count per query row in the last step, through the interleaved layout. A
        // query whose max came from an earlier tile has none; the golden knows which.
        for (uint32_t i = 0; i < BR; i++) {
            int ones = 0;
            for (uint32_t j = 0; j < BC; j++)
                if (p8t[P8_OFF(j, i)] == 127) ones++;
            int16_t want_ones = p8max_golden[i];
            if (want_ones >= 0 && ones != want_ones) {
                printf("query %2u: P8^T has %d x 127, expected %d  m=%04x rowsum=%04x l=%04x\n",
                       i, ones, (int)want_ones, fp16_at(m_l, i), RSUM_AT(i), fp16_at(l_l, i));
                err++;
            }
        }
        // ---- against the numerical golden -----------------------------------
        uint32_t worst_m = 0, worst_sum = 0, worst_l = 0;
        for (uint32_t qt = 0; qt < (uint32_t)NQ; qt++)
            for (uint32_t i = 0; i < BR; i++) {
                uint32_t u = fp16_ulp(fp16_at((volatile uint8_t *)M_OF(qt), i),
                                      m_golden[qt * BR + i]);
                if (u > worst_m) worst_m = u;
                u = fp16_ulp(fp16_at((volatile uint8_t *)L_OF(qt), i), l_golden[qt * BR + i]);
                if (u > worst_l) worst_l = u;
            }
        for (uint32_t i = 0; i < BR; i++) {
            uint32_t u = fp16_ulp(RSUM_AT(i), rowsum_golden[i]);
            if (u > worst_sum) worst_sum = u;
        }
        uint32_t p8_bad = 0;
        for (uint32_t b = 0; b < PGOLD_NBEATS; b++) {
            uint32_t key = (uint32_t)b * PGOLD_STRIDE;
            for (uint32_t i = 0; i < BR; i++) {
                int16_t want = p8_golden[b * BR + i];
                if (want < 0) continue;  // within 0.1 of a rounding boundary
                if ((int16_t)p8t[P8_OFF(key, i)] != want) p8_bad++;
            }
        }
        // ---- O, every query tile, against the attention math ----------------
        // O = sum_j (prod_{i>j} corr_i) V_j^T.P_j^T per query tile, within sum|terms|/64 + 32
        // (the SIMD exponential is a LUT). This sees the interleaved P layout, the n-major
        // B walk, the per-tile column scaler and the D32 map.
        uint32_t o_worst = 0;
        for (uint32_t qt = 0; qt < (uint32_t)NQ; qt++) {
            volatile int32_t *o32 = (volatile int32_t *)(l1 + OACC32_OF(qt));
            const int32_t *og = o32_golden + qt * (BR * DHEAD);
            const int32_t *ot = o32_tol + qt * (BR * DHEAD);
            for (uint32_t i = 0; i < (uint32_t)(BR * DHEAD); i++) {
                int32_t dlt = o32[i] - og[i];
                uint32_t e = (uint32_t)(dlt < 0 ? -dlt : dlt);
                if (e > o_worst) o_worst = e;
                if (e > (uint32_t)ot[i]) {
                    if (o_bad < 4)
                        printf("O[q%u][%4u] = %ld, expected %ld +- %ld\n", qt, i,
                               (long)o32[i], (long)og[i], (long)ot[i]);
                    o_bad++;
                }
            }
        }
        err += (int)o_bad;

        printf("  golden           m %u ULP  rowsum %u ULP  l %u ULP  P8 %u wrong"
               "  (limits %u/%u/%u/0)\n",
               worst_m, worst_sum, worst_l, p8_bad, GOLD_ULP_M, GOLD_ULP_SUM, GOLD_ULP_L);
        printf("  golden           O %u of %u outside tolerance (%u query tiles);"
               " worst |err| %u\n",
               o_bad, (unsigned)(NQ * BR * DHEAD), (unsigned)NQ, o_worst);
        printf("  data             running max moved in %u of %u (query, tile>=1) pairs,"
               " min corr 0.%03u\n",
               (unsigned)GOLD_MAX_MOVES, (unsigned)(NQ * BR * (NKV - 1)),
               (unsigned)GOLD_CORR_MIN_PERMILLE);
        if (worst_m > GOLD_ULP_M || p8_bad != 0 || worst_sum > GOLD_ULP_SUM ||
            worst_l > GOLD_ULP_L || o_bad) {
            printf("  golden MISMATCH: the arithmetic disagrees with the float model\n");
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
        printf("    init           %u B of m = -inf in %u cc  [%u -> %u]\n",
               (uint32_t)NQ * (uint32_t)(BR * 2),
               pub[227] - pub[226], pub[226], pub[227]);
        printf("    Q load         %u B in %u cc  [%u -> %u]\n",
               (uint32_t)NQ * (uint32_t)(N * K * tileSize * meshCol), pub[225] - pub[224], pub[224], pub[225]);
        printf("    store O        %u B in %u cc of transfer, staged over %u cc  [%u -> %u]"
               "   then m,l in %u cc  [%u -> %u]\n",
               (uint32_t)NQ * (uint32_t)(BR * DHEAD) * 4u, pub[238],
               pub[229] - pub[228], pub[228], pub[229],
               pub[231] - pub[230], pub[230], pub[231]);
        printf("    K(0) split     %u B in %u cc = %u.%u B/cc across TWO engines\n",
               KVBYTES, pub[237], KVBYTES / pub[237], ((KVBYTES * 10u) / pub[237]) % 10u);
        printf("    K per tile     %u B in %u cc = %u.%u B/cc of a 64 B/cc port  (iDMA, tiles 1+)\n",
               KVBYTES, pub[236] / (NKV - 1),
               (KVBYTES * (NKV - 1)) / pub[236],
               ((KVBYTES * (NKV - 1) * 10u) / pub[236]) % 10u);
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
