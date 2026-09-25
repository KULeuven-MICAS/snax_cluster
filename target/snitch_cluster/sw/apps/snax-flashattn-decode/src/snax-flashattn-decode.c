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
// (The operands are full-range INT8, so a score reaches 127^2 * d = 2,064,512 -- 31x past
// FP16's 65,504. The softmax temperature a -- 1/sqrt(d) times the operand scales -- is split:
// a power of two 2^-D32_FP16_SHIFT is applied by the D-port converter while it rounds to
// FP16, which keeps every score finite at no precision cost, and the remainder a' rides
// StreamMap's own scale operand in both exponentials. Neither costs a pass; see
// score_scale_split() in datagen.py.)
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
//                  corr16 = exp(a*(m16_old - m16))   2 -> 1  FUSED: EW0 (ADD) -> Map (EXP)
//
//                       [ corr16 ] .. [ l16_old ]
//                        corrS[j&1]     lrun
//                         \___ sticky MUL gives lsc16, below; and PV(j) copies the
//                              same beat into the C column scaler to rescale O
//
//                  P8^T   = int8(127*exp(a*(S16^T-m16)))     ) FUSED: EW0 (sticky ADD)
//                                                513 -> 257  ) written INTERLEAVED, as
//                                                            ) PV's B operand
//                  sum16  = sum of P16 over keys             ) -> Map (EXP) -> Reduce TAP
//                                                            ) -> Fp16ToInt8 (tail passthrough)
//
//                     THE WHOLE EPILOGUE IS ONE SWEEP. sum16 has no task of its own: it is
//                     the TAP beat, written after the tile. And the quantiser runs in the
//                     same pass because its tail passthrough leaves that trailing beat
//                     alone -- without it the scalar would be narrowed along with the data,
//                     and P would have to be written in FP16 and read back by a separate
//                     quantise task: a third trip over the tile, and an FP16 P buffer of
//                     512 beats written + 512 read per tile.
//
//                       read   [ -m16 ][ S16^T  x512 ]     the latch, then the tile
//                       write  [ P8^T  x256 ][ sum16 ]     into THIS tile's p8 buffer,
//                                              \___        one block per P8_SLOT
//                                                   lsc16 gets a slot of its own later,
//                                                   and the l16 row reads the pair
//                                                   [ sum16 ][ lsc16 ]. Both follow the
//                                                   p8 ping-pong.
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
//   have no reader -- not the GEMM, not the store, not the check -- see the O^T note in
//   the allocation block.
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
// An exact invariant: the number of 127s in each query row of the last tile's P8^T. A key
// that holds the row's running maximum subtracts to zero, exp(0) = 1 and 127 * 1.0 = 127,
// so the count follows from the data; it is read through P8_OFF, so a wrong layout moves
// it.
//
// A numerical golden, which pins the VALUES the invariant cannot see: m against the global
// maximum, the last tile's row sum, l after the whole recurrence, and P8 away from its
// rounding edges, against a float model in datagen.py that follows the hardware's
// precision stage by stage.
//
// Every KV tile has its OWN K and V, so the running max moves (GOLD_MAX_MOVES) and corr is
// not 1. O is checked against the attention math -- V^T.P^T summed over the tiles WITH the
// online rescale, P taken as a matrix rather than as the bytes the quantiser wrote --
// within a per-element tolerance, since the SIMD exponential is a LUT. So the O check sees
// both the P8 layout PV reads and the C column scaler. Build with -DFA_NO_COLSCALE or
// -DFA_NO_INTERLEAVE for the negative controls.
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
//     mrun  [32 FP16, 1 beat,  64 B]  --+
//                                        +-->  4 junction beats, 16/32 lanes live, 256 B
//     lrun  [32 FP16, 1 beat,  64 B]  --+      (128 B of payload, 128 B of identity pad)
//
// mrun and lrun are already one FP16 value per query row, which IS the slot axis -- but they
// are two separate 32-row beats, and the junction wants 8 rows of (m, l) per beat. That
// is a strided 2-beat xDMA task, the same shape as the -m_new fan-out this kernel issues.
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
//      (m, l) stays FP16 because mrun and lrun ARE FP16 already -- that fold needs no
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
// PRE-map elementwise (ADD, sticky), the row sum (ADD), and the tail passthrough that lets
// the same pass quantise the tile while the reduce appends its scalar. tailPassthrough is a
// BUILD-time feature, not a runtime select: without it csr(1) is not there at all and the
// SIMD_QUANT_TAIL write below lands on the next extension's CSR block.
//
// INTERLEAVE4 writes P8 directly in the B-operand layout PV reads (see "P8 IS WRITTEN AS
// PV's B OPERAND" in the loop). Without it the bit is ignored and P lands [key][query],
// which PV silently reads as the wrong keys for every query.
#if !defined(SIMD_EXT_STREAMREDUCE_HAS_MAX) ||          \
    !defined(SIMD_EXT_STREAMREDUCE_HAS_ADD) ||          \
    !defined(SIMD_EXT_STREAMMAP_HAS_EXP) ||             \
    !defined(SIMD_EXT_STREAMELEMENTWISE_0_HAS_ADD) ||   \
    !defined(SIMD_EXT_FP16TOINT8_HAS_TAILPASSTHROUGH) || \
    !defined(SIMD_EXT_FP16TOINT8_HAS_INTERLEAVE4)
#error \
    "This cluster's SIMD block cannot run the FlashAttention softmax: it needs StreamReduce MAX+ADD, StreamMap EXP, a PRE-map StreamElementwise ADD, and Fp16ToInt8 tailPassthrough + interleave."
#endif
#include "snax-versacore-to-lib.h"
// The online rescale of O runs on the GEMM's C read path: Int32ColumnScale in
// reader_writer slot 0 multiplies every INT32 column of C by an FP16 factor as it streams
// into the array. Its window is the host's enable word, N, and 16 factor words = 18 CSRs.
// No capability macro exists for streamer extensions, so the count is the detection.
#if !defined(READER_WRITER_EXTENSION_0_CSR_BASE) || READER_WRITER_EXTENSION_0_CSR_NUM != 18
#error \
    "This cluster's GEMM streamer has no Int32ColumnScale on the C read path (reader_writer slot 0): O cannot be rescaled when the running max moves."
#endif
#define COLSCALE_CSR READER_WRITER_EXTENSION_0_CSR_BASE
// The D-port converter must be the shift build: enable + extra-loop policy + k = 3 CSRs.
// Without it csr(1)+1 would land on the next streamer register, and full-range scores
// would overflow FP16 to Inf.
#if !defined(READER_WRITER_EXTENSION_1_CSR_NUM) || READER_WRITER_EXTENSION_1_CSR_NUM != 3
#error \
    "The GEMM's D-port Int32ToFp16Converter has no power-of-two shift (build it with shift: 1): full-range INT8 scores would overflow FP16."
#endif

// Negative controls, selected with EXTRA_CFLAGS: -DFA_NO_COLSCALE leaves O un-rescaled,
// -DFA_NO_INTERLEAVE writes P8 in the plain [key][query] pack and reads it n-major. Each
// must make the O check FAIL; that is what shows the check can see the error it guards
// against.
#ifdef FA_NO_COLSCALE
#define FA_COLSCALE_ON 0
#else
#define FA_COLSCALE_ON 1
#endif
#ifdef FA_NO_INTERLEAVE
#define FA_QUANT_ILV 0u
#else
#define FA_QUANT_ILV SIMD_QUANT_ILV4
#endif
#include "snax-xdma-lib.h"
#include "snrt.h"
#include "snax-perf-census.h"

// The matmul engine on hart 0 is VersaCore: a 1024-MAC INT8 array whose single spatial
// unrolling (Mu, Ku, Nu) is what this kernel calls meshRow/tileSize/meshCol = 16/4/16. It
// has ONE output port, carrying INT32 with an optional convert to FP16 on the way out, and
// no rescale unit -- this kernel needs neither a quantised output nor a rescale.
//
// meshRow, tileSize, meshCol, BR, BC, DHEAD, NKV and D32_FP16_SHIFT all arrive from data.h. They
// are derived there from the same M/N/K and array shape that produced the streamer
// descriptors, so the kernel's idea of the tile and the descriptors' idea of it cannot
// drift apart. Change them in data/params.hjson.

// BC and DHEAD are independent. Bc is the tiling knob, free to grow until TCDM is
// full; d is a property of the model. The two matmuls therefore have different
// shapes, which gemm_set_shape() switches between per dispatch.

#define SBEATS BC        // S^T / P beats: one per KEY, 32 query lanes each

// HOW MANY SCORE TILES CAN BE IN FLIGHT. With two, QK(j+2) and PV(j) block on the
// SAME event -- the buffer SM(j) is still reading -- so the array has no ready work
// of either kind and stalls. A third buffer decouples them: QK(j+2) can run while
// SM(j) still holds its tile. Costs (1+SBEATS)*BEAT = 32,832 B of scratchpad each.
#ifndef NSCORE
#define NSCORE 3
#endif
#define DBEATS DHEAD     // O^T beats:     one per HEAD element, 32 query lanes each
#define PBEATS (BC / 2)  // after Fp16ToInt8 halves them

// The cluster's TCDM, which is both the footprint guard's limit and the ceiling on the
// arena zero-fill below.
#define TCDM_BYTES (512u * 1024u)
// What the testbench's main-memory xDMA endpoint can address (TCDMAddrWidth = 19).
#define XDMA_MAIN_BASE  0x80000000u
#define XDMA_MAIN_REACH (512u * 1024u)

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

// A REAL transfer on the xDMA: main memory -> L1, one contiguous tile.
//
// This is the descriptor the fill above is not. Two differences carry all of it:
// the reader channels are ON, so the beat comes from memory rather than from a
// generated pattern, and the writer extension is OFF, so what the reader fetched is
// what lands. Leaving the memset armed from a previous fill would overwrite every
// byte of the tile with the pattern and still report success.
//
// The source address is a full main-memory address, which is what makes this a
// REMOTE transfer: the local engine sees the source is not its own, and forwards a
// cfg frame to the endpoint that owns it. On a testbench without that endpoint the
// task commits and never finishes.
//
// Hand-armed with constant CSR addresses for the same reason as the fill: a
// csrw_ss with a COMPUTED address degrades to a jump-table load from main memory,
// which costs ~92 cycles a write instead of one.
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
    // EXACTLY 8 channels, not 0xFFFFFFFF. The engine has eight of each, and the
    // memset above can set all 32 bits harmlessly only because its READER is off --
    // a reader told it owns 32 channels waits on twenty-four that never answer, and
    // the task simply never retires. `snax_xdma_local_wait` is unbounded, so that
    // presents as a simulation running for ever rather than as an error.
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, 0xFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, 0xFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, 0xFFFFFFFFu);
    // BOTH extension enables, not just the writer's. The memset that ran before this
    // leaves its writer extension armed, and a reader extension left over from any
    // earlier task would filter the tile on its way in.
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLE_PTR, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLE_PTR, 0);
}

// Re-point an armed tile descriptor at the next tile.
//
// BOTH base addresses have to be rewritten, not just the destination. The engine
// consumes its address registers as the transfer walks, so a second task issued
// without restoring them reads from wherever the first one finished. It still moves
// the right NUMBER of bytes at the right rate, so the kernel looks healthy and only
// the data is wrong: m, P8 and rowsum stay exact (they come from K) while O, which comes
// from V, does not. This mirrors the library's
// own snax_xdma_retask_1d, which rewrites the same five registers.
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

// WHERE THE RESULT LANDS. Its own array, not a data.h buffer: writing (O, m, l) past the
// end of C would put the state block on top of o32_golden, and the check would then
// compare against a golden the kernel had just overwritten.
static int32_t fa_out[BR * DHEAD + 128];

// A spin that cannot hang the simulation for ever. The two cores are hand-synchronised,
// so a deadlock is possible, and an infinite loop in Verilator burns wall-clock with no
// diagnosis.
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
// Composing them removes both problems. Once the counter has reached this id the task
// provably started, so the busy poll below can only be observing
// its end. Subtracting before the compare keeps the counter test correct across a wrap.
//
#ifndef SPIN_LIMIT
#define SPIN_LIMIT 2000000u
#endif

// Bounded, so a stuck engine reports itself instead of hanging the simulation. Returns
// non-zero on timeout, which the caller folds into its own timeout count.
__attribute__((always_inline)) static inline uint32_t gemm_wait(uint32_t task_id) {
    uint32_t spins = 0;
    // Shares SPIN_LIMIT with the handshake waits so that shortening it for a
    // deadlock hunt shortens THIS one too. A limit in the hundreds of thousands
    // costs a stuck dispatch a million cycles, and the run never reaches its own
    // diagnostic.
    while ((int32_t)(csrr_ss(GEMMX_FINISHED_TASK) - task_id) < 0) {
        if (++spins > SPIN_LIMIT) return 1;
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
// cannot fold them -- and .bss maps to DRAM on this target. Reading them here would cost
// an L3 round trip per field per dispatch: 134 cycles for 20 csrw, against the 1.0 cycle a
// folded `csrw imm` takes. So the caller hoists them into locals once, outside the loop,
// the same rule gemm_d32_emit_fp16() below follows.
//
// B's TWO strides move too, and not because of M or K. QK reads Q^T blocks n-major, block
// (k, n) at (n*K + k)*64: strides {64, K*64}. PV reads P^T, which the interleaving
// quantiser writes k-major, block (k, n) at (k*N + n)*P8_SLOT: strides {N*P8_SLOT,
// P8_SLOT} (see FA_PV_LAYOUT for the pitch). Both walk k innermost; only where each
// block is found differs.
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

#if FA_PV_LAYOUT == 2
// A's walk, which the PAIRED layout makes differ between the two shapes (see
// FA_PV_LAYOUT). QK's K tile is dense: k in one-block steps, m blocks K blocks apart,
// dim 3 unused. PV's V^T is stored with its m blocks in pairs whose k rows interleave,
// block (m, k) at (m/2)*2K + 2k + m%2 blocks, so k steps two blocks, the pair's second
// row sits one block up and the next pair 2K blocks on. The ORDER of m is unchanged --
// dims 2 and 3 are just m split in two -- so B, C, D and the array never notice.
__attribute__((always_inline)) static inline void gemm_set_a_walk(
    int32_t s0, int32_t b2, int32_t s2, int32_t b3, int32_t s3) {
    csrw_ss(T_STRIDE_READER_0_0, s0);
    csrw_ss(T_BOUND_READER_0_2, b2);
    csrw_ss(T_STRIDE_READER_0_2, s2);
    csrw_ss(T_BOUND_READER_0_3, b3);
    csrw_ss(T_STRIDE_READER_0_3, s3);
}
#define A_BLK ((int32_t)(tileSize * meshRow))  // one INT8 A block, bytes
#endif

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
    // The power-of-two output scale: S16 = RNE(S_int * 2^-k). Full-range INT8 scores reach
    // 2,064,512 and would overflow FP16 to Inf without it. Only meaningful when on; written
    // either way so no dispatch inherits a stale k.
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 2, on ? (uint32_t)D32_FP16_SHIFT : 0u);
    csrw_ss(T_BOUND_READER_WRITER_1_0, bound0);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, stride1);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, stride2);
}

// The C read path's Int32ColumnScale: y = rne(x * f[col]) on every INT32 lane of C, one
// FP16 factor per output column. O^T's columns are the queries, so loading corr_j as the
// factors makes PV(j) compute O^T = corr_j (.) O^T + V_j^T . P_j^T in ONE dispatch -- the
// online-softmax rescale, applied while O streams back into the array anyway.
//
// Like every streamer CSR these are latched at START, so arming for dispatch d happens
// while d-1 runs. The enable is restated for EVERY dispatch, QK included: a leftover
// enable would scale the next C, and QK's C is the (masked) zero bias.
//
// `nblk` is the output's column-block count (N = Br / meshCol), which tells the scaler
// which 16 of the 32 factors a beat uses: blocks arrive n-innermost, 8 beats each.
__attribute__((always_inline)) static inline void gemm_colscale_arm(uint32_t on,
                                                                    uint32_t nblk) {
    csrw_ss(COLSCALE_CSR + 1, nblk);
    csrw_ss(COLSCALE_CSR + 0, on);
}

// The 32 FP16 factors, two per word: exactly the byte layout of the 64 B corr beat the
// softmax writes, so the beat is copied verbatim. Unrolled by hand so each CSR address is
// a constant and every write folds to one `csrw imm`; a loop would put a jump-table load
// per write back on the critical path, which is where this runs (after the softmax
// publish, before PV's START).
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

// ---- PV's bank phase ---------------------------------------------------------------
//
// PV reads one A beat (V^T) and one B beat (P^T) per array pass, 64 B each, and TCDM
// is 32 banks x 8 B: a 256 B rotation of four 64 B groups. A walks k in 64 B steps. B
// walks k over P8 as the interleaving quantiser writes it, k-major, block (k, n) at
// (k*N + n)*P8_SLOT. At P8_SLOT = 64 that is 128 B steps: B's group advances twice as
// fast as A's, the distance between the two streams keeps turning, and every few passes
// they meet on the same banks.
//
// The quantiser cannot simply write n-major instead. One epilogue pass streams 2G + 1
// beats (G key groups of two B blocks, then the tapped rowsum), and an odd count has no
// factor of 2 for a nested address loop to split the n0 and n1 blocks with. The layouts
// below either make both streams advance at the SAME rate around the banks, or move the
// rowsum out of that pass (3):
//
//   0  dense:   P8 slots 64 B, V dense          rates differ     PV 11,863  pipe 23,297
//   1  spaced:  P8 slots 160 B, V dense         B steps 320 = 64 mod 256, as A does
//               (ping-pong buffers nested, P8_NEST)              PV  9,488  pipe 21,219
//   2  paired:  P8 dense, V's m blocks paired   A steps 128, as B does
//                                                                PV 10,628  pipe 22,626
//   3  n-major: P8 dense n-major, V dense       B steps 64, as A does
//                                                                PV  9,660  pipe 22,266
//
// (cycles, Verilator, NKV = 4, one source built four ways; 1.14 cycles a pass is also
// what the n-major walk of the FA_NO_INTERLEAVE control measures.) 3 works because the
// rowsum leaves the epilogue for a pass of its own after the publish, so the epilogue
// writes 2G beats, which a 2-D pattern can split into the two n halves. That pass
// re-reads the 32 KiB score tile while the GEMM runs QK(j+1), which it slows by ~900
// cycles, and it keeps the SIMD 63% busy, against 45% for 1.
//
// THE RATE IS THE LEVER, THE OFFSET BARELY IS. The streams are decoupled by their FIFOs,
// so a collision delays one of them, and at equal rates they then stay apart; at unequal
// rates no offset survives. The offset (P8_PAD) moves layout 1 by a percent or two: with
// the buffers apart, PV 9,905 / 9,483 / 9,733 / 9,583 at P8_PAD = 0 / 64 / 128 / 192 B;
// nested, 64 and 192 tie on decode (21,107 / 21,219) and 192 is faster on prefill
// (78,176 against 78,793), so 192 is the default. Layout 0 does not respond to it
// (PV 11,924 at 0, 11,912 at 64).
//
// 1 costs 8 KiB of scratchpad with the buffers nested (48 KiB with P8_NEST = 0, at the
// same speed) and leaves every other buffer where it was. 2 and 3 cost none and are
// slower. 2 relays V on its way in -- the xDMA writes it with a 3-D pattern, which runs
// at 31 B/cc instead of 51 -- and gives PV's A stream a 4-D walk; its B stream still
// stalls 1,850 cycles, against 860 for 1.

#ifndef FA_PV_LAYOUT
#ifdef FA_NO_INTERLEAVE
#define FA_PV_LAYOUT 0   // the plain-pack control has its own, n-major, B walk
#else
#define FA_PV_LAYOUT 1
#endif
#endif
#if defined(FA_NO_INTERLEAVE) && FA_PV_LAYOUT != 0
#error "FA_NO_INTERLEAVE writes the plain 2:1 pack densely; build it with FA_PV_LAYOUT=0"
#endif
#if FA_PV_LAYOUT == 1
#define P8_SLOT 160u   // the smallest pitch >= 64 whose double is 64 mod 256
#elif FA_PV_LAYOUT == 0 || FA_PV_LAYOUT == 2 || FA_PV_LAYOUT == 3
#define P8_SLOT 64u
#else
#error "FA_PV_LAYOUT must be 0, 1, 2 or 3"
#endif
// QK(t) overwrites the score buffer of tile t - NSCORE. The softmax is done with it at
// its publish -- except in FA_PV_LAYOUT 3, where the rowsum pass reads it once more
// AFTER the publish. It is certainly done by the NEXT publish, which waits for every
// task issued before it. In the dispatch order below QK(t) already follows PV(t-2),
// which waits for that same publish, so the later release costs nothing.
#if FA_PV_LAYOUT == 3
#define S_FREE_LAG (NSCORE - 2)
#else
#define S_FREE_LAG (NSCORE - 1)
#endif

// Byte offset of P8(key, query) in a P8 buffer, as the INTERLEAVE quantiser lays it: B
// block (key/4, query/16) k-major, one block per P8_SLOT, then one 4-byte atom per query
// holding four keys. datagen.py's p8_interleave_offset() is the dense (P8_SLOT = 64)
// formula, asserted there against the walk PV's B reader performs.
#define P8_NBLK (BR / meshCol)
#if FA_PV_LAYOUT == 3
// n-major: block (k, n) at (n*Bc/4 + k)*64 -- the n0 half, then the n1 half.
#define P8_OFF(key, q)                                                          \
    ((((q) / meshCol) * (BC / tileSize) + (key) / tileSize) * P8_SLOT +         \
     ((q) % meshCol) * tileSize + (key) % tileSize)
#else
#define P8_OFF(key, q)                                                          \
    ((((key) / tileSize) * P8_NBLK + (q) / meshCol) * P8_SLOT +                 \
     ((q) % meshCol) * tileSize + (key) % tileSize)
#endif

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
    // RECLAIM THE UNREAD OPERAND SPACE. data.h reserves four GEMM operand regions
    // below the arena, and two of them are never read on this kernel:
    //
    //   local_a  [0, 64 KiB)        K tile 0            -- used
    //   local_b  [64 KiB, +4 KiB)   Q                   -- used
    //   local_c  [68 KiB, +64 KiB)  C of QK             -- MASKED on every dispatch
    //   local_d32[132 KiB, +32 KiB) D of QK             -- D goes to d32_delta[] instead
    //
    // QK passes a C-enable mask of 0 (gemm_set_shape's last argument), and a disabled
    // channel presents zero without issuing a memory request, so base_c is never
    // dereferenced; the D32 writer is pointed at the score buffers in the arena. That
    // leaves 96 KiB reserved and untouched. Starting the arena at local_c hands it
    // back, which is what makes a third score buffer fit at all.
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
    uint32_t s_buf[NSCORE];                       // [ -m_new ][ S^T x Bc ] each
    for (uint32_t i = 0; i < NSCORE; i++) { s_buf[i] = top; top += (1 + SBEATS) * BEAT; }
    uint8_t *rmax  = l1 + top;  top += BEAT;          // rowmax, and later -m_new
    uint8_t *mrun  = l1 + top;  top += BEAT;          // running m   ] pair for max(m_old,rowmax)
    uint8_t *mnew  = l1 + top;  top += BEAT;
    // No beat for m_old - m_new: the difference is consumed inside the extension chain
    // by the fused task below and never reaches memory.
    // corr = exp(a*(m_old - m_new)) has TWO readers: the sticky MUL corr*l_old on the
    // SIMD, and PV(j) on the GEMM core, which copies it into the C column scaler. The GEMM
    // reads it only after the softmax publishes tile j -- by which time the SIMD may
    // already be on tile j+1 and computing the next corr. So corr ping-pongs like P8 does:
    // tile j writes slot j&1, and tile j+2 cannot start until QK(j+2), which the GEMM
    // issues only after launching PV(j). Both slots sit BELOW lrun so the sticky MUL
    // reaches lrun with a positive stride (the AGU's stride is unsigned).
    uint8_t *corrS = l1 + top;  top += 2u * BEAT;     // [corr, even j][corr, odd j]  ] latch for
    uint8_t *lrun  = l1 + top;  top += BEAT;          // running l                     ] corr*l_old
    uint8_t *lnew  = l1 + top;  top += BEAT;
    // No FP16 P buffer: the epilogue quantises the tile in the sweep that produces it,
    // so P exists only as INT8, inside the P8 buffers below. The tapped rowsum and
    // corr*l_old live there too -- see the P8 allocation.
    uint32_t oacc32 = top;      top += BR * DHEAD * 4; // O^T = [d, Br], INT32, the only O
    // P8 CARRIES ITS OWN TAIL: [ P8 x PBEATS ][ rowsum ], then corr*l_old.
    //
    // The epilogue pass narrows the tile and emits the tapped rowsum in ONE sweep
    // (Fp16ToInt8 tailPassthrough lets the reduce's trailing FP16 beat past the
    // quantiser), so the rowsum lands wherever the writer's flat stream puts it: one
    // P8_SLOT past the last P8 block. corr*l_old has a slot per buffer -- right behind
    // the rowsum when the buffers are apart -- and task 13 reads the pair
    // [rowsum][corr*l_old] as one LANEWISE 2-beat reduce, at whatever stride separates
    // them. Both are per tile, because the P8 buffer they belong to ping-pongs.
    // Each buffer is rounded up to whole 256 B rotations, and the pad behind them
    // undoes the one in front, so the K and V tiles allocated next keep their banks
    // whatever the P8 pitch and offset.
#define P8_BYTES (((PBEATS * P8_SLOT + 2u * BEAT) + 255u) & ~255u)
#ifndef P8_PAD
#if FA_PV_LAYOUT == 1
#define P8_PAD 192u  // see FA_PV_LAYOUT: with the buffers nested, 64 and 192 tie
#else
#define P8_PAD 0u
#endif
#endif
// NESTED PING-PONG. At a 160 B pitch each P8 block leaves a 96 B gap, room for a block
// of the OTHER buffer: p8_1 = p8_0 + P8_NEST puts its blocks in p8_0's gaps, so the two
// buffers share one region instead of taking 2 x 40 KiB. They never overlap in bytes,
// only in banks, which they do anyway. 0 keeps them apart.
#ifndef P8_NEST
#if FA_PV_LAYOUT == 1
#define P8_NEST 96u
#else
#define P8_NEST 0u
#endif
#endif
#if P8_NEST && (P8_NEST < 64u || P8_NEST + 64u > P8_SLOT)
#error "P8_NEST must leave a whole 64 B block between two of the other buffer's"
#endif
    top += P8_PAD;
#if P8_NEST
    // One region: p8_0's blocks, p8_1's in between, the two rowsums the same way, then
    // the two corr*l_old slots. corr*l_old cannot sit right behind its rowsum --
    // the other buffer's rowsum is there -- so task 13 reads the pair with a stride.
    uint32_t p8_0  = top;
    uint32_t p8_1  = top + P8_NEST;
    uint32_t p8_lsc = top + PBEATS * P8_SLOT + P8_NEST + BEAT;
    top += ((PBEATS * P8_SLOT + P8_NEST + 3u * BEAT) + 255u) & ~255u;
#else
    uint32_t p8_0  = top;       top += P8_BYTES;
    uint32_t p8_1  = top;       top += P8_BYTES;
#endif
    top += (256u - P8_PAD) & 255u;
    // O^T HAS ONE REPRESENTATION, and the online rescale is applied to IT.
    //
    // The GEMM accumulates P.V in INT32 in `oacc32`, in place (C and D32 both point
    // there, so the matmul computes O += P.V natively). That is the O that is stored and
    // that the golden checks.
    //
    // The rescale O <- corr (.) O happens on the way INTO that same matmul: PV(j) reads
    // O_{j-1} back through C, and the C read path's Int32ColumnScale multiplies column q
    // (= query q) by corr_j[q] as it streams. So PV(j) computes
    //     O^T = corr_j (.) O^T + V_j^T . P_j^T
    // with no extra pass, no extra TCDM traffic and no FP16 copy of O. PV(0) needs none:
    // its C is masked to zero.
    // K AND V STREAM. Every KV tile pulls a fresh 64 KiB K and a fresh 64 KiB V from main
    // memory, double-buffered so the load of tile j+1 overlaps the compute of tile j.
    //
    // The two have DIFFERENT lifetimes, and that is what makes two buffers each enough:
    // K(j) dies at the end of QK(j), V(j) only at the end of PV(j). Freeing them on
    // separate counters gives the iDMA three dispatches of slack on each; freeing both on
    // the later one would put every load straight onto the critical path.
    //
    // Every tile is DISTINCT: K(j) is A + j*Bc*d, V(j) is V + j*Bc*d. Replaying one tile
    // NKV times would freeze the running max after tile 0 and make corr = 1, which leaves
    // the O rescale and the P layout untested.
#define KVBYTES ((uint32_t)(BC * DHEAD))
#if FA_PV_LAYOUT == 2
#define V_K2 ((uint32_t)(BC / tileSize))     // PV's contraction blocks
#define V_M2 ((uint32_t)(DHEAD / meshRow))   // PV's output row blocks, paired
#define V_DST_B0 V_K2
#if tileSize * meshRow != 64 || (DHEAD / meshRow) % 2 != 0
#error "the paired V layout needs one A block per 64 B xDMA beat and an even m count"
#endif
#else
#define V_DST_B0 (KVBYTES / BEAT)
#endif
    uint32_t k_buf[2], v_buf[2];
    k_buf[0] = (uint32_t)delta_local_a;   // the staged buffer becomes buffer 0
    // BANK PHASE IS NOT THE LEVER. The K and V tiles are whole multiples of 256 B -- one
    // full 32-bank rotation -- so allocating one never shifts the bank the next one starts
    // on. Staggering their bases onto distinct banks does not help: the array's feed
    // re-balances, and a stall removed from one operand reappears on another. What moves
    // PV is the RATE at which its two streams step around the banks, not where they start
    // -- see FA_PV_LAYOUT.
    k_buf[1] = top; top += KVBYTES;
    v_buf[0] = top; top += KVBYTES;
    v_buf[1] = top; top += KVBYTES;

    volatile uint32_t *pub  = (volatile uint32_t *)(l1 + top); top += 1024;
    volatile uint32_t *sync = (volatile uint32_t *)(l1 + top); top += 64;

    // THE TASK GEOMETRIES LIVE IN THE ARENA, not in a .l1 static. TCDM is SRAM with no
    // reset: an address that has never been written reads X in RTL, and .l1 is NOLOAD, so
    // a linker-placed array there starts UNDEFINED rather than zero. Allocating them here
    // puts them inside the block the xDMA zeroes below, which is what makes them defined.
    // Still TCDM and not .bss: program_fast() reads every field of two of these per task,
    // and .bss maps to DRAM on this target, so each field would cost an L3 round trip.
    snax_simd_shape_t *shapes = (snax_simd_shape_t *)(l1 + top);
    top += (32u * sizeof(snax_simd_shape_t) + 63u) & ~63u;

    // WARM-UP SCRATCH. The softmax loop runs one pass over a two-beat dummy tile here
    // before S(0) exists -- same instructions, same CSR addresses, same operator
    // sequence, 1/256th of the data -- so that the first real tile does not pay the
    // cold-fetch cost on the critical path. See the loop.
    //
    // 24 beats, because the state commit's destination is a 2-beat STRIDED write whose
    // stride is (lrun - mrun) = 4 beats, and the warm copy of it has to land inside
    // this block rather than on the live running state.
    uint8_t *wscr = l1 + top;  top += 24u * BEAT;

    // THE GEMM DESCRIPTOR, STAGED OUT OF DRAM BEFORE THE CLOCK STARTS.
    // data.h's shape words are ordinary globals, and .bss/.rodata map to MAIN MEMORY on
    // this target: each read is an L3 round trip, and this core issues one outstanding
    // integer load at a time, so thirteen of them are thirteen serialised round trips.
    // Staging the dispatch needs all thirteen, and staging is the first thing the GEMM
    // lane does after the origin barrier -- so read here, outside the measured window,
    // where they cost nothing, and from TCDM inside it, where they cost a few cycles.
    volatile uint32_t *gdesc = (volatile uint32_t *)(l1 + top);  top += 64u;

    // s_hdr is the latch beat the SIMD reads from; d32_delta is the tile itself, one
    // beat past it, which is where the GEMM's D32 port writes.
    int32_t s_hdr[NSCORE], d32_delta[NSCORE];
    for (uint32_t i = 0; i < NSCORE; i++) {
        s_hdr[i]     = (int32_t)s_buf[i];
        d32_delta[i] = (int32_t)(s_buf[i] + BEAT);
    }
    const int32_t p8_delta[2] = {(int32_t)p8_0, (int32_t)p8_1};
    // The two tail slots of the P8 buffer tile j writes. The rowsum is the writer's beat
    // PBEATS, so it lands one P8_SLOT past the last P8 block. corr*l_old is one BEAT
    // behind it, or, with the buffers nested, in the region's own pair of slots.
#define RSUM_OF(j) (l1 + p8_delta[(j) & 1] + PBEATS * P8_SLOT)
#if P8_NEST
#define LSC_OF(j)  (l1 + p8_lsc + ((j) & 1) * BEAT)
#else
#define LSC_OF(j)  (RSUM_OF(j) + BEAT)
#endif
#define CORR_OF(j) (corrS + ((j) & 1) * BEAT)

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
        // Waiting for each one to be non-zero, for the reason t_org does below: the
        // barrier does not wait for the other harts' stores to land.
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
        printf("  PV layout       %d  (P8 pitch %u B, buffers %s, see FA_PV_LAYOUT)\n",
               FA_PV_LAYOUT, P8_SLOT, P8_NEST ? "nested" : "apart");
        if (arena_end > lowest) {
            printf("  TCDM OVERFLOW: the arena runs %ld B into the stacks -- the "
                   "preinit below would zero them. Reduce NSCORE or Bc.\n",
                   (long)(arena_end - lowest));
            cfg_err++;
        }
    }
    snrt_cluster_hw_barrier();
    if (cfg_err) { snrt_cluster_hw_barrier(); return cfg_err; }

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
    // Q, K and V are full-range INT8; the converter's 2^-D32_FP16_SHIFT keeps the scores
    // inside FP16.
    if (snax_is_gemm_core()) {
        // TCDM footprint guard. An overrun corrupts whatever follows rather than
        // faulting, so check the layout rather than trust the arithmetic.
        printf("  TCDM footprint  %lu bytes of %u  (Bc=%d, d=%d, S16 = S_int * 2^-%d)\n",
               (unsigned long)top, TCDM_BYTES, BC, DHEAD, D32_FP16_SHIFT);
        // THE xDMA SEES ONLY THE FIRST 512 KiB OF MAIN MEMORY. The testbench's main-
        // memory endpoint is the cluster's own xDMA wrapper with TCDMAddrWidth = 19, so
        // a source past XDMA_MAIN_REACH wraps to the DRAM base and reads .text -- at full
        // rate, with no error. Every byte the xDMA fetches (all V tiles, the upper half
        // of K tile 0) must sit below it; datagen.py emits V first for that reason.
        const uint32_t reach = XDMA_MAIN_BASE + XDMA_MAIN_REACH;
        if ((uint32_t)(uintptr_t)V + NKV * KVBYTES > reach ||
            (uint32_t)(uintptr_t)A + KVBYTES > reach) {
            printf("  xDMA REACH: V ends %08lx, K(0) ends %08lx, endpoint reaches %08lx\n",
                   (unsigned long)((uint32_t)(uintptr_t)V + NKV * KVBYTES),
                   (unsigned long)((uint32_t)(uintptr_t)A + KVBYTES),
                   (unsigned long)reach);
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
        // 6  corr lands in this tile's corr slot, where the sticky MUL of task 11 and
        //    the GEMM's PV(j) both find it. Base follows the ping-pong, set per tile.
        snax_simd_shape_flat(&sh[11], CORR_OF(0), 1);
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
        sh[17].stride[0] = P8_SLOT;  // one B block per slot; see FA_PV_LAYOUT
#if FA_PV_LAYOUT == 3
        // 9' the rowsum's own pass (layout 3) writes one beat, into the rowsum slot.
        snax_simd_shape_flat(&sh[30], RSUM_OF(0), 1);
#endif
        // 11 corr * l_old: sticky MUL over [corr_j][lrun] -- the latch, then l_old one
        //    stride (1 or 2 beats, by the ping-pong) above it.
        snax_simd_shape_2d(&sh[20], CORR_OF(0), 2, (uint32_t)(lrun - CORR_OF(0)), 1, 0);
        snax_simd_shape_flat(&sh[21], LSC_OF(0), 1);  // seed emits nothing, so just the result
        // 13 l_new = corr*l_old + rowsum: LANEWISE ADD over the pair [rsum][lsc].
        snax_simd_shape_2d(&sh[24], RSUM_OF(0), 2, (uint32_t)(LSC_OF(0) - RSUM_OF(0)), 1, 0);
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
    // core stamping its own after the barrier would leave them ~530 cycles apart -- enough
    // to make a cross-lane reading of the trace ("QK0 starts before K(0) lands") an
    // artefact of the offset rather than a fact about the machine. One core publishes the origin and
    // everyone subtracts that same value, so the lanes are comparable to the cycle.
    // The cluster contention counters are armed HERE, by one hart, so that the
    // window they cover is the same barrier-to-barrier span the three lanes
    // measure with mcycle. They are cluster-global, so arming them on the GEMM
    // hart covers the engines whose cores are asleep as well.
    // Outside the measured window on purpose: see the gdesc allocation above.
    if (snax_is_gemm_core()) {
        gdesc[0]  = (uint32_t)K;               gdesc[1]  = (uint32_t)M;
        gdesc[2]  = (uint32_t)N;               gdesc[3]  = (uint32_t)Atlstride2;
        gdesc[4]  = (uint32_t)Btlstride1;      gdesc[5]  = (uint32_t)S2_K;
        gdesc[6]  = (uint32_t)S2_M;            gdesc[7]  = (uint32_t)S2_Atlstride2;
        gdesc[8]  = (uint32_t)S2_Btlstride1;   gdesc[9]  = (uint32_t)delta_local_b;
        gdesc[10] = (uint32_t)D32tlbound0;     gdesc[11] = (uint32_t)D32tlstride1;
        gdesc[12] = (uint32_t)D32tlstride2;
        gdesc[13] = (uint32_t)Btlstride0;      gdesc[14] = (uint32_t)S2_Btlstride0;
    }

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
        // first key tile lands, and one DMA port moves it at ~55 B/cc. Both engines
        // are idle at cycle zero -- the iDMA has only Q to fetch and the xDMA's first
        // V is not wanted for thousands of cycles -- so K(0) is split down the middle
        // and carried by both at once. The two halves are contiguous in the same
        // buffer, so the GEMM's A stream never learns that two engines wrote it.
        //
        // Before the -inf fill, not after: `sync[7]` is not read until the softmax's
        // first tile, which is thousands of cycles away, while `sync[2]` gates the
        // very first dispatch.
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
        pub[94] = snrt_mcycle() - t0;

        uint32_t x0 = snrt_mcycle() - t0;
        xdma_fill_arm((uint32_t)mrun, BEAT, 1u, 0xFBFFFBFFu);
        snax_xdma_local_wait(snax_xdma_start());
        pub[58] = x0; pub[59] = snrt_mcycle() - t0;
        sync[7] = 1;   // m is live -- the SIMD may start tile 0

        // ---- the V stream, concurrent with hart 3's K stream ---------------
        // K and V are independent tiles of equal size that the pipeline needs at
        // the same rate, so putting them on one engine serialises two transfers
        // that have no reason to be ordered. Splitting them is what HeMAiA does,
        // and it needs the testbench's second xDMA endpoint on main memory.
        //
        // The descriptor is armed ONCE. Tiles differ only in their source (V + j*Bc*d)
        // and which of the two destination buffers they land in, so the per-tile cost
        // is the five-register retask rather than a re-arm.
        uint32_t xdma_busy = 0, xdma_block = 0;
        xdma_tile_arm((uint32_t)(l1 + v_buf[0]), (uint32_t)(uintptr_t)V, BEAT,
                      KVBYTES / BEAT);
#if FA_PV_LAYOUT == 2
        // PAIRED V (see FA_PV_LAYOUT). The reader walks the tile as main memory holds
        // it, block (m, k) at m*K2 + k; the writer lays each block where PV's paired A
        // walk looks for it, (m/2)*2K2 + 2k + m%2. One A block is one xDMA beat, so the
        // relayout is three loops on the destination and costs no extra pass.
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 0, V_K2);
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 0, 2u * BEAT);
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 1, 2u);
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 1, BEAT);
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 2, V_M2 / 2u);
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 2, 2u * V_K2 * BEAT);
#endif
        for (uint32_t j = 0; j < NKV; j++) {
            uint32_t w0 = snrt_mcycle() - t0;
            sync[11] = j * 10u + 1u;  // waiting: V buffer free
            if (j >= 2) SNAX_SPIN_UNTIL(sync[3] >= j - 1, timeouts);  // V buffer free
            sync[11] = j * 10u + 2u;  // transferring V(j)
            uint32_t v0 = snrt_mcycle() - t0;
            xdma_block += v0 - w0;
            xdma_tile_retask((uint32_t)(l1 + v_buf[j & 1]),
                             (uint32_t)(uintptr_t)V + j * KVBYTES, V_DST_B0);
            // A V tile is read from main memory, which the PEER endpoint serves, so
            // the hardware runs it as a REMOTE task. BOUNDED like every other wait in
            // this kernel: one unbounded wait anywhere means a deadlock elsewhere
            // presents as a run that never ends instead of one that reports.
            {
                snax_xdma_task_t tk = snax_xdma_start_task();
                uint32_t fp = tk.remote ? XDMA_FINISH_REMOTE_TASK_PTR
                                        : XDMA_FINISH_LOCAL_TASK_PTR;
                uint32_t spins = 0;
                while (snax_read_xdma_cfg_reg(fp) < tk.task_id) {
                    if (++spins > SPIN_LIMIT) { timeouts++; break; }
                }
            }
            uint32_t v1 = snrt_mcycle() - t0;
            sync[5] = j + 1;
            pub[36 + 4 * j] = v0; pub[37 + 4 * j] = v1;
            xdma_busy += v1 - v0;
            if (timeouts) break;
        }
        pub[62] = xdma_busy;
        pub[63] = snrt_mcycle() - t0;
        pub[77] = xdma_block;
    }

    if (snrt_is_dm_core()) {
        uint32_t t0 = t_org;
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
            // V is on the xDMA (hart 2); this lane carries K only, so the blocked time
            // is the wait for the GEMM to retire a QK and free a K buffer.
            uint32_t w0 = snrt_mcycle() - t0;
            sync[12] = j * 10u + 1u;  // waiting: K buffer free
            if (j >= 2) SNAX_SPIN_UNTIL(sync[4] >= j - 1, timeouts);  // K buffer free
            sync[12] = j * 10u + 2u;  // transferring K(j)
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
            if (j == 0u) pub[97] = k1 - k0; else pub[96] += k1 - k0;
            sync[2] = j + 1;
            if (j == 0) pub[93] = snrt_mcycle() - t0;   // K(0) published to the GEMM lane
            pub[34 + 4 * j] = k0; pub[35 + 4 * j] = k1;
            dma_busy += k1 - k0;
            dma_block += k0 - w0;
        }
        // THE STORE. A query tile's result leaving the cluster is (O, m, l): the INT32
        // accumulator plus the eight beats of running state. On one cluster it is a drain
        // tail; KV-sharded across clusters it is the partial each shard contributes to the
        // cross-cluster fold, so it is part of the measured pipeline.
        // That fold is the xDMA MonoidJunction, armed on this very transfer -- the DMA
        // below becomes the collective, and nothing above it moves. See "SHARDING THE KEYS
        // ACROSS CLUSTERS" at the top of this file for the geometry words, the `fmt`
        // trap, and why the numerator crosses in FP32 via Int32ToFp32 rather than FP16.
        // SPLIT. O is final the instant the last PV retires; only m and l wait on the SIMD's
        // trailing commit. Shipping them together would charge O's 16 KiB with the SIMD's
        // drain.
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
        const int32_t g_n = (int32_t)gdesc[2];
        const int32_t s1_k = (int32_t)gdesc[0], s1_m = (int32_t)gdesc[1];
        const int32_t s1_blk = s1_m * g_n;
        const int32_t s1_as2 = (int32_t)gdesc[3], s1_bs1 = (int32_t)gdesc[4];
        const int32_t s2_k = (int32_t)gdesc[5], s2_m = (int32_t)gdesc[6];
        const int32_t s2_blk = s2_m * g_n;
        const int32_t s2_as2 = (int32_t)gdesc[7];
        // B strides per shape, see gemm_set_shape(): QK's Q^T n-major as data.h emits it,
        // PV's P^T k-major because that is how the interleaving quantiser writes it.
        const int32_t s1_bs0 = (int32_t)gdesc[13];
        const int32_t s2_btile = (int32_t)gdesc[14];     // one 64 B B block
#ifdef FA_NO_INTERLEAVE
        const int32_t s2_bs0 = s2_btile, s2_bs1 = (int32_t)gdesc[8];
#elif FA_PV_LAYOUT == 3
        // n-major, the walk QK uses for Q^T: k one block, n half a buffer.
        const int32_t s2_bs0 = s2_btile, s2_bs1 = (int32_t)gdesc[8];
#else
        const int32_t s2_bs0 = g_n * (int32_t)P8_SLOT, s2_bs1 = (int32_t)P8_SLOT;
        (void)s2_btile;
#endif
        const uint32_t base_b = l1u + gdesc[9];
        // The arena owns the address range data.h names delta_local_c. QK masks C, so
        // this address is never dereferenced -- point it somewhere real anyway rather
        // than at memory that belongs to a score buffer.
        const uint32_t base_c = l1u + oacc32;
        const uint32_t base_o = l1u + oacc32;
        const uint32_t d32_b0_i = gdesc[10];  const uint32_t d32_b0_f = d32_b0_i / 2;
        const uint32_t d32_s1_i = gdesc[11];  const uint32_t d32_s1_f = d32_s1_i;
        const uint32_t d32_s2_i = gdesc[12];  const uint32_t d32_s2_f = d32_s2_i / 2;
        uint32_t g0 = t_org;
        pub[90] = snrt_mcycle() - g0;   // when this lane reached its own code

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
#if FA_PV_LAYOUT == 2
#define QK_A_WALK() gemm_set_a_walk(A_BLK, s1_m, s1_as2, 1, 0)
#define PV_A_WALK() gemm_set_a_walk(2 * A_BLK, 2, A_BLK, s2_m / 2, 2 * s2_k * A_BLK)
#else
#define QK_A_WALK() do { } while (0)
#define PV_A_WALK() do { } while (0)
#endif
#define STAGE_QK(t)                                                            \
        do {                                                                   \
            gemm_set_shape(s1_k, s1_m, s1_blk, s1_as2, s1_bs0, s1_bs1, 0u);    \
            QK_A_WALK();                                                       \
            gemm_set_bases(l1u + k_buf[(t) & 1], base_b, base_c,               \
                           l1u + (uint32_t)d32_delta[(t) % NSCORE]);                \
            gemm_d32_emit_fp16(1, d32_b0_f, d32_s1_f, d32_s2_f);               \
            gemm_colscale_arm(0u, (uint32_t)g_n);                              \
        } while (0)
        // Tile 0 has nothing to accumulate onto and masks the C reader off rather
        // than clearing oacc32: a disabled channel presents ZERO and issues no
        // TCDM request, which is exactly the seed the accumulator needs. For the same
        // reason it needs no rescale; every later PV arms the column scaler here and
        // loads corr_t's factors once the softmax has published it (PV_FACTORS).
#define STAGE_PV(t)                                                            \
        do {                                                                   \
            gemm_set_shape(s2_k, s2_m, s2_blk, s2_as2, s2_bs0, s2_bs1,         \
                           (t) == 0u ? 0u : 0xFFFFFFFFu);                      \
            PV_A_WALK();                                                       \
            gemm_set_bases(l1u + v_buf[(t) & 1],                               \
                           l1u + (uint32_t)p8_delta[(t) & 1], base_o, base_o); \
            gemm_d32_emit_fp16(0, d32_b0_i, d32_s1_i, d32_s2_i);               \
            gemm_colscale_arm(FA_COLSCALE_ON && (t) != 0u, (uint32_t)g_n);     \
        } while (0)
        // corr_t is final only once the softmax has published tile t, so unlike the rest
        // of the descriptor its 16 words cannot be staged early: they are the one part of
        // a PV's configuration that sits on the critical path, between the publish and
        // START. 16 TCDM loads + 16 csrw.
#define PV_FACTORS(t)                                                          \
        do {                                                                   \
            if (FA_COLSCALE_ON && (t) != 0u)                                   \
                gemm_colscale_factors((const volatile uint32_t *)CORR_OF(t));  \
        } while (0)

#ifdef GEMM_QUEUE
        // ---- QUEUED DISPATCH --------------------------------------------------
        //
        // With cfgQueueDepth = 2 in BOTH managers -- the streamer's and the array's --
        // a start write snapshots the register bank and retires, instead of stalling
        // the core until the accelerator is idle. So dispatch d can be armed AND
        // STARTED while d-1 is still running, and the array takes it on the cycle it
        // retires d-1, with no dead cycles at the boundary for the core to notice a
        // retirement and re-issue.
        //
        // `cfgFire` in ReqRspManager is a WRITE EVENT with a non-zero LSB, not an edge
        // on a stored level, so two consecutive `csrw START, 1` enqueue two tasks and
        // nothing has to clear the register in between.
        //
        // THE PER-DISPATCH COUNTERS ARE NOT READ HERE, deliberately. GEMMX_PERFORMANCE_
        // COUNTER and the three stall counters restart on every start pulse, so once
        // d+1 has begun they do not describe d, and there is no free-running array-busy
        // counter to read instead. Build without GEMM_QUEUE for the stall decomposition
        // and with it for the makespan.
        //
        // The schedule is built once rather than derived in the loop, because the order
        // is not a simple alternation: QK(0) leads, each KV tile then contributes QK(j)
        // and PV(j-1), and the final two dispatches are both PV.
        uint8_t sk[2 * NKV], si[2 * NKV];
        {
            uint32_t d = 0;
            sk[d] = 0; si[d++] = 0;
            for (uint32_t j = 1; j <= (uint32_t)NKV; j++) {
                if (j < (uint32_t)NKV) { sk[d] = 0; si[d++] = (uint8_t)j; }
                sk[d] = 1; si[d++] = (uint8_t)(j - 1);
            }
        }
        pub[91] = snrt_mcycle() - g0;

        uint32_t tid_q[2] = {0u, 0u};
        uint32_t have_prev = 0u, prev_d = 0u;
        for (uint32_t d = 0; d <= 2u * (uint32_t)NKV; d++) {
            if (d < 2u * (uint32_t)NKV) {
                const uint32_t kind = sk[d], t = si[d];
                // ARM FIRST, THEN WAIT. The predecessor snapshotted the CSR bank on its
                // own start write, so the bank is free the moment it was launched -- and
                // arming here puts the 21 writes inside the dependency wait rather than
                // after it.
                if (kind == 0u) STAGE_QK(t); else STAGE_PV(t);
                {
                    uint32_t w0 = snrt_mcycle();
                    if (kind == 0u) {
                        sync[9] = t * 10u + 1u;
                        if (t >= NSCORE)
                            SNAX_SPIN_UNTIL(sync[1] >= t - S_FREE_LAG, timeouts);
                        sync[9] = t * 10u + 2u;
                        SNAX_SPIN_UNTIL(sync[2] >= t + 1, timeouts);
                    } else {
                        sync[9] = t * 10u + 5u;
                        SNAX_SPIN_UNTIL(sync[1] >= t + 1, timeouts);
                        sync[9] = t * 10u + 6u;
                        SNAX_SPIN_UNTIL(sync[5] >= t + 1, timeouts);
                    }
                    gemm_stall += snrt_mcycle() - w0;
                }
                pub[(kind == 0u ? 10u : 12u) + 4u * t] = snrt_mcycle() - g0;
                sync[9] = t * 10u + (kind == 0u ? 3u : 7u);
                if (kind == 1u) PV_FACTORS(t);
                tid_q[d & 1u] = ++gemm_seq;
                gemm_launch();
                gemm_ack();
            }
            if (have_prev) {
                const uint32_t kind = sk[prev_d], t = si[prev_d];
                timeouts += gemm_wait(tid_q[prev_d & 1u]);
                pub[(kind == 0u ? 11u : 13u) + 4u * t] = snrt_mcycle() - g0;
                if (kind == 0u) {
                    sync[0] = t + 1;   // S(t) is published
                    sync[4] = t + 1;   // K(t) is dead
                } else {
                    sync[3] = t + 1;   // V(t) is dead
                }
            }
            have_prev = (d < 2u * (uint32_t)NKV);
            prev_d = d;
        }
#else
        // Dispatch 0 has no predecessor to hide behind. Writing it here puts its
        // cold instruction fetches inside the wait for K(0), which is dead time.
        STAGE_QK(0u);
        pub[91] = snrt_mcycle() - g0;   // dispatch 0 staged: the lane is ready to fire

        for (uint32_t j = 0; j <= NKV; j++) {
            uint32_t tid = 0;
            if (j < NKV) {
                // S(j) into the buffer the SIMD core is not reading: the one tile
                // j-NSCORE used.
                {
                    uint32_t w0 = snrt_mcycle();
                    // NSCORE tiles in flight: QK(j) may run once the softmax has
                    // released that buffer (see S_FREE_LAG).
                    sync[9] = j * 10u + 1u;   // waiting: score buffer free
                    if (j >= NSCORE)
                        SNAX_SPIN_UNTIL(sync[1] >= j - S_FREE_LAG, timeouts);
                    sync[9] = j * 10u + 2u;   // waiting: K(j) landed
                    SNAX_SPIN_UNTIL(sync[2] >= j + 1, timeouts);  // K(j) has landed
                    gemm_stall += snrt_mcycle() - w0;
                }
                pub[10 + 4 * j] = snrt_mcycle() - g0;
                sync[9] = j * 10u + 3u;   // QK issued
                tid = ++gemm_seq; gemm_launch();   // S^T = K.Q^T, staged one dispatch ago
                gemm_ack();
                // The successor is PV(j-1) in this same iteration, except at j == 0
                // where nothing precedes it and the next dispatch is QK(1).
                if (j == 0u) STAGE_QK(1u); else STAGE_PV(j - 1u);
                pub[64 + 2 * j] = snrt_mcycle() - g0;  // successor staged, array running
                timeouts += gemm_wait(tid);
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
                sync[9] = j * 10u + 5u;   // waiting: softmax j-1 published
                SNAX_SPIN_UNTIL(sync[1] >= j, timeouts);
                sync[9] = j * 10u + 6u;   // waiting: V(j-1) landed
                SNAX_SPIN_UNTIL(sync[5] >= j, timeouts);   // V(j-1) has landed
                sync[9] = j * 10u + 7u;   // PV issued
                gemm_stall += snrt_mcycle() - w0;
                pub[12 + 4 * (j - 1)] = snrt_mcycle() - g0;
                PV_FACTORS(j - 1u);
                tid = ++gemm_seq; gemm_launch();   // O^T = V^T.P^T, staged one dispatch ago
                gemm_ack();
                // The successor is QK(j+1) while KV tiles remain, otherwise the
                // next iteration's PV, and nothing at all after the last.
                if (j + 1u < (uint32_t)NKV)      STAGE_QK(j + 1u);
                else if (j < (uint32_t)NKV)      STAGE_PV(j);
                pub[65 + 2 * (j - 1)] = snrt_mcycle() - g0;
                timeouts += gemm_wait(tid);
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
#endif  /* GEMM_QUEUE */
        // The last PV left the scaler armed in the bank. Nothing else runs on this
        // streamer here, but the contract is that no one ever inherits an enable.
        gemm_colscale_arm(0u, (uint32_t)g_n);
        gemm_wall = snrt_mcycle() - g0;
#undef STAGE_QK
#undef STAGE_PV
#undef QK_A_WALK
#undef PV_A_WALK
    }

    if (snax_is_simd_core()) {
        uint32_t s0 = t_org;
        uint32_t simd_busy0 = snax_simd_busy_cycles();
        pub[92] = snrt_mcycle() - s0;   // when this lane reached its own code

        // ONE WARM ITERATION, THEN THE NKV REAL ONES.
        //
        // The first pass through this instruction sequence and its constant pool costs
        // several hundred cycles of instruction fetch that later passes do not pay, and
        // that cost is on the critical path: the array finishes QK(1) and then idles
        // until the softmax publishes tile 0, so the whole pipeline behind it starts
        // late by however long the first tile takes.
        //
        // The SIMD core has nothing to do until S(0) lands, so part of that idle window
        // runs the tile body once over a two-beat dummy tile in `wscr`. It must be the
        // SAME code, not a copy -- an inlined helper called twice warms the wrong lines
        // -- so the warm pass is iteration -1 of this loop, and everything it varies
        // (the tile base, the beat counts, the two reduce CSRs, the quantiser tail) is a
        // VARIABLE the real iterations set to the real geometry.
        //
        // Exactly one warm write would be destructive: the state commit lands on
        // mrun/lrun, which hold the -inf / 0 seed. It is redirected into wscr. Every
        // other warm write targets a slot tile 0 overwrites before anything reads it.
        for (uint32_t jw = 0; jw < NKV + 1u; jw++) {
            const uint32_t warm = (jw == 0u);
            const uint32_t j    = warm ? 0u : jw - 1u;
            uint32_t w0 = snrt_mcycle();
            sync[10] = j * 10u + 1u;  // waiting: QK(j) has produced S(j)
            if (!warm) {
                SNAX_SPIN_UNTIL(sync[0] >= j + 1, timeouts);
                if (j == 0) SNAX_SPIN_UNTIL(sync[7] >= 1, timeouts);
            }
            sync[10] = j * 10u + 2u;  // running the softmax chain
            simd_stall += snrt_mcycle() - w0;

            pub[26 + 2 * j] = snrt_mcycle() - s0;
            pub[72 + j] = snax_simd_busy_cycles() - simd_busy0;

            // The geometry of THIS pass. `nb` is the only thing the warm pass shrinks;
            // it has to be a multiple of 4, because the interleaving quantiser emits
            // only whole 4-beat groups (and asserts that tailPeriod is one).
            const uint32_t nb = warm ? 4u : (uint32_t)SBEATS;
            uint8_t *sbase = warm ? wscr : (l1 + s_hdr[j % NSCORE]);
            sh[0].base    = sbase + BEAT;   // the tile sits one beat past its latch
            sh[0].bound[0] = nb;
            // The latch beat and the tile behind it both live in the buffer the GEMM
            // just wrote, so the epilogue's read and the -m_new fan-out follow it too.
            sh[5].base      = warm ? wscr : (l1 + s_hdr[j % NSCORE]);
            sh[5].stride[0] = warm ? BEAT : (uint32_t)(rmax - (l1 + s_hdr[j % NSCORE]));
            sh[14].base     = sbase;
            sh[14].bound[0] = 1u + nb;
            // The epilogue writes [P8][rowsum] into this tile's P8 buffer, and corr*l_old
            // has a slot per buffer, so all three follow the ping-pong.
            sh[17].base      = warm ? (wscr + 8u * BEAT) : (l1 + p8_delta[j & 1]);
#if FA_PV_LAYOUT == 3
            // Innermost the two n blocks of a key group, half a buffer apart; dim 1 (the
            // key groups, one block apart) is written around the task, see below. Dense
            // in the warm pass, whose single group must stay inside wscr.
            sh[17].bound[0]  = 2u;
            sh[17].stride[0] = warm ? BEAT : (PBEATS / 2u) * BEAT;
            sh[30].base      = warm ? (wscr + 12u * BEAT) : RSUM_OF(j);
#else
            sh[17].bound[0]  = nb / 2u + 1u;
            // Dense in the warm pass, so its three beats stay clear of the warm slots
            // at 12 and 13 whatever the P8 pitch.
            sh[17].stride[0] = warm ? BEAT : P8_SLOT;
#endif
            sh[21].base = warm ? (wscr + 13u * BEAT) : LSC_OF(j);
            sh[24].base = warm ? (wscr + 12u * BEAT) : RSUM_OF(j);
            // [rowsum][corr*l_old]: adjacent unless the P8 buffers are nested.
            sh[24].stride[0] = warm ? BEAT : (uint32_t)(LSC_OF(j) - RSUM_OF(j));
            // THE ONE DESTRUCTIVE WRITE: the commit would otherwise overwrite the
            // running (m, l) seed with whatever the dummy tile produced.
            sh[29].base = warm ? (wscr + 16u * BEAT) : mrun;
            // corr goes to this tile's ping-pong slot, and the sticky MUL reads it with
            // l_old one stride above. The warm pass keeps both inside wscr.
            sh[11].base      = warm ? (wscr + 14u * BEAT) : CORR_OF(j);
            sh[20].base      = warm ? (wscr + 14u * BEAT) : CORR_OF(j);
            sh[20].stride[0] = warm ? BEAT : (uint32_t)(lrun - CORR_OF(j));

            // ---- the online softmax, in full -------------------------------
            // Every task below is one beat of arithmetic except 1, 8+9 and 9'.
            // That ratio is the point: the STATE UPDATE is dominated by per-task
            // start/drain, not by compute.

            // 1  rowmax over the tile the GEMM has already written as FP16. The tile is
            //    read where it lies and swallowed: one beat of per-lane maxima comes out.
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMREDUCE));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR, nb);
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
            // The chain's fixed order [EW0, Map, Reduce, EW1, Fp16ToInt8] decides which
            // pairs of operations can share a task. This pair can, because the combine
            // comes BEFORE the transform: EW0 with operandCount 2 folds the adjacent pair
            // [-m_new][m_old] into one beat and Map exponentiates it in flight, so the
            // difference is never written to memory and never read back.
            //
            // The l update's corr*l_old followed by + rowsum cannot share a task the same
            // way: its combine-after-combine would need a second operand fetched at EW1,
            // and the chain feeds EW1 only from the stage above it.
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                        (1u << SIMD_EXT_STREAMMAP));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 0, 2);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 1, SIMD_EW_ADD);
            // The temperature goes HERE as well as into P: corr must be the ratio of the
            // two P scales it bridges, exp(a*m_old) / exp(a*m_new).
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0, SCORE_SCALE_BITS);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 1, 0);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 2, SIMD_FUNC_EXP);
            snax_simd_program_1d(&sh[8], &sh[11]);
            snax_simd_fire();

            // 8+9 P = exp(S - m_new) AND rowsum, in ONE pass over the tile.
            // EW0 is upstream of Map, so the per-lane subtract happens before the exponential
            // and the tile is never written out in between.
#if FA_PV_LAYOUT == 3
            // No reduce in this pass: without its tapped beat the quantiser's output is
            // exactly 2G beats, and the rowsum gets a pass of its own after the publish.
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                        (1u << SIMD_EXT_STREAMMAP) |
                                        (1u << SIMD_EXT_FP16TOINT8));
#else
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                        (1u << SIMD_EXT_STREAMMAP) |
                                        (1u << SIMD_EXT_STREAMREDUCE) |
                                        (1u << SIMD_EXT_FP16TOINT8));
#endif
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 0, 1);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 1,
                                    SIMD_EW_ADD | SIMD_EW_STICKY_B);
            // exp(a * (S - m)): the map's own scale is the softmax temperature, so the
            // score scaling costs no task. m is the max of the UNSCALED scores, which is
            // the same key for any a > 0.
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0, SCORE_SCALE_BITS);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 1, 0);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 2, SIMD_FUNC_EXP);
#if FA_PV_LAYOUT != 3
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR, nb);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1,
                                    SIMD_RED_ADD | SIMD_RED_LANEWISE | SIMD_RED_TAP);
#endif
            // The quantiser narrows the SBEATS data beats and passes the reduce's trailing
            // rowsum beat through untouched, so this one task writes [P8 x PBEATS][rowsum].
            //
            // THE SCALE IS 127, NOT 1. P lies in (0, 1], so at 1.0 every element rounds to
            // 0 or 1 and P8 is a binary mask: PV then sums a handful of V rows and
            // computes nothing like attention. At 127 P8 uses the whole INT8 range. O
            // comes out 127x too large; l is summed from the UNSCALED FP16 P, so a final
            // O / l must divide by 127 as well.
            //
            // P8 IS WRITTEN AS PV's B OPERAND. PV reads P^T in B blocks of Ku x Nu = 4
            // keys x 16 queries, stored as one 4-byte atom PER QUERY holding four
            // consecutive keys. The plain 2:1 pack writes [key][query] -- 4 bytes are one
            // key's four queries -- and PV would read them as one query's four keys. The
            // INTERLEAVE bit makes each group of 4 key beats come out as two B blocks
            // (k = g, n = 0, 1), which PV walks with k-major strides. Same pass, same
            // rate, no relayout. The writer puts one block per P8_SLOT, which is what
            // keeps PV's two operand streams off each other's banks (FA_PV_LAYOUT).
            snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 0, 0x42FE0000u);  // 127.0f
#if FA_PV_LAYOUT == 3
            snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 1, FA_QUANT_ILV);
            // The one 2-D write in the loop: dim 1 steps the key groups. A task latches
            // the bank when it fires, so dim 1 goes back to the neutral bound 1 / stride 0
            // that program_1d assumes straight after.
            snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 1, nb / 4u);
            snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 1, BEAT);
            snax_simd_program_1d(&sh[14], &sh[17]);
            snax_simd_fire();
            snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 1, 1u);
            snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 1, 0u);
#else
            snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 1,
                                    SIMD_QUANT_TAIL(nb) | FA_QUANT_ILV);
            snax_simd_program_1d(&sh[14], &sh[17]);
            snax_simd_fire();
#endif

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
            if (!warm) sync[1] = j + 1;

            pub[27 + 2 * j] = snrt_mcycle() - s0;

#if FA_PV_LAYOUT == 3
            // 9' rowsum = sum over keys of exp(a*(S - m_new)), its own pass over the same
            //    [-m_new][S^T] the epilogue read. Off the GEMM's path: nothing the GEMM
            //    needs waits on it, only l does. EW0 and the map still hold the
            //    epilogue's configuration; only the enable mask and the reduce change.
            //    It reads S^T(j) after the publish, which is why the GEMM frees a score
            //    buffer one publish later in this layout (S_FREE_LAG).
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                        (1u << SIMD_EXT_STREAMMAP) |
                                        (1u << SIMD_EXT_STREAMREDUCE));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR, nb);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1,
                                    SIMD_RED_ADD | SIMD_RED_LANEWISE);
            snax_simd_program_1d(&sh[14], &sh[30]);
            snax_simd_fire();
#endif

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
            // The tile's ISSUE ends here, not at the publish above: tasks 9' (layout 3),
            // 11, 13 and 15+16 are all fired afterwards. Bracketing to the publish would attribute
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

    // Read the contention counters before anything below it runs: the golden
    // compare and the report walk TCDM and print, which is traffic the kernel
    // does not do and which would otherwise land inside the census.
    if (snax_is_gemm_core()) snax_perf_read(&perf);

    // ---- invariants for the FULL algorithm ----------------------------------
    //
    // Every KV tile is distinct, so these hold for REAL attention data, where the running
    // max moves and corr != 1 (GOLD_MAX_MOVES says how often):
    //
    //   P8^T count of 127 per query row        the row's maximum subtracts to zero and
    //                                            exp(0) = 1.0 quantises to exactly 127 --
    //                                            when that max is IN the last tile; read
    //                                            through P8_OFF, so a wrong layout moves it
    //   m    == the global max                  exact: a max of converted integers
    //   l, rowsum, P8 against the float model   ULPs, or exact away from rounding edges
    //   O    against V^T.P^T WITH the rescale   per-element tolerance, see datagen.py
    if (snax_is_simd_core()) {
        volatile int8_t *p8t = (volatile int8_t *)(l1 + p8_delta[(NKV - 1) & 1]);
        volatile uint8_t *rsum = RSUM_OF(NKV - 1);
        for (uint32_t i = 0; i < BR; i++) {   // i = query row = lane
            // The count, not just the presence: a dropped beat, a tail written into the
            // wrong slot or a shifted write all move it.
            int ones = 0;
            for (uint32_t j = 0; j < SBEATS; j++)
                if (p8t[P8_OFF(j, i)] == 127) ones++;
            int16_t want_ones = p8max_golden[i];
            // A query whose global max came from an EARLIER tile has no 127 here, so the
            // golden's count (which knows where the max is) is the reference.
            if (want_ones >= 0 && ones != want_ones) {
                printf("query %2u: P8^T has %d x 127, expected %d  m=%04x rowsum=%04x l=%04x\n",
                       i, ones, (int)want_ones, fp16_at(mrun, i), fp16_at(rsum, i),
                       fp16_at(lrun, i));
                err++;
            }
        }

        // ---- against the numerical golden -----------------------------------
        // m against the true maximum over all NKV*Bc keys, the last tile's row sum, l
        // after the whole recurrence, and P8 itself on one key in PGOLD_STRIDE.
        uint32_t worst_m = 0, worst_sum = 0, worst_l = 0;
        for (uint32_t i = 0; i < BR; i++) {
            uint32_t u = fp16_ulp(fp16_at(mrun, i), m_golden[i]);
            if (u > worst_m) worst_m = u;
            u = fp16_ulp(fp16_at(rsum, i), rowsum_golden[i]);
            if (u > worst_sum) worst_sum = u;
            u = fp16_ulp(fp16_at(lrun, i), l_golden[i]);
            if (u > worst_l) worst_l = u;
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
        // ---- O, the only check that reaches the SECOND matmul --------------
        // m, P8 and the sums all stop at the score tile. This compares the INT32
        // accumulator element by element against O built from the attention math,
        // O = sum_j (prod_{i>j} corr_i) V_j^T.P_j^T, so it sees the P8 layout PV reads,
        // the column scaler on C, a wrong shape-2 bound or stride, a broken
        // accumulate-in-place, and a D32 map that lands blocks in the wrong order.
        //
        // Not bit-exact, because the SIMD exponential is a LUT: a P8 can land one INT8
        // step off numpy's (moving an element by |V| <= 16) and corr one ULP off. The
        // tolerance is sum|terms|/64 + 32 per element. Without the rescale, or with the
        // plain P layout, thousands of elements are off by far more (see the FA_NO_*
        // controls).
        uint32_t o_worst = 0, o_worst_i = 0;
        {
            volatile int32_t *o32 = (volatile int32_t *)(l1 + oacc32);
            for (uint32_t i = 0; i < (uint32_t)(BR * DHEAD); i++) {
                int32_t d = o32[i] - o32_golden[i];
                uint32_t e = (uint32_t)(d < 0 ? -d : d);
                if (e > o_worst) { o_worst = e; o_worst_i = i; }
                if (e > (uint32_t)o32_tol[i]) {
                    if (o_bad < 4)
                        printf("O[%4u] = %ld, expected %ld +- %ld\n", i, (long)o32[i],
                               (long)o32_golden[i], (long)o32_tol[i]);
                    o_bad++;
                }
            }
            err += (int)o_bad;
        }

        printf("  golden           m %u ULP  rowsum %u ULP  l %u ULP  P8 %u wrong"
               "  (limits %u/%u/%u/0)\n",
               worst_m, worst_sum, worst_l, p8_bad, GOLD_ULP_M, GOLD_ULP_SUM, GOLD_ULP_L);
        printf("  golden           O %u of %u outside tolerance; worst |err| %u at [%u]"
               " (tol %ld, |O| %ld)\n",
               o_bad, (unsigned)(BR * DHEAD), o_worst, o_worst_i, (long)o32_tol[o_worst_i],
               (long)(o32_golden[o_worst_i] < 0 ? -o32_golden[o_worst_i]
                                                : o32_golden[o_worst_i]));
        printf("  data             running max moved in %u of %u (query, tile>=1) pairs,"
               " min corr 0.%03u\n",
               (unsigned)GOLD_MAX_MOVES, (unsigned)(BR * (NKV - 1)),
               (unsigned)GOLD_CORR_MIN_PERMILLE);
        if (worst_m > GOLD_ULP_M || p8_bad != 0 || worst_sum > GOLD_ULP_SUM ||
            worst_l > GOLD_ULP_L || o_bad) {
            printf("  golden MISMATCH: the arithmetic disagrees with the float model\n");
            err++;
        }
    }

    // On a deadlock every waiter times out and each blames its own condition; the
    // handshake as a WHOLE is what identifies the cycle, so dump it from each core.
    if (timeouts) {
        printf("  DEADLOCK core %lu timeouts=%d | counters S0=%lu S1=%lu K=%lu PV=%lu "
               "Kfree=%lu V=%lu | stopped at: gemm=%lu simd=%lu xdma=%lu idma=%lu"
               "  (j*10+step)\n",
               (unsigned long)snrt_cluster_core_idx(), timeouts,
               (unsigned long)sync[0], (unsigned long)sync[1], (unsigned long)sync[2],
               (unsigned long)sync[3], (unsigned long)sync[4], (unsigned long)sync[5],
               (unsigned long)sync[9], (unsigned long)sync[10],
               (unsigned long)sync[11], (unsigned long)sync[12]);
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
        if (pub[63] > pipeline) pipeline = pub[63];  // and so is the V stream's tail
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
        // K and V are on SEPARATE engines. Their transfers overlap, so the two
        // "moving" figures are concurrent and must not be added into one port's load.
        printf("  xDMA core        moving %5u (%2u%%)  blocked on a buffer %5u  wall %5u\n",
               pub[62], 100u * pub[62] / pipeline, pub[77], pub[63]);
        printf("    init           %u B cleared in %u cc  [%u -> %u]\n",
               2u * SIMD_BEAT_BYTES, pub[59] - pub[58], pub[58], pub[59]);
        printf("    Q load         %u B in %u cc  [%u -> %u]\n",
               (uint32_t)(N * K * tileSize * meshCol), pub[57] - pub[56], pub[56], pub[57]);
        printf("    store O        %u B in %u cc  [%u -> %u]   then m,l in %u cc  [%u -> %u]\n",
               (uint32_t)(BR * DHEAD) * 4u, pub[54] - pub[53], pub[53], pub[54],
               pub[61] - pub[60], pub[60], pub[61]);
        printf("    K(0) split     %u B in %u cc = %u.%u B/cc across TWO engines\n",
               KVBYTES, pub[97], KVBYTES / pub[97], ((KVBYTES * 10u) / pub[97]) % 10u);
        printf("    K per tile     %u B in %u cc = %u.%u B/cc of a 64 B/cc port  (iDMA, tiles 1+)\n",
               KVBYTES, pub[96] / (NKV - 1),
               (KVBYTES * (NKV - 1)) / pub[96],
               ((KVBYTES * (NKV - 1) * 10u) / pub[96]) % 10u);
        printf("    V per tile     %u B in %u cc = %u.%u B/cc of a 64 B/cc port  (xDMA)\n",
               KVBYTES, pub[62] / NKV,
               (KVBYTES * NKV) / (pub[62] ? pub[62] : 1u),
               ((KVBYTES * NKV * 10u) / (pub[62] ? pub[62] : 1u)) % 10u);
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
        printf("\n  TRACE head gemm_arrive %lu staged %lu qk0_launch %lu"
               "  simd_arrive %lu  k0_land %lu",
               (unsigned long)pub[90], (unsigned long)pub[91],
               (unsigned long)pub[10], (unsigned long)pub[92],
               (unsigned long)pub[93]);
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
        snax_perf_report(&perf, pipeline);
    }
    snrt_cluster_hw_barrier();
    return 0;
}
