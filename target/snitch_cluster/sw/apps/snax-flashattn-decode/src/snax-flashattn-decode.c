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
// splatted over a whole output beat. Softmax reduces per QUERY ROW, so the layout alone decides which of
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
// so it reduces each 512-beat plane down to one beat and writes nothing else. Each plane
// carries a spare beat in front for exactly that reason: the epilogue is a sticky-B pass,
// its broadcast operand has to sit immediately before the plane, and that beat is where
// the rowmax writes m16. A tap would have had to COPY the whole tile to put it behind a latch.
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
//                       stride, and SUB is x - B = m16_old - m16; corr16 lands in this tile's corr slot, where the l update
//                       and PV(j) -- which copies it into the C column scaler to rescale
//                       O -- both find it
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
//                  mold_h of the next tile <- m16    2 -> 2  a copy, no operator
//
//                     Old and new never need a "commit". m16_old lives in the planes' mold
//                     beats and m16 in their latch beats, so the rowmax writes the new value
//                     without touching the old one that corr still needs; l16_old is
//                     consumed into lsc16 before the l update overwrites it in place. The
//                     one copy carries m16 from the latches into the mold beats of the score
//                     buffer the next tile is being written into, which is all the state
//                     tile j+1 needs; after the last tile it carries m16 into the state
//                     vector m instead.
//
//   GEMM (hart 0)  O32^T += V8^T . P8^T    V8^T [128,512]  64 KiB  ]  2.10 M MAC
//                                          P8^T [512, 32]  16 KiB  ]  2048 cycles
//                                     -> O32^T [128, 32]  16 KiB     accumulated in place
//
//   O32^T is the ONLY representation of O. An FP16 copy rescaled by corr each tile would
//   have no reader -- not the GEMM, not the store, not the check -- see the O^T note in
//   the allocation block.
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
// form cuts the SAME sequence into P slices and gives one to each cluster. Q is replicated --
// every cluster attends the same query tile.
//
//      K, V [N, d]  INT8                  what cluster c holds when its slice is done
//       +---------------+  -+-           m_c [Br] FP16   l_c [Br] FP16   O_c [d, Br] INT32
//   c=0 |   keys 0..511 |   |    ->        m_st            l_st            oacc32
//       +---------------+   |            (64 B)          (64 B)          (16 KiB)
//   c=1 | keys 512..1023|   N    ->      m_1             l_1             O_1
//       +---------------+   |
//   c=2 |keys 1024..1535|   |    ->      m_2             l_2             O_2
//       +---------------+   |
//   c=3 |keys 1536..2047|   |    ->      m_3             l_3             O_3
//       +---------------+  -+-
//                                      four PARTIAL answers to the same question
//
// THESE DO NOT ADD AS THEY STAND. Each shard's l and O are expressed relative to ITS OWN
// maximum: if shard 0 saw a top score of 10 and shard 1 saw 2, shard 1's numbers are all
// scaled by e^2 where shard 0's are scaled by e^10, and adding them adds quantities in
// different units. Shard 1 must first be re-based onto the common maximum by exp(2 - 10):
//
//     m* = max_c m_c            alpha_c = exp(m_c - m*)      <- one scalar per shard, per row
//     l* = sum_c alpha_c . l_c                  O* = sum_c alpha_c . O_c       then O*/l*
//
// ---- THE REDUCE: SHARE THE MAXIMUM FIRST, THEN EVERYTHING IS A SUM ----
//
// alpha_c is the only thing that puts the partials in different units, and it is a function of
// the row maxima alone. If every shard knows m* BEFORE it exponentiates, its l_c and O_c are
// already relative to m*, alpha_c is 1 for every shard, and the reduces are plain sums:
//
//   1. m_c      each shard's row max over its whole slice (the rowmax tasks, run over every tile)
//   2. m*       ElementwiseJunction MAX, FP16, CSR word 0x02, one 64 B beat per shard, gathered at
//               the root; the root sends m* back to every shard (a ChainWrite of the same beat)
//   3. l_c, O_c the exp / quantise / PV passes, seeded with m* instead of the running max
//   4. l*       ElementwiseJunction ADD, FP16,  CSR word 0x00, 64 B
//      O*       ElementwiseJunction ADD, INT32, CSR word 0x60, oacc32 as it stands: 16 KiB = 256 beats
//   5. O*/l*    once, at the root
//
// (the elementwise word is [3:0] op, [6:4] fmt: op 0 ADD, 2 MAX; fmt 0 FP16, 6 INT32.)
//
// O never leaves INT32. The integer grid wraps, so the sum is exact and independent of the order
// of the route, and an elementwise fold needs no key in the beat, so there is no conversion and
// no layout to build: oacc32 is read as it lies. The sum cannot overflow while 127 * 127 * N <
// 2^31, i.e. for fewer than about 133,000 keys, and it is a sum of one quantity only if every
// shard quantises P8 and V8 with the same scales.
//
// Step 2 is one gather and one broadcast of a single beat, about 60 cycles a hop on the chain
// bench. The second pass's V tiles can stream in while it runs, because the first pass reads
// only K.
//
// ---- THE ROUTE: the fold happens IN the data movement, not after it ----
//
// Arming a junction is what turns a transfer into a GATHER (`collectiveMode :=
// junctionEnabled`). Each hop's WRITER folds the beats arriving from the previous cluster
// against the beats it reads from its own TCDM, and forwards the result. No cluster ever holds
// more than two partials, and the wire carries one partial's worth of traffic per hop:
//
//     cluster 0          cluster 1          cluster 2          cluster 3 (root)
//    +-----------+      +-----------+      +-----------+      +-----------+
//    |   X_0     |      |   X_1     |      |   X_2     |      |   X_3     |   <- local (B)
//    +-----+-----+      +-----+-----+      +-----+-----+      +-----+-----+
//          |                  | B                | B                | B
//          |  A         +-----v-----+  A   +-----v-----+  A   +-----v-----+
//          +----------->|  A (+) B  |----->|  A (+) B  |----->|  A (+) B  |
//                       +-----------+      +-----------+      +-----------+
//                                                                   |
//                                                              X* in TCDM
//
//   A = the stream arriving from the previous hop      B = this cluster's own buffer
//
// X is m, l or O: one such chain per reduce above. The HEAD is special: nothing arrives at it,
// so it gets a reader frame only and simply SENDS. Every other node gets both frames -- read
// your own buffer, arm your junction, forward. The chain is the writerPtr list in data order, and
// bit 0 of XDMA_DST_JCT_ENABLE_PTR selects the ElementwiseJunction (WRITER_JCT_ELEMENTWISEJUNCTION),
// whose word is XDMA_DST_JCT_CSR_PTR + 0. Contrast with shipping every shard to a root that
// folds P streams: that costs the root P-1 receives and P-1 buffers, this costs every node one of
// each, and the fold is hidden inside a transfer that had to happen anyway.
//
// ---- WHY NOT FOLD (m, l, O) IN ONE PASS ON THE MONOID JUNCTION ----
//
// The SoftmaxStatsJunction folds the pair (m, l) in one pass with no return trip, which is the
// right tool when the statistics are the whole answer and the maximum cannot be shared first --
// a router softmax is that case. It reads m_st and l_st as the SIMD leaves them: one group is a
// beat of 32 FP16 keys and a beat of 32 FP16 values, so the pair would have to be one contiguous
// 128 B region with m_st directly below l_st (today the corr slots sit between them). Here it is
// not the reduce, for three reasons:
//
//   - It gives m* and l* and leaves O_c scaled by its own m_c. O* still needs exp(m_c - m*)
//     applied per shard, which needs m* back at every shard: the return trip the one pass was
//     meant to avoid.
//   - Folding O in the same pass puts the key in every beat. The statistics junction has no value
//     field beyond l, and the general MonoidJunction cuts a row's 129 values into 9 chunks of 15
//     with the key replicated, so the beat has to be built in memory first. oacc32 is [d][Br]
//     with the query rows contiguous, the transpose of the d-contiguous chunk, and the xDMA
//     reader cannot make it: it moves 8-byte words (two adjacent rows of one d) from one base
//     with one spatial stride, and the cluster has no 32-bit transposer.
//   - O has to be FP32 on that route (FP16 saturates the numerator), but the INT32-to-FP32
//     reader extension only acts on a transfer this cluster reads itself: reader-side
//     configuration frames carry no extension words to another cluster, so a remote shard's
//     partial would cross as raw INT32 bits.
//
// FaGatherTester measures what does fit on the general MonoidJunction, against a float64
// online-softmax golden, relative to the largest term of each sum: FP32 raw 3.1e-06, the same
// with a power-of-two embedding 3.3e-06, the pre-scaled add 2.4e-07 (FP32) and 1.6e-03 (FP16),
// and the encoding a reasonable person writes first (normalise by l, key stays m) 3.0 -- finite,
// format-legal and wrong. The statistics junction of the shipped cluster is held bit for bit to
// that operator by SoftmaxStatsJunctionTester.
//
// ---- WHAT THIS KERNEL WOULD HAVE TO CHANGE ----
//
//   1. THE LOOP BECOMES TWO PASSES over the slice. Pass 1 runs QK and the rowmax for every tile
//      and keeps S in TCDM (BC * BR * 2 B a tile); pass 2 runs the exp / quantise / PV passes
//      seeded with m*. With m* known from the start nothing moves, so corr is 1 and the C
//      column scaler is not needed. The pipelined loop above is the one-cluster, online form.
//
//   2. THE STORE BECOMES THREE xDMA TASKS ON THE ElementwiseJunction, none of them needing a
//      reader or writer extension: MAX over m_st (before pass 2), ADD over l_st, and INT32 ADD
//      over oacc32 (both after the last PV retires). O is final the instant the last PV
//      retires; m and l wait on the SIMD's trailing commit, so the O gather can start first.
//
//   3. BEAT COUNTS MUST MATCH ON BOTH OPERAND STREAMS. The join is two independently dispatched
//      cfgs with no hardware backstop; a mismatch stalls for ever. The junction raises
//      XDMA_JCT_STATUS bit 1 (starved) rather than hanging silently, and bit 0 if the armed
//      operator cannot honour its word. Check both AFTER the transfer -- arming successfully
//      is not evidence that the fold ran.
//
// NONE OF THIS IS EXERCISABLE HERE. On a single cluster the junction sits on the crossing and
// localLoopback bypasses it, so no path reaches the operator: this app behaves identically
// whether the junction is built or not. The operators are covered by the Tier-1 junction suite;
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
// PRE-map elementwise (SUB, sticky and pairs), the row sum (ADD), and the tail passthrough that lets
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
#include "snax-tcdm-priority.h"

// The TCDM arbitration policy for the whole run (snax-tcdm-priority.h, SNAX_TCDM_POLICY_*): 0, the
// hardware's own. Build with EXTRA_CFLAGS=-DFA_TCDM_PRIO=n to try another.
#ifndef FA_TCDM_PRIO
#define FA_TCDM_PRIO 0
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
// meshRow, tileSize, meshCol, BR, BC, DHEAD, NKV and D32_FP16_SHIFT all arrive from data.h. They
// are derived there from the same M/N/K and array shape that produced the streamer
// descriptors, so the kernel's idea of the tile and the descriptors' idea of it cannot
// drift apart. Change them in data/params.hjson.

// BC and DHEAD are independent. Bc is the tiling knob, free to grow until TCDM is
// full; d is a property of the model. The two matmuls therefore have different
// shapes, which gemm_set_shape() switches between per dispatch.

// The query tile is NH SIMD beats wide: Br = NH * meshCol queries, one N column block of
// the array -- one beat -- per half. Every per-query vector (m, corr, l) is NH contiguous
// beats, and the score tile and P8 are NH planes (see THE SCORE PLANES in main()).
#define NH (BR / meshCol)

// HOW MANY SCORE TILES CAN BE IN FLIGHT. The GEMM runs QK(j+2) before PV(j) (see THE
// DISPATCH ORDER), so S(j+2) is written while the softmax still holds S(j) and S(j+1):
// three score buffers, NH*S_PITCH = 32,896 B of scratchpad each. P8 and corr rotate
// through three slots for the same reason -- tile j+3 is the first to reuse tile j's.
#define NSCORE 3
#define NP8 3

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
// B's TWO strides move too, and not because of M or K. QK reads Q^T blocks n-major, block
// (k, n) at (n*K + k)*64: strides {64, K*64}. PV reads P^T, which the epilogue writes
// n-major as well, one N half per task, block (k, n) at n*P_PITCH + k*64: strides {64,
// P_PITCH}. Both walk k innermost in 64 B steps; only the n stride differs.
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

// The Br FP16 factors, two per word: exactly the byte layout of the NH contiguous corr
// beats the softmax writes (half 0's queries, then half 1's), so they are copied verbatim. Unrolled by hand so each CSR address is
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
// P_PITCH = 64 mod 256 apart, so half 0 and half 1 meet V at P8_V_PHASE and P8_V_PHASE - 64.
#ifndef P8_V_PHASE
#define P8_V_PHASE 128u
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
    // RECLAIM THE UNREAD OPERAND SPACE. data.h reserves four GEMM operand regions
    // below the arena, and two of them are never read on this kernel:
    //
    //   local_a  [0, 64 KiB)        K tile 0            -- used
    //   local_b  [64 KiB, +4 KiB)   Q                   -- used
    //   local_c  [68 KiB, +64 KiB)  C of QK             -- MASKED on every dispatch
    //   local_d32[132 KiB, +32 KiB) D of QK             -- D goes to the score planes instead
    //
    // QK passes a C-enable mask of 0 (gemm_set_shape's last argument), and a disabled
    // channel presents zero without issuing a memory request, so base_c is never
    // dereferenced; the D32 writer is pointed at the score buffers in the arena. That
    // leaves 96 KiB reserved and untouched. Starting the arena at local_c hands it
    // back, which is what makes a third score buffer fit at all.
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
    // half there, so the tile never has to be copied to sit behind one. mold is the running maximum
    // the tile starts from, put there by the previous tile's last task; the rowmax reads
    // it as each plane's last beat, so ONE lanewise MAX of 1 + Bc beats per plane is
    // max(m_old, rowmax) -- the new running maximum, with no separate combine.
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
    // PV(j) on the GEMM core, which copies it into the C column scaler. The GEMM reads it
    // only after the softmax publishes tile j -- by which time the SIMD may already be on
    // tiles j+1 and j+2 and computing their corr. So corr rotates through NP8 slots like
    // P8 does: tile j writes slot j % NP8, and tile j+3 cannot start until QK(j+3), which the
    // GEMM issues only after launching PV(j) (see THE DISPATCH ORDER).
    //
    // m lives in the planes' latch beats (the rowmax writes it there every tile) and l is
    // rewritten in place by the l update, so nothing has to copy a "new" value over an
    // "old" one: m_old lives in the planes' mold beats, and l_old is consumed into
    // corr*l_old before l is rewritten. m_st holds the -inf seed, then the final m.
    uint8_t *m_st  = l1 + top;  top += NH * BEAT;       // m: seeded -inf by the xDMA, final m
    uint8_t *corrS = l1 + top;  top += NP8 * NH * BEAT; // corr, slot j % NP8
    uint8_t *l_st  = l1 + top;  top += NH * BEAT;       // l, seeded 0 by the preinit
#define CORR_OF(j) (l1u + (uint32_t)(corrS - l1) + ((j) % NP8) * NH * SIMD_BEAT_BYTES)
    // No FP16 P buffer: the epilogue quantises the tile in the sweep that produces it,
    // so P exists only as INT8, inside the P8 buffers below. The tapped rowsum and
    // corr*l_old live there too -- see the P8 allocation.
    uint32_t oacc32 = top;      top += BR * DHEAD * 4; // O^T = [d, Br], INT32, the only O
    // P8 CARRIES ITS OWN TAIL. Each epilogue task writes one half's B blocks and then its
    // tapped rowsum beat in one flat stream (Fp16ToInt8 tailPassthrough lets the reduce's
    // trailing FP16 beat past the quantiser), and corr*l_old gets the beat after it, so
    // the l update reads each half's pair [rowsum][corr*l_old] at a one-beat stride:
    //
    //   half n:  [ B block (0, n) ] .. [ B block (Bc/4-1, n) ][ rowsum_n ][ lsc_n ]
    //
    // NP8 P8 buffers rotate, and their tails with them: tile j writes buffer j % NP8, which
    // PV(j) reads, and tile j+3 is the first to write it again -- after QK(j+3), which the
    // GEMM dispatches only after PV(j) (see THE DISPATCH ORDER). See "P8" above for the
    // bank phase this layout gives PV.
    top = (top + 255u) & ~255u;
    uint32_t p8_buf[NP8];
    for (uint32_t i = 0; i < NP8; i++) { p8_buf[i] = top; top += P8_BUF; }
    top += P8_V_PHASE;
#define RSUM_OFF ((uint32_t)BC * meshCol)          // a half's rowsum beat
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
    uint32_t k_buf[2], v_buf[2];
    k_buf[0] = (uint32_t)delta_local_a;   // the staged buffer becomes buffer 0
    // The K and V tiles are whole multiples of 256 B -- one full 32-bank rotation -- so
    // allocating one never shifts the bank the next one starts on, and both V buffers meet
    // both P8 buffers at P8_V_PHASE.
    k_buf[1] = top; top += KVBYTES;
    v_buf[0] = top; top += KVBYTES;
    v_buf[1] = top; top += KVBYTES;

    volatile uint32_t *pub  = (volatile uint32_t *)(l1 + top); top += 1024;
    volatile uint32_t *sync = (volatile uint32_t *)(l1 + top); top += 64;

    // THE TWO SHAPES THAT SEED THE SIMD's AGU LIVE IN THE ARENA, not in a .l1 static. TCDM
    // is SRAM with no reset: an address that has never been written reads X in RTL, and .l1
    // is NOLOAD, so a linker-placed array there starts UNDEFINED rather than zero.
    // Allocating them here puts them inside the block the xDMA zeroes below.
    snax_simd_shape_t *shapes = (snax_simd_shape_t *)(l1 + top);
    top += (2u * sizeof(snax_simd_shape_t) + 63u) & ~63u;

    // WARM-UP SCRATCH. The softmax loop runs one pass over a four-key dummy tile here
    // before S(0) exists -- same instructions, same CSR addresses, same operator
    // sequence, 1/128th of the data -- so that the first real tile does not pay the
    // cold-fetch cost on the critical path. See the loop. Its own planes, P8 halves and
    // state vectors, so no warm write lands on live state:
#define W_KEYS 4u                                   // keys per warm plane: one 4-key group
#define WS_P   (NH * (W_KEYS + 2u))                 // beats: the warm P8 halves
#define WS_C   (WS_P + NH * (W_KEYS / 2u + 2u))     //        the warm corr, l

#define WS_L   (WS_C + NH)
    const uint32_t wscr = l1u + top;  top += (WS_L + NH) * BEAT;

    // THE GEMM DESCRIPTOR, STAGED OUT OF DRAM BEFORE THE CLOCK STARTS.
    // data.h's shape words are ordinary globals, and .bss/.rodata map to MAIN MEMORY on
    // this target: each read is an L3 round trip, and this core issues one outstanding
    // integer load at a time, so thirteen of them are thirteen serialised round trips.
    // Staging the dispatch needs all thirteen, and staging is the first thing the GEMM
    // lane does after the origin barrier -- so read here, outside the measured window,
    // where they cost nothing, and from TCDM inside it, where they cost a few cycles.
    volatile uint32_t *gdesc = (volatile uint32_t *)(l1 + top);  top += 64u;

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
        printf("  score planes    %u x %u B, P8 halves %u x %u B, V at bank phase %u vs P8\n",
               (unsigned)NH, (unsigned)S_PITCH, (unsigned)NH, (unsigned)P_PITCH,
               (unsigned)P8_V_PHASE);
        snax_tcdm_policy_install(FA_TCDM_PRIO, FA_TCDM_GUARD);
        printf("  TCDM arbitration %s (control 0x%x)\n", snax_tcdm_policy_name(FA_TCDM_PRIO),
               snax_tcdm_arb_ctrl_read());
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
        // no error. The xDMA moves a V tile, or half of K tile 0, per transfer.
        if (KVBYTES >= XDMA_TASK_SPAN) {
            printf("  xDMA SPAN: a V tile is %lu bytes, one transfer spans under %lu\n",
                   (unsigned long)KVBYTES, (unsigned long)XDMA_TASK_SPAN);
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
        snax_simd_shape_flat(&shapes[0], (void *)wscr, 1);
        snax_simd_shape_flat(&shapes[1], (void *)wscr, 1);
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
    // m starts at -inf, one 64 B vector -- exactly one xDMA beat -- per query tile. It is
    // a constant, and the xDMA's writer
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
        pub[94] = snrt_mcycle() - t0;

        uint32_t x0 = snrt_mcycle() - t0;
        xdma_fill_arm((uint32_t)m_st, XDMA_WIDTH, (BR * 2u) / XDMA_WIDTH, 0xFBFFFBFFu);
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
        xdma_tile_arm((uint32_t)(l1 + v_buf[0]), (uint32_t)(uintptr_t)V, XDMA_WIDTH,
                      KVBYTES / XDMA_WIDTH);
        for (uint32_t j = 0; j < NKV; j++) {
            uint32_t w0 = snrt_mcycle() - t0;
            sync[11] = j * 10u + 1u;  // waiting: V buffer free
            if (j >= 2) SNAX_SPIN_UNTIL(sync[3] >= j - 1, timeouts);  // V buffer free
            sync[11] = j * 10u + 2u;  // transferring V(j)
            uint32_t v0 = snrt_mcycle() - t0;
            xdma_block += v0 - w0;
            xdma_tile_retask((uint32_t)(l1 + v_buf[j & 1]),
                             (uint32_t)(uintptr_t)V + j * KVBYTES, KVBYTES / XDMA_WIDTH);
            // A V tile is read from main memory, which the PEER endpoint serves, so
            // the hardware runs it as a REMOTE task. BOUNDED like every other wait in
            // this kernel: one unbounded wait anywhere means a deadlock elsewhere
            // presents as a run that never ends instead of one that reports.
            {
                snax_xdma_task_t tk = snax_xdma_start_task();
                uint32_t fp = tk.remote ? XDMA_FINISH_REMOTE_TASK_PTR
                                        : XDMA_FINISH_LOCAL_TASK_PTR;
                uint32_t spins = 0;
                while (!snax_xdma_task_done(snax_read_xdma_cfg_reg(fp), tk.task_id)) {
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
        // accumulator plus the running-state block [m][corr x2][l]. On one cluster it is a drain
        // tail; KV-sharded across clusters it is the partial each shard contributes to the
        // cross-cluster reduce, so it is part of the measured pipeline.
        // That reduce is an ElementwiseJunction gather armed on a transfer like this one: O folds
        // as an INT32 ADD on oacc32 as it lies, once every shard's softmax has been seeded with
        // the shared maximum. See "SHARDING THE KEYS ACROSS CLUSTERS" at the top of this file.
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
                          (void *)m_st, (uint32_t)(l_st - m_st) + NH * BEAT);
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
        // The descriptor, hoisted out of DRAM once. Everything below is a local from
        // here on, so a dispatch is pure `csrw imm` and never touches .bss again.
        const int32_t g_n = (int32_t)gdesc[2];
        const int32_t s1_k = (int32_t)gdesc[0], s1_m = (int32_t)gdesc[1];
        const int32_t s1_blk = s1_m * g_n;
        const int32_t s1_as2 = (int32_t)gdesc[3], s1_bs1 = (int32_t)gdesc[4];
        const int32_t s2_k = (int32_t)gdesc[5], s2_m = (int32_t)gdesc[6];
        const int32_t s2_blk = s2_m * g_n;
        const int32_t s2_as2 = (int32_t)gdesc[7];
        // B strides per shape, see gemm_set_shape(): QK's Q^T n-major as data.h emits it,
        // PV's P^T n-major too, k one 64 B block, n one P8 half (P_PITCH) -- the order the
        // per-half epilogue writes it in.
        const int32_t s1_bs0 = (int32_t)gdesc[13];
        const int32_t s2_bs0 = (int32_t)gdesc[14], s2_bs1 = (int32_t)P_PITCH;
        const uint32_t base_b = l1u + gdesc[9];
        // The arena owns the address range data.h names delta_local_c. QK masks C, so
        // this address is never dereferenced -- point it somewhere real anyway rather
        // than at memory that belongs to a score buffer.
        const uint32_t base_c = l1u + oacc32;
        const uint32_t base_o = l1u + oacc32;
        // D's two walks (gemm_d32_emit_fp16): INT32 O^T as data.h emits it, FP16 S^T planes
        // from data.h's compile-time D16_* constants.
        const uint32_t d32_b0_i = gdesc[10], d32_s1_i = gdesc[11], d32_s2_i = gdesc[12];
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
        // QK's D32 base is plane 0's first key row, one beat past its latch.
#define STAGE_QK(t)                                                            \
        do {                                                                   \
            gemm_set_shape(s1_k, s1_m, s1_blk, s1_as2, s1_bs0, s1_bs1, 0u);    \
            gemm_set_bases(l1u + k_buf[(t) & 1], base_b, base_c,               \
                           l1u + s_buf[(t) % NSCORE] + BEAT);                  \
            gemm_d32_emit_fp16(1, D16_TLBOUND0, D16_TLSTRIDE1, D16_TLSTRIDE2); \
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
            gemm_set_bases(l1u + v_buf[(t) & 1], l1u + p8_buf[(t) % NP8],       \
                           base_o, base_o);                                    \
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

        // ---- THE DISPATCH ORDER: QK RUNS TWO TILES AHEAD OF PV ------------------
        //
        //     QK(0) QK(1) | QK(2) PV(0) | QK(3) PV(1) | ... | PV(NKV-2) PV(NKV-1)
        //
        // PV(j) waits for the softmax of tile j, which starts only once QK(j) has written
        // S(j). Issued right after QK(j+1), PV(j) would wait out whatever of SM(j) is longer
        // than QK(j+1) -- the head of the pipeline, where nothing else can fill the gap,
        // and the tail, where SM(NKV-1) overlaps only PV(NKV-2). Putting QK(j+2) in between
        // gives every softmax two dispatches of cover instead of one, so the array runs
        // back to back from QK(0) on. What it costs is buffering: QK(j+2) writes a third
        // score buffer while SM(j) and SM(j+1) still hold theirs (NSCORE = 3), and P8 and
        // corr keep three slots each, because SM(j+3) is the first to reuse tile j's, and
        // QK(j+3) -- which SM(j+3) waits for -- is dispatched only after PV(j).
        //
        // The schedule is built once rather than derived in the loop.
        uint8_t sk[2 * NKV], si[2 * NKV];   // kind (0 QK, 1 PV) and tile of dispatch d
        {
            uint32_t d = 0;
            for (uint32_t t = 0; t < 2u && t < (uint32_t)NKV; t++) { sk[d] = 0; si[d++] = (uint8_t)t; }
            for (uint32_t j = 0; j < (uint32_t)NKV; j++) {
                if (j + 2u < (uint32_t)NKV) { sk[d] = 0; si[d++] = (uint8_t)(j + 2u); }
                sk[d] = 1; si[d++] = (uint8_t)j;
            }
        }
        // What dispatch d waits for. A QK needs its K tile and a free score buffer (the one
        // tile t - NSCORE used, released when that tile's softmax publishes); a PV needs
        // its tile's softmax published and its V tile.
#define DISPATCH_WAIT(kind, t)                                                 \
        do {                                                                   \
            if ((kind) == 0u) {                                                \
                sync[9] = (t) * 10u + 1u;   /* waiting: score buffer free */   \
                if ((t) >= NSCORE)                                             \
                    SNAX_SPIN_UNTIL(sync[1] >= (t) - (NSCORE - 1), timeouts);  \
                sync[9] = (t) * 10u + 2u;   /* waiting: K(t) landed */         \
                SNAX_SPIN_UNTIL(sync[2] >= (t) + 1, timeouts);                 \
            } else {                                                           \
                sync[9] = (t) * 10u + 5u;   /* waiting: softmax t published */ \
                SNAX_SPIN_UNTIL(sync[1] >= (t) + 1, timeouts);                 \
                sync[9] = (t) * 10u + 6u;   /* waiting: V(t) landed */         \
                SNAX_SPIN_UNTIL(sync[5] >= (t) + 1, timeouts);                 \
            }                                                                  \
        } while (0)
        // What the retirement of dispatch d releases.
#define DISPATCH_DONE(kind, t)                                                 \
        do {                                                                   \
            if ((kind) == 0u) {                                                \
                sync[0] = (t) + 1;   /* S(t) is published */                   \
                sync[4] = (t) + 1;   /* K(t) is dead -- the iDMA may reuse it */ \
            } else {                                                           \
                sync[3] = (t) + 1;   /* V(t) is dead */                        \
            }                                                                  \
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
                    DISPATCH_WAIT(kind, t);
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
                DISPATCH_DONE(kind, t);
            }
            have_prev = (d < 2u * (uint32_t)NKV);
            prev_d = d;
        }
#else
        // Dispatch 0 has no predecessor to hide behind. Writing it here puts its
        // cold instruction fetches inside the wait for K(0), which is dead time.
        STAGE_QK(0u);
        pub[91] = snrt_mcycle() - g0;   // dispatch 0 staged: the lane is ready to fire

        for (uint32_t d = 0; d < 2u * (uint32_t)NKV; d++) {
            const uint32_t kind = sk[d], t = si[d];
            {
                uint32_t w0 = snrt_mcycle();
                DISPATCH_WAIT(kind, t);
                gemm_stall += snrt_mcycle() - w0;
            }
            pub[(kind == 0u ? 10u : 12u) + 4u * t] = snrt_mcycle() - g0;
            sync[9] = t * 10u + (kind == 0u ? 3u : 7u);
            if (kind == 1u) PV_FACTORS(t);
            const uint32_t tid = ++gemm_seq;
            gemm_launch();   // staged one dispatch ago
            gemm_ack();
            // Stage the successor while this one runs: the bank was snapshotted at START.
            if (d + 1u < 2u * (uint32_t)NKV) {
                if (sk[d + 1u] == 0u) STAGE_QK(si[d + 1u]); else STAGE_PV(si[d + 1u]);
            }
            pub[64 + 2 * t + kind] = snrt_mcycle() - g0;  // successor staged, array running
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
        }
#endif  /* GEMM_QUEUE */
#undef DISPATCH_WAIT
#undef DISPATCH_DONE
        // The last PV left the scaler armed in the bank. Nothing else runs on this
        // streamer here, but the contract is that no one ever inherits an enable.
        gemm_colscale_arm(0u, (uint32_t)g_n);
        gemm_wall = snrt_mcycle() - g0;
#undef STAGE_QK
#undef STAGE_PV
    }

    if (snax_is_simd_core()) {
        uint32_t s0 = t_org;
        uint32_t simd_busy0 = snax_simd_busy_cycles();
        pub[92] = snrt_mcycle() - s0;   // when this lane reached its own code
        const uint32_t m_u = (uint32_t)m_st, l_u = (uint32_t)l_st;

        // ONE WARM ITERATION, THEN THE NKV REAL ONES.
        //
        // The first pass through this instruction sequence and its constant pool costs
        // several hundred cycles of instruction fetch that later passes do not pay, and
        // that cost is on the critical path: the array finishes QK(1) and then idles
        // until the softmax publishes tile 0, so the whole pipeline behind it starts
        // late by however long the first tile takes.
        //
        // The SIMD core has nothing to do until S(0) lands, so part of that idle window
        // runs the tile body once over a four-key dummy tile in `wscr`. It must be the
        // SAME code, not a copy -- an inlined helper called twice warms the wrong lines
        // -- so the warm pass is iteration -1 of this loop, and everything it varies
        // (the bases, the key count, the reduce and quantiser counts) is a VARIABLE the
        // real iterations set to the real geometry.
        //
        // Every warm task reads and writes wscr except the last, which is real: it puts
        // the -inf seed into tile 0's mold beats (task 6 below, for "tile -1").
        for (uint32_t jw = 0; jw < NKV + 1u; jw++) {
            const uint32_t warm = (jw == 0u);
            const uint32_t j    = warm ? 0u : jw - 1u;
            uint32_t w0 = snrt_mcycle();
            sync[10] = j * 10u + 1u;  // waiting: QK(j) has produced S(j)
            if (!warm) SNAX_SPIN_UNTIL(sync[0] >= j + 1, timeouts);
            sync[10] = j * 10u + 2u;  // running the softmax chain
            simd_stall += snrt_mcycle() - w0;

            pub[26 + 2 * j] = snrt_mcycle() - s0;
            pub[72 + j] = snax_simd_busy_cycles() - simd_busy0;

            // The geometry of THIS pass. `nb` is the only thing the warm pass shrinks; it
            // has to be a multiple of 4, because the interleaving quantiser emits only
            // whole 4-beat groups (and asserts that tailPeriod is one).
            const uint32_t nb = warm ? W_KEYS : (uint32_t)BC;     // key beats per plane
            const uint32_t sp = (nb + 2u) * BEAT;                // plane pitch (S_PITCH)
            const uint32_t pp = nb * (BEAT / 2u) + 2u * BEAT;    // P8 half pitch (P_PITCH)
            const uint32_t ro = nb * (BEAT / 2u);                // half -> its rowsum beat
            const uint32_t sb = warm ? wscr : l1u + s_buf[j % NSCORE];   // plane 0's latch
            const uint32_t pb = warm ? wscr + WS_P * BEAT : l1u + p8_buf[j % NP8];
            const uint32_t cq = warm ? wscr + WS_C * BEAT : CORR_OF(j);
            const uint32_t lq = warm ? wscr + WS_L * BEAT : l_u;

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
            // The chain's fixed order [EW0, Map, Reduce, EW1, Fp16ToInt8] decides which
            // pairs of operations can share a task. This pair can, because the combine
            // comes BEFORE the transform: EW0 in PAIRS mode takes each [B][x] pair to x - B
            // in one beat and Map exponentiates it in flight, so the difference is never written
            // to memory and never read back. The temperature goes HERE as well as into P:
            // corr must be the ratio of the two P scales it bridges, exp(a*m_old) /
            // exp(a*m_new). corr lands in this tile's corr slot, where the MUL of task
            // 4 and the GEMM's PV(j) both find it.
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
            // meshCol queries of one plane. EW0 is upstream of Map, so the per-lane subtract
            // happens before the exponential and S - m_new is never written out; sticky-B
            // suppresses the seed beat, so EW0 emits exactly the Bc key beats. Fp16ToInt8
            // joins the SAME pass because its tail passthrough leaves the reduce's trailing
            // FP16 beat alone. Without it, narrowing would need its own task reading the
            // whole tile back out of TCDM, and an FP16 P buffer to read it from. The map
            // keeps task 2's a' and EXP.
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
            // INTERLEAVE bit makes each group of 4 key beats come out as the two halves of
            // B block (k = g, n = h), so the writer's flat stream is half h's blocks in k
            // order: same pass, same rate, no relayout.
            snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 0, 0x42FE0000u);  // 127.0f
            snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR + 1,
                                    SIMD_QUANT_TAIL(nb) | FA_QUANT_ILV);
            for (uint32_t h = 0; h < NH; h++)
                simd_task(sb + h * sp, nb + 1u, BEAT, 1u, 0u, pb + h * pp, nb / 2u + 1u, BEAT);

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

            // 6  the NEXT tile's m_old: m into the mold beats of the buffer tile jw scores
            //    into -- a plain copy, no operator armed. That buffer's previous reader
            //    (tile jw - NSCORE) is done, and QK(jw) writes only its key rows, so the
            //    copy may land while the GEMM fills the planes around it. The real tiles
            //    copy their latches; after the last one the copy lands in m_st, the final
            //    m. In the warm pass it is tile 0's -inf seed out of m_st, so it waits for
            //    the xDMA's fill.
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0u);
            if (warm) {
                SNAX_SPIN_UNTIL(sync[7] >= 1, timeouts);
                simd_task(m_u, NH, BEAT, 1u, 0u, l1u + s_buf[0] + S_MOLD, NH, S_PITCH);
            } else if (jw < (uint32_t)NKV) {
                simd_task(sb, NH, sp, 1u, 0u, l1u + s_buf[jw % NSCORE] + S_MOLD, NH, S_PITCH);
            } else {
                simd_task(sb, NH, sp, 1u, 0u, m_u, NH, BEAT);
            }
            // The tile's ISSUE ends here, not at the publish above: tasks 4 to 6 are fired
            // afterwards. Bracketing to the publish would attribute their engine time to an
            // interval that does not contain them.
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
        volatile int8_t *p8t = (volatile int8_t *)(l1 + p8_buf[(NKV - 1) % NP8]);
        // The last tile's rowsum: one beat behind each half's B blocks.
        volatile uint8_t *rsum0 = l1 + p8_buf[(NKV - 1) % NP8] + RSUM_OFF;
#define RSUM_AT(i) fp16_at(rsum0 + ((i) / meshCol) * P_PITCH, (i) % meshCol)
        for (uint32_t i = 0; i < BR; i++) {   // i = query row: lane i % 16 of half i / 16
            // The count, not just the presence: a dropped beat, a tail written into the
            // wrong slot or a shifted write all move it.
            int ones = 0;
            for (uint32_t j = 0; j < BC; j++)
                if (p8t[P8_OFF(j, i)] == 127) ones++;
            int16_t want_ones = p8max_golden[i];
            // A query whose global max came from an EARLIER tile has no 127 here, so the
            // golden's count (which knows where the max is) is the reference.
            if (want_ones >= 0 && ones != want_ones) {
                printf("query %2u: P8^T has %d x 127, expected %d  m=%04x rowsum=%04x l=%04x\n",
                       i, ones, (int)want_ones, fp16_at(m_st, i), RSUM_AT(i),
                       fp16_at(l_st, i));
                err++;
            }
        }

        // ---- against the numerical golden -----------------------------------
        // m against the true maximum over all NKV*Bc keys, the last tile's row sum, l
        // after the whole recurrence, and P8 itself on one key in PGOLD_STRIDE.
        uint32_t worst_m = 0, worst_sum = 0, worst_l = 0;
        for (uint32_t i = 0; i < BR; i++) {
            uint32_t u = fp16_ulp(fp16_at(m_st, i), m_golden[i]);
            if (u > worst_m) worst_m = u;
            u = fp16_ulp(RSUM_AT(i), rowsum_golden[i]);
            if (u > worst_sum) worst_sum = u;
            u = fp16_ulp(fp16_at(l_st, i), l_golden[i]);
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
        printf("    init           %u B of m = -inf in %u cc  [%u -> %u]\n",
               (unsigned)(BR * 2), pub[59] - pub[58], pub[58], pub[59]);
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
