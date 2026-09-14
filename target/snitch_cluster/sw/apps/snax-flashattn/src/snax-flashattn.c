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
//     O    = corr * O + P . V_j     [Br, d]    SIMD rescale + GEMM
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
// ONE KV TILE. Br = 32 query rows, Bc = 512 keys, d = 128, NKV = 4 tiles. Bc and d are
// INDEPENDENT, so the two matmuls have different shapes; gemm_set_shape() switches between
// them per dispatch, rewriting only the twelve CSRs that depend on M and K. A beat is
// 512 bits = 64 B = 32 FP16 lanes = one value per query row.
//
//   GEMM (hart 0)  S^T = K.Q^T          K [512,128] i8   64 KiB  ]  2.10 M MAC
//                                     Q^T [128, 32] i8    4 KiB  ]  2048 cycles
//                                  -> S^T [512, 32] f16  32 KiB     at 1024 MAC/cycle
//                     the mesh computes INT32, and Int32ToFp16 on the D32 writer port
//                     converts it FREE as the tile drains -- which also HALVES the
//                     beats written, so S^T reaches TCDM as 512 FP16 beats, not 1024
//   SIMD (hart 1)  m   = rowmax(S^T)  512 beats -> 513  StreamReduce MAX|LANEWISE|TAP
//                  -m  = negate m       2 beats -> 2    StreamMap LINEAR, a = -1
//                  corr= exp(m_old-m)   2 beats -> 2    StreamMap EXP
//                  P   = exp(S^T - m) 513 beats -> 513  ) FUSED: EW0 (sticky ADD)
//                  rowsum = sum over keys of P          ) -> Map (EXP) -> Reduce TAP
//                  P8^T= int8(P)      512 beats -> 256  Fp16ToInt8
//                  O  *= corr         129 beats -> 128  StreamElementwise sticky MUL
//                  l, m commit          8 beats -> 6    the online-softmax recurrence
//
//                  rowmax/sum over keys = the per-query-row reduction defined above:
//                  32 lanes in, 32 results out, one per query row, in ONE beat.
//
//                  TAP means the reduce does not swallow its input. The tile streams
//                  straight through to the output and the result beat is APPENDED, so
//                  512 beats in become 513 out. That is why one pass both delivers m
//                  and leaves the tile sitting where the next task needs it -- directly
//                  after the broadcast operand that the sticky-B chain reads first.
//   GEMM (hart 0)  O^T = V^T.P^T      V^T [128,512] i8  64 KiB  ]  2.10 M MAC
//                                     P^T [512, 32] i8  16 KiB  ]  2048 cycles
//                                  -> O^T [128, 32] i32 16 KiB     accumulated in place
//
// Per KV tile that is 4.19 M MAC and 4096 GEMM cycles against ~1680 beats read and ~1420
// written on the SIMD side, so the arithmetic floor is GEMM-bound and the softmax has to
// fit inside its shadow. The whole run is NKV of these. Every buffer above lives in TCDM
// at once; the footprint guard in main() reports the total against the 512 kB budget.
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
#include "snax-gemmx-lib.h"
#include "snax-gemmx-params.h"
#include "snax-simd-lib.h"
#include "snrt.h"

// BR, BC, DHEAD, NKV and QSHIFT arrive from data.h. They are derived there from the same
// M/N/K and mesh that produced the streamer descriptors, so the kernel's idea of the tile
// and the descriptors' idea of it cannot drift apart. Change them in data/params.hjson.

// BC and DHEAD are independent. Bc is the tiling knob, free to grow until TCDM is
// full; d is a property of the model. The two matmuls therefore have different
// shapes, which gemm_set_shape() switches between per dispatch.

#define SBEATS BC        // S^T / P beats: one per KEY, 32 query lanes each
#define DBEATS DHEAD     // O^T beats:     one per HEAD element, 32 query lanes each
#define PBEATS (BC / 2)  // after Fp16ToInt8 halves them

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

// The FULL streamer programming: bounds, strides, remap indices and the accelerator CSRs.
// Called ONCE, for shape 1. Everything that differs between the two matmul shapes is then
// patched per dispatch by gemm_set_shape() -- twelve registers instead of these ~84.
//
// Not for per-dispatch use: it costs ~2,150 cycles a call, because the library function
// is out of line and every CSR address inside it is opaque to the csrw_ss switch, so each
// access pays a jump-table load plus an indirect jump.
//
// The fifteen streamer arrays live in THIS frame rather than main's: SNRT_LOG2_STACK_SIZE
// is 10, so a hart has 1 KiB, and carrying these alongside main's shape structs and a
// printf frame overflows into the neighbouring hart's stack.
static void gemm_configure_once(void) {
    int32_t Aslstride[] = {Aslstride0};
    int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2, Atlbound3, Atlbound4, Atlbound5};
    int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2, Atlstride3, Atlstride4, Atlstride5};
    int32_t Bslstride[] = {Bslstride0};
    int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
    int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};
    int32_t D8slstride[] = {D8slstride0};
    int32_t D8tlbound[] = {D8tlbound0, D8tlbound1, D8tlbound2, D8tlbound3};
    int32_t D8tlstride[] = {D8tlstride0, D8tlstride1, D8tlstride2, D8tlstride3};
    // TWO spatial strides: the C port declares spatial_bounds [[8, 4]] and the streamer
    // reads S_STRIDE_NUM_READER_WRITER_0 = 2 of them. A 1-element array would feed the
    // second from off the end of the stack.
    int32_t Cslstride[] = {Cslstride0, Cslstride1};
    int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2, Ctlbound3};
    int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};
    int32_t D32slstride[] = {D32slstride0, D32slstride1};
    int32_t D32tlbound[] = {D32tlbound0, D32tlbound1, D32tlbound2, D32tlbound3};
    int32_t D32tlstride[] = {D32tlstride0, D32tlstride1, D32tlstride2, D32tlstride3};

    set_gemmx_streamer_csr(
        Aslstride, Atlbound, Atlstride, set_addr_remap_index_A,
        Bslstride, Btlbound, Btlstride, set_addr_remap_index_B,
        D8slstride, D8tlbound, D8tlstride, set_addr_remap_index_D8,
        Cslstride, Ctlbound, Ctlstride, set_addr_remap_index_C,
        D32slstride, D32tlbound, D32tlstride, set_addr_remap_index_D32,
        delta_local_a, delta_local_b, delta_local_d8, delta_local_c,
        delta_local_d32, bypassSIMD, transposed_A, transposed_B,
        channel_en_C, broadcast_C);
    uint32_t sub = gen_subtraction_config(0, 0);  // no zero-point in attention
    uint32_t csr0 = gen_csr0_config(input_zp_i, output_zp_i, max_int_i, min_int_i);
    uint32_t csr1 = gen_csr1_config(double_round_i);
    set_gemmx_csr(K, N, M, sub, csr0, csr1, shared_bitpacked_shift,
                  shared_multiplier, M * N, bypassSIMD);
}

// Switch the GEMM between the two matmul shapes.
//
//   S^T = K.Q^T    M1 = Bc/meshRow, K1 = d/tileSize
//   O^T = V^T.P^T  M2 = d/meshRow,  K2 = Bc/tileSize
//
// Only M and K move, so only the CSRs that depend on them are rewritten: nine stream
// registers plus three GEMM bounds, against the ~84 a full set_gemmx_streamer_csr()
// writes. Every address here is a compile-time constant, so each store folds through the
// always_inline csrw_ss to a single `csrw <imm>`.
#define GEMM_SHAPE_S1 0
#define GEMM_SHAPE_S2 1
static inline void gemm_set_shape(int shape) {
    const int32_t m  = shape ? S2_M : M;
    const int32_t k  = shape ? S2_K : K;
    const int32_t as2 = shape ? S2_Atlstride2 : Atlstride2;   // A: outer stride = K * row bytes
    const int32_t bs1 = shape ? S2_Btlstride1 : Btlstride1;   // B: mid   stride = K * col bytes

    csrw_ss(T_BOUND_K, k);                 // GEMM contraction depth
    csrw_ss(T_BOUND_M, m);                 // GEMM output rows      (N is identical in both)
    csrw_ss(TEMPORAL_LOOP_BOUND, m * N);   // retire count = M*N output blocks

    csrw_ss(T_BOUND_READER_0_0, k);        // A stream
    csrw_ss(T_BOUND_READER_0_2, m);
    csrw_ss(T_STRIDE_READER_0_2, as2);
    csrw_ss(T_BOUND_READER_1_0, k);        // B stream
    csrw_ss(T_BOUND_READER_1_2, m);
    csrw_ss(T_STRIDE_READER_1_1, bs1);
    csrw_ss(T_BOUND_READER_WRITER_0_2, m); // C stream
    csrw_ss(T_BOUND_READER_WRITER_1_2, m); // D32 stream
    csrw_ss(T_BOUND_WRITER_0_2, m);        // D8 (unused at bypassSIMD, kept consistent)
}

// The GEMM's D32 output port carries an Int32ToFp16Converter. Arm it for a matmul whose
// result is CONSUMED as floating point; disarm it for one that ACCUMULATES in place, since
// such a matmul reads its own previous output back through C as INT32.
//
// Arming also halves the beats the writer emits -- two INT32 beats merge into one FP16 beat
// -- so the D32 descriptor must halve with it, or the writer waits for beats that never
// arrive. Both layouts are contiguous, so halving the innermost bound and the two outer
// strides is the whole change.
//
// The descriptor values come in as ARGUMENTS rather than read from the D32tl* globals.
// Those globals live in .bss, which this target maps to DRAM, so reading them here would
// cost an L3 round trip per field per dispatch. The caller hoists them into locals once,
// where they stay in registers.
static inline void gemmx_d32_emit_fp16(int on, uint32_t bound0, uint32_t stride1,
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

    uint8_t *negmS = l1 + top;  top += BEAT;          // -m_new   ] latch for P = exp(S-m)
    uint8_t *s16   = l1 + top;  top += SBEATS * BEAT; // S^T      ]
    uint8_t *rmax  = l1 + top;  top += BEAT;          // tapped rowmax, and later -m_new
    uint8_t *mrun  = l1 + top;  top += BEAT;          // running m   ] pair for max(m_old,rowmax)
    uint8_t *mnew  = l1 + top;  top += BEAT;
    uint8_t *delta = l1 + top;  top += BEAT;          // m_old - m_new
    uint8_t *corrL = l1 + top;  top += BEAT;          // exp(delta)  ] latch for corr*l_old
    uint8_t *lrun  = l1 + top;  top += BEAT;          // running l   ]
    uint8_t *lnew  = l1 + top;  top += BEAT;
    uint8_t *p16   = l1 + top;  top += SBEATS * BEAT; // P = exp(S-m)
    // The tapped rowsum lands immediately after p16 and corr*l_old immediately after IT,
    // so task 13 reads the pair [rowsum][corr*l_old] directly. The adjacency a LANEWISE
    // 2-beat reduce needs comes from the ALLOCATION, not from moving a beat at run time.
    uint8_t *rsum  = l1 + top;  top += BEAT;          // tapped rowsum ] pair for l_new
    uint8_t *lsc   = l1 + top;  top += BEAT;          // corr*l_old    ]
    uint8_t *corrO = l1 + top;  top += BEAT;          // exp(delta)  ] latch for O *= corr
    uint8_t *oacc  = l1 + top;  top += DBEATS * BEAT; // O^T = [d, Br] ]
    uint32_t oacc32 = top;      top += BR * DHEAD * 4; // O^T as the GEMM sees it: INT32
    uint32_t d32_b = top;       top += BR * BC * 2;   // second S^T buffer (FP16)
    uint32_t p8_0  = top;       top += PBEATS * BEAT;
    uint32_t p8_1  = top;       top += PBEATS * BEAT;
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
    volatile uint32_t *pub  = (volatile uint32_t *)(l1 + top); top += 256;
    volatile uint32_t *sync = (volatile uint32_t *)(l1 + top); top += 64;

    const int32_t d32_delta[2] = {delta_local_d32, (int32_t)d32_b};
    const int32_t p8_delta[2] = {(int32_t)p8_0, (int32_t)p8_1};

    // The handoff. sync[0] counts S tiles the GEMM has finished producing;
    // sync[1] counts softmax tiles the SIMD core has finished consuming and
    // whose P8 is ready. Both only ever increase, so a reader never needs a
    // lock -- it just waits for the count to pass a threshold.

    uint32_t gemm_cycles = 0, simd_cycles = 0, gemm_stream_cycles = 0;
    uint32_t gemm_wall = 0, simd_wall = 0;
    uint32_t c_conv = 0, c_max = 0, c_exp = 0, c_quant = 0;
    uint32_t gemm_stall = 0, simd_stall = 0;  // time each core spent waiting
    int err = 0;      // invariant failures
    int cfg_err = 0;  // tasks the library refused to configure
    int timeouts = 0;

    // ---- stage Q, K, V and the zero bias ------------------------------------
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, M * K * meshRow * tileSize * sizeof(int8_t));
        snrt_dma_start_1d(local_b, B, N * K * tileSize * meshCol * sizeof(int8_t));
        snrt_dma_wait_all();

        // C is the GEMM's accumulator input. The data generator fills it with a random
        // bias, which attention does not have and which would dominate the scores. Zero it.
        for (uint32_t i = 0; i < (uint32_t)(M * N * meshRow * meshCol); i++)
            local_c[i] = 0;

        // A and B arrive ALREADY bounded by >>QSHIFT, so the scores stay inside FP16.
        // datagen.py applies the shift when it writes the data.

        // Online-softmax initial state. m starts at the most negative FINITE FP16
        // (-65504) rather than -inf: max(m, rowmax) is then just rowmax, and
        // exp(m - m_new) underflows to 0 as it should, without inf arithmetic
        // anywhere near the exponential.
        for (uint32_t i = 0; i < SIMD_BEAT_BYTES / 2; i++) {
            ((volatile uint16_t *)mrun)[i] = 0xFBFFu;   // -65504
            ((volatile uint16_t *)lrun)[i] = 0x0000u;   // l = 0
        }
        for (uint32_t i = 0; i < DBEATS * SIMD_BEAT_BYTES / 2; i++)
            ((volatile uint16_t *)oacc)[i] = 0x0000u;   // O = 0 (FP16 half)
        for (uint32_t i = 0; i < (uint32_t)(BR * DHEAD); i++)
            ((volatile int32_t *)(l1 + oacc32))[i] = 0;  // O = 0 (INT32 half)
        // The GEMM's C buffer is the O accumulator input; the rescale writes into it.
        for (uint32_t i = 0; i < (uint32_t)(M * N * meshRow * meshCol); i++)
            local_c[i] = 0;

        sync[0] = 0;
        sync[1] = 0;
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
    // against a rebuilt cluster cfg: the mesh comes from snax-gemmx-params.h and moves
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

        // 1  rowmax in one pass: the tile passes through (tap) and the statistic follows.
        snax_simd_shape_flat(&sh[0], l1 + d32_delta[0], SBEATS);
        snax_simd_shape_flat(&sh[1], s16, SBEATS + 1);
        // 2  m_new = max(m_old, rowmax): LANEWISE over the adjacent pair [rmax][mrun].
        snax_simd_shape_flat(&sh[2], rmax, 2);
        snax_simd_shape_flat(&sh[3], mnew, 1);
        // 3+4  -m_new into both latches in one task: read m_new twice (stride 0) and fan
        //      the negated value out, once before s16 (the latch for P) and once into the
        //      rmax slot, which task 2 has consumed so it now pairs with mrun. One 2-beat
        //      task pays one reader/extension/writer fill+drain where two 1-beat tasks
        //      pay two.
        snax_simd_shape_broadcast(&sh[4], mnew, 2);
        snax_simd_shape_2d(&sh[5], negmS, 2, (uint32_t)(rmax - negmS), 1, 0);
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
        //     read [negmS][S^T x Bc]  ->  EW0: sticky ADD -> Map: EXP -> Reduce: ADD|LANEWISE|TAP
        //     write [P x Bc][rowsum]
        //
        // Fusing them keeps S^T - m_new inside the chain, so it is never written out: a whole
        // read+write of the tile that does not happen.
        snax_simd_shape_flat(&sh[14], negmS, 1 + SBEATS);
        snax_simd_shape_flat(&sh[17], p16, SBEATS + 1);
        // 10 quantise P for the second matmul.
        snax_simd_shape_flat(&sh[18], p16, SBEATS);
        snax_simd_shape_flat(&sh[19], l1 + p8_delta[0], PBEATS);
        // 11 corr * l_old: sticky MUL over [corrL][lrun].
        snax_simd_shape_flat(&sh[20], corrL, 2);
        snax_simd_shape_flat(&sh[21], lsc, 1);   // seed emits nothing, so just the result
        // 13 l_new = corr*l_old + rowsum: LANEWISE ADD over the adjacent pair [rsum][lsc].
        snax_simd_shape_flat(&sh[24], rsum, 2);
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

    // ======================= the pipelined loop ==============================
    if (snax_is_gemm_core()) {
        // Both D32 descriptors, read out of DRAM ONCE. FP16 halves the beats the writer
        // emits, so the innermost bound and the two outer strides halve with it; both
        // layouts stay contiguous, which is why that is the whole difference.
        const uint32_t d32_b0_f = (uint32_t)D32tlbound0 / 2,  d32_b0_i = (uint32_t)D32tlbound0;
        const uint32_t d32_s1_f = (uint32_t)D32tlstride1 / 2, d32_s1_i = (uint32_t)D32tlstride1;
        const uint32_t d32_s2_f = (uint32_t)D32tlstride2 / 2, d32_s2_i = (uint32_t)D32tlstride2;
        uint32_t g0 = snrt_mcycle();
        for (uint32_t j = 0; j <= NKV; j++) {
            if (j < NKV) {
                // S(j) into the buffer the SIMD core is not reading. It last
                // held tile j-2, which is free once tile j-1 has been consumed.
                if (j >= 2) {
                    uint32_t w0 = snrt_mcycle();
                    SNAX_SPIN_UNTIL(sync[1] >= j - 1, timeouts);
                    gemm_stall += snrt_mcycle() - w0;
                }
                // C MUST be restated, not left at -1 ("keep current"): the O matmul below
                // points C at oacc32 for its in-place accumulation, so without this S would
                // be K.Q^T + O_accumulated from the second tile onward.
                gemm_set_shape(GEMM_SHAPE_S1);  // S^T = K.Q^T
                set_gemmx_bases(delta_local_a, delta_local_b, -1, delta_local_c,
                                d32_delta[j & 1]);
                gemmx_d32_emit_fp16(1, d32_b0_f, d32_s1_f, d32_s2_f);  // S consumed as FP16
                set_gemmx_streamer_start();
                set_gemmx_start();
                wait_gemmx_and_streamer();
                gemm_cycles += read_gemmx_perf_counter();
                gemm_stream_cycles += read_gemmx_streamer_perf_counter();
                sync[0] = j + 1;  // publish AFTER the streamer has drained
            }
            if (j > 0) {
                // O(j-1) = P(j-1).V. Waiting here is what lets S(j) above run
                // concurrently with the SIMD core's tile j-1.
                uint32_t w0 = snrt_mcycle();
                SNAX_SPIN_UNTIL(sync[1] >= j, timeouts);
                gemm_stall += snrt_mcycle() - w0;
                // Transposed: O^T = V^T.P^T, so P^T is the B operand (it has
                // exactly B's shape) and V^T stays in A.
                //
                // C AND D32 BOTH POINT AT oacc32, so the matmul computes
                // O += P.V in place -- the accumulation across KV tiles is the
                // GEMM's own C input, costing nothing extra. The first tile
                // needs oacc32 zeroed, which the DM core does before the loop.
                gemm_set_shape(GEMM_SHAPE_S2);  // O^T = V^T.P^T
                set_gemmx_bases(delta_local_a, p8_delta[(j - 1) & 1], -1,
                                (int32_t)oacc32, (int32_t)oacc32);
                gemmx_d32_emit_fp16(0, d32_b0_i, d32_s1_i, d32_s2_i);  // O accumulates in place as INT32
                set_gemmx_streamer_start();
                set_gemmx_start();
                wait_gemmx_and_streamer();
                gemm_cycles += read_gemmx_perf_counter();
                gemm_stream_cycles += read_gemmx_streamer_perf_counter();
            }
        }
        gemm_wall = snrt_mcycle() - g0;
    }

    if (snax_is_simd_core()) {
        uint32_t s0 = snrt_mcycle();
        uint32_t simd_busy0 = snax_simd_busy_cycles();
        for (uint32_t j = 0; j < NKV; j++) {
            uint32_t w0 = snrt_mcycle();
            SNAX_SPIN_UNTIL(sync[0] >= j + 1, timeouts);
            simd_stall += snrt_mcycle() - w0;

            sh[0].base = l1 + d32_delta[j & 1];
            sh[19].base = l1 + p8_delta[j & 1];

            // ---- the online softmax, in full -------------------------------
            // Every task below is one beat of arithmetic except 1, 8, 9, 10 and
            // 14. That ratio is the point: the STATE UPDATE is dominated by
            // per-task start/drain, not by compute.

            // 1  rowmax over the tile the GEMM has already written as FP16 (the tap
            //    appends the per-lane maxima after the tile passes through)
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMREDUCE));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR, SBEATS);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1,
                                    SIMD_RED_MAX | SIMD_RED_LANEWISE | SIMD_RED_TAP);
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
                                        (1u << SIMD_EXT_STREAMREDUCE));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 0, 1);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 1,
                                    SIMD_EW_ADD | SIMD_EW_STICKY_B);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0, SIMD_F32_ONE);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 1, 0);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 2, SIMD_FUNC_EXP);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR, SBEATS);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1,
                                    SIMD_RED_ADD | SIMD_RED_LANEWISE | SIMD_RED_TAP);
            snax_simd_program_1d(&sh[14], &sh[17]);
            snax_simd_fire();

            // 10 quantise P
            snax_simd_use1(SIMD_EXT_FP16TOINT8, SIMD_EXT_FP16TOINT8_CSR, SIMD_F32_ONE);
            snax_simd_program_1d(&sh[18], &sh[19]);
            snax_simd_fire();

            // 14 O *= corr. HOISTED to here, directly after the quantise, because these two
            // are the ONLY tasks the GEMM waits on: p8 is its B operand and the rescaled O is
            // its accumulator. It depends on corr (task 6+7) and nothing later, so nothing
            // stops it running now.
            snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1,
                           SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1,
                           SIMD_EW_MUL | SIMD_EW_STICKY_B);
            snax_simd_program_1d(&sh[26], &sh[27]);
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
        simd_cycles = snax_simd_busy_cycles() - simd_busy0;
        simd_wall = snrt_mcycle() - s0;
    }
    snrt_cluster_hw_barrier();

    // ---- invariants for the FULL algorithm ----------------------------------
    //
    // Three independent checks, each pinning a different part of the online softmax:
    //
    //   P    contains exactly 1.0   per query row   the running max, the subtract AND the exp
    //   P^T  contains exactly 1     per query row   the quantiser
    //   l    == NKV * rowsum(one tile)              the TAPPED rowsum and the l recurrence
    //
    // The third is the sharpest. Every KV tile is fed the same K, so after the first tile
    // the running max stops changing, corr = exp(0) = 1, and l must accumulate exactly NKV
    // identical row sums; a wrong tap, correction or l update makes it drift.
    if (snax_is_simd_core()) {
        volatile int8_t *p8t = (volatile int8_t *)(l1 + p8_delta[(NKV - 1) & 1]);
        for (uint32_t i = 0; i < BR; i++) {   // i = query row = lane
            // exp(S - m) == 1.0 for the maximal key pins the running max, the per-lane
            // subtract in EW0 and the exponential in Map with a single test.
            int found_zero = 0, found_one = 0;
            for (uint32_t j = 0; j < SBEATS; j++) {
                if (fp16_at(p16 + j * SIMD_BEAT_BYTES, i) == 0x3C00u) found_zero = 1; // exp(0) = 1.0
                if (p8t[j * BR + i] == 1) found_one = 1;
            }
            if (!found_zero || !found_one) {
                printf("query %2u: p16_one=%d p8_one=%d m=%04x rowsum=%04x l=%04x\n", i,
                       found_zero, found_one, fp16_at(mrun, i), fp16_at(rsum, i),
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
        uint32_t worst_m = 0, worst_sum = 0, worst_p = 0;
        for (uint32_t i = 0; i < BR; i++) {
            uint32_t u = fp16_ulp(fp16_at(mrun, i), m_golden[i]);
            if (u > worst_m) worst_m = u;
            u = fp16_ulp(fp16_at(rsum, i), rowsum_golden[i]);
            if (u > worst_sum) worst_sum = u;
        }
        for (uint32_t b = 0; b < PGOLD_NBEATS; b++) {
            volatile uint8_t *beat = p16 + (uint32_t)b * PGOLD_STRIDE * SIMD_BEAT_BYTES;
            for (uint32_t i = 0; i < BR; i++) {
                uint32_t u = fp16_ulp(fp16_at(beat, i), p16_golden[b * BR + i]);
                if (u > worst_p) worst_p = u;
            }
        }
        printf("  golden           m %u ULP  P %u ULP  rowsum %u ULP  (limits %u/%u/%u)\n",
               worst_m, worst_p, worst_sum, GOLD_ULP_M, GOLD_ULP_P, GOLD_ULP_SUM);
        if (worst_m > GOLD_ULP_M || worst_p > GOLD_ULP_P || worst_sum > GOLD_ULP_SUM) {
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
        uint32_t busy = gemm_cycles + simd_total;

        printf("\n=== FlashAttention on the four-engine cluster ===\n");
        printf("  tile             Br=%d Bc=%d d=%d, %d KV tiles, mesh %dx%dx%d\n",
               BR, BC, DHEAD, NKV, meshRow, tileSize, meshCol);
        printf("  GEMM streamer    %5u cycles (operand feed + D32 drain)\n",
               gemm_stream_cycles);
        printf("  pipeline         %5u cycles, %u per KV tile\n", pipeline,
               pipeline / NKV);
        // Per core: what it spent running its engine, blocked on the other
        // core, and writing CSRs. The three must sum to that core's wall.
        printf("  GEMM core        busy %5u (%2u%%)  peer-wait %5u  config %5u\n",
               gemm_cycles, 100u * gemm_cycles / pipeline, gemm_stall,
               gemm_wall > gemm_cycles + gemm_stall
                   ? gemm_wall - gemm_cycles - gemm_stall : 0);
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
