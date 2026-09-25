// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The attention front half of an LLM layer, from the layer input to FlashAttention's
// operands, on the four-engine cluster:
//
//     x [T, d] fp16 --RMSNorm--> quantise --Q/K/V projections--> FA-ready Q, K, V^T (int8)
//
// This is the stretch of HeMAiA's llm_layer_4cluster (main_bingo.py) that ends in a
// "checked dead end": its projections produce D-layout FP16, FlashAttention wants INT8 in
// the array's own operand layouts, and the layer stages FA's operands from the datagen
// instead of feeding them. This app closes that gap, and then removes most of the passes
// the naive fix needs.
//
// ======================================================================================
// WHAT FLASHATTENTION ACTUALLY READS
// ======================================================================================
//
// FlashAttention runs attention TRANSPOSED (snax-flashattn-decode.c), because that is what
// gives its softmax a LANEWISE rowmax:
//
//     S^T = K . Q^T        A = K    [Bc, d]  A-layout (m, k, r, s)
//                          B = Q^T  [d, Br]  B-layout (n, k, c, s)
//     O^T = V^T . P^T      A = V^T  [d, Bc]  A-layout
//
// Write the two index maps out (mesh (meshRow, tileSize, meshCol) = (16, 4, 16)):
//
//     A of X [M, K]:  X[m*16 + r, k*4 + s]  at ((m*K_T + k)*16 + r)*4 + s
//     B of Y [K, N]:  Y[k*4 + s, n*16 + c]  at ((n*K_T + k)*16 + c)*4 + s
//
// Put Y = Q^T, so Y[k*4+s, n*16+c] = Q[n*16+c, k*4+s], and the B map of Q^T IS the A map
// of Q, symbol for symbol, because meshRow == meshCol. So:
//
//     Q  -- NOT a transpose. B(Q^T) is byte-identical to A(Q), and the projections
//           already produce Q row-major. HeMAiA's comment calls this a transpose because it
//           converts from PACKED to B; nobody has to produce B from packed.
//     K  -- A-layout of the projection output. A strided nest at FP16, then a quantise.
//     V  -- A-layout of V^T. THIS one really is a transpose of the projection output:
//           a 4-element run of V^T is four TOKENS of one feature, which D-layout V never
//           holds contiguously.
//
// (HeMAiA's datagen stages FA's V as row_major_to_a(V) -- A-layout of V, not of V^T. The
// FA block's golden draws V at random, so nothing catches it; a real V would be read with
// its axes swapped.)
//
// ======================================================================================
// THE THREE PATHS THIS APP RUNS, ON THE SAME DATA
// ======================================================================================
//
// PATH H -- HeMAiA's graph, stage for stage, plus the smallest fix that reaches FA.
//
//     SIMD   reduce(SUMSQ) -> bcast_map(1/D, RSQRT) -> ew2(MUL)       norm1, row-major
//     xDMA   packed -> A (fp16 nest)                                  n1_to_a
//     SIMD   quantise                                                  n1_q
//     GEMM   x3, D port -> D-layout fp16                               proj_q|k|v
//     SIMD   x3 dequantise                                             proj_*_dq
//   --- the fix ---
//     xDMA   Q, K: D -> A (fp16 nest)       V: D -> A(V^T) through the 8x8 transposer
//     SIMD   x3 quantise                                               -> FA operands
//
//   10 SIMD passes, 4 xDMA passes, 3 GEMMs.
//
// PATH O -- the same arithmetic, with every layout change folded into an address generator
// that was running anyway, and every elementwise pair fused into one pass. No xDMA.
//
//     SIMD   reduce(SUMSQ) -> bcast_map(1/D, RSQRT)                    as in path H
//     SIMD   ew2(MUL) -> Fp16ToInt8, reading in A-ORDER               norm + reshape + quant
//     GEMM   Q, K: D port writes A-layout directly
//            V^T = Wv^T . Xn^T: the SWAPPED projection, D port writes A(V^T) directly
//     SIMD   x3 map(dequant) -> Fp16ToInt8, each behind its own GEMM  -> FA operands
//
//   6 SIMD passes, 0 xDMA passes, 3 GEMMs.
//
// PATH T -- O's projections behind snax-simd-rmsnorm's TRANSPOSED norm, which trades the
// reduce's cross-lane fold (~35 cc a row) for two xDMA transposes on an otherwise idle
// engine. The second transpose goes straight from y^T to A-layout, the same one-pass trick
// as path H's V fix.
//
//     xDMA   x -> x^T
//     SIMD   reduce(SUMSQ|LANEWISE) -> map(RSQRT), 1 beat -> ew(MUL|STICKY_B)
//     xDMA   y^T -> A (transposer)
//     SIMD   quantise
//     GEMM + SIMD   as path O
//
// All three produce the same bytes: every check below is bit-exact.
//
// MEASURED, warm, Verilator, snax_split_cluster, T = 32, d = 128. Makespan is first op to
// last op, engines running concurrently and handing off through TCDM flags; every path
// programs a dependent task before waiting for its input, so the paths differ in dataflow
// only.
//
//                 makespan    GEMM busy   SIMD busy   xDMA busy
//     H             6,941       1,840       4,929         844
//     O             4,890       1,759       3,761           0      -29%
//     T             3,487       1,751       1,901         447      -49%
//
// T is bounded by the array: 1,751 of its 3,487 cycles are the three projections at
// ~92% utilisation, behind a ~1,400-cycle norm prologue and one 330-cycle requant.
//
// THE THREE FOLDS, and why each is legal.
//
// 1. THE D PORT CAN WRITE A-LAYOUT. Its writer is an ordinary AGU: 16 channels of 8 B, a
//    [4, 4] spatial grid, 3 temporal loops. With the INT32->FP16 converter armed one beat is
//    4 output rows x 16 columns, and channel i carries row i/4, columns 4*(i%4)..+3 -- FOUR
//    consecutive fp16 of one row, which is EXACTLY one A-layout atom (tileSize = 4). So the
//    reshape HeMAiA runs as a separate xDMA node is just a different set of writer strides:
//
//                     sl0 (col grp)  sl1 (row)   t0 (beat)   t1 (n)    t2 (m)
//        D-layout         8            32          128         512       N*512
//        packed           8            pitch       4*pitch     32        16*pitch
//        A-layout         128          8           32          512       N*512
//
//    (packed is FlashAttention's own S^T descriptor.) HeMAiA's note that
//    "between two blocks that both live in L1 there is no load for the conversion to fold
//    into" misses the producer's own writer.
//
// 2. V^T COMES FROM THE SWAPPED GEMM. V^T = (Xn.Wv)^T = Wv^T . Xn^T: A = Wv^T, staged
//    offline in A-layout because it is a weight; B = Xn^T in B-layout, which by the
//    identity above is the SAME BYTES as the A-layout Xn the Q and K projections read. So
//    the transpose costs nothing at run time -- not a pass, not even a second copy of the
//    activation -- and the D port then writes A(V^T) by fold 1.
//
// 3. NORM -> A-LAYOUT -> QUANTISE IS ONE SIMD PASS. The SIMD reader's lane stride is
//    programmable. Read x and the replicated scale plane with lane stride = one ROW, and
//    each 8 B lane is one A atom of a different row: a beat is rows 8g..8g+7 at one k,
//    which is half an A block. EW1 multiplies, Fp16ToInt8 pairs two such beats into one
//    INT8 beat -- rows 16m..16m+15 at k, exactly one 64 B A/i8 block -- and the writer
//    drops it at block (m, k). Reader {operand, g, k}, writer {m, k}: three and two loops,
//    inside the SIMD's three. The quantise has to come AFTER the reshape because the int8
//    atom is 4 B against the 8 B lane -- and here it does, inside the same pass.
//
//    The row pitch is padded 256 -> 264 B for this. At 256 every lane of an A-order beat
//    sits in the SAME TCDM bank (32 banks x 8 B = 256 B), an 8-way conflict on every beat;
//    at 264 = 33 words the eight rows land in eight banks.
//
// ======================================================================================
// THE PRECISION FINDING, which no layout fix would have surfaced
// ======================================================================================
//
// FlashAttention narrows S = K.Q^T from INT32 to FP16 on the D port with no scale between.
// At the layer's shared s_qk this data reaches |S| = 289,505 -- 4.4x past FP16, so the
// scores become inf and the softmax NaN. The datagen therefore gives Q its OWN scale, the
// largest power of two that keeps |S| in FP16 (here x4 against K and V's x32; real
// attention divides S by sqrt(d) anyway).
//
// ======================================================================================
// PROVING FA CAN USE IT
// ======================================================================================
//
// After the paths the GEMM runs FlashAttention's two matmul shapes on path O's outputs, with
// FlashAttention's own descriptors (the generic programming below reproduces
// snax-flashattn-decode's datagen for M, N, K):
//
//     S^T = K.Q^T     A = K8, B = Q8          -> [key][query] fp16, FA's score layout
//     O^T = V^T.P^T   A = V^T8, B = staged P^T  (a small P, so O fits FP16 and can be read)
//
// and the SIMD core recomputes sampled outputs from the DEVICE's own operand bytes. That
// check does not depend on the operands matching any golden: if the layouts were wrong the
// matmul would contract the wrong elements and the recompute would disagree.
//
// ALSO CHECKED HERE: the two blocked routes of HeMAiA's __snax_bingo_kernel_simd_rmsnorm,
// ported pass for pass -- row_major -> A and col_major -> B, fp16 and int8 -- since that
// kernel cannot run on this testbench. Both are fold 3 generalised: a row_major tile's lane
// runs are A atoms, a col_major tile's are B atoms.

#include "data.h"
#include "snax-core-roles.h"
#include "snax-simd-lib.h"
#include "snax-versacore-to-lib.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

// The D-port converter must be the shift build (enable + extra-loop policy + shift = 3 CSRs):
// FlashAttention's scores at the layer's shared Q/K scale are past FP16's range.
#if !defined(READER_WRITER_EXTENSION_1_CSR_NUM) || READER_WRITER_EXTENSION_1_CSR_NUM != 3
#error "The GEMM's D-port Int32ToFp16Converter has no power-of-two shift (build it with shift: 1)."
#endif

#if !defined(SIMD_EXT_STREAMREDUCE_HAS_SUMSQ) || !defined(SIMD_EXT_STREAMMAP_HAS_RSQRT) || \
    !defined(SIMD_EXT_STREAMELEMENTWISE_1_HAS_MUL) || !defined(SIMD_EXT_FP16TOINT8)
#error "this kernel needs StreamReduce SUMSQ, StreamMap RSQRT, a post-map MUL and Fp16ToInt8"
#endif
#if !SNAX_HAS_GEMM_CORE || !SNAX_HAS_SIMD_CORE || !SNAX_HAS_XDMA_CORE
#error "this kernel needs the four-engine cluster"
#endif

// Every descriptor below is derived for this mesh and this tile. The derivations are in
// the header; these are the assumptions they rest on.
#if MESH_ROW != MESH_COL
#error "B(Q^T) == A(Q) needs meshRow == meshCol"
#endif
#if MESH_ROW != 16 || TILE_SIZE != 4
#error "the D-port descriptors assume a 16x4x16 mesh: a 4-row x 16-col FP16 beat, 8 B atoms"
#endif
#if T_TOK != 32 || D_MODEL % 32
#error "T must be 32 (one FP16 lane per token) and d a multiple of 32"
#endif

#define BEAT SIMD_BEAT_BYTES
#define ROWB (D_MODEL * 2u)         // one packed fp16 row
#define PITCH (ROWB + 8u)           // the padded row: 33 words, one bank apart per row
#define F16B (T_TOK * D_MODEL * 2u) // an fp16 [T, d] tile
#define I8B (T_TOK * D_MODEL)       // its int8 twin
#define WB (D_MODEL * D_MODEL)      // a weight
#define M_T (T_TOK / MESH_ROW)      // token row-tiles              = 2
#define K_T (D_MODEL / TILE_SIZE)   // feature k-tiles              = 32
#define N_D (D_MODEL / MESH_COL)    // feature col-tiles (x.W)      = 8
#define M_D (D_MODEL / MESH_ROW)    // feature row-tiles (V^T rows) = 8
#define N_TK (T_TOK / MESH_COL)     // token col-tiles              = 2
#define K_TK (T_TOK / TILE_SIZE)    // token k-tiles (P.V)          = 8

// ============================================================ the GEMM (hart 0)

// D-port output layouts. See the table in the header.
enum { OUT_D = 0, OUT_PACKED = 1, OUT_A = 2 };

// Arm one VersaCore dispatch: every streamer and accelerator CSR. The descriptor is
// generic in (M, N, K) and reproduces snax-flashattn-decode's datagen for its shapes.
// always_inline so every CSR address is a constant and folds to a single `csrw imm` --
// out of line each write is a jump-table load from L2.
//
// Both CSR managers queue two configurations (snax_cfg_queue_depth 2): a start write
// snapshots the bank, so the NEXT dispatch can be armed while this one runs. That is how
// the projections are issued -- the ~60 writes of dispatch n+1 hide behind dispatch n.
//
// C is read with ALL channels masked: a disabled channel presents zero and issues no TCDM
// request, which is the fresh-accumulator seed every matmul here wants. take_in_new_c must
// stay 1 -- at 0 the array stops draining the C reader and its FIFO never empties.
//
// `shift` is the D-port converter's power-of-two output scale: D = RNE(acc * 2^-shift). 0 for
// the projections and the P.V check, FA_S_SHIFT for FlashAttention's scores, whose INT32 range
// is past FP16's.
__attribute__((always_inline)) static inline void gemm_cfg(
    const void *a, const void *b, void *d, uint32_t M, uint32_t N, uint32_t K,
    uint32_t layout, uint32_t shift) {
    const uint32_t blk = MESH_ROW * TILE_SIZE;  // one A or B block, int8 = 64 B = one beat
    uint32_t dsl0, dsl1, dt0, dt1, dt2;
    if (layout == OUT_A) {
        uint32_t atom = TILE_SIZE * 2u, ablk = MESH_ROW * atom;  // 8 B, 128 B
        dsl0 = ablk;                    // next 4 columns = next k
        dsl1 = atom;                    // next row in the beat = next r
        dt0 = 4u * atom;                // next beat: 4 rows further
        dt1 = (MESH_COL / TILE_SIZE) * ablk;       // next n: 4 k's further
        dt2 = N * (MESH_COL / TILE_SIZE) * ablk;   // next m: a whole k-row of blocks
    } else if (layout == OUT_PACKED) {
        uint32_t pitch = N * MESH_COL * 2u;
        dsl0 = 8u;
        dsl1 = pitch;
        dt0 = 4u * pitch;
        dt1 = MESH_COL * 2u;
        dt2 = MESH_ROW * pitch;
    } else {
        uint32_t bb = MESH_ROW * MESH_COL * 2u;    // one 16x16 fp16 block
        dsl0 = 8u;
        dsl1 = MESH_COL * 2u;
        dt0 = 4u * MESH_COL * 2u;
        dt1 = bb;
        dt2 = N * bb;
    }

    // A -- reader 0: k inner, n broadcast, m outer.
    csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)a);
    csrw_ss(S_STRIDE_READER_0_0, 8);
    csrw_ss(T_BOUND_READER_0_0, K);
    csrw_ss(T_STRIDE_READER_0_0, blk);
    csrw_ss(T_BOUND_READER_0_1, N);
    csrw_ss(T_STRIDE_READER_0_1, 0);
    csrw_ss(T_BOUND_READER_0_2, M);
    csrw_ss(T_STRIDE_READER_0_2, K * blk);
    csrw_ss(T_BOUND_READER_0_3, 1);
    csrw_ss(T_STRIDE_READER_0_3, 0);
    csrw_ss(T_BOUND_READER_0_4, 1);
    csrw_ss(T_STRIDE_READER_0_4, 0);
    csrw_ss(T_BOUND_READER_0_5, 1);
    csrw_ss(T_STRIDE_READER_0_5, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0);
    // B -- reader 1: k inner, n, m broadcast.
    csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)b);
    csrw_ss(S_STRIDE_READER_1_0, 8);
    csrw_ss(T_BOUND_READER_1_0, K);
    csrw_ss(T_STRIDE_READER_1_0, blk);
    csrw_ss(T_BOUND_READER_1_1, N);
    csrw_ss(T_STRIDE_READER_1_1, K * blk);
    csrw_ss(T_BOUND_READER_1_2, M);
    csrw_ss(T_STRIDE_READER_1_2, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0);
    // C -- masked, so its addresses are never issued; the bounds still have to match the
    // eight INT32 beats per output block the array consumes.
    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, (uint32_t)d);
    csrw_ss(S_STRIDE_READER_WRITER_0_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_0_1, N * MESH_COL * 2u);
    csrw_ss(T_BOUND_READER_WRITER_0_0, MESH_ROW * MESH_COL * 32u / 1024u);
    csrw_ss(T_STRIDE_READER_WRITER_0_0, 128u * N);
    csrw_ss(T_BOUND_READER_WRITER_0_1, N);
    csrw_ss(T_STRIDE_READER_WRITER_0_1, MESH_COL * 2u);
    csrw_ss(T_BOUND_READER_WRITER_0_2, M);
    csrw_ss(T_STRIDE_READER_WRITER_0_2, 1024u * N);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, 0);
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_0, 0);
    // D -- the writer, with the converter armed: half the INT32 beats, 4 rows each.
    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, (uint32_t)d);
    csrw_ss(S_STRIDE_READER_WRITER_1_0, dsl0);
    csrw_ss(S_STRIDE_READER_WRITER_1_1, dsl1);
    csrw_ss(T_BOUND_READER_WRITER_1_0, MESH_ROW / 4u);
    csrw_ss(T_STRIDE_READER_WRITER_1_0, dt0);
    csrw_ss(T_BOUND_READER_WRITER_1_1, N);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, dt1);
    csrw_ss(T_BOUND_READER_WRITER_1_2, M);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, dt2);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, 0);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 0, 1u);  // Int32ToFp16 on
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 0u);  // 2:1 merge
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 2, shift);  // RNE(acc * 2^-shift)
    // The array: output-stationary, K pairs per block, M*N blocks.
    csrw_ss(OVERWRITE_ACCUM, 1);
    csrw_ss(ACCUM_BOUND, K);
    csrw_ss(OUTPUT_BOUND, M * N);
    csrw_ss(SUBTRACTIONS, 0);
    csrw_ss(ARRAY_SHAPE_CFG, 0);
    csrw_ss(DATA_TYPE_CFG, 0);
}

// Submit the armed dispatch. `cfgFire` is a write EVENT, so two back-to-back fires enqueue
// two tasks and nothing needs clearing in between.
__attribute__((always_inline)) static inline void gemm_fire(void) {
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(GEMMX_START, 1);
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
}

// Wait for the dispatch with this id. The ARRAY's retired-task counter is the only
// completion signal that works: the streamer's counter rises before the array is done, and
// BUSY neither tells "not started" from "finished" nor falls between queued dispatches.
// Bounded; returns 1 on timeout.
__attribute__((always_inline)) static inline uint32_t gemm_wait(uint32_t id) {
    uint32_t spins = 0;
    while ((int32_t)(csrr_ss(GEMMX_FINISHED_TASK) - id) < 0)
        if (++spins > 200000u) return 1u;
    csrw_ss(GEMMX_START, 0);
    return 0u;
}

// ============================================================ the SIMD block (hart 1)
//
// Every pass takes `how`:
//   PREP   program the CSR bank only. The bank is snapshotted into the task queue on the
//          fire, so a pass can be programmed BEFORE its input exists and fired the moment
//          the producer publishes it -- the programming leaves the critical path.
//   FIRE   program and submit. The block runs its 4-entry queue strictly in order and
//          starts a task only once the previous one has retired with its writer drained,
//          so a chain of DEPENDENT passes can be queued back to back.
//   DRAIN  program, submit, and wait for the whole queue.
#define PREP 0u
#define FIRE 1u
#define DRAIN 2u

// A task that never retires must not cost a whole simulation: bound every wait and latch
// the failure, so the passes after a wedged one short-circuit.
static inline void simd_go(volatile uint32_t *hung, const char *what,
                           const snax_simd_shape_t *in, const snax_simd_shape_t *out,
                           uint32_t how) {
    if (*hung) return;
    snax_simd_program_fast(in, out);
    if (how == PREP) return;
    snax_simd_fire();
    if (how == DRAIN && snax_simd_wait_all_checked(what, SIMD_WAIT_BUDGET)) *hung = 1;
}

// Submit a PREP'ed pass and drain.
static inline void simd_launch(volatile uint32_t *hung, const char *what) {
    if (*hung) return;
    snax_simd_fire();
    if (snax_simd_wait_all_checked(what, SIMD_WAIT_BUDGET)) *hung = 1;
}

// Per-row SUMSQ: rows of `beats`, `row_stride` apart -> one splatted scalar beat per row.
static void pass_reduce(volatile uint32_t *hung, void *src, void *dst, uint32_t row_stride,
                        uint32_t how) {
    snax_simd_shape_t in, out;
    snax_simd_shape_rows(&in, src, T_TOK, ROWB / BEAT, row_stride);
    snax_simd_shape_flat(&out, dst, T_TOK);
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, ROWB / BEAT,
                   SIMD_RED_SUMSQ);
    simd_go(hung, "reduce", &in, &out, how);
}

// Each row's scalar beat, re-presented once per beat of the row (inner stride 0), through
// map(a = 1/D, RSQRT): the replicated 1/rms plane, rows `row_stride` apart.
static void pass_bcast_rsqrt(volatile uint32_t *hung, void *bt, void *dst,
                             uint32_t row_stride, uint32_t how) {
    snax_simd_shape_t in, out;
    snax_simd_shape_2d(&in, bt, ROWB / BEAT, 0u, T_TOK, BEAT);
    snax_simd_shape_rows(&out, dst, T_TOK, ROWB / BEAT, row_stride);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR,
                   0x3F800000u - ((uint32_t)LOG2D << 23), 0u, SIMD_FUNC_RSQRT);
    simd_go(hung, "bcast_rsqrt", &in, &out, how);
}

// Row-major x * plane -> packed n1 (path H). Reader {operand, beat, row}. The AGU stride
// is unsigned, so the base is the LOWER operand; MUL commutes.
static void pass_mul(volatile uint32_t *hung, void *x, void *plane, void *dst,
                     uint32_t how) {
    uint32_t xa = (uint32_t)x, pa = (uint32_t)plane;
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, (void *)(xa < pa ? xa : pa), 1);
    in.bound[0] = 2u;
    in.stride[0] = xa < pa ? pa - xa : xa - pa;
    in.bound[1] = ROWB / BEAT;
    in.stride[1] = BEAT;
    in.bound[2] = T_TOK;
    in.stride[2] = ROWB;
    snax_simd_shape_flat(&out, dst, F16B / BEAT);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 2u,
                   SIMD_EW_MUL);
    simd_go(hung, "mul", &in, &out, how);
}

// Arm Fp16ToInt8 behind whatever else is enabled. tailPeriod 0: no passthrough.
#define ARM_QUANT(scale)                                            \
    do {                                                            \
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 0, (scale));  \
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 1, 0u);       \
    } while (0)

// FP16 -> INT8, flat, layout-preserving: `in_beats` in, half as many out.
static void pass_quant(volatile uint32_t *hung, void *src, void *dst, uint32_t in_beats,
                       uint32_t scale, uint32_t how) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, src, in_beats);
    snax_simd_shape_flat(&out, dst, in_beats / 2u);
    snax_simd_use0(SIMD_EXT_FP16TOINT8);
    ARM_QUANT(scale);
    simd_go(hung, "quant", &in, &out, how);
}

// out = a*x, flat (path H's dequantise).
static void pass_scale(volatile uint32_t *hung, void *src, void *dst, uint32_t beats,
                       uint32_t a, uint32_t how) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, src, beats);
    snax_simd_shape_flat(&out, dst, beats);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, a, 0u, SIMD_FUNC_LINEAR);
    simd_go(hung, "dequant", &in, &out, how);
}

// Dequantise THEN quantise in one sweep (path O). Both stages narrow to FP16 between them
// exactly as two passes would, so this is bit-identical to pass_scale + pass_quant.
static void pass_requant(volatile uint32_t *hung, void *src, void *dst, uint32_t in_beats,
                         uint32_t a, uint32_t scale, uint32_t how) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, src, in_beats);
    snax_simd_shape_flat(&out, dst, in_beats / 2u);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                            (1u << SIMD_EXT_STREAMMAP) | (1u << SIMD_EXT_FP16TOINT8));
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 0, a);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 1, 0u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 2, SIMD_FUNC_LINEAR);
    ARM_QUANT(scale);
    simd_go(hung, "requant", &in, &out, how);
}

// Fold 3: x * plane -> Fp16ToInt8 -> A/i8, reading in A-ORDER. Both operands sit at the
// padded pitch. Reader, innermost first:
//   lane    8 lanes, PITCH apart     rows 8g..8g+7, one 8 B atom each (4 features)
//   dim 0   operand, 2               x, then its plane beat
//   dim 1   g, 4, 8*PITCH            the next 8 rows (so g = 2m + half)
//   dim 2   k, K_T, 8 B              the next 4 features
// Fp16ToInt8 pairs (g = 2m, 2m+1) into rows 16m..16m+15 at k: one 64 B A/i8 block.
// Writer: dim 0 m (M_T, K_T*64), dim 1 k (K_T, 64).
static void pass_norm_quant_a(volatile uint32_t *hung, void *x, void *plane, void *dst,
                              uint32_t scale, uint32_t how) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, x, 1);
    in.lane_stride = PITCH;
    in.bound[0] = 2u;
    in.stride[0] = (uint32_t)plane - (uint32_t)x;
    in.bound[1] = 4u;
    in.stride[1] = 8u * PITCH;
    in.bound[2] = K_T;
    in.stride[2] = TILE_SIZE * 2u;
    snax_simd_shape_2d(&out, dst, M_T, K_T * BEAT, K_T, BEAT);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_1) |
                                                     (1u << SIMD_EXT_FP16TOINT8));
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 0, 2u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1, SIMD_EW_MUL);
    ARM_QUANT(scale);
    simd_go(hung, "norm_quant_a", &in, &out, how);
}

// ---- the transposed RMSNorm (path T) -- snax-simd-rmsnorm's fast path -----------------
//
// On x^T [d, T] a beat is one feature of ALL 32 tokens, lane t = token t, so the per-token
// sum of squares is what the per-lane accumulators hold after d beats: LANEWISE emits it
// as one beat with no fold. One map turns that beat into the 32 scales, and STICKY-B
// latches it as operand B for the whole tile -- no replicated plane at all.

// SUMSQ | LANEWISE over x^T: d beats in, ONE beat out (lane t = token t's sum).
static void pass_ssq_lanewise(volatile uint32_t *hung, void *xt, void *dst, uint32_t how) {
    snax_simd_shape_t in, out;
    snax_simd_shape_rows(&in, xt, 1u, D_MODEL, D_MODEL * BEAT);
    snax_simd_shape_flat(&out, dst, 1u);
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, D_MODEL,
                   SIMD_RED_SUMSQ | SIMD_RED_LANEWISE);
    simd_go(hung, "ssq_lanewise", &in, &out, how);
}

// map(a = 1/D, RSQRT) over one beat: the 32 per-token scales.
static void pass_rsqrt1(volatile uint32_t *hung, void *src, void *dst, uint32_t how) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, src, 1u);
    snax_simd_shape_flat(&out, dst, 1u);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR,
                   0x3F800000u - ((uint32_t)LOG2D << 23), 0u, SIMD_FUNC_RSQRT);
    simd_go(hung, "rsqrt1", &in, &out, how);
}

// ew(MUL | STICKY_B): beat 0 (the scales, IMMEDIATELY below x^T) is latched and emits
// nothing; beats 1..d emit x^T * scales. d+1 in, d out.
static void pass_sticky_mul(volatile uint32_t *hung, void *seed_then_xt, void *dst,
                            uint32_t how) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, seed_then_xt, D_MODEL + 1u);
    snax_simd_shape_flat(&out, dst, D_MODEL);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1u,
                   SIMD_EW_MUL | SIMD_EW_STICKY_B);
    simd_go(hung, "sticky_mul", &in, &out, how);
}

// ---- the HeMAiA rmsnorm's blocked routes, ported pass for pass ------------------------
//
// HeMAiA's __snax_bingo_kernel_simd_rmsnorm writes a GEMM operand layout directly:
// row_major -> A and col_major -> B, fp16 or int8. That kernel cannot run on this testbench,
// so its two routes are reproduced here with the same shapes and checked against what the
// paths above already produce. `blocked` is its simd_pass_blocked.
//
// The reader's lane stride is one ROW of `src`, so lane c carries one 4-element run of row
// 8g+c and beats g = 2b, 2b+1 at the same k are operand block (b, k): int8 packs them into
// one 64 B block, fp16 writes them as the block's two 64 B halves. row_major src -> A,
// col_major src -> B. pitch/8 must be odd so the eight lanes hit eight banks.
static void pass_blocked(volatile uint32_t *hung, void *src, void *mul_by, uint32_t pitch,
                         uint32_t nrows, uint32_t kt, void *dst, uint32_t out_i8,
                         uint32_t scale, uint32_t how) {
    const uint32_t lanes = 8u, blocks = nrows / 16u;
    snax_simd_shape_t in, out;
    uint32_t mask = 0u, d = 0u;
    snax_simd_shape_flat(&in, src, 1);
    in.lane_stride = pitch;
    if (mul_by) {
        uint32_t sa = (uint32_t)src, ma = (uint32_t)mul_by;
        in.base = (void *)(sa < ma ? sa : ma);
        in.bound[0] = 2u;
        in.stride[0] = sa < ma ? ma - sa : sa - ma;
        d = 1u;
        mask |= 1u << SIMD_EXT_STREAMELEMENTWISE_1;
    }
    in.bound[d] = nrows / lanes;
    in.stride[d] = lanes * pitch;
    in.bound[d + 1u] = kt;
    in.stride[d + 1u] = 8u;
    if (out_i8) {
        snax_simd_shape_2d(&out, dst, blocks, kt * BEAT, kt, BEAT);
        mask |= 1u << SIMD_EXT_FP16TOINT8;
    } else {
        snax_simd_shape_flat(&out, dst, 1);
        out.bound[0] = 2u;
        out.stride[0] = BEAT;
        out.bound[1] = blocks;
        out.stride[1] = kt * 2u * BEAT;
        out.bound[2] = kt;
        out.stride[2] = 2u * BEAT;
    }
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, mask);
    if (mul_by) {
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 0, 2u);
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1, SIMD_EW_MUL);
    }
    if (out_i8) ARM_QUANT(scale);
    simd_go(hung, "blocked", &in, &out, how);
}

// row_major -> A: reduce(SUMSQ|TAP) re-lays x at a (beats+1)*64+8 B pitch with each row's
// scalar right after it, bcast(RSQRT) writes the plane at the same pitch, blocked(xs*plane).
#define TAP_PITCH ((ROWB / BEAT + 1u) * BEAT + 8u)
static void route_row_to_a(volatile uint32_t *hung, void *x, uint8_t *xs, uint8_t *plane,
                           void *dst, uint32_t out_i8) {
    snax_simd_shape_t in, out;
    snax_simd_shape_rows(&in, x, T_TOK, ROWB / BEAT, ROWB);
    snax_simd_shape_rows(&out, xs, T_TOK, ROWB / BEAT + 1u, TAP_PITCH);
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, ROWB / BEAT,
                   SIMD_RED_SUMSQ | SIMD_RED_TAP);
    simd_go(hung, "tap_reduce", &in, &out, FIRE);
    snax_simd_shape_2d(&in, xs + ROWB, ROWB / BEAT, 0u, T_TOK, TAP_PITCH);
    snax_simd_shape_rows(&out, plane, T_TOK, ROWB / BEAT, TAP_PITCH);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR,
                   0x3F800000u - ((uint32_t)LOG2D << 23), 0u, SIMD_FUNC_RSQRT);
    simd_go(hung, "tap_bcast", &in, &out, FIRE);
    pass_blocked(hung, xs, plane, TAP_PITCH, T_TOK, K_T, dst, out_i8, SCALE_N1_BITS, DRAIN);
}

// col_major -> B: lanewise SUMSQ, 1-beat RSQRT into the seed, sticky MUL writing y^T at a
// 72 B pitch, blocked(y^T) -- lanes are eight features, each carrying four tokens.
#define YT_PITCH (2u * T_TOK + 8u)
static void route_col_to_b(volatile uint32_t *hung, uint8_t *xt_seed, uint8_t *ssq,
                           uint8_t *ytp, void *dst, uint32_t out_i8) {
    snax_simd_shape_t in, out;
    pass_ssq_lanewise(hung, xt_seed + BEAT, ssq, FIRE);
    pass_rsqrt1(hung, ssq, xt_seed, FIRE);
    snax_simd_shape_flat(&in, xt_seed, D_MODEL + 1u);
    snax_simd_shape_rows(&out, ytp, D_MODEL, 1u, YT_PITCH);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1u,
                   SIMD_EW_MUL | SIMD_EW_STICKY_B);
    simd_go(hung, "sticky_mul_pitch", &in, &out, FIRE);
    pass_blocked(hung, ytp, (void *)0, YT_PITCH, D_MODEL, T_TOK / 4u, dst, out_i8,
                 SCALE_N1_BITS, DRAIN);
}

// HeMAiA's descriptor-driven simd_pass_blocked, for ANY mesh: the host derives the order
// (data/blocked_nest.py, a mirror of HeMAiA's kernels/blocked_nest.py) and this executes
// it without interpreting it -- lane stride and up to three loops per side, an operand
// loop added innermost when multiplying, and an optional repeat. Drains the queue.
static void pass_blocked_desc(volatile uint32_t *hung, void *src, void *mul_by, void *dst,
                              const blk_case_t *c, uint32_t scale) {
    uint32_t sa = (uint32_t)src, ma = (uint32_t)mul_by;
    uint8_t *rbase = (uint8_t *)(mul_by && ma < sa ? ma : sa);
    uint32_t opstride = mul_by ? (ma > sa ? ma - sa : sa - ma) : 0u;
    uint32_t mask = (c->out_i8 ? (1u << SIMD_EXT_FP16TOINT8) : 0u) |
                    (mul_by ? (1u << SIMD_EXT_STREAMELEMENTWISE_1) : 0u);
    for (uint32_t r = 0; r < c->reps; r++) {
        snax_simd_shape_t in, out;
        snax_simd_shape_flat(&in, rbase + r * c->rep_rd, 1);
        in.lane_stride = c->rd_lane;
        uint32_t d = 0;
        if (mul_by) {
            in.bound[0] = 2u;
            in.stride[0] = opstride;
            d = 1;
        }
        for (uint32_t k = 0; d < 3u; k++, d++) {
            in.bound[d] = c->rd[2 * k];
            in.stride[d] = c->rd[2 * k + 1];
        }
        snax_simd_shape_flat(&out, (uint8_t *)dst + r * c->rep_wr, 1);
        out.lane_stride = c->wr_lane;
        for (uint32_t k = 0; k < 3u; k++) {
            out.bound[k] = c->wr[2 * k];
            out.stride[k] = c->wr[2 * k + 1];
        }
        snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, mask);
        if (mul_by) {
            snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 0, 2u);
            snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1, SIMD_EW_MUL);
        }
        if (c->out_i8) ARM_QUANT(scale);
        simd_go(hung, "blocked_desc", &in, &out, r + 1u == c->reps ? DRAIN : FIRE);
    }
}

// Element (i, j) of an [.., cols] tile in the A-layout of mb x kb blocks, in elements.
static inline uint32_t blk_at(uint32_t i, uint32_t j, uint32_t cols, uint32_t mb,
                              uint32_t kb) {
    return ((i / mb * (cols / kb) + j / kb) * mb + i % mb) * kb + j % kb;
}

// ============================================================ the xDMA (hart 2)
//
// Programmed with every CSR address a compile-time constant (a computed one lowers to a
// jump-table load from L2 per write) -- the same unrolled descriptor snax-simd-rmsnorm uses.

#define XD_W(addr, val) snax_write_xdma_cfg_reg((addr), (val))
#define XDMA_LANE_BYTES (XDMA_WIDTH / XDMA_SPATIAL_CHAN)

__attribute__((always_inline)) static inline void xdma_program_fast(
    const void *src, void *dst, uint32_t sp_src, uint32_t sp_dst, const uint32_t *bs,
    const uint32_t *ss, const uint32_t *bd, const uint32_t *sd) {
    uint32_t hi = (uint32_t)snrt_cluster_base_addrh();
    XD_W(XDMA_SRC_ADDR_PTR_LSB, (uint32_t)src);
    XD_W(XDMA_SRC_ADDR_PTR_MSB, hi);
    XD_W(XDMA_DST_ADDR_PTR_LSB, (uint32_t)dst);
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

// Exactly one reader extension or none; the transposer's one CSR picks 16-bit blocks.
__attribute__((always_inline)) static inline void xdma_arm(uint32_t src_mask,
                                                           uint32_t xpose_mode) {
    XD_W(XDMA_SRC_ENABLE_PTR, src_mask);
    XD_W(XDMA_SRC_EXT_CSR_PTR, xpose_mode);
    XD_W(XDMA_DST_ENABLE_PTR, 0u);
}

// Start the programmed task and wait for it. The conversions below take `go`: 0 programs
// the descriptor only, so it can be written before the producer publishes the input.
static uint32_t xdma_run(void) {
    snax_xdma_local_wait(snax_xdma_start());
    return snax_xdma_last_task_cycle();
}

// Zero [dst, dst + beats*64) with the writer memset -- reader channels off, so nothing is
// read. TCDM has no reset: on RTL an unwritten word reads X, and Verilator's zeroes hide it.
static void xdma_fill_zero(void *dst, uint32_t beats) {
    uint32_t b5[5] = {beats, 1, 1, 1, 1}, s5[5] = {BEAT, 0, 0, 0, 0};
    xdma_program_fast(dst, dst, 8, 8, b5, s5, b5, s5);
    XD_W(XDMA_SRC_ENABLED_CHAN_PTR, 0);
    XD_W(XDMA_SRC_ENABLE_PTR, 0);
    XD_W(XDMA_DST_ENABLE_PTR, 1u << WRITER_EXT_VERILOGMEMSET);
    XD_W(XDMA_DST_EXT_CSR_PTR, 0u);
    (void)xdma_run();
}

// packed fp16 [T, d] -> A-layout (path H's n1_to_a): the 8 lanes walk rows, the atom is
// 4 features = 8 B on both sides.
static uint32_t xdma_packed_to_a(const void *src, void *dst, uint32_t go) {
    const uint32_t ablk = MESH_ROW * TILE_SIZE * 2u;
    uint32_t bs[5] = {MESH_ROW / 8u, K_T, M_T, 1, 1};
    uint32_t ss[5] = {8u * ROWB, TILE_SIZE * 2u, MESH_ROW * ROWB, 0, 0};
    uint32_t sd[5] = {8u * TILE_SIZE * 2u, ablk, K_T * ablk, 0, 0};
    xdma_arm(0u, 0u);
    xdma_program_fast(src, dst, ROWB, TILE_SIZE * 2u, bs, ss, bs, sd);
    return go ? xdma_run() : 0u;
}

// D-layout fp16 [T, d] -> A-layout (path H's fix for Q and K). HeMAiA's nest.d_to_a_args,
// RTL-validated there: lanes on r, then r-group, the tileSize groups j of one meshCol, n, m.
static uint32_t xdma_d_to_a(const void *src, void *dst, uint32_t go) {
    const uint32_t ablk = MESH_ROW * TILE_SIZE * 2u, dblk = MESH_ROW * MESH_COL * 2u;
    uint32_t bs[5] = {MESH_ROW / 8u, MESH_COL / TILE_SIZE, N_D, M_T, 1};
    uint32_t ss[5] = {8u * MESH_COL * 2u, TILE_SIZE * 2u, dblk, N_D * dblk, 0};
    uint32_t sd[5] = {8u * TILE_SIZE * 2u, ablk, (MESH_COL / TILE_SIZE) * ablk, K_T * ablk, 0};
    xdma_arm(0u, 0u);
    xdma_program_fast(src, dst, MESH_COL * 2u, TILE_SIZE * 2u, bs, ss, bs, sd);
    return go ? xdma_run() : 0u;
}

// D-layout V [T, d] -> A-layout of V^T [d, T] in ONE pass through the 8x8 transposer
// (path H's fix for V). A block's two input beats are its columns 0-3 and 4-7, lane c its
// row c; the two output beats are its transposed rows' first and last four elements.
//                    src (V, D-layout)             dst (V^T, A-layout, K' = T/4)
//   lane             next token row   32 B         next feature row   8 B
//   dim 0  beat, 2   features +4       8 B         tokens +4 -> k'+1  128 B
//   dim 1  2         features +8      16 B         features +8        64 B
//   dim 2  n, N_D    features +16    512 B         features +16      K'*128 B
//   dim 3  2         tokens +8       256 B         tokens +8 -> k'+2  256 B
//   dim 4  m, M_T    tokens +16    N_D*512 B       tokens +16 -> k'+4 512 B
static uint32_t xdma_d_to_at(const void *src, void *dst, uint32_t go) {
    const uint32_t ablk = MESH_ROW * TILE_SIZE * 2u, dblk = MESH_ROW * MESH_COL * 2u;
    uint32_t bs[5] = {2, 2, N_D, 2, M_T};
    uint32_t ss[5] = {8u, 16u, dblk, 8u * MESH_COL * 2u, N_D * dblk};
    uint32_t sd[5] = {ablk, 8u * TILE_SIZE * 2u, K_TK * ablk, 2u * ablk, 4u * ablk};
    xdma_arm(1u << READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16, 1u);
    xdma_program_fast(src, dst, MESH_COL * 2u, TILE_SIZE * 2u, bs, ss, bs, sd);
    return go ? xdma_run() : 0u;
}

// Full [rows, cols] -> [cols, rows] fp16 transpose (the same as snax-simd-rmsnorm's): 8x8
// blocks, two beats each, the block grid walked so each lands at its transposed position.
static uint32_t xdma_transpose_fp16(const void *src, void *dst, uint32_t rows,
                                    uint32_t cols, uint32_t go) {
    uint32_t bs[5] = {2, cols / 8u, rows / 8u, 1, 1};
    uint32_t ss[5] = {XDMA_LANE_BYTES, 16u, cols * 16u, 0, 0};
    uint32_t sd[5] = {XDMA_LANE_BYTES, rows * 16u, 16u, 0, 0};
    xdma_arm(1u << READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16, 1u);
    xdma_program_fast(src, dst, cols * 2u, rows * 2u, bs, ss, bs, sd);
    return go ? xdma_run() : 0u;
}

// y^T [d, T] packed -> A-layout of y [T, d], in ONE transposer pass: a transposed block's
// output lane is one token's 4 consecutive features, which is exactly an A atom.
//                    src (y^T, packed, 64 B rows)   dst (A of y, K_T = d/4)
//   lane             next feature row   64 B        next token   -> r+1    8 B
//   dim 0  beat, 2   tokens +4           8 B        features +4  -> k+1  128 B
//   dim 1  d/8       features +8       512 B        features +8  -> k+2  256 B
//   dim 2  2         tokens +8          16 B        tokens +8    -> r+8   64 B
//   dim 3  2         tokens +16         32 B        tokens +16   -> m+1  K_T*128 B
static uint32_t xdma_yt_to_a(const void *src, void *dst, uint32_t go) {
    const uint32_t ablk = MESH_ROW * TILE_SIZE * 2u;
    uint32_t bs[5] = {2, D_MODEL / 8u, 2, T_TOK / 16u, 1};
    uint32_t ss[5] = {8u, 8u * T_TOK * 2u, 16u, 32u, 0};
    uint32_t sd[5] = {ablk, 2u * ablk, 8u * TILE_SIZE * 2u, K_T * ablk, 0};
    xdma_arm(1u << READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16, 1u);
    xdma_program_fast(src, dst, T_TOK * 2u, TILE_SIZE * 2u, bs, ss, bs, sd);
    return go ? xdma_run() : 0u;
}

// ============================================================ checks (hart 1)

static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

static inline uint32_t ulp16(uint16_t a, uint16_t b) {
    uint32_t x = fp16_mono(a), y = fp16_mono(b);
    return x > y ? x - y : y - x;
}

// The D port's INT32 -> FP16 (round to nearest even), in integer arithmetic, with the
// converter's power-of-two output scale: RNE(v * 2^-shift), shift clamped to 14 like the RTL.
// The shift only lowers the exponent; with shift <= 14 no integer lands in the subnormals.
static uint16_t i32_to_f16(int32_t v, uint32_t shift) {
    uint32_t sign = v < 0, mag = sign ? (uint32_t)(-v) : (uint32_t)v;
    if (!mag) return (uint16_t)(sign << 15);
    if (shift > 14u) shift = 14u;
    uint32_t msb = 31u - (uint32_t)__builtin_clz(mag);
    if (msb + 15u - shift >= 31u) return (uint16_t)((sign << 15) | 0x7C00u);
    uint32_t norm = mag << (31u - msb);
    uint32_t frac = (norm >> 21) & 0x3FFu, g = (norm >> 20) & 1u;
    uint32_t rs = (norm & 0xFFFFFu) != 0, e = msb + 15u - shift;
    if (g && (rs || (frac & 1u))) {
        if (++frac == 1024u) {
            frac = 0;
            if (++e >= 31u) return (uint16_t)((sign << 15) | 0x7C00u);
        }
    }
    return (uint16_t)((sign << 15) | (e << 10) | frac);
}

// Element (row, col) of a [rows, cols] matrix in A-layout / B-layout, in elements.
static inline uint32_t a_at(uint32_t row, uint32_t col, uint32_t cols) {
    return (((row / MESH_ROW) * (cols / TILE_SIZE) + col / TILE_SIZE) * MESH_ROW +
            row % MESH_ROW) * TILE_SIZE + col % TILE_SIZE;
}
static inline uint32_t b_at(uint32_t row, uint32_t col, uint32_t rows) {
    return (((col / MESH_COL) * (rows / TILE_SIZE) + row / TILE_SIZE) * MESH_COL +
            col % MESH_COL) * TILE_SIZE + row % TILE_SIZE;
}

// Count the int8 elements that differ, and the largest difference.
static uint32_t cmp_i8(const int8_t *got, const int8_t *want, uint32_t n, uint32_t *worst) {
    uint32_t bad = 0;
    *worst = 0;
    for (uint32_t i = 0; i < n; i++) {
        int32_t dlt = (int32_t)got[i] - (int32_t)want[i];
        uint32_t ad = (uint32_t)(dlt < 0 ? -dlt : dlt);
        if (ad) bad++;
        if (ad > *worst) *worst = ad;
    }
    return bad;
}

static uint32_t cmp_f16(const uint16_t *got, const uint16_t *want, uint32_t n,
                        uint32_t *worst) {
    uint32_t bad = 0;
    *worst = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t u = ulp16(got[i], want[i]);
        if (u) bad++;
        if (u > *worst) *worst = u;
    }
    return bad;
}

// ============================================================ main
//
// THE SCHEDULE. Each engine runs its own ordered list of operations; one that consumes
// another engine's output spins on that producer's DONE flag in TCDM. There is no barrier
// inside a path: a barrier would serialise engines that are independent, and what this app
// measures is the makespan a layer actually gets -- e.g. path O requantises Q on the SIMD
// while the array computes K and V^T.

// Shared words at the bottom of the arena.
#define TS(i) tm[(i)]           // op start, relative to the path's origin
#define TE(i) tm[64 + (i)]      // op end: its output is in TCDM
#define DONE(i) tm[128 + (i)]   // == epoch once op i has ended
#define ORG tm[192]             // the path's time origin, stamped by one hart for all
#define HUNG (tm + 193)         // latched by a SIMD task that never retired
#define TMO tm[194]             // bounded waits that ran out

enum {
    H_NORM, H_RS, H_Q1, H_PQ, H_PK, H_PV, H_DQQ, H_DQK, H_DQV, H_CQ, H_CK, H_CV,
    H_QQ, H_QK, H_QV, H_END,
    O_NORM = 16, O_PQ, O_PK, O_PV, O_RQ, O_RK, O_RV, O_END,
    T_XT = 24, T_NS, T_YA, T_Q1, T_PQ, T_PK, T_PV, T_RQ, T_RK, T_RV, T_END,
    F_QK = 36, F_PV, F_END,
    V_A8 = 40, V_A16, V_B8, V_B16, V_END,
};

static inline void spin_done(volatile uint32_t *tm, uint32_t i, uint32_t epoch) {
    uint32_t n = 0;
    while (DONE(i) != epoch)
        if (++n > 2000000u) {
            TMO++;
            return;
        }
}

#define WAIT(i) spin_done(tm, (i), epoch)
#define BEGIN(i) (TS(i) = snrt_mcycle() - org)
#define END(i)                             \
    do {                                   \
        TE(i) = snrt_mcycle() - org;       \
        DONE(i) = epoch;                   \
    } while (0)

// The projections and the requantise, shared by paths O and T. `pq` names three
// consecutive GEMM ops (Q, K, V^T) and `rq` three SIMD ones. Each dispatch is armed while
// its predecessor runs; each requant waits for exactly its own projection, so Q's and K's
// hide behind the next GEMM and only V^T's is left on the tail.
#define GEMM_PROJ_TAIL(dep, pq, n1q, yo)                                              \
    do {                                                                               \
        uint32_t id_ = csrr_ss(GEMMX_FINISHED_TASK), iq_, ik_, iv_;                    \
        gemm_cfg((n1q), wq, (yo), M_T, N_D, K_T, OUT_A, 0u); /* before its operand exists */ \
        WAIT(dep);                                                                     \
        BEGIN(pq); gemm_fire(); iq_ = ++id_;                                           \
        gemm_cfg((n1q), wk, (yo) + F16B, M_T, N_D, K_T, OUT_A, 0u);                        \
        BEGIN((pq) + 1); gemm_fire(); ik_ = ++id_;                                     \
        TMO += gemm_wait(iq_); END(pq);                                                \
        /* V^T = Wv^T . Xn^T: A = Wv^T [d, d], B = Xn^T -- the A-layout Xn bytes again */ \
        gemm_cfg(wvt, (n1q), (yo) + 2 * F16B, M_D, N_TK, K_T, OUT_A, 0u);                   \
        BEGIN((pq) + 2); gemm_fire(); iv_ = ++id_;                                     \
        TMO += gemm_wait(ik_); END((pq) + 1);                                          \
        TMO += gemm_wait(iv_); END((pq) + 2);                                          \
    } while (0)

#define SIMD_REQUANT_TAIL(pq, rq, yo, qkv)                                             \
    do {                                                                               \
        pass_requant(HUNG, (yo), (qkv), F16B / BEAT, DQ_PROJ_BITS, SCALE_Q_BITS, PREP); \
        WAIT(pq);                                                                      \
        BEGIN(rq); simd_launch(HUNG, "requant q"); END(rq);                            \
        pass_requant(HUNG, (yo) + F16B, (qkv) + I8B, F16B / BEAT, DQ_PROJ_BITS,        \
                     SCALE_KV_BITS, PREP);                                             \
        WAIT((pq) + 1);                                                                \
        BEGIN((rq) + 1); simd_launch(HUNG, "requant k"); END((rq) + 1);                \
        pass_requant(HUNG, (yo) + 2 * F16B, (qkv) + 2 * I8B, F16B / BEAT, DQ_PROJ_BITS, \
                     SCALE_KV_BITS, PREP);                                             \
        WAIT((pq) + 2);                                                                \
        BEGIN((rq) + 2); simd_launch(HUNG, "requant v^T"); END((rq) + 2);              \
    } while (0)

// One path's timeline. Ops of one engine appear in `eng` in their program order, so an
// op's effective start is the later of its own start and its engine's previous end: a
// queued GEMM dispatch is fired while its predecessor runs, but the array only takes it
// when that one retires.
static uint32_t report_path(volatile uint32_t *tm, char tag, uint32_t lo, uint32_t hi,
                            const char *const *names, const uint8_t *eng) {
    static const char *const en[3] = {"GEMM", "SIMD", "xDMA"};
    uint32_t last[3] = {0, 0, 0}, busy[3] = {0, 0, 0}, t0 = 0xFFFFFFFFu, t1 = 0;
    for (uint32_t i = lo; i < hi; i++) {
        uint32_t e = eng[i - lo], s = TS(i) > last[e] ? TS(i) : last[e];
        busy[e] += TE(i) - s;
        last[e] = TE(i);
        if (TS(i) < t0) t0 = TS(i);
        if (TE(i) > t1) t1 = TE(i);
    }
    for (uint32_t i = lo; i < hi; i++)
        printf("[QKV]  %c %-26s %s  %5u .. %5u\n", tag, names[i - lo], en[eng[i - lo]],
               TS(i) - t0, TE(i) - t0);
    printf("[QKV]  %c makespan %u   busy: GEMM %u  SIMD %u  xDMA %u\n", tag, t1 - t0,
           busy[0], busy[1], busy[2]);
    return t1 - t0;
}

int main() {
    // ---- TCDM layout, derived identically on every hart ----------------------------------
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 63u) & ~63u);
    uint8_t *arena = p;
#define TAKE(n) (p += (n), p - (n))
    volatile uint32_t *tm = (volatile uint32_t *)TAKE(1024);
    uint8_t *x_p = TAKE(F16B);                  // packed x          (path H)
    uint8_t *x_w = TAKE(T_TOK * PITCH);         // padded x          (path O) -- BELOW bc_w
    int8_t *wq = (int8_t *)TAKE(WB);
    int8_t *wk = (int8_t *)TAKE(WB);
    int8_t *wv = (int8_t *)TAKE(WB);
    int8_t *wvt = (int8_t *)TAKE(WB);
    int8_t *ptb = (int8_t *)TAKE(T_TOK * T_TOK);
    uint8_t *bt = TAKE(T_TOK * BEAT);           // per-row SUMSQ, one splatted beat each
    uint8_t *bc_p = TAKE(F16B);                 // 1/rms plane, packed
    uint8_t *bc_w = TAKE(T_TOK * PITCH);        // 1/rms plane, padded -- ABOVE x_w
    uint8_t *n1_p = TAKE(F16B);                 // norm1, packed     (path H)
    uint8_t *n1a = TAKE(F16B);                  // norm1, A/f16      (path H)
    int8_t *n1q_h = (int8_t *)TAKE(I8B);        // norm1, A/i8       (path H)
    int8_t *n1q_o = (int8_t *)TAKE(I8B);        // norm1, A/i8       (path O)
    uint8_t *yh = TAKE(3 * F16B);               // H: projections, D/f16 x3
    uint8_t *dqh = TAKE(3 * F16B);              // H: dequantised, D/f16 x3
    uint8_t *ah = TAKE(3 * F16B);               // H: Q, K in A/f16; V^T in A/f16
    int8_t *qkv_h = (int8_t *)TAKE(3 * I8B);    // H: FA operands  Q8 | K8 | V^T8
    uint8_t *yo = TAKE(3 * F16B);               // O: Q | K | V^T in A/f16, ADJACENT
    int8_t *qkv_o = (int8_t *)TAKE(3 * I8B);    // O: FA operands  Q8 | K8 | V^T8
    // path T. The scale beat MUST sit immediately below x^T: the sticky task reads one
    // flat stream of 1 + d beats and the latch is simply its first.
    uint8_t *xt_seed = TAKE(BEAT + D_MODEL * BEAT);
    uint8_t *xt = xt_seed + BEAT;               // x^T [d, T]
    uint8_t *ssq = TAKE(BEAT);                  // per-token SUMSQ, lane t = token t
    uint8_t *yt = TAKE(F16B);                   // norm1^T [d, T]
    uint8_t *ya = TAKE(F16B);                   // norm1, A/f16
    int8_t *n1q_t = (int8_t *)TAKE(I8B);        // norm1, A/i8       (path T)
    uint8_t *yt3 = TAKE(3 * F16B);              // T: Q | K | V^T in A/f16
    int8_t *qkv_t = (int8_t *)TAKE(3 * I8B);    // T: FA operands  Q8 | K8 | V^T8
    // the HeMAiA rmsnorm's blocked routes
    uint8_t *xs = TAKE(T_TOK * TAP_PITCH);      // x re-laid by the TAP reduce  -- BELOW pl
    uint8_t *pl = TAKE(T_TOK * TAP_PITCH);      // the 1/rms plane, same pitch
    uint8_t *ytp = TAKE(D_MODEL * YT_PITCH);    // y^T at a 72 B pitch
    int8_t *va8 = (int8_t *)TAKE(I8B);          // row_major -> A, int8
    uint8_t *va16 = TAKE(F16B);                 // row_major -> A, fp16
    int8_t *vb8 = (int8_t *)TAKE(I8B);          // col_major -> B, int8
    uint8_t *vb16 = TAKE(F16B);                 // col_major -> B, fp16
    uint16_t *s16 = (uint16_t *)TAKE(T_TOK * T_TOK * 2u);   // S^T [key][query]
    uint16_t *o16 = (uint16_t *)TAKE(D_MODEL * T_TOK * 2u); // O^T [d][query]
    // goldens, staged into L1 so the checks do not pay a DRAM round trip per element
    uint16_t *gl_n1 = (uint16_t *)TAKE(F16B);
    int8_t *gl_n1q = (int8_t *)TAKE(I8B);
    uint16_t *gl_pq = (uint16_t *)TAKE(F16B);
    int8_t *gl_qkv = (int8_t *)TAKE(3 * I8B);
    uint16_t *gl_s = (uint16_t *)TAKE(T_TOK * T_TOK * 2u);
    uint16_t *gl_o = (uint16_t *)TAKE(D_MODEL * T_TOK * 2u);
    uint32_t arena_beats = (uint32_t)(p - arena) / BEAT;
#undef TAKE

    // ---- define every byte, then load ----------------------------------------------------
    if (snax_is_xdma_core()) xdma_fill_zero(arena, arena_beats);
    snrt_cluster_hw_barrier();
    if (snax_is_idma_core()) {
        snrt_dma_start_1d(x_p, x_in, F16B);
        snrt_dma_start_2d(x_w, x_in, ROWB, PITCH, ROWB, T_TOK);
        snrt_dma_start_1d(wq, wq_b, WB);
        snrt_dma_start_1d(wk, wk_b, WB);
        snrt_dma_start_1d(wv, wv_b, WB);
        snrt_dma_start_1d(wvt, wvt_a, WB);
        snrt_dma_start_1d(ptb, pt_b, T_TOK * T_TOK);
        snrt_dma_start_1d(gl_n1, g_n1, F16B);
        snrt_dma_start_1d(gl_n1q, g_n1q_a, I8B);
        snrt_dma_start_1d(gl_pq, g_projq_d, F16B);
        snrt_dma_start_1d(gl_qkv, g_q8, I8B);
        snrt_dma_start_1d(gl_qkv + I8B, g_k8, I8B);
        snrt_dma_start_1d(gl_qkv + 2 * I8B, g_vt8, I8B);
        snrt_dma_start_1d(gl_s, g_s16, T_TOK * T_TOK * 2u);
        snrt_dma_start_1d(gl_o, g_o16, D_MODEL * T_TOK * 2u);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    const int isG = snax_is_gemm_core(), isS = snax_is_simd_core(), isX = snax_is_xdma_core();

    // Two iterations: 0 is cold (icache, first AGU program), 1 is warm and is reported.
    for (uint32_t epoch = 1; epoch <= 2; epoch++) {
        uint32_t org;

        // ================= PATH H: HeMAiA's graph + the minimum fix =================
        if (isG) ORG = snrt_mcycle();
        snrt_cluster_hw_barrier();
        org = ORG;
        if (isG) {
            uint32_t id = csrr_ss(GEMMX_FINISHED_TASK), iq, ik, iv;
            gemm_cfg(n1q_h, wq, yh, M_T, N_D, K_T, OUT_D, 0u);  // armed before its operand exists
            WAIT(H_Q1);
            BEGIN(H_PQ); gemm_fire(); iq = ++id;
            gemm_cfg(n1q_h, wk, yh + F16B, M_T, N_D, K_T, OUT_D, 0u);
            BEGIN(H_PK); gemm_fire(); ik = ++id;
            TMO += gemm_wait(iq); END(H_PQ);
            gemm_cfg(n1q_h, wv, yh + 2 * F16B, M_T, N_D, K_T, OUT_D, 0u);
            BEGIN(H_PV); gemm_fire(); iv = ++id;
            TMO += gemm_wait(ik); END(H_PK);
            TMO += gemm_wait(iv); END(H_PV);
        }
        if (isS) {
            BEGIN(H_NORM);
            pass_reduce(HUNG, x_p, bt, ROWB, FIRE);
            pass_bcast_rsqrt(HUNG, bt, bc_p, ROWB, FIRE);
            pass_mul(HUNG, x_p, bc_p, n1_p, DRAIN);
            END(H_NORM);
            // Every later pass depends on another engine: program it, THEN wait.
            pass_quant(HUNG, n1a, n1q_h, F16B / BEAT, SCALE_N1_BITS, PREP);
            WAIT(H_RS);
            BEGIN(H_Q1); simd_launch(HUNG, "quant n1"); END(H_Q1);
            pass_scale(HUNG, yh, dqh, F16B / BEAT, DQ_PROJ_BITS, PREP);
            WAIT(H_PQ);
            BEGIN(H_DQQ); simd_launch(HUNG, "dequant q"); END(H_DQQ);
            pass_scale(HUNG, yh + F16B, dqh + F16B, F16B / BEAT, DQ_PROJ_BITS, PREP);
            WAIT(H_PK);
            BEGIN(H_DQK); simd_launch(HUNG, "dequant k"); END(H_DQK);
            pass_scale(HUNG, yh + 2 * F16B, dqh + 2 * F16B, F16B / BEAT, DQ_PROJ_BITS, PREP);
            WAIT(H_PV);
            BEGIN(H_DQV); simd_launch(HUNG, "dequant v"); END(H_DQV);
            // -- the fix: into FA's layouts, then int8 --
            pass_quant(HUNG, ah, qkv_h, F16B / BEAT, SCALE_Q_BITS, PREP);
            WAIT(H_CQ);
            BEGIN(H_QQ); simd_launch(HUNG, "quant q"); END(H_QQ);
            pass_quant(HUNG, ah + F16B, qkv_h + I8B, F16B / BEAT, SCALE_KV_BITS, PREP);
            WAIT(H_CK);
            BEGIN(H_QK); simd_launch(HUNG, "quant k"); END(H_QK);
            pass_quant(HUNG, ah + 2 * F16B, qkv_h + 2 * I8B, F16B / BEAT, SCALE_KV_BITS, PREP);
            WAIT(H_CV);
            BEGIN(H_QV); simd_launch(HUNG, "quant v^T"); END(H_QV);
        }
        if (isX) {
            (void)xdma_packed_to_a(n1_p, n1a, 0);
            WAIT(H_NORM); BEGIN(H_RS); (void)xdma_run(); END(H_RS);
            (void)xdma_d_to_a(dqh, ah, 0);
            WAIT(H_DQQ); BEGIN(H_CQ); (void)xdma_run(); END(H_CQ);
            (void)xdma_d_to_a(dqh + F16B, ah + F16B, 0);
            WAIT(H_DQK); BEGIN(H_CK); (void)xdma_run(); END(H_CK);
            (void)xdma_d_to_at(dqh + 2 * F16B, ah + 2 * F16B, 0);
            WAIT(H_DQV); BEGIN(H_CV); (void)xdma_run(); END(H_CV);
        }
        snrt_cluster_hw_barrier();

        // ================= PATH O: layouts folded into the AGUs =================
        if (isG) ORG = snrt_mcycle();
        snrt_cluster_hw_barrier();
        org = ORG;
        if (isG) GEMM_PROJ_TAIL(O_NORM, O_PQ, n1q_o, yo);
        if (isS) {
            BEGIN(O_NORM);
            pass_reduce(HUNG, x_w, bt, PITCH, FIRE);
            pass_bcast_rsqrt(HUNG, bt, bc_w, PITCH, FIRE);
            pass_norm_quant_a(HUNG, x_w, bc_w, n1q_o, SCALE_N1_BITS, DRAIN);
            END(O_NORM);
            SIMD_REQUANT_TAIL(O_PQ, O_RQ, yo, qkv_o);
        }
        snrt_cluster_hw_barrier();

        // ================= PATH T: transposed RMSNorm, then as O =================
        if (isG) ORG = snrt_mcycle();
        snrt_cluster_hw_barrier();
        org = ORG;
        if (isX) {
            BEGIN(T_XT); (void)xdma_transpose_fp16(x_p, xt, T_TOK, D_MODEL, 1); END(T_XT);
            (void)xdma_yt_to_a(yt, ya, 0);
            WAIT(T_NS); BEGIN(T_YA); (void)xdma_run(); END(T_YA);
        }
        if (isS) {
            pass_ssq_lanewise(HUNG, xt, ssq, PREP);
            WAIT(T_XT);
            BEGIN(T_NS);
            snax_simd_fire();
            pass_rsqrt1(HUNG, ssq, xt_seed, FIRE);
            pass_sticky_mul(HUNG, xt_seed, yt, DRAIN);
            END(T_NS);
            pass_quant(HUNG, ya, n1q_t, F16B / BEAT, SCALE_N1_BITS, PREP);
            WAIT(T_YA);
            BEGIN(T_Q1); simd_launch(HUNG, "quant n1"); END(T_Q1);
            SIMD_REQUANT_TAIL(T_PQ, T_RQ, yt3, qkv_t);
        }
        if (isG) GEMM_PROJ_TAIL(T_Q1, T_PQ, n1q_t, yt3);
        snrt_cluster_hw_barrier();

        // ================= the HeMAiA rmsnorm's blocked routes, one at a time ==============
        if (isG) ORG = snrt_mcycle();
        snrt_cluster_hw_barrier();
        org = ORG;
        if (isS) {
            BEGIN(V_A8); route_row_to_a(HUNG, x_p, xs, pl, va8, 1u); END(V_A8);
            BEGIN(V_A16); route_row_to_a(HUNG, x_p, xs, pl, va16, 0u); END(V_A16);
            BEGIN(V_B8); route_col_to_b(HUNG, xt_seed, ssq, ytp, vb8, 1u); END(V_B8);
            BEGIN(V_B16); route_col_to_b(HUNG, xt_seed, ssq, ytp, vb16, 0u); END(V_B16);
        }
        snrt_cluster_hw_barrier();

        // ================= FlashAttention's two matmuls, on path O's operands =============
        if (isG) ORG = snrt_mcycle();
        snrt_cluster_hw_barrier();
        org = ORG;
        if (isG) {
            uint32_t id = csrr_ss(GEMMX_FINISHED_TASK);
            BEGIN(F_QK);
            gemm_cfg(qkv_o + I8B, qkv_o, s16, M_T, N_TK, K_T, OUT_PACKED, FA_S_SHIFT);  // A = K, B = Q
            gemm_fire();
            TMO += gemm_wait(++id);
            END(F_QK);
            BEGIN(F_PV);
            gemm_cfg(qkv_o + 2 * I8B, ptb, o16, M_D, N_TK, K_TK, OUT_PACKED, 0u);  // A = V^T
            gemm_fire();
            TMO += gemm_wait(++id);
            END(F_PV);
        }
        snrt_cluster_hw_barrier();
    }

    if (!isS) return 0;

    // ============================================================ report
    static const char *const hname[H_END] = {
        "norm: reduce,bcast,mul", "xDMA packed->A", "quant n1", "GEMM proj_q (D)",
        "GEMM proj_k (D)", "GEMM proj_v (D)", "dequant q", "dequant k", "dequant v",
        "xDMA q D->A", "xDMA k D->A", "xDMA v D->A(V^T) xpose", "quant q", "quant k",
        "quant v^T"};
    static const uint8_t heng[H_END] = {1, 2, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1};
    static const char *const oname[O_END - O_NORM] = {
        "norm: reduce,bcast,mul+q A", "GEMM Q   -> A", "GEMM K   -> A",
        "GEMM V^T -> A (swapped)", "requant Q", "requant K", "requant V^T"};
    static const uint8_t oeng[O_END - O_NORM] = {1, 0, 0, 0, 1, 1, 1};
    static const char *const tname[T_END - T_XT] = {
        "xDMA x -> x^T", "norm^T: lanewise,rsqrt,mul", "xDMA y^T -> A (xpose)", "quant n1",
        "GEMM Q   -> A", "GEMM K   -> A", "GEMM V^T -> A (swapped)", "requant Q",
        "requant K", "requant V^T"};
    static const uint8_t teng[T_END - T_XT] = {2, 1, 2, 1, 0, 0, 0, 1, 1, 1};

    printf("[QKV] T=%u d=%u, warm. Timeline in cycles from the path's first op.\n", T_TOK,
           D_MODEL);
    printf("[QKV] ---- PATH H: HeMAiA's graph + the minimum fix to reach FA ----\n");
    uint32_t mh = report_path(tm, 'H', 0, H_END, hname, heng);
    printf("[QKV] ---- PATH O: layouts folded into the AGUs ----\n");
    uint32_t mo = report_path(tm, 'O', O_NORM, O_END, oname, oeng);
    printf("[QKV] ---- PATH T: transposed RMSNorm (xDMA transposes), then as O ----\n");
    uint32_t mt = report_path(tm, 'T', T_XT, T_END, tname, teng);
    printf("[QKV] === makespan H %u | O %u (%u%% less) | T %u (%u%% less)\n", mh, mo,
           mh ? 100u * (mh - mo) / mh : 0u, mt, mh ? 100u * (mh - mt) / mh : 0u);
    printf("[QKV] FA  S^T = K.Q^T  %u cc | O^T = V^T.P^T  %u cc  (cfg + run + wait)\n",
           TE(F_QK) - TS(F_QK), TE(F_PV) - TS(F_PV));

    // ============================================================ checks
    int err = 0;
    uint32_t w, bad;

    // 1) norm1 against the golden: the rsqrt ROM is within 1 ULP, so allow 2 (HeMAiA's own
    //    norm1 check, in ULPs rather than an absolute 0.05).
    bad = cmp_f16((uint16_t *)n1_p, gl_n1, T_TOK * D_MODEL, &w);
    printf("[QKV] norm1 (H, packed) vs golden: %u/%u differ, worst %u ULP (%s)\n", bad,
           T_TOK * D_MODEL, w, w > 2 ? "FAIL" : "ok");
    if (w > 2) err++;

    // 2) the quantised norm. Against the golden it can move by 1 wherever the ROM's ULP
    //    crosses a rounding boundary; path O against path H must be EXACT -- fold 3 is a
    //    reordering of the same arithmetic.
    bad = cmp_i8(n1q_h, gl_n1q, I8B, &w);
    printf("[QKV] n1_q A/i8 (H) vs golden: %u/%u differ, worst %u\n", bad, I8B, w);
    if (w > 1) err++;
    bad = cmp_i8(n1q_o, n1q_h, I8B, &w);
    printf("[QKV] n1_q A/i8: O (one A-order pass) vs H (3 stages): %s (%u differ)\n",
           bad ? "FAIL" : "bit-exact", bad);
    if (bad) err++;
    bad = cmp_i8(n1q_t, n1q_h, I8B, &w);
    printf("[QKV] n1_q A/i8: T (transposed norm) vs H: %s (%u differ, worst %u)\n",
           bad ? "DIFFERS" : "bit-exact", bad, w);
    if (w > 1) err++;

    // 3) HeMAiA's proj_q check: the dequantised projection in D-layout.
    bad = cmp_f16((uint16_t *)dqh, gl_pq, T_TOK * D_MODEL, &w);
    printf("[QKV] proj_q_dq D/f16 (H) vs golden: %u/%u differ, worst %u ULP\n", bad,
           T_TOK * D_MODEL, w);

    // 4) FA's operands. O vs H must be bit-exact: different GEMM orientation for V,
    //    different writers, different pass counts -- same numbers.
    static const char *const qn[3] = {"Q8 (B of Q^T)", "K8 (A)", "V^T8 (A)"};
    for (uint32_t t = 0; t < 3; t++) {
        uint32_t b1 = cmp_i8(qkv_o + t * I8B, qkv_h + t * I8B, I8B, &w);
        uint32_t w3, b3 = cmp_i8(qkv_t + t * I8B, qkv_h + t * I8B, I8B, &w3);
        uint32_t w2, b2 = cmp_i8(qkv_o + t * I8B, gl_qkv + t * I8B, I8B, &w2);
        printf("[QKV] %-14s O vs H: %s | T vs H: %s | vs golden: %u differ, worst %u\n",
               qn[t], b1 ? "FAIL" : "bit-exact", b3 ? "FAIL" : "bit-exact", b2, w2);
        if (b1 || b3) err++;
    }

    // 5) FlashAttention consumes them. Recompute sampled outputs from the DEVICE's own
    //    operand bytes, through the layouts' index maps.
    const int8_t *q8 = qkv_o, *k8 = qkv_o + I8B, *vt8 = qkv_o + 2 * I8B;
    uint32_t sbad = 0, sn = 0;
    for (uint32_t key = 0; key < T_TOK; key += 7)
        for (uint32_t q = 0; q < T_TOK; q++) {
            int32_t acc = 0;
            for (uint32_t f = 0; f < D_MODEL; f++)
                acc += (int32_t)k8[a_at(key, f, D_MODEL)] * (int32_t)q8[a_at(q, f, D_MODEL)];
            sn++;
            if (s16[key * T_TOK + q] != i32_to_f16(acc, FA_S_SHIFT)) {
                if (sbad < 4)
                    printf("[QKV]   S^T[%u][%u] = %04x, recomputed %04x\n", key, q,
                           s16[key * T_TOK + q], i32_to_f16(acc, FA_S_SHIFT));
                sbad++;
            }
        }
    printf("[QKV] FA S^T=K.Q^T vs recompute from device K8,Q8: %s (%u/%u sampled)\n",
           sbad ? "FAIL" : "exact", sbad, sn);
    if (sbad) err++;
    bad = cmp_f16(s16, gl_s, T_TOK * T_TOK, &w);
    printf("[QKV] FA S^T vs golden: %u/%u differ, worst %u ULP\n", bad, T_TOK * T_TOK, w);

    uint32_t obad = 0, on = 0;
    for (uint32_t f = 0; f < D_MODEL; f += 9)
        for (uint32_t q = 0; q < T_TOK; q++) {
            int32_t acc = 0;
            for (uint32_t key = 0; key < T_TOK; key++)
                acc += (int32_t)vt8[a_at(f, key, T_TOK)] * (int32_t)ptb[b_at(key, q, T_TOK)];
            on++;
            if (o16[f * T_TOK + q] != i32_to_f16(acc, 0u)) {
                if (obad < 4)
                    printf("[QKV]   O^T[%u][%u] = %04x, recomputed %04x\n", f, q,
                           o16[f * T_TOK + q], i32_to_f16(acc, 0u));
                obad++;
            }
        }
    printf("[QKV] FA O^T=V^T.P^T vs recompute from device V^T8: %s (%u/%u sampled)\n",
           obad ? "FAIL" : "exact", obad, on);
    if (obad) err++;
    bad = cmp_f16(o16, gl_o, D_MODEL * T_TOK, &w);
    printf("[QKV] FA O^T vs golden: %u/%u differ, worst %u ULP\n", bad, D_MODEL * T_TOK, w);

    // 6) the HeMAiA rmsnorm's blocked routes, each against a result the paths above
    //    produced by the long way round -- through the element index maps, not a golden.
    printf("[QKV] HeMAiA rmsnorm routes, one kernel each (cc incl. programming):\n");
    printf("[QKV]   row_major->A i8 %u | f16 %u   col_major->B i8 %u | f16 %u\n",
           TE(V_A8) - TS(V_A8), TE(V_A16) - TS(V_A16), TE(V_B8) - TS(V_B8),
           TE(V_B16) - TS(V_B16));
    bad = cmp_i8(va8, n1q_h, I8B, &w);
    printf("[QKV]   row_major->A int8 vs H's rmsnorm+reshape+quant: %s (%u differ)\n",
           bad ? "FAIL" : "bit-exact", bad);
    if (bad) err++;
    bad = cmp_f16((uint16_t *)va16, (uint16_t *)n1a, T_TOK * D_MODEL, &w);
    printf("[QKV]   row_major->A fp16 vs H's rmsnorm+reshape: %s (%u differ)\n",
           bad ? "FAIL" : "bit-exact", bad);
    if (bad) err++;
    {
        uint32_t b8 = 0, b16 = 0;
        for (uint32_t t = 0; t < T_TOK; t++)
            for (uint32_t f = 0; f < D_MODEL; f++) {
                uint32_t bi = b_at(t, f, T_TOK);  // B of y [K = T, N = d]
                if (vb8[bi] != n1q_h[a_at(t, f, D_MODEL)]) b8++;
                if (((uint16_t *)vb16)[bi] != ((uint16_t *)n1_p)[t * D_MODEL + f]) b16++;
            }
        printf("[QKV]   col_major->B int8 vs H's n1_q through the B map: %s (%u differ)\n",
               b8 ? "FAIL" : "bit-exact", b8);
        printf("[QKV]   col_major->B fp16 vs H's norm1 through the B map: %s (%u differ)\n",
               b16 ? "FAIL" : "bit-exact", b16);
        if (b8 || b16) err++;
    }

    // 7) THE SAME PASS FOR OTHER MESHES. The SIMD writes a layout and never talks to the
    //    array, so the operand layouts of meshes this cluster does not have are testable
    //    here. xs/pl (the TAP re-laid x and its 1/rms plane) and ytp (y^T at 72 B) are
    //    still in TCDM from the routes above; each case writes one operand into vb16 and is
    //    checked element by element against the index map.
    {
        uint32_t fails = 0;
        printf("[QKV] blocked pass on other meshes (cc incl. programming; 0 differ = exact):\n");
        for (uint32_t n = 0; n < N_BLK_CASES; n++) {
            const blk_case_t *c = &blk_cases[n];
            volatile uint32_t *o32 = (volatile uint32_t *)vb16;
            for (uint32_t k = 0; k < F16B / 4u; k++) o32[k] = 0xA5A5A5A5u;  // no stale hits
            uint32_t t0 = snrt_mcycle();
            if (c->kind == 0)
                pass_blocked_desc(HUNG, xs, pl, vb16, c, SCALE_N1_BITS);
            else
                pass_blocked_desc(HUNG, ytp, (void *)0, vb16, c, SCALE_N1_BITS);
            uint32_t cyc = snrt_mcycle() - t0;
            uint32_t bad = 0;
            for (uint32_t t = 0; t < T_TOK; t++)
                for (uint32_t f = 0; f < D_MODEL; f++) {
                    // A of y: S = y, (i, j) = (t, f). B of y: S = y^T, (i, j) = (f, t).
                    uint32_t at = c->kind == 0 ? blk_at(t, f, D_MODEL, c->mb, c->ku)
                                               : blk_at(f, t, T_TOK, c->mb, c->ku);
                    if (c->out_i8) {
                        if (((int8_t *)vb16)[at] != n1q_h[a_at(t, f, D_MODEL)]) bad++;
                    } else if (((uint16_t *)vb16)[at] != ((uint16_t *)n1_p)[t * D_MODEL + f]) {
                        bad++;
                    }
                }
            printf("[QKV]   mesh (%2u,%2u,%2u) %c %s  %u task%s  %4u cc  %u differ\n", c->mu,
                   c->ku, c->nu, c->kind ? 'B' : 'A', c->out_i8 ? "i8 " : "f16", c->reps,
                   c->reps > 1 ? "s" : " ", cyc, bad);
            if (bad) fails++;
        }
        printf("[QKV]   %u/%u mesh cases bit-exact\n", N_BLK_CASES - fails, N_BLK_CASES);
        if (fails) err++;
    }

    if (*HUNG || TMO) {
        printf("[QKV] FAIL: %s%u bounded wait(s) ran out; every figure after is meaningless\n",
               *HUNG ? "a SIMD task never retired; " : "", TMO);
        err++;
    }
    printf(err ? "[QKV] FAIL\n" : "[QKV] PASS\n");
    return err != 0;
}
