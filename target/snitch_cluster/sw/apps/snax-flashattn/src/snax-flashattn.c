// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// FlashAttention inner loop on the four-engine cluster, instrumented for
// hardware utilisation.
//
// One (query tile, KV tile) pair: Br = 32, Bc = d = 64. d == Bc makes S = Q.K^T
// and O = P.V the same shape, so ONE streamer configuration drives both matmuls
// and only the five base pointers move between them.
//
//   GEMM (hart 0)   S^T   = K.Q^T                INT8 x INT8 -> INT32
//   SIMD (hart 1)   S16   = fp16(S^T)            Int32ToFp16    ) FUSED: one pass,
//                   m     = lanewise max(S16)    StreamReduce   ) tap appends m
//                   -m    = replicate, negate    StreamMap LINEAR, a = -1
//                   sm    = S16 + (-m)           StreamElementwise ADD, 2 operands
//                   P8^T  = int8(exp(sm))        StreamMap EXP + Fp16ToInt8, FUSED
//   GEMM (hart 0)   O^T   = V^T.P^T
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
//    cores and drag harts 2 and 3 in. Within a tile the five SIMD tasks are
//    fired back to back with no wait between them; see the note in the loop.
//
// 2. THE TRANSPOSE. Everything is computed in [Bc, Br] rather than [Br, Bc], so
//    one beat holds one value per QUERY ROW and the softmax reduction runs along
//    BEATS instead of across lanes. See the block above the tile constants: it is
//    what lets StreamReduce skip its horizontal fold entirely.
//
// WHAT IS AND IS NOT VALIDATED. The softmax is checked against an exact
// invariant in two halves -- sm must contain +-0.0 and P8^T must contain 1, for
// every query row -- which together cover the reduce, the subtract, the
// exponential and the quantiser. Every KV tile is fed the same K, so the final
// tile stands for all of them. See the check itself for why it is split.
//
// The second GEMM is shape- and dataflow-accurate but its operands are not a
// numerically validated attention output, and O is overwritten per tile rather
// than rescaled and accumulated the way real FlashAttention does. Treat O as a
// cycle measurement -- one P.V matmul per KV tile, which is the right load --
// not as a result.

#include <stdint.h>
#include "data.h"
#include "snax-core-roles.h"
#include "snax-gemmx-lib.h"
#include "snax-gemmx-params.h"
#include "snax-simd-lib.h"
#include "snrt.h"

#define BR 32  // query rows in this tile
#define BC 128  // key columns in this tile  (d == BC, so both matmuls have one shape)
#define NKV 4  // key/value tiles to stream through the pipeline

// EVERYTHING IS TRANSPOSED. The score tile is stored [Bc, Br], not [Br, Bc]:
//
//   S^T = (Q.K^T)^T = K.Q^T
//
// which is the same GEMM with its two operands swapped -- M and N exchanged in
// params.hjson, no extra pass, no transposer. One beat of S^T is 32 FP16 = one
// value per QUERY ROW at a fixed key, so:
//
//   rowmax   = LANEWISE MAX over the Bc beats. The reduction runs ALONG BEATS,
//              which is the accumulator StreamReduce already keeps for free, so
//              the horizontal fold, its treeBuf serialisation and the scalar
//              drain are all skipped. One task, one output beat, no per-row
//              bubble -- against ~40 cycles per row on the folding path.
//   m        = ONE beat (32 lanes = 32 query rows), not BR scalars.
//   exp(S-m) = the same broadcast/add/exp chain, but the broadcast operand is a
//              single beat replicated, not a per-row scalar re-armed BR times.
//
// This is Track C 7.3 of the decoupling plan. The measured cost before it was
// rowmax = 1296 of 2407 SIMD engine cycles per KV tile, i.e. 54% of all engine
// time spent folding 32 short rows.
#define SBEATS BC        // S^T beats: one per key, 32 query lanes each
#define PBEATS (BC / 2)  // after Fp16ToInt8 halves them

// A spin that cannot hang the simulation for ever. A deadlock here is a real
// possibility -- the two cores are hand-synchronised -- and an infinite loop in
// Verilator just burns wall-clock with no diagnosis.
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

// The twelve task geometries. In TCDM, not on the stack and not in .bss:
// program_fast() reads every field of two of these per task, so a DRAM-backed
// .bss object costs an L3 round trip per field -- 4500 cycles a run when this
// was plain `static`. Only the SIMD core touches them.
SNRT_L1_DATA static snax_simd_shape_t shapes[32];

// SNRT_L1_DATA lives in .l1, which is NOLOAD -- it is NOT zero-initialised. A
// shape that is declared but never filled therefore programs the AGU from
// whatever was in TCDM, and the task still completes: it reads and writes the
// wrong addresses and the result is silently wrong rather than faulted. That
// cost a full bisect (the reduce looked broken in the cluster while passing
// every unit test, because its output shape was garbage). Clear them once, so
// an unfilled shape is an empty task the library REFUSES instead.
static inline void snax_simd_shapes_clear(snax_simd_shape_t *sh, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint8_t *b = (uint8_t *)&sh[i];
        for (uint32_t k = 0; k < sizeof(snax_simd_shape_t); k++) b[k] = 0;
    }
}

static inline uint16_t fp16_at(volatile uint8_t *p, uint32_t i) {
    return ((volatile uint16_t *)p)[i];
}

// Configure the GEMM once. Every matmul in the loop has the same shape, so the
// streamer bounds, strides, remap indices and the accelerator CSRs are written
// exactly once here and only the five base pointers move per tile. Re-issuing
// the full config per matmul was ~1000 cycles, nearly all of it instruction
// fetch rather than the ~84 CSR writes themselves.
//
// The fifteen streamer arrays live in THIS frame rather than main's on purpose:
// SNRT_LOG2_STACK_SIZE is 10, so a hart has 1 KiB, and carrying these alongside
// main's shape structs and a printf frame overflowed into the neighbouring
// hart's stack. main then returned with a corrupted s2, which surfaced as a
// misaligned amoadd in snrt_main's exit path -- a trap with no visible relation
// to its cause.
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
    // TWO spatial strides: the C port declares spatial_bounds [[8, 4]] and the
    // streamer reads S_STRIDE_NUM_READER_WRITER_0 = 2 of them. A 1-element array
    // here fed the second one from off the end of the stack.
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
    uint32_t top = ((uint32_t)delta_local_d32 + BR * BC * 4 + 63u) & ~63u;
    const uint32_t BEAT = SIMD_BEAT_BYTES;

    uint8_t *negmS = l1 + top;  top += BEAT;          // -m_new   ] latch for P = exp(S-m)
    uint8_t *s16   = l1 + top;  top += SBEATS * BEAT; // S^T      ]
    uint8_t *rmax  = l1 + top;  top += BEAT;          // tapped rowmax, and later -m_new
    uint8_t *mrun  = l1 + top;  top += BEAT;          // running m   ] pair for max(m_old,rowmax)
    uint8_t *mnew  = l1 + top;  top += BEAT;
    uint8_t *delta = l1 + top;  top += BEAT;          // m_old - m_new
    uint8_t *corrL = l1 + top;  top += BEAT;          // exp(delta)  ] latch for corr*l_old
    uint8_t *lrun  = l1 + top;  top += BEAT;          // running l   ]
    // Task 11 writes its two beats as [pad][corr*l_old], and task 13 then reads
    // [rowsum][corr*l_old] -- so the pad slot and the rowsum copy are the SAME beat,
    // written by 11 and overwritten by 12. That is what puts them adjacent to lsc.
    uint8_t *rsum2 = l1 + top;  top += BEAT;          // pad, then the rowsum copy ]
    uint8_t *lsc   = l1 + top;  top += BEAT;          // corr*l_old                ] pair for l_new
    uint8_t *lnew  = l1 + top;  top += BEAT;
    uint8_t *smpad = l1 + top;  top += BEAT;
    uint8_t *sm    = l1 + top;  top += SBEATS * BEAT; // S^T - m_new
    uint8_t *p16   = l1 + top;  top += SBEATS * BEAT; // P = exp(S-m)
    uint8_t *rsum  = l1 + top;  top += BEAT;          // tapped rowsum, right after p16
    uint8_t *corrO = l1 + top;  top += BEAT;          // exp(delta)  ] latch for O *= corr
    uint8_t *oacc  = l1 + top;  top += SBEATS * BEAT; // O^T         ]
    uint32_t oacc32 = top;      top += BR * BC * 4;   // O^T as the GEMM sees it: INT32
    uint32_t d32_b = top;       top += BR * BC * 4;   // second S^T buffer
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

        // C is the GEMM's accumulator input. The matmul data generator fills it
        // with a RANDOM bias, which attention does not have -- and which dominated
        // the scores: with Q,K bounded to +-16 the products cannot exceed 16384,
        // yet S came out at -78529, i.e. essentially all C. Zero it.
        for (uint32_t i = 0; i < (uint32_t)(M * N * meshRow * meshCol); i++)
            local_c[i] = 0;

        // Bound the operands so the scores stay inside FP16.
        //
        // A score is a sum of d = 64 INT8 products, so with full-range operands it
        // reaches ~1e6 -- far past FP16's 65504. Int32ToFp16 then returns inf for
        // those rows, exp(inf - inf) is NaN, and the softmax invariant fails on
        // exactly the overflowing rows (3 of 32 were, before this). Real attention
        // scales scores by 1/sqrt(d) for the same reason; here it is cheaper to
        // bound the inputs. >>2 was NOT enough: it gives |q|,|k| <= 32, so
        // |S| <= 32*32*64 = 65536 -- just past FP16's 65504, and three rows of 32
        // did overflow to -inf (diagnosed as hw_max == sw_max == 0xfc00, i.e. the
        // reduce was right and the conversion had already saturated). >>3 gives
        // |q|,|k| <= 16 and |S| <= 16384, comfortably inside the format.
        for (uint32_t i = 0; i < (uint32_t)(M * K * meshRow * tileSize); i++)
            local_a[i] >>= 3;
        for (uint32_t i = 0; i < (uint32_t)(N * K * tileSize * meshCol); i++)
            local_b[i] >>= 3;

        // Online-softmax initial state. m starts at the most negative FINITE FP16
        // (-65504) rather than -inf: max(m, rowmax) is then just rowmax, and
        // exp(m - m_new) underflows to 0 as it should, without inf arithmetic
        // anywhere near the exponential.
        for (uint32_t i = 0; i < SIMD_BEAT_BYTES / 2; i++) {
            ((volatile uint16_t *)mrun)[i] = 0xFBFFu;   // -65504
            ((volatile uint16_t *)lrun)[i] = 0x0000u;   // l = 0
        }
        for (uint32_t i = 0; i < SBEATS * SIMD_BEAT_BYTES / 2; i++)
            ((volatile uint16_t *)oacc)[i] = 0x0000u;   // O = 0 (FP16 half)
        for (uint32_t i = 0; i < (uint32_t)(BR * BC); i++)
            ((volatile int32_t *)(l1 + oacc32))[i] = 0;  // O = 0 (INT32 half)
        // The GEMM's C buffer is the O accumulator input; the rescale writes into it.
        for (uint32_t i = 0; i < (uint32_t)(M * N * meshRow * meshCol); i++)
            local_c[i] = 0;

        sync[0] = 0;
        sync[1] = 0;
    }
    snrt_cluster_hw_barrier();

    // The kernel's tile constants must agree with what data.h was generated from.
    // d == BC in particular is load-bearing: it is what makes S^T = K.Q^T and
    // O^T = V^T.P^T the same shape, so one streamer config drives both matmuls and
    // only the five base pointers move. Break it and the second matmul silently
    // uses the first one's bounds.
    if (snax_is_gemm_core()) {
        if (M * meshRow != BC || N * meshCol != BR || K * tileSize != BC) {
            printf("tile mismatch: M*meshRow=%ld (BC=%d) N*meshCol=%ld (BR=%d) "
                   "K*tileSize=%ld (d must equal BC)\n",
                   (long)(M * meshRow), BC, (long)(N * meshCol), BR,
                   (long)(K * tileSize));
            cfg_err++;
        }
    }

    if (snax_is_gemm_core()) gemm_configure_once();

    // The six softmax tasks likewise have fixed geometry. Build every shape up
    // front so the loop body is arm + program_fast + launch and nothing else.
    snax_simd_shape_t *sh = shapes;
    if (snax_is_simd_core()) {
        snax_simd_shapes_clear(shapes, 32);
        const uint32_t D32B = BR * BC * 4 / SIMD_BEAT_BYTES;  // INT32 beats of one S^T tile

        // 1  convert + rowmax, one pass: D32 -> s16[SBEATS] then the tapped rowmax.
        snax_simd_shape_flat(&sh[0], l1 + d32_delta[0], D32B);
        snax_simd_shape_flat(&sh[1], s16, SBEATS + 1);
        // 2  m_new = max(m_old, rowmax): LANEWISE over the adjacent pair [rmax][mrun].
        snax_simd_shape_flat(&sh[2], rmax, 2);
        snax_simd_shape_flat(&sh[3], mnew, 1);
        // 3,4  -m_new, written TWICE: once before s16 (latch for P) and once into the
        //      rmax slot, which task 2 has already consumed, so it pairs with mrun.
        snax_simd_shape_flat(&sh[4], mnew, 1);
        snax_simd_shape_flat(&sh[5], negmS, 1);
        snax_simd_shape_flat(&sh[6], mnew, 1);
        snax_simd_shape_flat(&sh[7], rmax, 1);
        // 5  delta = m_old - m_new: LANEWISE ADD over [-m_new][m_old].
        snax_simd_shape_flat(&sh[8], rmax, 2);
        snax_simd_shape_flat(&sh[9], delta, 1);
        // 6,7  corr = exp(delta), twice: one before l_old, one before O.
        snax_simd_shape_flat(&sh[10], delta, 1);
        snax_simd_shape_flat(&sh[11], corrL, 1);
        snax_simd_shape_flat(&sh[12], delta, 1);
        snax_simd_shape_flat(&sh[13], corrO, 1);
        // 8  S^T - m_new: sticky, latch negmS then the whole tile.
        snax_simd_shape_flat(&sh[14], negmS, 1 + SBEATS);
        snax_simd_shape_flat(&sh[15], smpad, 1 + SBEATS);
        // 9  P = exp(sm) AND rowsum, one pass: Map(EXP) feeding Reduce(ADD|LANEWISE|TAP).
        snax_simd_shape_flat(&sh[16], sm, SBEATS);
        snax_simd_shape_flat(&sh[17], p16, SBEATS + 1);
        // 10 quantise P for the second matmul.
        snax_simd_shape_flat(&sh[18], p16, SBEATS);
        snax_simd_shape_flat(&sh[19], l1 + p8_delta[0], PBEATS);
        // 11 corr * l_old: sticky MUL over [corrL][lrun].
        snax_simd_shape_flat(&sh[20], corrL, 2);
        snax_simd_shape_flat(&sh[21], rsum2, 2);
        // 12 copy the tapped rowsum next to corr*l_old (adjacency, see the layout note).
        snax_simd_shape_flat(&sh[22], rsum, 1);
        snax_simd_shape_flat(&sh[23], rsum2, 1);
        // 13 l_new = corr*l_old + rowsum: LANEWISE ADD over [rsum2][lsc]... written to lnew.
        snax_simd_shape_flat(&sh[24], rsum2, 2);
        snax_simd_shape_flat(&sh[25], lnew, 1);
        // 14 O *= corr: sticky MUL, latch corrO then the O tile, into the GEMM's C buffer.
        snax_simd_shape_flat(&sh[26], corrO, 1 + SBEATS);
        snax_simd_shape_flat(&sh[27], corrO, 1 + SBEATS);  // in place; corrO is consumed
        // 15,16 commit the running state for the next KV tile.
        snax_simd_shape_flat(&sh[28], mnew, 1);
        snax_simd_shape_flat(&sh[29], mrun, 1);
        snax_simd_shape_flat(&sh[30], lnew, 1);
        snax_simd_shape_flat(&sh[31], lrun, 1);
    }
    snrt_cluster_hw_barrier();

    // ======================= the pipelined loop ==============================
    if (snax_is_gemm_core()) {
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
                // C MUST be restated, not left at -1 ("keep current"): the O matmul
                // below points C at oacc32 for its in-place accumulation, and without
                // this S would be computed as K.Q^T + O_accumulated from the second
                // tile onward -- which showed up as the running max drifting upward.
                set_gemmx_bases(delta_local_a, delta_local_b, -1, delta_local_c,
                                d32_delta[j & 1]);
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
                set_gemmx_bases(delta_local_a, p8_delta[(j - 1) & 1], -1,
                                (int32_t)oacc32, (int32_t)oacc32);
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

            // 1  convert + rowmax in one pass (tap appends the per-lane maxima)
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_INT32TOFP16CONVERTER_512) |
                                        (1u << SIMD_EXT_STREAMREDUCE));
            snax_write_simd_cfg_reg(SIMD_EXT_INT32TOFP16CONVERTER_512_CSR, 0);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR, SBEATS);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1,
                                    SIMD_RED_MAX | SIMD_RED_LANEWISE | SIMD_RED_TAP);
            snax_simd_program_fast(&sh[0], &sh[1]);
            snax_simd_fire();

            // 2  m_new = max(m_old, rowmax)
            snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                           SIMD_RED_MAX | SIMD_RED_LANEWISE);
            snax_simd_program_fast(&sh[2], &sh[3]);
            snax_simd_fire();

            // 3,4  -m_new, to both places a latch is needed
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR,
                           snax_simd_f32_neg(SIMD_F32_ONE), 0, SIMD_FUNC_LINEAR);
            snax_simd_program_fast(&sh[4], &sh[5]);
            snax_simd_fire();
            snax_simd_program_fast(&sh[6], &sh[7]);
            snax_simd_fire();

            // 5  delta = m_old - m_new
            snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                           SIMD_RED_ADD | SIMD_RED_LANEWISE);
            snax_simd_program_fast(&sh[8], &sh[9]);
            snax_simd_fire();

            // 6,7  corr = exp(delta), to both places a latch is needed
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_ONE,
                           0, SIMD_FUNC_EXP);
            snax_simd_program_fast(&sh[10], &sh[11]);
            snax_simd_fire();
            snax_simd_program_fast(&sh[12], &sh[13]);
            snax_simd_fire();

            // 8  S^T - m_new
            snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE,
                           SIMD_EXT_STREAMELEMENTWISE_CSR, 1,
                           SIMD_EW_ADD | SIMD_EW_STICKY_B);
            snax_simd_program_fast(&sh[14], &sh[15]);
            snax_simd_fire();

            // 9  P = exp(sm) AND rowsum, one pass
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMMAP) |
                                        (1u << SIMD_EXT_STREAMREDUCE));
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 0, SIMD_F32_ONE);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 1, 0);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMMAP_CSR + 2, SIMD_FUNC_EXP);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR, SBEATS);
            snax_write_simd_cfg_reg(SIMD_EXT_STREAMREDUCE_CSR + 1,
                                    SIMD_RED_ADD | SIMD_RED_LANEWISE | SIMD_RED_TAP);
            snax_simd_program_fast(&sh[16], &sh[17]);
            snax_simd_fire();

            // 10 quantise P
            snax_simd_use1(SIMD_EXT_FP16TOINT8, SIMD_EXT_FP16TOINT8_CSR, SIMD_F32_ONE);
            snax_simd_program_fast(&sh[18], &sh[19]);
            snax_simd_fire();

            // 11 corr * l_old
            snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE,
                           SIMD_EXT_STREAMELEMENTWISE_CSR, 1,
                           SIMD_EW_MUL | SIMD_EW_STICKY_B);
            snax_simd_program_fast(&sh[20], &sh[21]);
            snax_simd_fire();

            // 12 copy the rowsum next to it (sticky latch adjacency)
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_ONE,
                           0, SIMD_FUNC_LINEAR);
            snax_simd_program_fast(&sh[22], &sh[23]);
            snax_simd_fire();

            // 13 l_new = corr*l_old + rowsum
            snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                           SIMD_RED_ADD | SIMD_RED_LANEWISE);
            snax_simd_program_fast(&sh[24], &sh[25]);
            snax_simd_fire();

            // 14 O *= corr, straight into the GEMM's C buffer so the matmul adds it
            snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE,
                           SIMD_EXT_STREAMELEMENTWISE_CSR, 1,
                           SIMD_EW_MUL | SIMD_EW_STICKY_B);
            snax_simd_program_fast(&sh[26], &sh[27]);
            snax_simd_fire();

            // 15,16 commit the running state
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_ONE,
                           0, SIMD_FUNC_LINEAR);
            snax_simd_program_fast(&sh[28], &sh[29]);
            snax_simd_fire();
            snax_simd_program_fast(&sh[30], &sh[31]);
            snax_simd_fire();


            // Bounded: an engine that never retires is a real possibility while the
            // armed combinations are still being explored, and an unbounded spin turns
            // that into a 900-second simulator timeout with no diagnosis. This reports
            // it as a failure with the counters instead.
            {
                uint32_t want = snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR);
                uint32_t spins = 0;
                while (snax_read_simd_cfg_reg(SIMD_FINISHED_TASK_PTR) < want) {
                    if (++spins > 40000u) {
                        printf("SIMD STALL tile %u: submitted=%u finished=%u status=%08x\n",
                               j, want, snax_read_simd_cfg_reg(SIMD_FINISHED_TASK_PTR),
                               snax_read_simd_cfg_reg(SIMD_STATUS));
                        timeouts++;
                        break;
                    }
                }
            }
            if (timeouts) break;
            sync[1] = j + 1;
        }
        simd_cycles = snax_simd_busy_cycles() - simd_busy0;
        simd_wall = snrt_mcycle() - s0;
    }
    snrt_cluster_hw_barrier();

    // ---- invariants for the FULL algorithm ----------------------------------
    //
    // Three independent checks, each pinning a different part of the online softmax:
    //
    //   sm   contains exactly +-0.0 per query row   the running max and the subtract
    //   P^T  contains exactly 1     per query row   exp(0) = 1, and the quantiser
    //   l    == NKV * rowsum(one tile)              the TAPPED rowsum and the l recurrence
    //
    // The third is the new one and it is the sharpest: every KV tile is fed the same K,
    // so after the first tile the running max stops changing, corr = exp(0) = 1, and l
    // must accumulate exactly NKV identical row sums. If the tap, the correction or the
    // l update were wrong, l would drift.
    if (snax_is_simd_core()) {
        volatile int8_t *p8t = (volatile int8_t *)(l1 + p8_delta[(NKV - 1) & 1]);
        for (uint32_t i = 0; i < BR; i++) {   // i = query row = lane
            int found_zero = 0, found_one = 0;
            for (uint32_t j = 0; j < SBEATS; j++) {
                uint16_t v = fp16_at(sm + j * SIMD_BEAT_BYTES, i);
                if (v == 0x0000u || v == 0x8000u) found_zero = 1;
                if (p8t[j * BR + i] == 1) found_one = 1;
            }
            if (!found_zero || !found_one) {
                printf("query %2u: zero=%d one=%d m=%04x rowsum=%04x l=%04x\n", i,
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
            const uint32_t shift = (NKV == 4) ? 2u : (NKV == 2 ? 1u : 3u);
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
        printf("  tile             Br=%d Bc=d=%d, %d KV tiles, mesh %dx%dx%d\n",
               BR, BC, NKV, meshRow, tileSize, meshCol);
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
