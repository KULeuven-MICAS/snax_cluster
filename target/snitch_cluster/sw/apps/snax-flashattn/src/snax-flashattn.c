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
//   SIMD (hart 1)   S16   = fp16(S^T)            Int32ToFp16
//                   m     = lanewise max(S16)    StreamReduce MAX | LANEWISE
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
#define BC 64  // key columns in this tile  (d == BC, so both matmuls have one shape)
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
SNRT_L1_DATA static snax_simd_shape_t shapes[12];

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
    int32_t Cslstride[] = {Cslstride0};
    int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2, Ctlbound3};
    int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};
    int32_t D32slstride[] = {D32slstride0};
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

    // Scratch laid out at run time from where data.h actually ends, not from
    // hardcoded offsets: swapping M and N moves every delta_local_*, and a
    // hardcoded map silently overlapped once already.
    uint32_t top = ((uint32_t)delta_local_d32 + BR * BC * 4 + 63u) & ~63u;
    uint8_t *s16 = l1 + top;    top += SBEATS * SIMD_BEAT_BYTES;   // S^T, FP16
    uint8_t *stat = l1 + top;   top += SIMD_BEAT_BYTES;            // m, ONE beat
    uint8_t *mrep = l1 + top;   top += SBEATS * SIMD_BEAT_BYTES;   // -m replicated
    uint8_t *sm = l1 + top;     top += SBEATS * SIMD_BEAT_BYTES;   // S^T - m
    uint32_t d32_b = top;       top += BR * BC * 4;                // 2nd S^T buffer
    uint32_t p8_0 = top;        top += PBEATS * SIMD_BEAT_BYTES;   // P^T, INT8
    uint32_t p8_1 = top;        top += PBEATS * SIMD_BEAT_BYTES;
    uint32_t oacc = top;        top += BR * BC * 4;                // O^T, per tile
    volatile uint32_t *pub = (volatile uint32_t *)(l1 + top); top += 256;
    volatile uint32_t *sync = (volatile uint32_t *)(l1 + top);

    // Double buffers, indexed by tile parity.
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

        sync[0] = 0;
        sync[1] = 0;
    }
    snrt_cluster_hw_barrier();

    if (snax_is_gemm_core()) gemm_configure_once();

    // The six softmax tasks likewise have fixed geometry. Build every shape up
    // front so the loop body is arm + program_fast + launch and nothing else.
    snax_simd_shape_t *cv_in = &shapes[0], *cv_out = &shapes[1];
    snax_simd_shape_t *mx_in = &shapes[2], *mx_out = &shapes[3];
    snax_simd_shape_t *b_in = &shapes[4], *b_out = &shapes[5];
    snax_simd_shape_t *ew_in = &shapes[6], *ew_out = &shapes[7];
    snax_simd_shape_t *ex_in = &shapes[8], *ex_out = &shapes[9];
    snax_simd_shape_t *qz_in = &shapes[10], *qz_out = &shapes[11];
    if (snax_is_simd_core()) {
        // (a) INT32 S^T -> FP16: 128 beats in, SBEATS out.
        snax_simd_shape_flat(cv_in, l1 + d32_delta[0], BR * BC * 4 / SIMD_BEAT_BYTES);
        snax_simd_shape_flat(cv_out, s16, SBEATS);

        // (b) the row maxima, LANEWISE: SBEATS beats in, ONE beat out.
        //
        // operandCount is the whole tile, not a row: the "row" being reduced is
        // the key axis, which runs along beats. Lane i of the result is the max
        // over all Bc keys for query row i. No fold, no per-row bubble, and the
        // operandCount==1 assert that forced the old 2-beat row shape is moot.
        snax_simd_shape_flat(mx_in, s16, SBEATS);
        snax_simd_shape_flat(mx_out, stat, 1);

        // (c) exp(S^T - m). E1 replicates the single m beat SBEATS times with a
        // stride-0 inner loop, negating it via the LINEAR `a` coefficient.
        snax_simd_shape_2d(b_in, stat, SBEATS, 0, 1, 0);
        snax_simd_shape_flat(b_out, mrep, SBEATS);
        // E2 feeds both operands from ONE address stream: inner loop of 2
        // stepping by (mrep - s16), which must be POSITIVE -- the AGU stride is
        // unsigned, and an operand below its partner wraps the address, reads
        // outside TCDM and writes X while the task still reports complete.
        snax_simd_shape_2d(ew_in, s16, 2, (uint32_t)(mrep - s16), SBEATS,
                           SIMD_BEAT_BYTES);
        snax_simd_shape_flat(ew_out, sm, SBEATS);
        // E3 exponentiates AND quantises in one armed task: StreamMap is
        // extension 1, Fp16ToInt8 is 4, so Map is upstream. SBEATS in, PBEATS out.
        snax_simd_shape_flat(ex_in, sm, SBEATS);
        snax_simd_shape_flat(qz_out, l1 + p8_delta[0], PBEATS);
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
                set_gemmx_bases(delta_local_a, delta_local_b, -1, -1,
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
                // Transposed: O^T = V^T.P^T, so P^T is the B operand (it is
                // [Bc, Br] = [64, 32], exactly B's shape at M=8, N=4) and V^T
                // stays in A. In the untransposed form P was A instead.
                set_gemmx_bases(delta_local_a, p8_delta[(j - 1) & 1], -1, -1,
                                (int32_t)oacc);
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

            cv_in->base = l1 + d32_delta[j & 1];
            qz_out->base = l1 + p8_delta[j & 1];

            // FIVE tasks, submitted back to back with NO wait between them.
            //
            // SimdTop snapshots the whole task -- both AGU configs, the enable
            // mask and the operator CSRs -- into a 2-entry queue on the start
            // pulse, so arming and programming task i+1 overlaps task i running.
            // The engine still executes them strictly in order, which is what
            // makes the read-after-write chain between them safe. The start CSR
            // write back-pressures when the queue is full, so the core paces
            // itself without polling.
            //
            // Waiting after each task instead cost ~178 cycles of config on the
            // critical path six times over -- 1070 of a 3477-cycle tile. The
            // only reason the loop was written that way is that
            // snax_simd_last_task_cycle() restarts per task, so reading a
            // breakdown REQUIRED serialising. SIMD_BUSY_CYCLES is free-running
            // and is sampled once per tile instead.

            // (a) INT32 -> FP16.
            snax_simd_use1(SIMD_EXT_INT32TOFP16CONVERTER_512,
                           SIMD_EXT_INT32TOFP16CONVERTER_512_CSR, 0);
            snax_simd_program_fast(cv_in, cv_out);
            snax_simd_fire();

            // (b) row maxima.
            snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR,
                           SBEATS, SIMD_RED_MAX | SIMD_RED_LANEWISE);
            snax_simd_program_fast(mx_in, mx_out);
            snax_simd_fire();

            // (c) exp(S - m). E1 broadcasts and negates the row maxima with a
            // stride-0 inner loop; E2 adds the two operands from one address
            // stream; E3 exponentiates AND quantises in a single armed task,
            // because StreamMap is extension 1 and Fp16ToInt8 is 4, so Map sits
            // upstream of the quantiser in the chain. That fusion removes both a
            // task and a 4 KiB round trip through p16.
            //
            // E2 and E3 cannot be fused the same way: StreamElementwise is 3,
            // DOWNSTREAM of Map, so the exp would be applied to the -m operand
            // as well.
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR,
                           snax_simd_f32_neg(SIMD_F32_ONE), 0, SIMD_FUNC_LINEAR);
            snax_simd_program_fast(b_in, b_out);
            snax_simd_fire();

            snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE,
                           SIMD_EXT_STREAMELEMENTWISE_CSR, 2, SIMD_EW_ADD);
            snax_simd_program_fast(ew_in, ew_out);
            snax_simd_fire();

            // exp + quantise, one task: read sm, write P8 directly.
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR,
                           SIMD_F32_ONE, 0, SIMD_FUNC_EXP);
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR,
                                    (1u << SIMD_EXT_STREAMMAP) |
                                        (1u << SIMD_EXT_FP16TOINT8));
            snax_write_simd_cfg_reg(SIMD_EXT_FP16TOINT8_CSR, SIMD_F32_ONE);
            snax_simd_program_fast(ex_in, qz_out);
            snax_simd_fire();

            snax_simd_wait_all();
            sync[1] = j + 1;
        }
        simd_cycles = snax_simd_busy_cycles() - simd_busy0;
        simd_wall = snrt_mcycle() - s0;
    }
    snrt_cluster_hw_barrier();

    // ---- the invariant, in two halves ---------------------------------------
    //
    // exp(x - max x) contains exactly 1.0 in every query row, because exp(0)=1.
    // Fusing exp with the quantiser means no FP16 P tile is ever written, so the
    // same ground is covered either side of the fusion:
    //
    //   sm  must contain exactly +-0.0 for every query row   the reduce + subtract
    //   P^T must contain exactly 1    for every query row    the exp + quantiser
    //                                                        (exp(0)=1, inv_scale=1)
    //
    // Transposed, a query row is a LANE and the scan runs over beats (keys) --
    // the mirror image of the old row-major walk. That the check had to be
    // rewritten this way is the clearest statement of what LANEWISE changed.
    if (snax_is_simd_core()) {
        volatile int8_t *p8t = (volatile int8_t *)(l1 + p8_delta[(NKV - 1) & 1]);
        for (uint32_t i = 0; i < BR; i++) {  // i = query row = lane
            int found_zero = 0, found_one = 0;
            for (uint32_t j = 0; j < SBEATS; j++) {  // j = key = beat
                uint16_t v = fp16_at(sm + j * SIMD_BEAT_BYTES, i);
                if (v == 0x0000u || v == 0x8000u) found_zero = 1;
                if (p8t[j * BR + i] == 1) found_one = 1;
            }
            if (!found_zero || !found_one) {
                // Diagnose: is the lane-wise max wrong, or does the subtract/exp
                // not land exactly on 0 / 1?
                uint16_t hw_max = fp16_at(stat, i);
                uint16_t sw_max = fp16_at(s16, i);
                for (uint32_t j = 1; j < SBEATS; j++) {
                    uint16_t v = fp16_at(s16 + j * SIMD_BEAT_BYTES, i);
                    // FP16 compare via sign/magnitude, integer only.
                    int32_t x = (sw_max & 0x8000u) ? -(int32_t)(sw_max & 0x7FFFu)
                                                   : (int32_t)sw_max;
                    int32_t y = (v & 0x8000u) ? -(int32_t)(v & 0x7FFFu) : (int32_t)v;
                    if (y > x) sw_max = v;
                }
                printf("query %2u: zero=%d one=%d hw_max=%04x sw_max=%04x "
                       "sm[k0..3]=%04x %04x %04x %04x\n",
                       i, found_zero, found_one, hw_max, sw_max,
                       fp16_at(sm, i), fp16_at(sm + SIMD_BEAT_BYTES, i),
                       fp16_at(sm + 2 * SIMD_BEAT_BYTES, i),
                       fp16_at(sm + 3 * SIMD_BEAT_BYTES, i));
                err++;
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
