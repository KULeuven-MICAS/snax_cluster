// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// FlashAttention inner loop on the four-engine cluster, instrumented for
// hardware utilisation.
//
// One (query tile, KV tile) pair: Br = 32, Bc = d = 64, i.e. 4x8x8 GEMM tiles of
// 8x8. A score row is 64 FP16 = 128 B = 2 SIMD beats, deliberately: StreamReduce
// asserts on operandCount = 1, so a one-beat row cannot be reduced at all. And
// d == Bc makes S = Q.K^T and O = P.V the same shape, so one streamer
// configuration drives both matmuls.
//
//   GEMM (hart 0)   S = Q.K^T                   INT8 x INT8 -> INT32
//   SIMD (hart 1)   S16   = fp16(S)             Int32ToFp16
//                   m[r]  = rowmax(S16[r])      StreamReduce MAX
//                   P16   = exp(S16 - m)        broadcast / EW ADD / StreamMap EXP
//                   P8    = int8(P16)           Fp16ToInt8
//   GEMM (hart 0)   O = P.V
//
// THE POINT IS THE OVERLAP. These are two engines on two harts, and the whole
// reason FlashAttention is the driving workload is that its softmax can run on
// the SIMD engine while the GEMM works on the next KV tile. So the loop below
// is software-pipelined across NKV key/value tiles:
//
//   GEMM core:  S(0) ; then per j:  S(j+1)  ||  <SIMD does tile j>  ; O(j)
//   SIMD core:  per j: wait for S(j), softmax it, publish P8(j)
//
// S and P8 are double-buffered so producer and consumer never touch the same
// memory, and the handoff is two monotonically increasing counters in TCDM
// rather than snrt_cluster_hw_barrier(), which would serialise the two cores
// again -- and would also drag harts 2 and 3 in.
//
// WHAT IS AND IS NOT VALIDATED. The softmax is checked against an exact
// invariant: exp(x - max(x)) must contain exactly 1.0 in every row, because
// exp(0) = 1. That one check covers the reduce, the per-row subtract and the
// exponential together, and it is what the numbers below rest on. Every KV tile
// is fed the same K, so checking the final tile's P16 checks them all.
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

#define BR 32  // query rows in this tile      (M * meshRow)
#define BC 64  // key columns in this tile      (N * meshCol)
// d == BC so both matmuls have the same shape and one streamer config serves both.
#define BEATS_PER_ROW (BC * 2 / SIMD_BEAT_BYTES)
#define NKV 4  // key/value tiles to stream through the pipeline

// TCDM map, above the GEMM buffers that data.h lays out (which end at
// delta_local_d32 + BR*BC*4 = 22528). Contiguous and in address order, so a
// collision is visible by reading down the column -- OFF_PUB was originally
// placed inside OFF_MREP and survived only because the report is written after
// the broadcast buffer is dead.
//
//   S16/P16/MREP/SM   64 beats  = 4 KiB each
//   STAT              BR beats  = 2 KiB
//   a D32 tile        BR*BC*4   = 8 KiB
//   a P8 tile         BR*BC     = 2 KiB
#define OFF_S16 22528     // ..26624
#define OFF_P16 26624     // ..30720
#define OFF_STAT 30720    // ..32768
#define OFF_MREP 32768    // ..36864
#define OFF_SM 36864      // ..40960
#define OFF_D32_B 40960   // ..49152  second S buffer; the first is delta_local_d32
#define OFF_P8_0 49152    // ..51200
#define OFF_P8_1 51200    // ..53248
#define OFF_OACC 53248    // ..61440  O destination, rewritten per tile
#define OFF_PUB 61440     // ..61696  the report handoff
#define OFF_SYNC 61696    // ..61760  the two pipeline counters

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
    uint8_t *s16 = l1 + OFF_S16;
    uint8_t *p16 = l1 + OFF_P16;
    uint8_t *stat = l1 + OFF_STAT;
    uint8_t *mrep = l1 + OFF_MREP;
    uint8_t *sm = l1 + OFF_SM;

    // Double buffers, indexed by tile parity.
    const int32_t d32_delta[2] = {delta_local_d32, OFF_D32_B};
    const int32_t p8_delta[2] = {OFF_P8_0, OFF_P8_1};

    // The handoff. sync[0] counts S tiles the GEMM has finished producing;
    // sync[1] counts softmax tiles the SIMD core has finished consuming and
    // whose P8 is ready. Both only ever increase, so a reader never needs a
    // lock -- it just waits for the count to pass a threshold.
    volatile uint32_t *sync = (volatile uint32_t *)(l1 + OFF_SYNC);

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
    const uint32_t sbeats = BR * BEATS_PER_ROW;
    if (snax_is_simd_core()) {
        // (a) INT32 scores -> FP16: 64 beats in, 32 beats out.
        snax_simd_shape_flat(cv_in, l1 + d32_delta[0], BR * BC * 4 / SIMD_BEAT_BYTES);
        snax_simd_shape_flat(cv_out, s16, sbeats);
        // (b) per-row max, one task: BR rows of BEATS_PER_ROW -> BR scalar beats.
        snax_simd_shape_rows(mx_in, s16, BR, BEATS_PER_ROW,
                             BEATS_PER_ROW * SIMD_BEAT_BYTES);
        snax_simd_shape_flat(mx_out, stat, BR);
        // (c) exp(S - m) in three tasks. A stride-0 inner loop re-presents each
        // row's max beat, and the reduce already splatted the scalar across the
        // beat, so no lane fixup is needed. The negation is the `a` coefficient,
        // which is what removes the 24-cycle software f16->f32 per row.
        snax_simd_shape_2d(b_in, stat, BEATS_PER_ROW, 0, BR, SIMD_BEAT_BYTES);
        snax_simd_shape_flat(b_out, mrep, sbeats);
        // The two operands of the elementwise add come from ONE address stream:
        // the inner loop of 2 steps by (mrep - s16). That delta must be
        // positive -- the AGU stride is unsigned, and an operand placed below
        // its partner wraps the address, reads outside TCDM and writes X while
        // the task still reports complete.
        snax_simd_shape_2d(ew_in, s16, 2, (uint32_t)(mrep - s16), sbeats,
                           SIMD_BEAT_BYTES);
        snax_simd_shape_flat(ew_out, sm, sbeats);
        snax_simd_shape_flat(ex_in, sm, sbeats);
        snax_simd_shape_flat(ex_out, p16, sbeats);
        // (d) quantise P to INT8 for the second GEMM.
        snax_simd_shape_flat(qz_in, p16, sbeats);
        snax_simd_shape_flat(qz_out, l1 + p8_delta[0], BR * BC / SIMD_BEAT_BYTES);
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
                set_gemmx_bases(p8_delta[(j - 1) & 1], delta_local_b, -1, -1,
                                OFF_OACC);
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
        for (uint32_t j = 0; j < NKV; j++) {
            uint32_t w0 = snrt_mcycle();
            SNAX_SPIN_UNTIL(sync[0] >= j + 1, timeouts);
            simd_stall += snrt_mcycle() - w0;

            cv_in->base = l1 + d32_delta[j & 1];
            qz_out->base = l1 + p8_delta[j & 1];

            // (a) INT32 -> FP16.
            snax_simd_use1(SIMD_EXT_INT32TOFP16CONVERTER_512,
                           SIMD_EXT_INT32TOFP16CONVERTER_512_CSR, 0);
            snax_simd_program_fast(cv_in, cv_out);
            snax_simd_wait(snax_simd_launch_async());
            c_conv = snax_simd_last_task_cycle();

            // (b) row maxima.
            snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR,
                           BEATS_PER_ROW, SIMD_RED_MAX);
            snax_simd_program_fast(mx_in, mx_out);
            snax_simd_wait(snax_simd_launch_async());
            c_max = snax_simd_last_task_cycle();

            // (c) exp(S - m), three tasks.
            //
            // It used to be one task per row, because StreamMap's `b` is a
            // scalar CSR: 2320 cycles for 800 of engine time, all of it a
            // software f16->f32 of the max, a repoint and a discarded counter
            // read, 32 times over. StreamMap sits UPSTREAM of
            // StreamElementwise in the chain (ids 2 and 4), so E2 and E3 cannot
            // be fused -- the exp would be applied to the -m operand too.
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR,
                           snax_simd_f32_neg(SIMD_F32_ONE), 0, SIMD_FUNC_LINEAR);
            snax_simd_program_fast(b_in, b_out);
            snax_simd_wait(snax_simd_launch_async());
            c_exp = snax_simd_last_task_cycle();

            snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE,
                           SIMD_EXT_STREAMELEMENTWISE_CSR, 2, SIMD_EW_ADD);
            snax_simd_program_fast(ew_in, ew_out);
            snax_simd_wait(snax_simd_launch_async());
            c_exp += snax_simd_last_task_cycle();

            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR,
                           SIMD_F32_ONE, 0, SIMD_FUNC_EXP);
            snax_simd_program_fast(ex_in, ex_out);
            snax_simd_wait(snax_simd_launch_async());
            c_exp += snax_simd_last_task_cycle();

            // (d) quantise for the second GEMM.
            snax_simd_use1(SIMD_EXT_FP16TOINT8, SIMD_EXT_FP16TOINT8_CSR,
                           SIMD_F32_ONE);
            snax_simd_program_fast(qz_in, qz_out);
            snax_simd_wait(snax_simd_launch_async());
            c_quant = snax_simd_last_task_cycle();

            simd_cycles += c_conv + c_max + c_exp + c_quant;
            sync[1] = j + 1;
        }
        simd_wall = snrt_mcycle() - s0;
    }
    snrt_cluster_hw_barrier();

    // ---- the invariant: exp(x - max x) contains exactly 1.0 in every row ----
    // Every KV tile is fed the same K, so the final tile's P16 stands for all
    // of them. Checked after the loop so the check is not inside the clock.
    if (snax_is_simd_core()) {
        for (uint32_t r = 0; r < BR; r++) {
            volatile uint8_t *row = p16 + r * BEATS_PER_ROW * SIMD_BEAT_BYTES;
            int found_one = 0;
            for (uint32_t i = 0; i < BC; i++) {
                if (fp16_at(row, i) == 0x3C00u) found_one = 1;
            }
            if (!found_one) {
                // Diagnose rather than just report: is the hardware max wrong, or
                // is exp(0) not returning exactly 1.0?
                uint16_t hw_max = fp16_at(stat + r * SIMD_BEAT_BYTES, 0);
                volatile uint8_t *srow = s16 + r * BEATS_PER_ROW * SIMD_BEAT_BYTES;
                uint16_t sw_max = fp16_at(srow, 0);
                for (uint32_t i = 1; i < BC; i++) {
                    uint16_t v = fp16_at(srow, i);
                    // FP16 compare via sign/magnitude, integer only.
                    int32_t a = (sw_max & 0x8000u) ? -(int32_t)(sw_max & 0x7FFFu)
                                                   : (int32_t)sw_max;
                    int32_t b = (v & 0x8000u) ? -(int32_t)(v & 0x7FFFu) : (int32_t)v;
                    if (b > a) sw_max = v;
                }
                printf("row %2u: hw_max=%04x sw_max=%04x s16[0..3]=%04x %04x %04x %04x\n",
                       r, hw_max, sw_max, fp16_at(srow, 0), fp16_at(srow, 1),
                       fp16_at(srow, 2), fp16_at(srow, 3));
                err++;
            }
        }
    }

    // ---- report -------------------------------------------------------------
    // Each core owns its own totals; publish through TCDM so one core prints.
    volatile uint32_t *pub = (volatile uint32_t *)(l1 + OFF_PUB);
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
        printf("  engine busy      GEMM %5u   SIMD %5u\n", gemm_cycles, simd_total);
        printf("  SIMD per tile    convert %u  rowmax %u  exp(3 tasks) %u  quant %u\n",
               pub[4], pub[5], pub[6], pub[7]);
        printf("  core wall        GEMM %5u   SIMD %5u   pipeline %u\n",
               gemm_wall, simd_wall_r, pipeline);
        printf("  waiting on peer  GEMM %5u   SIMD %5u\n", gemm_stall, pub[8]);
        if (pipeline) {
            printf("  utilisation      GEMM %2u%%     SIMD %2u%%     overall %u%%\n",
                   100u * gemm_cycles / pipeline, 100u * simd_total / pipeline,
                   100u * busy / pipeline);
            uint32_t orch = pipeline > busy ? pipeline - busy : 0;
            printf("  orchestration    %u cycles (%u%% of the pipeline)\n", orch,
                   100u * orch / pipeline);
        }
        if (pub[9]) printf("  WARNING: %u spin timeouts -- the handoff deadlocked\n", pub[9]);
        printf("  %s (%u invariant, %u config)\n",
               (pub[1] == 0 && pub[2] == 0 && pub[9] == 0) ? "PASS" : "FAIL",
               pub[1], pub[2]);
    }
    snrt_cluster_hw_barrier();
    return 0;
}
