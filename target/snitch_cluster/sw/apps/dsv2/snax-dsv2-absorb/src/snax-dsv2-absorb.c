// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The absorbed per-head GEMVs of DeepSeek-V2-Lite's multi-head latent attention:
//
//     W_UK   q~_h = RNE(q_nope_h (1 x 128) . W_UK_h (128 x 512) * 2^-5)   the query into
//                   the latent space, where one 576-long dot product with a cached row
//                   [c | k_pe] is a score
//     W_UV   o_h  = RNE(o~_h (1 x 512) . W_UV_h (512 x 128) * 2^-7)       the weighted sum of
//                   latents back to the head's 128 values
//
// for all 16 heads, then the per-column dequantisation y (.) s as in snax-dsv2-gemv. Each
// projection is 1 MiB of INT8 weights; the inputs are the layer's own (the golden pack in
// sw/apps/dsv2/dsv2), with the latent norm's gain folded into the weights.
//
// ======================================================================================
// THE DATAFLOW
// ======================================================================================
//
// The grouped GEMV of snax-dsv2.h with one group per HEAD: every head has its own x (its
// q_nope or o~, in row 0 of its A operand) and its own weight, so a task walks
//
//     A   (head, n-block, k)   head stride 16 K: the heads' A operands are consecutive
//     B   (head, n-block, k)   head stride K N: the heads' weights are consecutive
//     D   (head, n-block, beat) head stride one output row (N FP16)
//
// and head h's outputs land in row h of a [16, N] result.
//
// THE INPUTS are the 16 vectors x_h, K bytes each, one after another. The heads' A operands
// are consecutive too, so one 2-D transfer of 16 K / 4 runs of 4 bytes, 64 apart, puts every
// x_h in row 0 of its own; the other rows are never read and stay zero.
//
// A projection's weights (1 MiB) are twice the L1, so the iDMA streams them ABS_HPT heads
// per task into two buffers, each refilled as soon as the task that read it retires. On
// four clusters a cluster's 4 heads (256 KiB) are one task.
//
//   iDMA (hart 3)   A operands and scales, then the weight blocks, then the goldens
//   GEMM (hart 0)   16 / ABS_HPT tasks per projection
//   SIMD (hart 1)   one dequantisation task per projection, then the checks
//   xDMA (hart 2)   zeroes the arena first (TCDM has no reset; RTL reads X)
//
// CHECKS (main returns the number that failed): the final results once both projections have
// run, yd (the dequantisation) bit for bit and no Inf in it. With DSV2_STAGE_CHECKS = 1 each
// projection is checked as it finishes, y (the array and the D port) as well. Every projection
// keeps its yd; the goldens are staged into the weight buffers once the tasks that read them
// have retired. The engines' spans follow ([SPAN], snax-dsv2-trace.h).

#include "data.h"
#include "snax-core-roles.h"
#include "snax-dsv2.h"
#include "snax-dsv2-trace.h"

#if !SNAX_HAS_GEMM_CORE || !SNAX_HAS_SIMD_CORE || !SNAX_HAS_XDMA_CORE || !SNAX_HAS_IDMA_CORE
#error "this kernel needs the four-engine cluster"
#endif

#define TAG "[ABSORB]"
#define NTASK (ABS_HEADS / ABS_HPT)

// ---- shared words ----------------------------------------------------------------------
#define LOADED sy[0]  // weight blocks in L1, over both projections  (iDMA -> GEMM)
#define FREED sy[1]   // blocks whose task retired: buffer free      (GEMM -> iDMA)
#define OPS sy[2]     // projections whose A operands and s are in L1 (iDMA -> GEMM)
#define GDONE sy[3]   // projections whose tasks all retired          (GEMM -> SIMD, iDMA)
#define GOLD sy[4]    // projections whose goldens are staged         (iDMA -> SIMD)
#define TMO sy[5]     // bounded waits that ran out
#define T_FIRST(p) sy[16 + 8 * (p) + 0]
#define T_LAST(p) sy[16 + 8 * (p) + 1]
#define D_BUSY(p) sy[16 + 8 * (p) + 2]
#define D_WAIT(p) sy[16 + 8 * (p) + 3]
#define T_DQ(p) sy[16 + 8 * (p) + 4]
#define SPIN(w, v) dsv2_spin_ge(&(w), (v), &TMO)

// A projection's cycles against the array's floor, and the GEMM's and the iDMA's rates.
static void proj_report(volatile uint32_t *sy, uint32_t pj) {
    const abs_proj_t *pr = &abs_projs[pj];
    const uint32_t K = pr->K, N = pr->N, k = pr->k;
    const uint32_t cyc = T_LAST(pj) - T_FIRST(pj);
    // array passes, one per cycle at best: shape 1 takes two column blocks a pass
    const uint32_t shape = dsv2_gemv_shape(), nb = N / DSV2_NU;
    const uint32_t floor = ABS_HEADS * (K / DSV2_KU) * (nb / dsv2_gemv_blocks_per_pass(nb));
    const uint32_t dbusy = D_BUSY(pj);
    printf(TAG " %s  16 x (1 x %u x %u)  k=%u  %u tasks: %u cc (floor %u) GEMM %u%%  "
               "iDMA %u B/cc, waited %u cc  dequant %u cc  [shape %u]\n",
           pr->name, K, N, k, NTASK, cyc, floor, cyc ? (100u * floor) / cyc : 0u,
           dbusy ? (ABS_HEADS * K * N) / dbusy : 0u, D_WAIT(pj), T_DQ(pj), shape);
}

// A projection's final result: yd bit for bit, and no Inf in it.
static uint32_t check_yd(uint32_t pj, const uint16_t *yd, const uint16_t *gyd) {
    const abs_proj_t *pr = &abs_projs[pj];
    const uint32_t n = ABS_HEADS * pr->N;
    uint32_t fails = dsv2_check(TAG, !dsv2_cmp16(TAG, "yd", yd, gyd, n), pr->name,
                                "yd = y (.) s, bit-exact");
    return fails + dsv2_check(TAG, !dsv2_count_inf16(yd, n), pr->name, "no Inf in yd");
}

int main() {
    // ---- L1, derived identically on every hart -----------------------------------------
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 63u) & ~63u);
    uint8_t *arena = p;
    uint32_t ntot = 0;  // both projections' outputs
    for (uint32_t pj = 0; pj < ABS_NPROJ; pj++) ntot += ABS_HEADS * abs_projs[pj].N;
#define TAKE(n) (p += (((n) + 63u) & ~63u), p - (((n) + 63u) & ~63u))
    volatile uint32_t *sy = (volatile uint32_t *)TAKE(512);
    dsv2_trace_t *tr = (dsv2_trace_t *)TAKE(sizeof(dsv2_trace_t));
    int8_t *abuf = (int8_t *)TAKE(ABS_HEADS * DSV2_MR * ABS_KMAX);  // every head's A operand
    uint16_t *ybuf = (uint16_t *)TAKE(ABS_HEADS * 2u * ABS_NMAX + DSV2_SPILL(ABS_NMAX / DSV2_NU));
    uint16_t *sbuf = (uint16_t *)TAKE(ABS_HEADS * 2u * ABS_NMAX);   // the dequant factors
    uint16_t *ydall = (uint16_t *)TAKE(2u * ntot);                  // both projections' yd
    int8_t *bbuf[2];
    bbuf[0] = (int8_t *)TAKE(ABS_HPT * ABS_HEAD_BYTES);
    bbuf[1] = (int8_t *)TAKE(ABS_HPT * ABS_HEAD_BYTES);
    uint16_t *gold = (uint16_t *)bbuf[0];  // goldens, once the tasks reading the buffers retired
    const uint32_t arena_beats = (uint32_t)(p - arena) / DSV2_BEAT;
#undef TAKE

    const int isG = snax_is_gemm_core(), isS = snax_is_simd_core();
    const int isX = snax_is_xdma_core(), isD = snax_is_idma_core();

    if (isS) dsv2_print_l1((uint32_t)(p - arena), DSV2_L1_TOP - (uint32_t)arena);
    if ((uint32_t)p > DSV2_L1_TOP) {
        if (isS)
            printf(TAG " FAIL: the arena ends at 0x%08x, past the stacks at 0x%08x\n",
                   (uint32_t)p, DSV2_L1_TOP);
        return isS ? 1 : 0;
    }
    const uint32_t f0 = snrt_mcycle();
    if (isX) dsv2_xdma_fill_zero(arena, arena_beats);
    if (isX) dsv2_span(tr, DSV2_TR_XDMA, "0 zero the arena", f0, snrt_mcycle());
    snrt_cluster_hw_barrier();

    if (isD) {
        // ============================== the iDMA
        for (uint32_t pj = 0; pj < ABS_NPROJ; pj++) {
            const abs_proj_t *pr = &abs_projs[pj];
            const uint32_t K = pr->K, N = pr->N, tbytes = ABS_HPT * K * N;
            const int8_t *w = pr->w;
            uint32_t o0 = snrt_mcycle();
            snrt_dma_start_2d(abuf, pr->a, DSV2_KU, DSV2_MR * DSV2_KU, DSV2_KU,
                              ABS_HEADS * K / DSV2_KU);
            snrt_dma_start_1d(sbuf, pr->s, ABS_HEADS * 2u * N);
            snrt_dma_wait_all();
            OPS = pj + 1u;
            dsv2_span(tr, DSV2_TR_IDMA, "0 operands", o0, snrt_mcycle());
            uint32_t busy = 0, waited = 0;
            for (uint32_t t = 0; t < NTASK; t++) {
                uint32_t t0 = snrt_mcycle();
                if (t >= 2) SPIN(FREED, pj * NTASK + t - 1u);  // task t-2 retired
                uint32_t t1 = snrt_mcycle();
                snrt_dma_start_1d(bbuf[t & 1u], w + t * tbytes, tbytes);
                snrt_dma_wait_all();
                uint32_t t2 = snrt_mcycle();
                dsv2_span(tr, DSV2_TR_IDMA, pr->name, t1, t2);
                waited += t1 - t0;
                busy += t2 - t1;
                LOADED = pj * NTASK + t + 1u;
            }
            D_BUSY(pj) = busy;
            D_WAIT(pj) = waited;
#if DSV2_STAGE_CHECKS
            // this projection's goldens: y, then yd
            SPIN(GDONE, pj + 1u);
            o0 = snrt_mcycle();
            snrt_dma_start_1d(gold, pr->y, ABS_HEADS * 2u * N);
            snrt_dma_start_1d(gold + ABS_HEADS * N, pr->yd, ABS_HEADS * 2u * N);
            snrt_dma_wait_all();
            GOLD = pj + 1u;
            dsv2_span(tr, DSV2_TR_IDMA, "0 goldens", o0, snrt_mcycle());
#endif
            snrt_cluster_hw_barrier();
        }
#if !DSV2_STAGE_CHECKS
        // after the kernel: both projections' yd goldens, in order
        uint32_t off = 0;
        for (uint32_t pj = 0; pj < ABS_NPROJ; pj++) {
            snrt_dma_start_1d(gold + off, abs_projs[pj].yd, ABS_HEADS * 2u * abs_projs[pj].N);
            off += ABS_HEADS * abs_projs[pj].N;
        }
        snrt_dma_wait_all();
        GOLD = ABS_NPROJ;
#endif
        return 0;
    }

    if (isG) {
        // ============================== the GEMM: ABS_HPT heads per task
        for (uint32_t pj = 0; pj < ABS_NPROJ; pj++) {
            const abs_proj_t *pr = &abs_projs[pj];
            const uint32_t K = pr->K, N = pr->N, k = pr->k, nb = N / DSV2_NU;
            const uint32_t astep = DSV2_MR * K;
            SPIN(OPS, pj + 1u);
            dsv2_gemv_arm(K / DSV2_KU, nb, ABS_HPT, astep, k);
            uint32_t id = csrr_ss(GEMMX_FINISHED_TASK);
            for (uint32_t t = 0; t < NTASK; t++) {
                SPIN(LOADED, pj * NTASK + t + 1u);
                dsv2_gemv_fire(abuf + t * ABS_HPT * astep, bbuf[t & 1u],
                               (uint8_t *)ybuf + t * ABS_HPT * DSV2_SLOT(nb));
                if (t == 0) T_FIRST(pj) = snrt_mcycle();
                ++id;
                if (t) {
                    TMO += dsv2_gemm_wait(id - 1u);
                    FREED = pj * NTASK + t;
                }
            }
            TMO += dsv2_gemm_wait(id);
            T_LAST(pj) = snrt_mcycle();
            dsv2_span(tr, DSV2_TR_GEMM, pr->name, T_FIRST(pj), T_LAST(pj));
            FREED = (pj + 1u) * NTASK;
            GDONE = pj + 1u;
            snrt_cluster_hw_barrier();
        }
        return 0;
    }

    if (isX) {
        for (uint32_t pj = 0; pj < ABS_NPROJ; pj++) snrt_cluster_hw_barrier();
        return 0;
    }

    // ============================== the SIMD: dequantise; the checks
    uint32_t fails = 0, hung = 0, yoff = 0;
    printf(TAG " %u heads, %u per task; %s\n", ABS_HEADS, ABS_HPT,
           DSV2_STAGE_CHECKS ? "y and yd checked after each projection"
                             : "yd checked once both projections have run");
    for (uint32_t pj = 0; pj < ABS_NPROJ; pj++) {
        const abs_proj_t *pr = &abs_projs[pj];
        const uint32_t n = ABS_HEADS * pr->N;
        dsv2_dequant_prep(ybuf, sbuf, ydall + yoff, n);
        SPIN(GDONE, pj + 1u);
        uint32_t t0 = snrt_mcycle();
        snax_simd_fire();
        hung += snax_simd_wait_all_checked("dequant", SIMD_WAIT_BUDGET);
        T_DQ(pj) = snrt_mcycle() - t0;
        dsv2_span(tr, DSV2_TR_SIMD, "G3 dequant", t0, snrt_mcycle());
#if DSV2_STAGE_CHECKS
        SPIN(GOLD, pj + 1u);
        const uint32_t c0 = snrt_mcycle();
        proj_report(sy, pj);
        fails += dsv2_check(TAG, !dsv2_cmp16(TAG, "y", ybuf, gold, n), pr->name,
                            "y = RNE(x_h . W_h * 2^-k), all heads, bit-exact");
        fails += dsv2_check(TAG, !dsv2_count_inf16(ybuf, n), pr->name, "no Inf at the chosen k");
        fails += check_yd(pj, ydall + yoff, gold + n);
        dsv2_span(tr, DSV2_TR_SIMD, "check (core)", c0, snrt_mcycle());
#endif
        yoff += n;
        snrt_cluster_hw_barrier();
    }
#if !DSV2_STAGE_CHECKS
    // the final results, once both projections have run
    SPIN(GOLD, ABS_NPROJ);
    yoff = 0;
    for (uint32_t pj = 0; pj < ABS_NPROJ; pj++) {
        proj_report(sy, pj);
        fails += check_yd(pj, ydall + yoff, gold + yoff);
        yoff += ABS_HEADS * abs_projs[pj].N;
    }
#endif
    if (hung || TMO) {
        printf(TAG " FAIL: a task never retired or %u bounded wait(s) ran out\n", TMO);
        fails++;
    }
    printf(fails ? TAG " FAIL (%u checks)\n" : TAG " PASS\n", fails);
    dsv2_trace_print(tr);
    return (int)fails;
}
