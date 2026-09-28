// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V2-Lite's MoE gate for one token (V5), from the layer's own post-attention state:
//
//   GEMM   logits_raw = RNE(hn8 . W_r * 2^-k)            the router GEMV, one 64-column chunk
//   SIMD   logits = logits_raw (.) s                     G3's dequantisation
//          p = softmax(logits)                           one row of 2 beats, five tasks
//   core   ids = the 6 largest p, w = p[ids]              FP16 bit patterns as integers; ties
//                                                        to the lower index; w NOT renormalised
//
// The softmax and the top-k are snax-dsv2.h's (dsv2_softmax_row, dsv2_top_k16). The top-k
// runs on the SIMD hart, in integer code: no hart here has an FPU.
//
// CHECKS, bit for bit against the device model (which reproduces the exp and rsqrt tables),
// once the kernel is done and its goldens loaded: the ids and the weights; with
// DSV2_STAGE_CHECKS = 1 the logits and p as well. main returns the number of failed checks.
// The engines' spans and the cluster census over the kernel follow ([SPAN], [UTIL];
// snax-dsv2-trace.h).

#include "data.h"
#include "snax-core-roles.h"
#include "snax-dsv2.h"
#include "snax-dsv2-trace.h"

#if !SNAX_HAS_GEMM_CORE || !SNAX_HAS_SIMD_CORE || !SNAX_HAS_XDMA_CORE || !SNAX_HAS_IDMA_CORE
#error "this kernel needs the four-engine cluster"
#endif

#define TAG "[ROUTER]"
#define NB (N_EXP / DSV2_NU)       // n-blocks: 4
#define BEATS (N_EXP / 32)         // FP16 beats: 2
#define READY sy[0]  // the operands in L1                 (iDMA -> GEMM, SIMD)
#define G_DONE sy[1] // the GEMV retired                   (GEMM -> SIMD)
#define TMO sy[2]
#define DONE sy[3]   // the top-6 chosen                   (SIMD -> iDMA)
#define GOLD sy[4]   // the goldens in L1                  (iDMA -> SIMD)
#define SPIN(w, v) dsv2_spin_ge(&(w), (v), &TMO)

static uint32_t chk16(const char *what, const uint16_t *got, const uint16_t *want, uint32_t n,
                      uint32_t max_ulp) {
    n = dsv2_check_terms(n);
    uint32_t worst, bad = dsv2_ulp16(got, want, n, &worst);
    printf(TAG "   %s %-22s %2u/%2u differ, worst %u ULP (allowed %u)\n",
           worst <= max_ulp ? "PASS" : "FAIL", what, bad, n, worst, max_ulp);
    return worst <= max_ulp ? 0u : 1u;
}

int main() {
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 63u) & ~63u);
    uint8_t *arena = p;
#define TAKE(n) (p += (((n) + 63u) & ~63u), p - (((n) + 63u) & ~63u))
    volatile uint32_t *sy = (volatile uint32_t *)TAKE(256);
    dsv2_trace_t *tr = (dsv2_trace_t *)TAKE(sizeof(dsv2_trace_t));
    int8_t *abuf = (int8_t *)TAKE(DSV2_MR * K_IN);
    int8_t *wbuf = (int8_t *)TAKE(K_IN * N_EXP);
    uint16_t *y = (uint16_t *)TAKE(2 * N_EXP + DSV2_SPILL(NB));
    uint16_t *s = (uint16_t *)TAKE(2 * N_EXP);                 // the dequant factors
    uint8_t *lgl = TAKE(DSV2_BEAT + 2 * N_EXP);                // [latch][logits]
    uint16_t *lg = (uint16_t *)(lgl + DSV2_BEAT);
    uint8_t *tmp = TAKE(DSV2_BEAT);
    uint8_t *el = TAKE(DSV2_BEAT + 2 * N_EXP + DSV2_BEAT);     // [latch][e][sum]
    uint16_t *e = (uint16_t *)(el + DSV2_BEAT);
    uint16_t *pr = (uint16_t *)TAKE(2 * N_EXP);
    uint16_t *glg = (uint16_t *)TAKE(2 * N_EXP), *gp = (uint16_t *)TAKE(2 * N_EXP);
    uint32_t *gids = (uint32_t *)TAKE(4 * TOP_K);
    uint16_t *gw = (uint16_t *)TAKE(2 * TOP_K);
    const uint32_t arena_beats = (uint32_t)(p - arena) / DSV2_BEAT;
#undef TAKE

    const int isG = snax_is_gemm_core(), isS = snax_is_simd_core();
    const int isX = snax_is_xdma_core(), isD = snax_is_idma_core();
    if (isS) dsv2_print_l1((uint32_t)(p - arena), DSV2_L1_TOP - (uint32_t)arena);
    const uint32_t f0 = snrt_mcycle();
    if (isX) dsv2_xdma_fill_zero(arena, arena_beats);
    if (isX) dsv2_span(tr, DSV2_TR_XDMA, "0 zero the arena", f0, snrt_mcycle());
    if (isS) snax_perf_arm();  // the census covers the kernel
    snrt_cluster_hw_barrier();

    if (isD) {
        const uint32_t i0 = snrt_mcycle();
        // hn's vector into row 0 of the A buffer, 4 values in each 64-byte block
        snrt_dma_start_2d(abuf, hq_a, DSV2_KU, DSV2_MR * DSV2_KU, DSV2_KU, K_IN / DSV2_KU);
        snrt_dma_start_1d(wbuf, wr, K_IN * N_EXP);
        snrt_dma_start_1d(s, wr_s, 2 * N_EXP);
        snrt_dma_wait_all();
        dsv2_span(tr, DSV2_TR_IDMA, "15 A operand, W_r, factors", i0, snrt_mcycle());
        READY = 1u;
        // the goldens, once the kernel is done
        SPIN(DONE, 1u);
        snrt_dma_start_1d(gids, g_ids, 4 * TOP_K);
        snrt_dma_start_1d(gw, g_w, 2 * TOP_K);
#if DSV2_STAGE_CHECKS
        snrt_dma_start_1d(glg, g_logits, 2 * N_EXP);
        snrt_dma_start_1d(gp, g_p, 2 * N_EXP);
#endif
        snrt_dma_wait_all();
        GOLD = 1u;
        return 0;
    }
    if (isG) {
        SPIN(READY, 1u);
        const uint32_t g0 = snrt_mcycle();
        dsv2_gemv_arm(K_IN / DSV2_KU, NB, 1u, 0u, D_SHIFT);
        uint32_t id = csrr_ss(GEMMX_FINISHED_TASK);
        dsv2_gemv_fire(abuf, wbuf, y);
        TMO += dsv2_gemm_wait(id + 1u);
        dsv2_span(tr, DSV2_TR_GEMM, "15 router GEMV", g0, snrt_mcycle());
        G_DONE = 1u;
        return 0;
    }
    if (!isS) return 0;

    SPIN(READY, 1u);
    const uint32_t busy0 = snax_simd_busy_cycles();
    dsv2_dequant_prep(y, s, lg, N_EXP);
    SPIN(G_DONE, 1u);
    uint32_t t0 = snrt_mcycle();
    snax_simd_fire();
    dsv2_softmax_row(lg, tmp, e, pr, BEATS);
    uint32_t hung = snax_simd_wait_all_checked("router", SIMD_WAIT_BUDGET);
    uint32_t t1 = snrt_mcycle();
    uint32_t ids[TOP_K];
    uint16_t w[TOP_K];
    dsv2_top_k16(pr, N_EXP, TOP_K, ids);
    for (uint32_t i = 0; i < TOP_K; i++) w[i] = pr[ids[i]];
    uint32_t t2 = snrt_mcycle();
    const uint32_t simd_busy = snax_simd_busy_cycles() - busy0;
    snax_perf_snapshot_t census;
    snax_perf_read(&census);
    dsv2_span(tr, DSV2_TR_SIMD, "16 dequant, softmax", t0, t1);
    dsv2_span(tr, DSV2_TR_SIMD, "16 top-6 (core)", t1, t2);
    DONE = 1u;
    SPIN(GOLD, 1u);

    uint32_t fails = 0, idbad = 0;
    printf(TAG " dequant + softmax %u cc, top-%u %u cc; ids", t1 - t0, TOP_K, t2 - t1);
    for (uint32_t i = 0; i < TOP_K; i++) {
        printf(" %u", ids[i]);
        idbad += ids[i] != gids[i];
    }
    printf("\n");
#if DSV2_STAGE_CHECKS
    fails += chk16("logits", lg, glg, N_EXP, 0);
    fails += chk16("softmax", pr, gp, N_EXP, 0);
#endif
    fails += dsv2_check(TAG, idbad == 0, "top-6", "the golden's ids, in order");
    fails += chk16("weights p[ids]", w, gw, TOP_K, 0);
    if (hung || TMO) {
        printf(TAG " FAIL: a task never retired or a wait ran out\n");
        fails++;
    }
    printf(fails ? TAG " FAIL (%u checks)\n" : TAG " PASS\n", fails);
    dsv2_trace_print(tr);
    dsv2_util_print(&census, simd_busy);
    return (int)fails;
}
