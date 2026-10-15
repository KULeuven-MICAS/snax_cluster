// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GEMMs of HeMAiA's MoE prefill on VersaCore's array shape 3, (8, 8, 16), in isolation, on one
// cluster and checked:
//
//     D (8 mt x 16 nt, FP16) = RNE(A . B * 2^-k)    A: row-major INT8, B: INT4 weights
//
// TWO QUESTIONS, two kinds of case (data/datagen.py; GEMM3_BENCH_SET in the Makefile):
//
// 1. How do the weights sit against a row-major A (g3_cases)? A case is a shape, the layout of B
//    (G3_LAY_B, G3_LAY_PAIRED, G3_LAY_PARITY: snax-dsv2-g3.h) and B's offset past a 256-byte
//    boundary (A sits on one). The case runs alone, G3B_REPS times back to back; the last run is
//    the measurement. The report: cycles and cycles per array pass (M K N passes), the array's
//    busy cycles and those it waited for A, for B and for D (VersaCore's stall census of that
//    run), and the first run's cycles.
//
// 2. What does launching a GEMM cost (g3b_launch)? A launch case is G back-to-back GEMMs of one
//    shape, programmed one of six ways: the arm (the generic path of HeMAiA's gemm_start, the lean
//    straight-line arm of every register, or the lean arm once and only the bases per GEMM) and
//    the issue (sync: each GEMM retires before the next is armed; async: the next is armed and
//    queued while the previous runs, two in flight -- snax_cfg_queue_depth 2). The report:
//    cycles for the run, the core's cycles launching the first GEMM (cold instruction fetch) and
//    the others (warm), and the run's cycles a GEMM against the array's own.
//
// Each case runs with nothing else touching L1: the loads, the clear and the check sit between
// barriers around it. The check compares a quarter of D per hart with the golden, bit for bit.
//
// main returns the number of cases whose D differs, plus one if a task never retired.

#include "data.h"
#include "snax-core-roles.h"
#include "snax-dsv2-g3.h"

#if !SNAX_HAS_GEMM_CORE || !SNAX_HAS_SIMD_CORE || !SNAX_HAS_XDMA_CORE || !SNAX_HAS_IDMA_CORE
#error "this bench needs the four-engine cluster"
#endif
#if G3B_REPS < 1 || G3B_REPS > 8
#error "G3B_REPS: 1 to 8 runs a case"
#endif

#define TAG "[G3B]"
#define G3B_GEN 0u    // the generic arm, every GEMM
#define G3B_LEAN 1u   // the lean arm, every GEMM
#define G3B_DELTA 2u  // the lean arm once; the bases per GEMM
#define G3B_ARM(kind) ((kind) % 3u)
#define G3B_ASYNC(kind) ((kind) / 3u)

// ---- shared words ----------------------------------------------------------------------
#define TMO sy[0]                // bounded waits that ran out
#define CYC(r) sy[8 + (r)]       // GEMM: run r's cycles, fire to retirement
#define BUSY sy[16]              // the array's busy cycles, last run
#define WAIT_A sy[17]            //   of which waiting for A
#define WAIT_B sy[18]            //   for B (A there)
#define WAIT_D sy[19]            //   for the array (A and B there)
#define L_TOTAL sy[20]           // a launch case: cycles for the whole run
#define L_FIRST sy[21]           //   launching the first GEMM
#define L_WARM sy[22]            //   launching the others, summed
#define L_ARM1 sy[23]            //   arming once (delta)
#define BAD(h) sy[24 + (h)]      // hart h: the outputs of its quarter that differ
#define ULP(h) sy[28 + (h)]      // hart h: their largest distance, in ULPs

static const char *const g3b_lay[] = {"?", "b     ", "paired", "parity"};
static const char *const g3b_alay[] = {"row", "blk", "ilv"};
static const char *const g3b_arm[] = {"generic", "lean   ", "delta  "};

// One case's line; returns 1 if its D differs from the golden.
static uint32_t g3b_report(volatile uint32_t *sy, const g3b_case_t *cs) {
    const uint32_t passes = cs->mt * cs->kt * cs->nt, cyc = CYC(G3B_REPS - 1);
    const uint32_t milli = cyc * 1000u / passes, bmilli = BUSY * 1000u / passes;
    uint32_t bad = 0, worst = 0;
    for (uint32_t h = 0; h < 4; h++) {
        bad += BAD(h);
        if (ULP(h) > worst) worst = ULP(h);
    }
    printf(TAG " %-15s A %s B %s +%3u  %u x %3u x %u  %6u passes %6u cc  %u.%03u cc/pass  busy %6u (%u.%03u)  "
               "wait A %4u B %4u D %4u  first %6u cc  %s\n",
           cs->name, g3b_alay[cs->alay], g3b_lay[cs->lay], cs->boff, cs->mt, cs->kt, cs->nt, passes, cyc,
           milli / 1000u, milli % 1000u, BUSY, bmilli / 1000u, bmilli % 1000u, WAIT_A, WAIT_B,
           WAIT_D, CYC(0), bad ? "FAIL" : "PASS");
    if (bad)
        printf(TAG "   %u of %u outputs differ, worst %u ULP\n", bad, cs->mt * cs->nt * 128u, worst);
    return bad ? 1u : 0u;
}

// One launch case's line.
static uint32_t g3b_lreport(volatile uint32_t *sy, const g3b_launch_t *lc) {
    const uint32_t passes = lc->mt * lc->kt * lc->nt, g = lc->reps;
    const uint32_t per = L_TOTAL / g, warm = g > 1u ? L_WARM / (g - 1u) : 0u;
    uint32_t bad = 0, worst = 0;
    for (uint32_t h = 0; h < 4; h++) {
        bad += BAD(h);
        if (ULP(h) > worst) worst = ULP(h);
    }
    printf(TAG " %-10s %s %s %u x %4u x %u  %2u GEMMs %6u cc  %6u a GEMM (array %5u)  launch: first %5u "
               "warm %5u  arm once %4u  %s\n",
           lc->name, g3b_arm[G3B_ARM(lc->kind)], G3B_ASYNC(lc->kind) ? "async" : "sync ", lc->mt,
           lc->kt, lc->nt, g, L_TOTAL, per, passes, L_FIRST, warm, L_ARM1, bad ? "FAIL" : "PASS");
    return bad ? 1u : 0u;
}

int main() {
    // ---- L1, derived identically on every hart: every buffer on a 256-byte boundary --------
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 255u) & ~255u);
#define TAKE(n) (p += (((n) + 255u) & ~255u), p - (((n) + 255u) & ~255u))
    volatile uint32_t *sy = (volatile uint32_t *)TAKE(256);
    int8_t *abuf = (int8_t *)TAKE(G3B_ABYTES);
    int8_t *bbuf = (int8_t *)TAKE(G3B_BBYTES + 256u);  // B at bbuf + its offset
    uint16_t *dbuf = (uint16_t *)TAKE(G3B_DBYTES);
    uint16_t *gold = (uint16_t *)TAKE(G3B_DBYTES);
    g3_shape_t *shape = (g3_shape_t *)TAKE(64);
#undef TAKE

    const int isG = snax_is_gemm_core(), isS = snax_is_simd_core();
    const int isX = snax_is_xdma_core(), isD = snax_is_idma_core();
    const uint32_t hart = isG ? 0u : isS ? 1u : isX ? 2u : 3u;

    if ((uint32_t)p > DSV2_L1_TOP) {
        if (isS) printf(TAG " FAIL: the arena ends at 0x%08x, past the stacks at 0x%08x\n",
                        (uint32_t)p, DSV2_L1_TOP);
        return isS ? 1 : 0;
    }
    if (isS) {
        for (uint32_t i = 0; i < 64; i++) sy[i] = 0;
        dsv2_tcdm_prio_install(DSV2_TCDM_PRIO);
        dsv2_print_l1((uint32_t)(p - (uint8_t *)sy), DSV2_L1_TOP - (uint32_t)sy);
        printf(TAG " %u cases, %u launch cases, %u runs each, k = %u\n", G3B_NCASES, G3B_NLAUNCH,
               G3B_REPS, G3B_K);
    }
    if (isG) {  // the generic arm's table, in L1
        for (uint32_t i = 0; i < sizeof(g3_shape_t) / 4u; i++)
            ((volatile uint32_t *)shape)[i] = ((const uint32_t *)&g3_shape_rom)[i];
    }
    snrt_cluster_hw_barrier();

    uint32_t fails = 0, id = isG ? csrr_ss(GEMMX_FINISHED_TASK) : 0u;
    const int8_t *la = 0, *lb = 0;  // the blobs in L1, and how many of their bytes
    uint32_t labytes = 0, lbbytes = 0, lboff = ~0u;

    // ---- the layout cases ---------------------------------------------------------------------
    for (uint32_t c = 0; c < G3B_NCASES; c++) {
        const g3b_case_t *cs = &g3b_cases[c];
        const uint32_t dbytes = cs->mt * cs->nt * 256u;
        int8_t *b = bbuf + cs->boff;
        if (isD) {  // the operands in, D cleared
            if (cs->a != la || cs->abytes > labytes) {
                snrt_dma_start_1d(abuf, cs->a, cs->abytes);
                la = cs->a;
                labytes = cs->abytes;
            }
            if (cs->b != lb || cs->boff != lboff || cs->bbytes > lbbytes) {
                snrt_dma_start_1d(b, cs->b, cs->bbytes);
                lb = cs->b;
                lboff = cs->boff;
                lbbytes = cs->bbytes;
            }
            snrt_dma_start_1d(dbuf, g3_nan, dbytes);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
        if (isG) {  // the GEMM, alone
            g3_arm(cs->mt, cs->kt, cs->nt, G3B_K, cs->lay, cs->alay, cs->pitch);
            for (uint32_t r = 0; r < G3B_REPS; r++) {
                const uint32_t t0 = snrt_mcycle();
                g3_fire(abuf, b, dbuf);
                TMO += dsv2_gemm_wait(++id);
                CYC(r) = snrt_mcycle() - t0;
            }
            BUSY = csrr_ss(GEMMX_PERFORMANCE_COUNTER);
            WAIT_A = csrr_ss(GEMMX_STALL_A);
            WAIT_B = csrr_ss(GEMMX_STALL_B);
            WAIT_D = csrr_ss(GEMMX_STALL_D);
        }
        snrt_cluster_hw_barrier();
        if (isD) {  // the golden in
            snrt_dma_start_1d(gold, cs->y, dbytes);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
        {  // a quarter of D per hart
            const uint32_t q = dbytes / 2u / 4u;  // FP16 outputs a hart
            uint32_t worst;
            BAD(hart) = dsv2_ulp16(dbuf + hart * q, gold + hart * q, q, &worst);
            ULP(hart) = worst;
        }
        snrt_cluster_hw_barrier();
        if (isS) fails += g3b_report(sy, cs);
    }

    // ---- the launch cases ---------------------------------------------------------------------
    for (uint32_t c = 0; c < G3B_NLAUNCH; c++) {
        const g3b_launch_t *lc = &g3b_launch[c];
        const uint32_t dbytes = lc->mt * lc->nt * 256u;
        int8_t *b = bbuf + lc->boff;
        if (isD) {
            if (lc->a != la || lc->abytes > labytes) {
                snrt_dma_start_1d(abuf, lc->a, lc->abytes);
                la = lc->a;
                labytes = lc->abytes;
            }
            if (lc->b != lb || lc->boff != lboff || lc->bbytes > lbbytes) {
                snrt_dma_start_1d(b, lc->b, lc->bbytes);
                lb = lc->b;
                lboff = lc->boff;
                lbbytes = lc->bbytes;
            }
            snrt_dma_start_1d(dbuf, g3_nan, dbytes);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
        if (isG) {
            const uint32_t arm = G3B_ARM(lc->kind), async = G3B_ASYNC(lc->kind);
            uint32_t first = 0, warm = 0, arm1 = 0, tmo = 0;
            const uint32_t t0 = snrt_mcycle();
            if (arm == G3B_DELTA) {
                g3_arm(lc->mt, lc->kt, lc->nt, G3B_K, lc->lay, lc->alay, lc->pitch);
                arm1 = snrt_mcycle() - t0;
            }
            for (uint32_t g = 0; g < lc->reps; g++) {
                const uint32_t ta = snrt_mcycle();
                if (arm == G3B_GEN) {
                    g3_arm_generic(shape, (uint32_t)abuf, (uint32_t)b, (uint32_t)dbuf, lc->mt,
                                   lc->kt, lc->nt, G3B_K, lc->lay, lc->alay, lc->pitch);
                    g3_start();
                } else if (arm == G3B_LEAN) {
                    g3_arm(lc->mt, lc->kt, lc->nt, G3B_K, lc->lay, lc->alay, lc->pitch);
                    g3_fire(abuf, b, dbuf);
                } else {
                    g3_fire(abuf, b, dbuf);
                }
                const uint32_t tb = snrt_mcycle();
                if (g == 0) first = tb - ta;
                else warm += tb - ta;
                ++id;
                if (!async) tmo += dsv2_gemm_wait(id);
                else if (g) tmo += dsv2_gemm_wait(id - 1u);
            }
            tmo += dsv2_gemm_wait(id);
            L_TOTAL = snrt_mcycle() - t0;
            L_FIRST = first;
            L_WARM = warm;
            L_ARM1 = arm1;
            TMO += tmo;
        }
        snrt_cluster_hw_barrier();
        if (isD) {
            snrt_dma_start_1d(gold, lc->y, dbytes);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
        {
            const uint32_t q = dbytes / 2u / 4u;
            uint32_t worst;
            BAD(hart) = dsv2_ulp16(dbuf + hart * q, gold + hart * q, q, &worst);
            ULP(hart) = worst;
        }
        snrt_cluster_hw_barrier();
        if (isS) fails += g3b_lreport(sy, lc);
    }
    if (!isS) return 0;
    if (TMO) {
        printf(TAG " FAIL: %u GEMM task(s) never retired\n", TMO);
        fails++;
    }
    printf(fails ? TAG " FAIL (%u)\n" : TAG " PASS\n", fails);
    return (int)fails;
}
