// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GEMM shapes in isolation on the (16, 4, 16) array, the shapes of HeMAiA's FFN prefill GEMM
// bench, on one cluster and checked:
//
//     D (16 M x 16 N, FP16) = RNE(A . B * 2^-k)     A: [16 M, 4 K] INT8, B: [4 K, 16 N]
//
// for the cases data.h lists (data/datagen.py; GEMM_BENCH_SET in the Makefile). A case is a
// shape, the way B is laid out and read (its mode), and where B sits against A:
//
//   i8    INT8 weights in the B-layout, 64 B a pass
//   i4    INT4 weights in the paired layout, 32 B a pass, consecutive passes 64 B apart: B
//         advances at A's stride (snax-dsv2.h, THE GEMM)
//   i4b   INT4 weights in the B-layout, 32 B a pass, 32 B apart: B's four banks walk through
//         A's eight, two passes in every eight -- the bank-conflict control
//
// The three modes of one shape compute the same D, so they share one golden.
//
// Each case runs alone, GB_REPS times back to back; the last run is the measurement, the first
// also pays the instruction fetch. Nothing else touches L1 while it runs: the loads, the clear
// and the check sit between barriers around it.
//
// PER CASE
//   iDMA   A and B into L1 unless the last case's already cover them: A on a 256-byte
//          boundary, B the case's offset past one
//   xDMA   D filled with FP16 NaN, so a tile the GEMM never writes cannot pass for an earlier
//          case's
//   GEMM   dsv2_gemm_arm, B's walk replaced for i4b; GB_REPS tasks, each waited for
//   iDMA   the golden D into L1
//   all    each hart compares a quarter of D with the golden, bit for bit
//   SIMD   prints the case: the last run's cycles and cycles per array pass (M K N passes), the
//          array's busy cycles and those it waited for A, for B and for D (VersaCore's stall
//          census of that run), and the first run's cycles
//
// main returns the number of cases whose D differs, plus one if a task never retired.

#include "data.h"
#include "snax-core-roles.h"
#include "snax-dsv2.h"

#if !SNAX_HAS_GEMM_CORE || !SNAX_HAS_SIMD_CORE || !SNAX_HAS_XDMA_CORE || !SNAX_HAS_IDMA_CORE
#error "this bench needs the four-engine cluster"
#endif
#if GB_REPS < 1 || GB_REPS > 8
#error "GB_REPS: 1 to 8 runs a case"
#endif

#define TAG "[GBENCH]"
#define GB_I8 0u
#define GB_I4 1u
#define GB_I4B 2u
#define GB_NAN16X2 0x7E007E00u  // two FP16 quiet NaNs, the D clear

// ---- shared words ----------------------------------------------------------------------
#define TMO sy[0]                // bounded waits that ran out
#define CYC(r) sy[8 + (r)]       // GEMM: run r's cycles, fire to retirement
#define BUSY sy[16]              // the array's busy cycles, last run
#define WAIT_A sy[17]            //   of which waiting for A
#define WAIT_B sy[18]            //   for B (A there)
#define WAIT_D sy[19]            //   for the array (A and B there)
#define BAD(h) sy[24 + (h)]      // hart h: the outputs of its quarter that differ
#define ULP(h) sy[28 + (h)]      // hart h: their largest distance, in ULPs

static const char *const gb_mode[] = {"i8 ", "i4 ", "i4b"};

// i4b: B in the B-layout, nibble-packed -- k inner at 32 B, tile n 32 kt bytes on, the row tile
// at stride 0. Replaces the paired walk dsv2_gemm_arm programmed; its mask and converter stay.
__attribute__((always_inline)) static inline void gb_b_int4_blayout(uint32_t mt, uint32_t kt,
                                                                    uint32_t nt) {
    csrw_ss(T_BOUND_READER_1_0, kt);
    csrw_ss(T_STRIDE_READER_1_0, 32);
    csrw_ss(T_BOUND_READER_1_1, nt);
    csrw_ss(T_STRIDE_READER_1_1, 32u * kt);
    csrw_ss(T_BOUND_READER_1_2, mt);
    csrw_ss(T_STRIDE_READER_1_2, 0);
    csrw_ss(T_BOUND_READER_1_3, 1);
    csrw_ss(T_STRIDE_READER_1_3, 0);
}

// One case's line; returns 1 if its D differs from the golden.
static uint32_t gb_report(volatile uint32_t *sy, const gb_case_t *cs) {
    const uint32_t passes = cs->mt * cs->kt * cs->nt, cyc = CYC(GB_REPS - 1);
    const uint32_t milli = cyc * 1000u / passes;
    uint32_t bad = 0, worst = 0;
    for (uint32_t h = 0; h < 4; h++) {
        bad += BAD(h);
        if (ULP(h) > worst) worst = ULP(h);
    }
    printf(TAG " %-9s %s B+%3u  %u x %3u x %2u  %6u passes %6u cc  %u.%03u cc/pass  busy %6u  "
               "wait A %4u B %4u D %4u  first %6u cc  %s\n",
           cs->name, gb_mode[cs->mode], cs->boff, cs->mt, cs->kt, cs->nt, passes, cyc,
           milli / 1000u, milli % 1000u, BUSY, WAIT_A, WAIT_B, WAIT_D, CYC(0),
           bad ? "FAIL" : "PASS");
    if (bad)
        printf(TAG "   %u of %u outputs differ, worst %u ULP\n", bad,
               cs->mt * cs->nt * 256u, worst);
    return bad ? 1u : 0u;
}

int main() {
    // ---- L1, derived identically on every hart: every buffer on a 256-byte boundary --------
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 255u) & ~255u);
    uint8_t *arena = p;
#define TAKE(n) (p += (((n) + 255u) & ~255u), p - (((n) + 255u) & ~255u))
    volatile uint32_t *sy = (volatile uint32_t *)TAKE(256);
    int8_t *abuf = (int8_t *)TAKE(GB_ABYTES);
    int8_t *bbuf = (int8_t *)TAKE(GB_BBYTES + 256u);  // B at bbuf + its offset
    uint16_t *dbuf = (uint16_t *)TAKE(GB_DBYTES);
    uint16_t *gold = (uint16_t *)TAKE(GB_DBYTES);
#undef TAKE
    const uint32_t arena_beats = (uint32_t)(p - arena) / DSV2_BEAT;

    const int isG = snax_is_gemm_core(), isS = snax_is_simd_core();
    const int isX = snax_is_xdma_core(), isD = snax_is_idma_core();
    const uint32_t hart = isG ? 0u : isS ? 1u : isX ? 2u : 3u;

    if (isS) dsv2_print_l1((uint32_t)(p - arena), DSV2_L1_TOP - (uint32_t)arena);
    if ((uint32_t)p > DSV2_L1_TOP) {
        if (isS) printf(TAG " FAIL: the arena ends at 0x%08x, past the stacks at 0x%08x\n",
                        (uint32_t)p, DSV2_L1_TOP);
        return isS ? 1 : 0;
    }
    if (isS) {
        dsv2_tcdm_prio_install(DSV2_TCDM_PRIO);
        printf(TAG " %u cases, %u runs each, the last measured; k = %u; TCDM arbitration: %s\n",
               GB_NCASES, GB_REPS, GB_K, snax_tcdm_policy_name(DSV2_TCDM_PRIO));
    }
    if (isX) dsv2_xdma_fill_zero(arena, arena_beats);
    snrt_cluster_hw_barrier();

    uint32_t fails = 0, id = isG ? csrr_ss(GEMMX_FINISHED_TASK) : 0u, lboff = ~0u;
    const int8_t *la = 0, *lb = 0;  // the blobs in L1, and how many of their bytes
    uint32_t labytes = 0, lbbytes = 0;
    for (uint32_t c = 0; c < GB_NCASES; c++) {
        const gb_case_t *cs = &gb_cases[c];
        const uint32_t dbytes = cs->mt * cs->nt * 512u;
        int8_t *b = bbuf + cs->boff;
        // ---- the operands in, D cleared
        if (isD) {
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
            snrt_dma_wait_all();
        }
        if (isX) dsv2_xdma_fill(dbuf, dbytes / DSV2_BEAT, GB_NAN16X2);
        snrt_cluster_hw_barrier();
        // ---- the GEMM, alone
        if (isG) {
            dsv2_gemm_arm(cs->mt, cs->kt, cs->nt, GB_K, cs->mode != GB_I8);
            if (cs->mode == GB_I4B) gb_b_int4_blayout(cs->mt, cs->kt, cs->nt);
            for (uint32_t r = 0; r < GB_REPS; r++) {
                const uint32_t t0 = snrt_mcycle();
                dsv2_gemv_fire(abuf, b, dbuf);
                TMO += dsv2_gemm_wait(++id);
                CYC(r) = snrt_mcycle() - t0;
            }
            BUSY = csrr_ss(GEMMX_PERFORMANCE_COUNTER);
            WAIT_A = csrr_ss(GEMMX_STALL_A);
            WAIT_B = csrr_ss(GEMMX_STALL_B);
            WAIT_D = csrr_ss(GEMMX_STALL_D);
        }
        snrt_cluster_hw_barrier();
        // ---- the golden in, then a quarter of D per hart
        if (isD) {
            snrt_dma_start_1d(gold, cs->y, dbytes);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
        {
            const uint32_t q = dbytes / 2u / 4u;  // FP16 outputs a hart, a multiple of 64
            uint32_t worst;
            BAD(hart) = dsv2_ulp16(dbuf + hart * q, gold + hart * q, q, &worst);
            ULP(hart) = worst;
        }
        snrt_cluster_hw_barrier();
        if (isS) fails += gb_report(sy, cs);
    }
    if (!isS) return 0;
    if (TMO) {
        printf(TAG " FAIL: %u GEMM task(s) never retired\n", TMO);
        fails++;
    }
    printf(fails ? TAG " FAIL (%u)\n" : TAG " PASS\n", fails);
    return (int)fails;
}
