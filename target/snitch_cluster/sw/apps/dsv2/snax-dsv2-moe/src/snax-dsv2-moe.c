// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V2-Lite layer 1's mixture of experts for one token (I2, with D3 and V7),
// h -> out = h + MoE(h): the block kernel of snax-dsv2-moe.h (stages 14 to 23), run once on
// the golden pack's h.
//
// CHECKS, against the device model's chained values, bit for bit, once out is computed and the
// goldens loaded: the block's results, out and the router's decision, the ids (in order, all
// with weights in the ELF) and their weights. With DSV2_STAGE_CHECKS = 1 every stage as well: hn
// and its A operand (the row the GEMVs read), the logits, p, per slot the dequantised gate|up
// and the down output, and the shared ones. main returns the number of failed checks. After
// the report, the block's engine spans and the cluster census over it ([ORIGIN], [SPAN],
// [UTIL]; snax-dsv2-trace.h).

#include "data.h"
#include "snax-core-roles.h"
#include "snax-perf-census.h"
#include "snax-dsv2-moe.h"

#define TAG "[MOE]"
#define DONE_CHECK sy[0]  // goldens staged (iDMA -> SIMD)

static uint32_t chk16(const char *what, const uint16_t *got, const uint16_t *want, uint32_t n) {
    n = dsv2_check_terms(n);
    uint32_t worst, bad = dsv2_ulp16(got, want, n, &worst);
    printf(TAG "   %s %-26s %5u/%5u differ, worst %u ULP\n", bad ? "FAIL" : "PASS", what, bad, n,
           worst);
    return bad ? 1u : 0u;
}

// A GEMV's A operand against the INT8 values of its row 0.
static uint32_t chka(const char *what, const int8_t *a, const int8_t *want, uint32_t n) {
    n = dsv2_check_terms(n);
    uint32_t bad = dsv2_a_row_diff(a, want, n, 0u);
    printf(TAG "   %s %-26s %5u/%5u differ in row 0\n", bad ? "FAIL" : "PASS", what, bad, n);
    return bad ? 1u : 0u;
}

int main() {
    // ---- L1, derived identically on every hart ------------------------------------------
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 63u) & ~63u);
    uint8_t *arena = p;
    volatile uint32_t *sy = (volatile uint32_t *)p;  // this app's words
    volatile uint32_t *bsy = sy + 256;               // the block's
    p += 1024 + MOE_SY_BYTES;
    uint16_t *h = (uint16_t *)(p + DSV2_BEAT);       // [the norm's seed][h]
    p += DSV2_BEAT + 2 * HID;
    dsv2_trace_t *tr = (dsv2_trace_t *)p;           // the block's spans (snax-dsv2-trace.h)
    p += (sizeof(dsv2_trace_t) + 63u) & ~63u;
    dsv2_moe_t m;
    p = dsv2_moe_carve(&m, p, bsy, h);
    m.tr = tr;
    const uint32_t arena_beats = (uint32_t)(p - arena) / DSV2_BEAT;
    // goldens, staged into the weight buffers once the last job has retired
    uint8_t *gb = (uint8_t *)m.bbuf[0];
    uint16_t *g_hn_l = (uint16_t *)gb;          gb += 2 * HID;
    int8_t *g_ha_l = (int8_t *)gb;              gb += HID;
    uint16_t *g_lg_l = (uint16_t *)gb;          gb += 2 * N_EXP;
    uint16_t *g_p_l = (uint16_t *)gb;           gb += 2 * N_EXP;
    uint32_t *g_ids_l = (uint32_t *)gb;         gb += 4 * 8;
    uint16_t *g_w_l = (uint16_t *)gb;           gb += 2 * 8;
    uint16_t *g_gu_l = (uint16_t *)gb;          gb += TOP_K * 2 * 2 * I_EXP;
    uint16_t *g_y_l = (uint16_t *)gb;           gb += TOP_K * 2 * HID;
    uint16_t *g_sgu_l = (uint16_t *)gb;         gb += 2 * 2 * I_SH;
    uint16_t *g_sy_l = (uint16_t *)gb;          gb += 2 * HID;
    uint16_t *g_out_l = (uint16_t *)gb;         gb += 2 * HID;

    const int isG = snax_is_gemm_core(), isS = snax_is_simd_core();
    const int isX = snax_is_xdma_core(), isD = snax_is_idma_core();
    if (isS) dsv2_print_l1((uint32_t)(p - arena), DSV2_L1_TOP - (uint32_t)arena);
    if ((uint32_t)p > DSV2_L1_TOP || gb > (uint8_t *)m.bbuf[0] + MOE_NBUF * MOE_BUF) {
        if (isS) printf(TAG " FAIL: the arena (to 0x%08x) or the goldens overflow\n", (uint32_t)p);
        return isS ? 1 : 0;
    }
    if (isS) {
        dsv2_tcdm_prio_install(DSV2_TCDM_PRIO);
        printf(TAG " TCDM arbitration: %s (control 0x%x)\n",
               snax_tcdm_policy_name(DSV2_TCDM_PRIO), snax_tcdm_arb_ctrl_read());
    }
    if (isX) dsv2_xdma_fill_zero(arena, arena_beats);
    snrt_cluster_hw_barrier();
    if (isG) MOE_T_ORG(&m) = snrt_mcycle();
    if (isS) snax_perf_arm();  // the contention census covers the block
    snrt_cluster_hw_barrier();
    while (MOE_T_ORG(&m) == 0u) {  // the barrier does not order the store
    }

    if (isX) {
#if MOE_DUAL
        dsv2_moe_xdma(&m);
#endif
        return 0;
    }
    if (isG) {
        dsv2_moe_gemm(&m);
        return 0;
    }
    if (isD) {
        dsv2_moe_idma(&m, h16);
        MOE_SPIN(&m, MOE_DONE(&m), 1u);
        snrt_dma_start_1d(g_ids_l, g_ids, 4 * TOP_K);
        snrt_dma_start_1d(g_w_l, g_w, 2 * TOP_K);
        snrt_dma_start_1d(g_out_l, g_out, 2 * HID);
#if DSV2_STAGE_CHECKS
        snrt_dma_start_1d(g_hn_l, g_hn, 2 * HID);
        snrt_dma_start_1d(g_ha_l, g_ha, HID);
        snrt_dma_start_1d(g_lg_l, g_logits, 2 * N_EXP);
        snrt_dma_start_1d(g_p_l, g_p, 2 * N_EXP);
        snrt_dma_start_1d(g_gu_l, g_gu, TOP_K * 2 * 2 * I_EXP);
        snrt_dma_start_1d(g_y_l, g_y, TOP_K * 2 * HID);
        snrt_dma_start_1d(g_sgu_l, g_sh_gu, 2 * 2 * I_SH);
        snrt_dma_start_1d(g_sy_l, g_sh_y, 2 * HID);
#endif
        snrt_dma_wait_all();
        DONE_CHECK = 1u;
        return 0;
    }

    // ============================== the SIMD
    dsv2_moe_simd(&m);
    snax_perf_snapshot_t census;
    snax_perf_read(&census);
    dsv2_moe_report(&m, TAG);
    snax_perf_report(&census, MOE_T_JOB(&m, MOE_NJOBS(&m) - 1u));
    printf("[ORIGIN] %u\n", MOE_T_ORG(&m));
    dsv2_trace_print(tr);
    dsv2_util_print(&census, MOE_P(&m, MOE_P_SBUSY));
    dsv2_spin_ge(&DONE_CHECK, 1u, &MOE_TMO(&m));

    uint32_t fails = 0;
#if DSV2_STAGE_CHECKS
    fails += chk16("hn", m.hn, g_hn_l, HID);
    fails += chka("hn A operand", m.ha, g_ha_l, HID);
    fails += chk16("router logits", m.lg, g_lg_l, N_EXP);
    fails += chk16("softmax", m.pr, g_p_l, N_EXP);
#endif
    uint32_t idbad = 0;
    for (uint32_t i = 0; i < TOP_K; i++) idbad += MOE_IDS(&m, 0, i) != g_ids_l[i];
    fails += dsv2_check(TAG, idbad == 0 && !MOE_REFUSED(&m), "top-6",
                        "the golden's ids, in order, all present");
    fails += chk16("weights", m.w, g_w_l, TOP_K);
#if DSV2_STAGE_CHECKS
    fails += chk16("slots: gate|up (dequant)", m.gu, g_gu_l, TOP_K * 2 * I_EXP);
    fails += chk16("slots: down outputs e_i", m.es, g_y_l, TOP_K * HID);
    fails += chk16("shared: gate|up (dequant)", m.sgu, g_sgu_l, 2 * I_SH);
    fails += chk16("shared: down output", m.sh, g_sy_l, HID);
#endif
    fails += chk16("out = h + s + sum w_i e_i", dsv2_moe_out(&m, 0), g_out_l, HID);
    if (MOE_TMO(&m)) {
        printf(TAG " FAIL: a task never retired or a wait ran out\n");
        fails++;
    }
    printf(fails ? TAG " FAIL (%u checks)\n" : TAG " PASS\n", fails);
    return (int)fails;
}
