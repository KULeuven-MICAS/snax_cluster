// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V2-Lite layer 1's MLA block for one new token (I1), x -> h = x + MLA(x), with the
// token's row appended to the latent cache: the block kernel of snax-dsv2-mla.h (stages 1 to
// 13), run once on the golden pack's token, and every stage checked.
//
// CHECKS against the device model's chained values (sw/apps/dsv2/dsv2), bit for bit, once h is
// computed and the goldens loaded: the block's results, h = x + attn and the appended row read
// back from both cache copies. With DSV2_STAGE_CHECKS = 1 every stage as well: xn, ckv, cn, the
// cache row, q, the rotated q_pe | k_pe, Q8 (q~ quantised: q~ itself shares its L1 with the
// attention), the last two tiles' scores, m, l, the last P8, c / l, o~, the heads' outputs,
// W_O's A operand (the row the GEMV reads) and raw output, and attn. main returns the number
// of failed checks.
// After the report, the block's engine spans and the cluster census over it ([ORIGIN], [SPAN],
// [UTIL]; snax-dsv2-trace.h).

#include "data.h"
#include "snax-core-roles.h"
#include "snax-perf-census.h"
#include "snax-dsv2-mla.h"

#define TAG "[MLA-BLOCK]"
#define DONE_CHECK sy[0]  // goldens staged (iDMA -> SIMD)

static uint32_t chk16(const char *what, const uint16_t *got, const uint16_t *want, uint32_t n) {
    n = dsv2_check_terms(n);
    uint32_t worst, bad = dsv2_ulp16(got, want, n, &worst);
    printf(TAG "   %s %-36s %5u/%5u differ, worst %u ULP\n", bad ? "FAIL" : "PASS", what, bad, n,
           worst);
    return bad ? 1u : 0u;
}

static uint32_t chk8(const char *what, const int8_t *got, const int8_t *want, uint32_t n) {
    n = dsv2_check_terms(n);
    uint32_t worst, bad = dsv2_lsb8(got, want, n, &worst);
    printf(TAG "   %s %-36s %5u/%5u differ, worst %u LSB\n", bad ? "FAIL" : "PASS", what, bad, n,
           worst);
    return bad ? 1u : 0u;
}

// A GEMV's A operand against the INT8 values of its row 0.
static uint32_t chka(const char *what, const int8_t *a, const int8_t *want, uint32_t n) {
    n = dsv2_check_terms(n);
    uint32_t bad = dsv2_a_row_diff(a, want, n, 0u);
    printf(TAG "   %s %-36s %5u/%5u differ in row 0\n", bad ? "FAIL" : "PASS", what, bad, n);
    return bad ? 1u : 0u;
}

int main() {
    // ---- L1, derived identically on every hart ------------------------------------------
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 63u) & ~63u);
    uint8_t *arena = p;
    volatile uint32_t *sy = (volatile uint32_t *)p;  // this app's words
    volatile uint32_t *bsy = sy + 256;               // the block's
    p += 2048;
    uint16_t *h = (uint16_t *)(p + DSV2_BEAT);       // [free beat][h]
    p += DSV2_BEAT + 2 * HID;
    dsv2_trace_t *tr = (dsv2_trace_t *)p;           // the block's spans (snax-dsv2-trace.h)
    p += (sizeof(dsv2_trace_t) + 63u) & ~63u;
    dsv2_mla_t m;
    p = dsv2_mla_carve(&m, p, bsy, h);
    m.tr = tr;
    const uint32_t arena_beats = (uint32_t)(p - arena) / DSV2_BEAT;
    static const dsv2_mla_pass_t passes[NP] = MLA_PASSES;
    const dsv2_mla_pass_t *ps = &passes[0];
    // the goldens, staged into the slabs once h is computed; the cache read-back after them
    uint8_t *gb = m.slab[0];
    uint16_t *g_xn_l = (uint16_t *)gb;          gb += 2 * HID;
    uint16_t *g_ckv_l = (uint16_t *)gb;         gb += 2 * MLA_DQK;
    uint16_t *g_cn_l = (uint16_t *)gb;          gb += 2 * MLA_DV;
    int8_t *g_row_l = (int8_t *)gb;             gb += MLA_DQK;
    uint16_t *g_q_l = (uint16_t *)gb;           gb += 2 * MLA_Q_N;
    uint16_t *g_rot_l = (uint16_t *)gb;         gb += MLA_ROPE_B;
    int8_t *g_q8_l = (int8_t *)gb;              gb += MLA_BR * MLA_DQK;
    uint16_t *g_s_l = (uint16_t *)gb;           gb += 2 * MLA_SBEATS * DSV2_BEAT;
    uint16_t *g_m_l = (uint16_t *)gb;           gb += DSV2_BEAT;
    uint16_t *g_l_l = (uint16_t *)gb;           gb += DSV2_BEAT;
    int8_t *g_p8_l = (int8_t *)gb;              gb += MLA_PBEATS * DSV2_BEAT;
    uint16_t *g_rsc_l = (uint16_t *)gb;         gb += DSV2_BEAT;
    uint16_t *g_ot_l = (uint16_t *)gb;          gb += 2 * HEADS * MLA_DV;
    uint16_t *g_oh_l = (uint16_t *)gb;          gb += 2 * HID;
    uint16_t *g_attn_l = (uint16_t *)gb;        gb += 2 * HID;
    uint16_t *g_h_l = (uint16_t *)gb;           gb += 2 * HID;
    const uint32_t kblk = (MLA_DQK / DSV2_KU) * 64u;                  // one 16-token key block
    const uint32_t vrun = 64u, vbytes = (MLA_DV / DSV2_MR) * vrun;    // the 4-token value blocks
    int8_t *kgot = (int8_t *)gb;                gb += kblk;
    int8_t *kwant = (int8_t *)gb;               gb += kblk;
    int8_t *vgot = (int8_t *)gb;                gb += vbytes;
    int8_t *vwant = (int8_t *)gb;               gb += vbytes;
    int8_t *g_oa_l = (int8_t *)m.region;                              // R's first view is dead by then
    uint16_t *g_ya_l = (uint16_t *)(m.region + DSV2_MR * HID);

    const int isG = snax_is_gemm_core(), isS = snax_is_simd_core();
    const int isX = snax_is_xdma_core(), isD = snax_is_idma_core();
    if (isS) dsv2_print_l1((uint32_t)(p - arena), DSV2_L1_TOP - (uint32_t)arena);
    if ((uint32_t)p > DSV2_L1_TOP || gb > m.slab[0] + 2 * MLA_SLAB) {
        if (isS)
            printf(TAG " FAIL: the arena (to 0x%08x, stacks at 0x%08x) or the goldens overflow\n",
                   (uint32_t)p, DSV2_L1_TOP);
        return isS ? 1 : 0;
    }
    if (isS) {
        dsv2_tcdm_prio_install(DSV2_TCDM_PRIO);
        printf(TAG " TCDM arbitration: %s (control 0x%x)\n",
               snax_tcdm_policy_name(DSV2_TCDM_PRIO), snax_tcdm_arb_ctrl_read());
    }
    if (isX) dsv2_xdma_fill_zero(arena, arena_beats);
    snrt_cluster_hw_barrier();
    if (isG) MLA_T_ORG(&m) = snrt_mcycle();
    if (isS) snax_perf_arm();  // the contention census covers the block
    snrt_cluster_hw_barrier();
    while (MLA_T_ORG(&m) == 0u) {  // the barrier does not order the store
    }
    dsv2_mla_begin(&m, ps);

    if (isX) {
        dsv2_mla_xdma(&m);
        return 0;
    }
    if (isG) {
        dsv2_mla_gemm(&m, ps);
        return 0;
    }
    if (isD) {
        dsv2_mla_idma(&m, ps);
        // the goldens, and the cache blocks the append touched, once h is computed
        MLA_SPIN(&m, MLA_DONE(&m), 1u);
#if DSV2_STAGE_CHECKS
        snrt_dma_start_1d(g_xn_l, g_xn, 2 * HID);
        snrt_dma_start_1d(g_ckv_l, g_ckv, 2 * MLA_DQK);
        snrt_dma_start_1d(g_cn_l, g_cn, 2 * MLA_DV);
        snrt_dma_start_1d(g_row_l, g_row, MLA_DQK);
        snrt_dma_start_1d(g_q_l, g_q, 2 * MLA_Q_N);
        snrt_dma_start_1d(g_rot_l, g_rot, MLA_ROPE_B);
        snrt_dma_start_1d(g_q8_l, g_q8, MLA_BR * MLA_DQK);
        snrt_dma_start_1d(g_s_l, g_s, 2 * MLA_SBEATS * DSV2_BEAT);
        snrt_dma_start_1d(g_m_l, g_m, DSV2_BEAT);
        snrt_dma_start_1d(g_l_l, g_l, DSV2_BEAT);
        snrt_dma_start_1d(g_p8_l, g_p8, MLA_PBEATS * DSV2_BEAT);
        snrt_dma_start_1d(g_rsc_l, g_rsc, DSV2_BEAT);
        snrt_dma_start_1d(g_ot_l, g_ot, 2 * HEADS * MLA_DV);
        snrt_dma_start_1d(g_oh_l, g_oh, 2 * HID);
        snrt_dma_start_1d(g_oa_l, g_oa, HID);
        snrt_dma_start_1d(g_ya_l, g_ya, 2 * HID);
        snrt_dma_start_1d(g_attn_l, g_attn, 2 * HID);
#endif
        snrt_dma_start_1d(g_h_l, g_h, 2 * HID);
        snrt_dma_start_1d(kgot, key + (ps->pos / 16u) * kblk, kblk);
        snrt_dma_start_1d(kwant, g_key + (ps->pos / 16u) * kblk, kblk);
        snrt_dma_start_2d(vgot, val + (ps->pos / 4u) * 64u, vrun, vrun, CAP * 16u,
                          MLA_DV / DSV2_MR);
        snrt_dma_start_2d(vwant, g_val + (ps->pos / 4u) * 64u, vrun, vrun, CAP * 16u,
                          MLA_DV / DSV2_MR);
        snrt_dma_wait_all();
        DONE_CHECK = 1u;
        return 0;
    }

    // ============================== the SIMD
    dsv2_mla_simd(&m, ps);
    snax_perf_snapshot_t census;
    snax_perf_read(&census);
    dsv2_mla_report(&m, TAG);
    snax_perf_report(&census, MLA_T_SS(&m, 7));
    printf("[ORIGIN] %u\n", MLA_T_ORG(&m));
    dsv2_trace_print(tr);
    dsv2_util_print(&census, MLA_P(&m, MLA_P_SBUSY));
    dsv2_spin_ge(&DONE_CHECK, 1u, &MLA_TMO(&m));

    uint32_t fails = 0, w;
#if DSV2_STAGE_CHECKS
    printf(TAG " 1-5: norms, projections, RoPE, the cache row\n");
    fails += chk16("xn", m.xn, g_xn_l, HID);
    fails += chk16("ckv = xn . W_DKV (dequantised)", m.ckv, g_ckv_l, MLA_DQK);
    fails += chk16("cn", m.cn, g_cn_l, MLA_DV);
    fails += chk8("row [c8 | kpe8]", m.row, g_row_l, MLA_DQK);
    fails += chk16("q = xn . W_Q (dequantised)", m.q16, g_q_l, MLA_Q_N);
    fails += chk16("q_pe, k_pe rotated", m.rot, g_rot_l, MLA_ROPE_N);
#endif
    printf(TAG " 7: the append\n");
    fails += dsv2_check(TAG, dsv2_lsb8(kgot, kwant, kblk, &w) == 0, "key copy",
                        "the appended row's 16-token block, byte-exact");
    fails += dsv2_check(TAG, dsv2_lsb8(vgot, vwant, vbytes, &w) == 0, "value copy",
                        "the appended row's 4-token blocks, byte-exact");
#if DSV2_STAGE_CHECKS
    printf(TAG " 8-10: attention\n");
    fails += chk8("Q8 (A-layout of [32 x 576])", m.q8, g_q8_l, MLA_BR * MLA_DQK);
    for (uint32_t k = 0; k < 2u; k++) {
        const uint32_t j = m.nt - 2u + k;
        fails += chk16(j == m.nt - 1u ? "S^T, last tile" : "S^T, tile before last",
                       (const uint16_t *)(m.sbuf + (j & 1u) * (1 + MLA_SBEATS) * DSV2_BEAT +
                                          DSV2_BEAT),
                       g_s_l + k * MLA_SBEATS * MLA_BR, MLA_SBEATS * MLA_BR);
    }
    fails += chk16("m", (const uint16_t *)m.mrun, g_m_l, MLA_BR);
    fails += chk16("l", (const uint16_t *)m.lrun, g_l_l, MLA_BR);
    fails += chk8("P8, last tile (interleaved)", (const int8_t *)m.p8[(m.nt - 1u) & 1u], g_p8_l,
                  MLA_PBEATS * DSV2_BEAT);
    fails += chk16("c / l per query", (const uint16_t *)m.rsc, g_rsc_l, MLA_BR);
    fails += chk16("o~ (16 heads x 512)", m.xt, g_ot_l, HEADS * MLA_DV);
    printf(TAG " 11-13: W_UV, W_O, the residual\n");
    fails += chk16("o = o~ . W_UV (dequantised)", m.oh, g_oh_l, HID);
    fails += chka("W_O A operand", m.oa, g_oa_l, HID);
    fails += chk16("attn raw = RNE(o8 . W_O * 2^-k)", m.ya, g_ya_l, HID);
    fails += chk16("attn = o . W_O (dequantised)", m.attn, g_attn_l, HID);
#else
    printf(TAG " 13: the block's output\n");
#endif
    fails += chk16("h = x + attn", m.h, g_h_l, HID);
    if (MLA_TMO(&m)) {
        printf(TAG " FAIL: %u bounded wait(s) ran out\n", MLA_TMO(&m));
        fails++;
    }
    printf(fails ? TAG " FAIL (%u checks)\n" : TAG " PASS\n", fails);
    return (int)fails;
}
