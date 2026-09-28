// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V2-Lite layer 1, pass after pass (I3; I4 for consecutive passes; I7 with two tokens
// per pass, as snax-dsv2-spec): per pass of NTOK tokens,
//
//     h   = x + MLA(x)      snax-dsv2-mla.h, stages 1 to 13; appends the tokens' cache rows
//     out = h + MoE(h)      snax-dsv2-moe.h, stages 14 to 23
//
// Pass p starts at position MLA_PASSES[p].pos, so its attention reads the rows every earlier
// pass appended; a pass starting below the end of the one before overwrites that pass's last
// rows (a rejected draft). The data's passes (data/params.hjson) decide which.
//
// L1. The two blocks take turns in ONE region; each carves it its own way. Between them, only h
// (with a free beat below it, the MoE norm's seed) and the sync words live outside it. Before
// each block every hart meets at a barrier, the xDMA zeroes the region (TCDM has no reset, and
// the blocks need Q8's zero rows and l = 0), and each block gets fresh sync words, so no pass
// sees a counter or a stale store of the one before.
//
// CHECKS, per pass and token, against the device model's values bit for bit: h, the top-6 ids
// and weights, out; after the last pass, the key and value blocks every append touched. These
// are the layer's results, so DSV2_STAGE_CHECKS changes nothing here: snax-dsv2-mla and
// snax-dsv2-moe check the blocks' intermediates. main returns the number of failed checks.
//
// TRACE. After each block the SIMD hart prints the block's engine spans and the cluster census
// over it ([BLOCK], [SPAN], [UTIL]; snax-dsv2-trace.h), while the other harts wait at the next
// barrier, and empties the trace for the next block.

#include "data.h"
#include "snax-core-roles.h"
#include "snax-dsv2-mla.h"
#include "snax-dsv2-moe.h"

#define TAG "[LAYER]"
#define GOLD(p) sy[1 + (p)]  // pass p's goldens staged (iDMA -> SIMD)
#define CACHE_IN sy[15]      // the cache blocks read back (iDMA -> SIMD)
#define APP_TMO sy[16]

typedef struct {
    const uint16_t *h;
    const uint32_t *ids;
    const uint16_t *w, *out;
} layer_gold_t;

static uint32_t chk16(uint32_t tok, const char *what, const uint16_t *got, const uint16_t *want,
                      uint32_t n) {
    n = dsv2_check_terms(n);
    uint32_t worst, bad = dsv2_ulp16(got, want, n, &worst);
    printf(TAG "   %s token %u: %-22s %5u/%5u differ, worst %u ULP\n", bad ? "FAIL" : "PASS", tok,
           what, bad, n, worst);
    return bad ? 1u : 0u;
}

// Every hart: the barrier that ends one step and the region zeroing that starts the next.
static void next_block(int isX, uint8_t *region, uint32_t beats) {
    snrt_cluster_hw_barrier();
    if (isX) dsv2_xdma_fill_zero(region, beats);
    snrt_cluster_hw_barrier();
}

// A block's spans and utilisation, then an empty trace for the next block.
static void layer_trace(dsv2_trace_t *tr, uint32_t pass, const char *block, uint32_t org,
                        const snax_perf_snapshot_t *census, uint32_t simd_busy) {
    printf("[BLOCK] %u %s %u\n", pass, block, org);
    dsv2_trace_print(tr);
    dsv2_util_print(census, simd_busy);
    dsv2_trace_clear(tr);
}

// The block's cycle origin, published by the GEMM hart.
static void set_origin(int isG, volatile uint32_t *org) {
    if (isG) *org = snrt_mcycle();
    snrt_cluster_hw_barrier();
    while (*org == 0u) {  // the barrier does not order the store
    }
}

int main() {
    static const dsv2_mla_pass_t passes[NP] = MLA_PASSES;
    static const layer_gold_t golds[NP] = LAYER_GOLDENS;
    // ---- L1, derived identically on every hart ------------------------------------------
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 63u) & ~63u);
    uint8_t *arena = p;
    volatile uint32_t *sy = (volatile uint32_t *)p;  // this app's words
    p += 1024;
    volatile uint32_t *bsy = (volatile uint32_t *)p;  // per pass: the MLA's words, the MoE's
    p += NP * (1024 + MOE_SY_BYTES);
    dsv2_trace_t *tr = (dsv2_trace_t *)p;            // a block's spans
    p += (sizeof(dsv2_trace_t) + 63u) & ~63u;
    uint16_t *h = (uint16_t *)(p + DSV2_BEAT);        // [the norm's seed][h] per token
    p += NTOK * MLA_HP;
    uint8_t *region = p;
    dsv2_mla_t ma;
    dsv2_moe_t mo;
    uint8_t *end_a = dsv2_mla_carve(&ma, region, bsy, h);
    uint8_t *end_o = dsv2_moe_carve(&mo, region, bsy, h);
    ma.tr = tr;
    mo.tr = tr;
    uint8_t *end = end_a > end_o ? end_a : end_o;
    const uint32_t region_beats = (uint32_t)(end - region) / DSV2_BEAT;
    // a pass's goldens, staged into the MoE's weight buffers once out is computed
    uint16_t *g_h_l = (uint16_t *)mo.bbuf[0];
    uint16_t *g_out_l = g_h_l + NTOK * HID;
    uint32_t *g_ids_l = (uint32_t *)(g_out_l + NTOK * HID);
    uint16_t *g_w_l = (uint16_t *)(g_ids_l + NTOK * 8);
    // after the last pass, the key and value blocks the appends touched, and the expected ones
    const uint32_t kblk = (MLA_DQK / DSV2_KU) * 64u;
    const uint32_t k0 = passes[0].pos / 16u, k1 = (END_ROWS - 1u) / 16u;
    const uint32_t v0 = passes[0].pos / 4u, v1 = (END_ROWS - 1u) / 4u;
    const uint32_t kbytes = (k1 - k0 + 1u) * kblk, vrun = (v1 - v0 + 1u) * 64u;
    const uint32_t vbytes = (MLA_DV / DSV2_MR) * vrun;
    int8_t *kgot = (int8_t *)region, *kwant = kgot + kbytes;
    int8_t *vgot = kwant + kbytes, *vwant = vgot + vbytes;

    const int isG = snax_is_gemm_core(), isS = snax_is_simd_core();
    const int isX = snax_is_xdma_core(), isD = snax_is_idma_core();
    if (isS) dsv2_print_l1((uint32_t)(end - arena), DSV2_L1_TOP - (uint32_t)arena);
    if (isS) {
        dsv2_tcdm_prio_install(DSV2_TCDM_PRIO);
        printf(TAG " TCDM arbitration: %s (control 0x%x)\n",
               snax_tcdm_policy_name(DSV2_TCDM_PRIO), snax_tcdm_arb_ctrl_read());
    }
    if ((uint32_t)end > DSV2_L1_TOP) {
        if (isS)
            printf(TAG " FAIL: the region ends at 0x%08x, past the stacks at 0x%08x\n",
                   (uint32_t)end, DSV2_L1_TOP);
        return isS ? 1 : 0;
    }
    if (isX) dsv2_xdma_fill_zero(arena, (uint32_t)(end - arena) / DSV2_BEAT);

    uint32_t fails = 0;
    snax_perf_snapshot_t census;
    for (uint32_t pp = 0; pp < NP; pp++) {
        const dsv2_mla_pass_t *ps = &passes[pp];
        // ---- the MLA block: x -> h, the row appended ------------------------------------
        ma.sy = bsy + pp * (1024 + MOE_SY_BYTES) / 4u;
        next_block(isX, region, region_beats);
        if (isS) snax_perf_arm();  // the census covers the block
        set_origin(isG, &MLA_T_ORG(&ma));
        dsv2_mla_begin(&ma, ps);
        if (isX) dsv2_mla_xdma(&ma);
        if (isD) dsv2_mla_idma(&ma, ps);
        if (isG) dsv2_mla_gemm(&ma, ps);
        if (isS) {
            dsv2_mla_simd(&ma, ps);
            snax_perf_read(&census);
            printf(TAG " pass %u, %u token(s) from position %u\n", pp, NTOK, ps->pos);
            dsv2_mla_report(&ma, TAG);
            layer_trace(tr, pp, "mla", MLA_T_ORG(&ma), &census, MLA_P(&ma, MLA_P_SBUSY));
        }
        // ---- the MoE block: h -> out ----------------------------------------------------
        mo.sy = bsy + (pp * (1024 + MOE_SY_BYTES) + 1024) / 4u;
        next_block(isX, region, region_beats);
        if (isS) snax_perf_arm();
        set_origin(isG, &MOE_T_ORG(&mo));
        if (isD) {
            dsv2_moe_idma(&mo, (const uint16_t *)0);
            MOE_SPIN(&mo, MOE_DONE(&mo), 1u);
            snrt_dma_start_1d(g_h_l, golds[pp].h, NTOK * 2 * HID);
            snrt_dma_start_1d(g_out_l, golds[pp].out, NTOK * 2 * HID);
            snrt_dma_start_1d(g_ids_l, golds[pp].ids, NTOK * 4 * 8);
            snrt_dma_start_1d(g_w_l, golds[pp].w, NTOK * 2 * 8);
            snrt_dma_wait_all();
            GOLD(pp) = 1u;
        }
        if (isG) dsv2_moe_gemm(&mo);
#if MOE_DUAL
        if (isX) dsv2_moe_xdma(&mo);
#endif
        if (isS) {
            dsv2_moe_simd(&mo);
            snax_perf_read(&census);
            dsv2_moe_report(&mo, TAG);
            layer_trace(tr, pp, "moe", MOE_T_ORG(&mo), &census, MOE_P(&mo, MOE_P_SBUSY));
            dsv2_spin_ge(&GOLD(pp), 1u, &APP_TMO);
            printf(TAG " pass %u: MLA %u cc + MoE %u cc\n", pp, MLA_T_SS(&ma, 7),
                   MOE_T_JOB(&mo, MOE_NJOBS(&mo) - 1u));
            for (uint32_t t = 0; t < NTOK; t++) {
                fails += chk16(t, "h = x + MLA(x)", MOE_TOK(h, t, MLA_HP), g_h_l + t * HID, HID);
                uint32_t idbad = 0;
                for (uint32_t i = 0; i < TOP_K; i++) idbad += MOE_IDS(&mo, t, i) != g_ids_l[8 * t + i];
                fails += dsv2_check(TAG, idbad == 0 && !MOE_REFUSED(&mo), "top-6",
                                    "the golden's ids, in order, all present");
                fails += chk16(t, "weights", MOE_TOK(mo.w, t, DSV2_BEAT), g_w_l + 8 * t, TOP_K);
                fails += chk16(t, "out = h + MoE(h)", dsv2_moe_out(&mo, t), g_out_l + t * HID, HID);
            }
            if (MLA_TMO(&ma) || MOE_TMO(&mo)) {
                printf(TAG " FAIL: a task never retired or a wait ran out\n");
                fails++;
            }
        }
    }

    // ---- after the last pass: every appended row, in both copies ---------------------------
    snrt_cluster_hw_barrier();
    if (isD) {
        snrt_dma_start_1d(kgot, key + k0 * kblk, kbytes);
        snrt_dma_start_1d(kwant, g_key + k0 * kblk, kbytes);
        snrt_dma_start_2d(vgot, val + v0 * 64u, vrun, vrun, CAP * 16u, MLA_DV / DSV2_MR);
        snrt_dma_start_2d(vwant, g_val + v0 * 64u, vrun, vrun, CAP * 16u, MLA_DV / DSV2_MR);
        snrt_dma_wait_all();
        CACHE_IN = 1u;
    }
    if (!isS) return 0;
    dsv2_spin_ge(&CACHE_IN, 1u, &APP_TMO);
    uint32_t w;
    fails += dsv2_check(TAG, dsv2_lsb8(kgot, kwant, kbytes, &w) == 0, "key copy",
                        "every appended row's 16-token block, byte-exact");
    fails += dsv2_check(TAG, dsv2_lsb8(vgot, vwant, vbytes, &w) == 0, "value copy",
                        "every appended row's 4-token blocks, byte-exact");
    if (APP_TMO) {
        printf(TAG " FAIL: %u bounded wait(s) ran out\n", APP_TMO);
        fails++;
    }
    printf(fails ? TAG " FAIL (%u checks)\n" : TAG " PASS\n", fails);
    return (int)fails;
}
