// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The SIMD stages of DeepSeek-V2-Lite layer 1, each on the layer's own tensors (the golden
// pack in sw/apps/dsv2/dsv2), from its golden input:
//
//   V1+V2  input norm      x (1 x 2048)        -> xn (FP16) -> the W_Q / W_DKV A operand
//   V1+V2  latent norm     c (1 x 512)         -> cn (FP16) -> c8, the cache row's latent
//   V3     RoPE at L       [q_pe x 16 | k_pe]  one 1,088-wide row -> rotated
//   V1+V2  post-attn norm  h (1 x 2048)        -> hn (FP16) -> the router / expert A operand
//   V6+V2  SwiGLU          [gate | up] of an expert (2 x 1408) and of the shared (2 x 2816)
//                          -> silu(gate) (.) up (FP16) -> the down projection's A operand
//   V1+V2  unit rows       mean square exactly 1: 1/rms is exactly 1, so bit-exact
//
// The kernels are snax-dsv2.h's. The A operands are the GEMV's, x in row 0, written there by the
// quantiser directly; the check compares that row.
//
// THE RoPE ROW. q_pe of head h is q[192 h + 128 .. 192 h + 192), so the 16 heads' q_pe are
// strided in q; the iDMA gathers them and k_pe into one contiguous row (a 2-D copy), then
// writes its pair-swapped copy (two 2-D copies of 2-byte elements). The tables for position
// L are the datagen's: cos repeated per pair, sin with the pair's sign, tiled per vector.
//
// CHECKS, against the device model (hwmodel.py), bit for bit: the model reproduces the SIMD's
// rsqrt and silu tables, its fused multiply-add and its order of summation (simd.py), so the
// norms, SwiGLU and their INT8 operands match exactly, as do RoPE and the unit rows. The goldens
// load once every stage has run. By default the checks cover each stage's final result: the
// GEMV A operands, c8, the rotated row, the down projections' A operands. With
// DSV2_STAGE_CHECKS = 1 they cover the intermediates too -- xn, cn, hn, silu(gate), the
// SwiGLU products, the unit rows' norms -- and the far-operand read below.
//
// FAR OPERANDS. Last, a 2-operand MUL whose operands sit more than 256 KiB apart -- the loop
// stride of the pair has its top address bit set -- must read the far one: a pattern at the
// arena's end times 1.0 at its start comes back unchanged.
//
// TRACE. The engines' spans ([SPAN], snax-dsv2-trace.h), and the cluster census over the SIMD
// stages ([UTIL]), print after the checks.

#include "data.h"
#include "snax-core-roles.h"
#include "snax-dsv2.h"
#include "snax-dsv2-trace.h"

#if !SNAX_HAS_SIMD_CORE || !SNAX_HAS_XDMA_CORE || !SNAX_HAS_IDMA_CORE
#error "this kernel needs the SIMD block, the xDMA and the iDMA"
#endif

#define TAG "[SIMD]"
#define ROPE_N ((ROPE_HEADS + 1) * ROPE_DIM)  // 1,088: 16 q_pe and one k_pe
#define ROPE_B (2u * ROPE_N)                    // one operand's bytes, 34 beats
#define READY sy[0]   // inputs and the RoPE row in L1      (iDMA -> SIMD)
#define TMO sy[1]
#define STAGED sy[2]  // every stage has run               (SIMD -> iDMA)
#define GOLD sy[3]    // the goldens are in L1             (iDMA -> SIMD)
#define SPIN(w, v) dsv2_spin_ge(&(w), (v), &TMO)

// FP16 against the model: `max_ulp` allowed.
static uint32_t chk16(const char *what, const uint16_t *got, const uint16_t *want, uint32_t n,
                      uint32_t max_ulp) {
    n = dsv2_check_terms(n);
    uint32_t worst, bad = dsv2_ulp16(got, want, n, &worst);
    printf(TAG "   %s %-30s %4u/%4u differ, worst %u ULP (allowed %u)\n",
           worst <= max_ulp ? "PASS" : "FAIL", what, bad, n, worst, max_ulp);
    return worst <= max_ulp ? 0u : 1u;
}

// INT8 against the model: `max_lsb` allowed.
static uint32_t chk8(const char *what, const int8_t *got, const int8_t *want, uint32_t n,
                     uint32_t max_lsb) {
    n = dsv2_check_terms(n);
    uint32_t worst, bad = dsv2_lsb8(got, want, n, &worst);
    printf(TAG "   %s %-30s %5u/%5u differ, worst %u LSB (allowed %u)\n",
           worst <= max_lsb ? "PASS" : "FAIL", what, bad, n, worst, max_lsb);
    return worst <= max_lsb ? 0u : 1u;
}

// A GEMV's A operand against the INT8 values of its row 0, the row the GEMV reads.
static uint32_t chka(const char *what, const int8_t *a, const int8_t *want, uint32_t n) {
    n = dsv2_check_terms(n);
    uint32_t bad = dsv2_a_row_diff(a, want, n, 0u);
    printf(TAG "   %s %-30s %5u/%5u differ in row 0 (allowed 0)\n", bad ? "FAIL" : "PASS", what,
           bad, n);
    return bad ? 1u : 0u;
}

// Run what is queued, and return its cycles; a task that never retires latches *hung.
static uint32_t drain(uint32_t t0, const char *what, uint32_t *hung) {
    if (snax_simd_wait_all_checked(what, SIMD_WAIT_BUDGET)) *hung = 1;
    return snrt_mcycle() - t0;
}

int main() {
    // ---- L1, derived identically on every hart. A norm's input has its seed beat below. -
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 63u) & ~63u);
    uint8_t *arena = p;
#define TAKE(n) (p += (((n) + 63u) & ~63u), p - (((n) + 63u) & ~63u))
    volatile uint32_t *sy = (volatile uint32_t *)TAKE(256);
    dsv2_trace_t *tr = (dsv2_trace_t *)TAKE(sizeof(dsv2_trace_t));
    uint8_t *ssq = TAKE(DSV2_BEAT);
    // input norm
    TAKE(DSV2_BEAT);
    uint16_t *x = (uint16_t *)TAKE(4096);
    uint16_t *xn = (uint16_t *)TAKE(4096), *gxn = (uint16_t *)TAKE(4096);
    int8_t *xa = (int8_t *)TAKE(16 * 2048), *gxa = (int8_t *)TAKE(2048);
    // latent norm: ckv = [c 512 | k_pe 64]
    TAKE(DSV2_BEAT);
    uint16_t *ckv = (uint16_t *)TAKE(2 * (KV_RANK + ROPE_DIM));
    uint16_t *cn = (uint16_t *)TAKE(2 * KV_RANK), *gcn = (uint16_t *)TAKE(2 * KV_RANK);
    int8_t *c8 = (int8_t *)TAKE(KV_RANK), *gc8 = (int8_t *)TAKE(KV_RANK);
    // RoPE: four equally spaced operands, then the output
    uint16_t *q = (uint16_t *)TAKE(2 * ROPE_HEADS * Q_HEAD);
    uint8_t *rope = TAKE(4 * ROPE_B);  // x | cos | swap(x) | sin
    uint16_t *ry = (uint16_t *)TAKE(ROPE_B), *gry = (uint16_t *)TAKE(ROPE_B);
    // post-attention norm
    TAKE(DSV2_BEAT);
    uint16_t *h = (uint16_t *)TAKE(4096);
    uint16_t *hn = (uint16_t *)TAKE(4096), *ghn = (uint16_t *)TAKE(4096);
    int8_t *ha = (int8_t *)TAKE(16 * 2048), *gha = (int8_t *)TAKE(2048);
    // SwiGLU: silu(gate), then [gate | up]
    uint16_t *esg = (uint16_t *)TAKE(2 * I_EXPERT), *eg = (uint16_t *)TAKE(4 * I_EXPERT);
    uint16_t *ea16 = (uint16_t *)TAKE(2 * I_EXPERT);
    int8_t *ea8 = (int8_t *)TAKE(16 * I_EXPERT);
    uint16_t *gesg = (uint16_t *)TAKE(2 * I_EXPERT), *gea16 = (uint16_t *)TAKE(2 * I_EXPERT);
    int8_t *gea8 = (int8_t *)TAKE(I_EXPERT);
    uint16_t *ssg = (uint16_t *)TAKE(2 * I_SHARED), *sg = (uint16_t *)TAKE(4 * I_SHARED);
    uint16_t *sa16 = (uint16_t *)TAKE(2 * I_SHARED);
    int8_t *sa8 = (int8_t *)TAKE(16 * I_SHARED);
    uint16_t *gssg = (uint16_t *)TAKE(2 * I_SHARED), *gsa16 = (uint16_t *)TAKE(2 * I_SHARED);
    int8_t *gsa8 = (int8_t *)TAKE(I_SHARED);
    // the unit rows
    TAKE(DSV2_BEAT);
    uint16_t *u2k = (uint16_t *)TAKE(4096);
    uint16_t *u2kn = (uint16_t *)TAKE(4096);
    int8_t *u2ka = (int8_t *)TAKE(16 * 2048), *gu2ka = (int8_t *)TAKE(2048);
    TAKE(DSV2_BEAT);
    uint16_t *u512 = (uint16_t *)TAKE(1024);
    uint16_t *u512n = (uint16_t *)TAKE(1024);
    int8_t *u512a = (int8_t *)TAKE(16 * 512), *gu512a = (int8_t *)TAKE(512);
    // far operands: a pattern and its product, more than 256 KiB past x
    if (p < (uint8_t *)x + (1u << 18) + DSV2_BEAT) p = (uint8_t *)x + (1u << 18) + DSV2_BEAT;
    uint16_t *pat = (uint16_t *)TAKE(4096), *prod = (uint16_t *)TAKE(4096);
    const uint32_t arena_beats = (uint32_t)(p - arena) / DSV2_BEAT;
#undef TAKE

    const int isS = snax_is_simd_core(), isX = snax_is_xdma_core(), isD = snax_is_idma_core();
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
    if (isS) snax_perf_arm();  // the census covers the SIMD stages
    snrt_cluster_hw_barrier();

    if (isD) {
        // the inputs
        const uint32_t i0 = snrt_mcycle();
        snrt_dma_start_1d(x, x16, 4096);
        snrt_dma_start_1d(ckv, ckv16, 2 * (KV_RANK + ROPE_DIM));
        snrt_dma_start_1d(q, q16, 2 * ROPE_HEADS * Q_HEAD);
        snrt_dma_start_1d(rope + ROPE_B, rope_cos, ROPE_B);
        snrt_dma_start_1d(rope + 3 * ROPE_B, rope_sin, ROPE_B);
        snrt_dma_start_1d(h, h16, 4096);
        snrt_dma_start_1d(eg, e_g16, 4 * I_EXPERT);
        snrt_dma_start_1d(sg, s_g16, 4 * I_SHARED);
        snrt_dma_start_1d(u2k, unit2048, 4096);
        snrt_dma_start_1d(u512, unit512, 1024);
        // the RoPE row: the heads' q_pe (strided in q), then k_pe; then its pair swap
        uint32_t t0 = snrt_mcycle();
        snrt_dma_start_2d(rope, (uint8_t *)q + 2 * Q_NOPE, 2 * ROPE_DIM, 2 * ROPE_DIM,
                          2 * Q_HEAD, ROPE_HEADS);
        snrt_dma_start_1d(rope + 2 * ROPE_HEADS * ROPE_DIM, (uint8_t *)ckv + 2 * KV_RANK,
                          2 * ROPE_DIM);
        snrt_dma_wait_all();
        snrt_dma_start_2d(rope + 2 * ROPE_B, rope + 2, 2, 4, 4, ROPE_N / 2);
        snrt_dma_start_2d(rope + 2 * ROPE_B + 2, rope, 2, 4, 4, ROPE_N / 2);
        snrt_dma_wait_all();
        sy[8] = snrt_mcycle() - t0;
        dsv2_span(tr, DSV2_TR_IDMA, "0 inputs; 5 RoPE row and its swap", i0, snrt_mcycle());
        READY = 1;
        // the goldens, once every stage has run: the final results', then the intermediates'
        SPIN(STAGED, 1u);
        snrt_dma_start_1d(gxa, g_xa, 2048);
        snrt_dma_start_1d(gc8, g_c8, KV_RANK);
        snrt_dma_start_1d(gry, g_rope, ROPE_B);
        snrt_dma_start_1d(gha, g_ha, 2048);
        snrt_dma_start_1d(gea8, g_e_a8, I_EXPERT);
        snrt_dma_start_1d(gsa8, g_s_a8, I_SHARED);
        snrt_dma_start_1d(gu2ka, g_unit2048a, 2048);
        snrt_dma_start_1d(gu512a, g_unit512a, 512);
#if DSV2_STAGE_CHECKS
        snrt_dma_start_1d(gxn, g_xn, 4096);
        snrt_dma_start_1d(gcn, g_cn, 2 * KV_RANK);
        snrt_dma_start_1d(ghn, g_hn, 4096);
        snrt_dma_start_1d(gesg, g_e_sg, 2 * I_EXPERT);
        snrt_dma_start_1d(gea16, g_e_a16, 2 * I_EXPERT);
        snrt_dma_start_1d(gssg, g_s_sg, 2 * I_SHARED);
        snrt_dma_start_1d(gsa16, g_s_a16, 2 * I_SHARED);
#endif
        snrt_dma_wait_all();
        GOLD = 1;
        snrt_cluster_hw_barrier();
        return 0;
    }
    if (!isS) {
        snrt_cluster_hw_barrier();
        return 0;
    }

    SPIN(READY, 1u);
    uint32_t hung = 0, fails = 0, t;
    uint32_t c_in, c_lat, c_rope, c_post, c_e, c_s, c_u;
    const uint32_t busy0 = snax_simd_busy_cycles();
#define SIMD_SPAN(label, c) dsv2_span(tr, DSV2_TR_SIMD, (label), t, t + (c))

    t = snrt_mcycle();
    dsv2_rmsnorm_row(x, ssq, xn, 11);
    dsv2_quant_a(xn, xa, 2048, INV_X);
    c_in = drain(t, "input norm", &hung);
    SIMD_SPAN("1 input norm, A operand", c_in);

    t = snrt_mcycle();
    dsv2_rmsnorm_row(ckv, ssq, cn, 9);
    dsv2_quant_flat(cn, c8, KV_RANK, INV_C);
    c_lat = drain(t, "latent norm", &hung);
    SIMD_SPAN("4 latent norm, c8", c_lat);

    t = snrt_mcycle();
    dsv2_rope(rope, ROPE_B, ry, ROPE_N);
    c_rope = drain(t, "rope", &hung);
    SIMD_SPAN("5 RoPE", c_rope);

    t = snrt_mcycle();
    dsv2_rmsnorm_row(h, ssq, hn, 11);
    dsv2_quant_a(hn, ha, 2048, INV_H);
    c_post = drain(t, "post-attention norm", &hung);
    SIMD_SPAN("14 post-attention norm, A operand", c_post);

    t = snrt_mcycle();
    dsv2_swiglu(eg, esg, ea16, ea8, I_EXPERT, INV_AE, 0u);
    c_e = drain(t, "expert swiglu", &hung);
    SIMD_SPAN("18 SwiGLU, expert", c_e);

    t = snrt_mcycle();
    dsv2_swiglu(sg, ssg, sa16, sa8, I_SHARED, INV_AS, 0u);
    c_s = drain(t, "shared swiglu", &hung);
    SIMD_SPAN("21 SwiGLU, shared", c_s);

    t = snrt_mcycle();
    dsv2_rmsnorm_row(u2k, ssq, u2kn, 11);
    dsv2_quant_a(u2kn, u2ka, 2048, INV_X);
    dsv2_rmsnorm_row(u512, ssq, u512n, 9);
    dsv2_quant_a(u512n, u512a, 512, INV_X);
    c_u = drain(t, "unit rows", &hung);
    SIMD_SPAN("1 unit rows (exact norms)", c_u);
    const uint32_t simd_busy = snax_simd_busy_cycles() - busy0;
    snax_perf_snapshot_t census;
    snax_perf_read(&census);
    STAGED = 1u;

    printf(TAG " cycles: input norm+quant %u | latent norm+quant %u | RoPE %u (iDMA staging %u) "
               "| post-attn norm+quant %u | SwiGLU expert %u, shared %u | unit rows %u\n",
           c_in, c_lat, c_rope, sy[8], c_post, c_e, c_s, c_u);

    SPIN(GOLD, 1u);
    printf(TAG " V1+V2 input norm (1 x 2048)\n");
#if DSV2_STAGE_CHECKS
    fails += chk16("xn", xn, gxn, 2048, 0);
#endif
    fails += chka("W_Q/W_DKV A operand", xa, gxa, 2048);
    printf(TAG " V1+V2 latent norm (1 x 512)\n");
#if DSV2_STAGE_CHECKS
    fails += chk16("cn", cn, gcn, KV_RANK, 0);
#endif
    fails += chk8("c8 (cache row)", c8, gc8, KV_RANK, 0);
    printf(TAG " V3 RoPE at position L (16 x 64 q_pe, 1 x 64 k_pe)\n");
    fails += chk16("q_pe, k_pe rotated", ry, gry, ROPE_N, 0);
    printf(TAG " V1+V2 post-attention norm (1 x 2048)\n");
#if DSV2_STAGE_CHECKS
    fails += chk16("hn", hn, ghn, 2048, 0);
#endif
    fails += chka("router/expert A operand", ha, gha, 2048);
    printf(TAG " V6+V2 SwiGLU, expert (1 x 1408)\n");
#if DSV2_STAGE_CHECKS
    fails += chk16("silu(gate)", esg, gesg, I_EXPERT, 0);
    fails += chk16("silu(gate) * up", ea16, gea16, I_EXPERT, 0);
#endif
    fails += chka("down A operand", ea8, gea8, I_EXPERT);
    printf(TAG " V6+V2 SwiGLU, shared (1 x 2816)\n");
#if DSV2_STAGE_CHECKS
    fails += chk16("silu(gate)", ssg, gssg, I_SHARED, 0);
    fails += chk16("silu(gate) * up", sa16, gsa16, I_SHARED, 0);
#endif
    fails += chka("down A operand", sa8, gsa8, I_SHARED);
    printf(TAG " V1+V2 unit rows (mean square exactly 1)\n");
#if DSV2_STAGE_CHECKS
    fails += chk16("2048: norm = input", u2kn, u2k, 2048, 0);
    fails += chk16("512: norm = input", u512n, u512, 512, 0);
#endif
    fails += chka("2048: A operand", u2ka, gu2ka, 2048);
    fails += chka("512: A operand", u512a, gu512a, 512);

#if DSV2_STAGE_CHECKS
    // far operands: ones at the arena's start (x), a pattern at its end
    const uint32_t d = (uint32_t)pat - (uint32_t)x;
    for (uint32_t i = 0; i < 2048; i++) x[i] = 0x3C00u;  // 1.0
    for (uint32_t i = 0; i < 2048; i++) pat[i] = (uint16_t)(0x3000u + 7u * i);  // finite, normal
    t = snrt_mcycle();
    dsv2_dequant_prep(x, pat, prod, 2048);
    snax_simd_fire();
    SIMD_SPAN("0 far operands", drain(t, "far operands", &hung));
    printf(TAG " far operands: a 2-operand read with its operands %u B apart\n", d);
    fails += dsv2_check(TAG, d >= (1u << 18), "far operands", "the pair is past 256 KiB");
    fails += chk16("pattern (.) 1.0 = pattern", prod, pat, 2048, 0);
#else
    (void)pat;
    (void)prod;
#endif
    if (hung || TMO) {
        printf(TAG " FAIL: a task never retired or a wait ran out\n");
        fails++;
    }
    printf(fails ? TAG " FAIL (%u checks)\n" : TAG " PASS\n", fails);
    dsv2_trace_print(tr);
    dsv2_util_print(&census, simd_busy);
#undef SIMD_SPAN
    snrt_cluster_hw_barrier();
    return (int)fails;
}
