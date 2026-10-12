// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V2-Lite's multi-head latent attention for one new token (A1, with D2 and V4):
//
//     S^T = K . Q^T        over [c | k_pe], 576      K: the key copy, Bc tokens per tile
//     P   = softmax(a S)   online, per key tile      FlashAttention's state (m, l)
//     O^T += V^T . P^T     over the latent, 512      V^T: the value copy
//     o~  = O / l                                    16 heads x 512, then W_UV's operands
//
// All 16 heads share the one cache, so a query tile is the token's 16 heads -- one SIMD beat of
// 16 FP16 lanes, one array column block: Br = 16 -- and the two head sizes differ: d_qk = 576
// for the scores, d_v = 512 for the weighted sum. The golden pack (sw/apps/dsv2/util) supplies
// the cache (L cached rows plus the token's own), q~ and the rotated q_pe.
//
// ======================================================================================
// PER KEY TILE j -- snax-flashattn-decode's pipeline at MLA's shapes
// ======================================================================================
//
// The scores are a SCORE PLANE, one beat per key, with a spare beat either side, and the
// softmax state and outputs are laid out around it (snax-dsv2-mla.h, THE ATTENTION):
//
//   S^T(j)  [latch][S^T: BC key beats][mold]          the latch takes m_new, the mold m_old
//   P8      [B blocks (k): BC / 4 x 64 B][rowsum][lsc]
//   O16     [c / l][O16^T: 512 latent beats]
//
//   iDMA (hart 3)   K(j): one contiguous Bc x 576 block of the key copy
//                   V(j): 32 runs of 16 Bc bytes of the value copy (V^T's tile)
//   GEMM (hart 0)   QK(j) = K(j) . Q^T -> S^T(j) FP16, k_s; then PV(j-1) behind it
//   SIMD (hart 1)   the online softmax of tile j, FlashAttention's chain:
//                     1   m_new = max(m_old, rowmax)            Reduce MAX|LANEWISE over
//                                                               [keys][mold], into the latch
//                     2   corr = exp(a' (m_old - m_new))        EW0 SUB over [latch][mold] ->
//                                                               Map EXP
//                     3   P8 and the rowsum, one sweep          EW0 sticky SUB -> Map EXP ->
//                                                               Reduce ADD|TAP -> Fp16ToInt8
//                                                               (interleaved: PV's B operand)
//                     4   corr * l_old                          EW1 MUL over [corr][l]
//                     5   l = rowsum + corr l_old               Reduce ADD|LANEWISE
//                     6   m_new into the next tile's mold       a copy
//                   and P8(j) is published after task 3.
//   GEMM            PV(j): O^T += V(j) . P8^T, C scaled per query by corr_j (masked at
//                   j = 0); the LAST tile's D leaves as FP16 through the converter (k_o).
//
// AFTER THE LAST TILE, the SIMD scales O16 by c / l per query, c = 2^k_o s_c / 127, in two
// tasks -- RSQRT(l / c) on the l beat read twice, squared by EW1 MUL (l^2 would overflow
// FP16; sqrt(c / l) squared does not), into O16's latch, then a sticky MUL over [c/l][O16] --
// the xDMA transposes it, and ONE quantise task writes W_UV's 16 A operands (x_h in all 16
// rows, as snax-dsv2-absorb reads them).
//
// D2, THE QUERY, comes first: q~ (16 x 512) and q_pe (16 x 64) arrive FP16, row-major, and
// two quantise tasks read them in A-ORDER -- lane stride = one row, so a beat is one row per
// channel, four values each -- and pack the beats of 16 rows into Q8's A blocks, each segment
// with its own scale (s_q~ s_c = s_qpe s_kpe keeps a score one exact INT32 dot product). The row
// pitches are padded to an odd word count so the rows of a beat fall in distinct banks.
//
// CHECKS (main returns the number that fail), bit for bit against the device model, which
// reproduces the exp and rsqrt tables, once the kernel is done and its goldens loaded: the
// outputs o~ and the W_UV operands; with DSV2_STAGE_CHECKS = 1 the intermediates as well, Q8,
// every tile's S^T, m, l, the last tile's P8, O16 and c / l. The engines' spans and the cluster
// census over the kernel follow ([SPAN], [UTIL]; snax-dsv2-trace.h).
//
// TEST SIZE (data/params.hjson): a short cache, two key tiles; the full-length attention runs on
// HeMAiA.

#include "data.h"
#include "snax-core-roles.h"
#include "snax-dsv2.h"
#include "snax-dsv2-trace.h"

#if !SNAX_HAS_GEMM_CORE || !SNAX_HAS_SIMD_CORE || !SNAX_HAS_XDMA_CORE || !SNAX_HAS_IDMA_CORE
#error "this kernel needs the four-engine cluster"
#endif
#if !defined(SIMD_EXT_STREAMREDUCE_HAS_MAX) || !defined(SIMD_EXT_STREAMREDUCE_HAS_ADD) || \
    !defined(SIMD_EXT_STREAMMAP_HAS_EXP) || !defined(SIMD_EXT_STREAMELEMENTWISE_0_HAS_SUB)
#error "the online softmax needs StreamReduce MAX/ADD, StreamMap EXP and a pre-map SUB"
#endif
#if !defined(READER_WRITER_EXTENSION_0_CSR_BASE) || READER_WRITER_EXTENSION_0_CSR_NUM != 18
#error "the O rescale needs the C read path's Int32ColumnScale (18 CSRs)"
#endif
#define COLSCALE_CSR READER_WRITER_EXTENSION_0_CSR_BASE

#define TAG "[MLA]"
#define HEADS 16
#if DSV2_LANES != DSV2_NU || HEADS != DSV2_LANES
#error "the score plane is one SIMD beat, one array column block and the token's 16 heads"
#endif
#define BR DSV2_LANES            // query lanes: the token's 16 heads, one plane
#define DQK 576                  // [c | k_pe]
#define DV 512                   // c
#define KT_S (DQK / DSV2_KU)     // 144: QK's contraction blocks
#define M_S (BC / DSV2_MR)       // QK's key blocks
#define N_Q (BR / DSV2_NU)       // 1: query blocks
#define KT_P (BC / DSV2_KU)      // PV's contraction blocks
#define M_P (DV / DSV2_MR)       // 32: PV's latent blocks
#define KTILE (BC * DQK)         // bytes of a K tile
#define VTILE (DV * BC)          // bytes of a V^T tile
#define BLK 64u                  // one A or B block, 16 x 4 or 4 x 16 INT8
#define BEAT DSV2_SBEAT
#define SP ((BC + 2u) * BEAT)    // a score plane: [latch][BC keys][mold]
#define S_MOLD ((BC + 1u) * BEAT)
#define PP (BC * DSV2_NU + 2u * BEAT)  // P8: [B blocks][rowsum][lsc]
#define P_RS (BC * DSV2_NU)            // P8 -> its rowsum beat
#define QT_PITCH (2 * DV + 8)    // q~ rows, padded to an odd word count
#define QP_PITCH (2 * 64 + 8)    // q_pe rows

// ---- shared words ----------------------------------------------------------------------
#define K_IN sy[0]    // K tiles landed          (iDMA -> GEMM)
#define V_IN sy[1]    // V tiles landed          (iDMA -> GEMM)
#define K_FREE sy[2]  // QKs retired             (GEMM -> iDMA)
#define V_FREE sy[3]  // PVs retired             (GEMM -> iDMA)
#define S_OUT sy[4]   // S tiles written         (GEMM -> SIMD)
#define P_OUT sy[5]   // P8 tiles published      (SIMD -> GEMM)
#define Q_OUT sy[6]   // Q8 assembled            (SIMD -> GEMM)
#define M_INIT sy[7]  // m seeded to -65504      (xDMA -> SIMD)
#define O_OUT sy[8]   // the last PV retired     (GEMM -> SIMD)
#define N_OUT sy[9]   // o~^T written            (SIMD -> xDMA)
#define X_OUT sy[10]  // o~ transposed           (xDMA -> SIMD)
#define G1 sy[11]     // goldens, round 1        (iDMA -> SIMD)
#define DONE1 sy[12]  // round 1 checked         (SIMD -> iDMA)
#define G2 sy[13]     // goldens, round 2        (iDMA -> SIMD)
#define KDONE sy[14]  // the kernel is done      (SIMD -> iDMA)
#define TMO sy[15]
#define T_ORG sy[16]
#define T_QK(j) sy[32 + (j)]   // QK retired, cycles from the origin
#define T_SM(j) sy[64 + (j)]   // softmax published
#define T_PV(j) sy[96 + (j)]   // PV retired
#define SPIN(w, v) dsv2_spin_ge(&(w), (v), &TMO)

// ============================================================ the GEMM (hart 0)

// The port's 16 channels are four groups of 32 contiguous bytes, so one transaction is 128
// contiguous bytes. C and D share the INT32 walk, block (m, n) at (m N + n) 1 KiB, row-major
// inside: O^T accumulates in place. The FP16 walk puts block (m, n) at n np + m 512, one 32 B
// beat per key or latent: a plane; its converter halves the beats per block.
__attribute__((always_inline)) static inline void cd_walk(uint32_t m, uint32_t c_on,
                                                          uint32_t d_fp16, uint32_t k,
                                                          uint32_t np) {
    csrw_ss(S_STRIDE_READER_WRITER_0_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_0_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_0_0, 8);
    csrw_ss(T_STRIDE_READER_WRITER_0_0, 128);
    csrw_ss(T_BOUND_READER_WRITER_0_1, N_Q);
    csrw_ss(T_STRIDE_READER_WRITER_0_1, 1024);
    csrw_ss(T_BOUND_READER_WRITER_0_2, m);
    csrw_ss(T_STRIDE_READER_WRITER_0_2, N_Q * 1024u);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, 0);
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_0, c_on ? 0xFFFFFFFFu : 0u);
    csrw_ss(S_STRIDE_READER_WRITER_1_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_1_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_1_0, d_fp16 ? 4u : 8u);
    csrw_ss(T_STRIDE_READER_WRITER_1_0, 128);
    csrw_ss(T_BOUND_READER_WRITER_1_1, N_Q);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, d_fp16 ? np : 1024u);
    csrw_ss(T_BOUND_READER_WRITER_1_2, m);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, d_fp16 ? 512u : N_Q * 1024u);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, 0);
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_1, 0xFFFFFFFFu);  // shape 0: full output beats
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 0, d_fp16 ? 1u : 0u);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 0u);
    (void)set_versacore_d_shift(d_fp16 ? k : 0u);
}

// A and B walks: k innermost, then n, then m. A broadcasts over n; B over m.
__attribute__((always_inline)) static inline void ab_walk(uint32_t kt, uint32_t m,
                                                          uint32_t a_ms, uint32_t b_ks,
                                                          uint32_t b_ns) {
    csrw_ss(S_STRIDE_READER_0_0, 8);
    csrw_ss(T_BOUND_READER_0_0, kt);
    csrw_ss(T_STRIDE_READER_0_0, 64);
    csrw_ss(T_BOUND_READER_0_1, N_Q);
    csrw_ss(T_STRIDE_READER_0_1, 0);
    csrw_ss(T_BOUND_READER_0_2, m);
    csrw_ss(T_STRIDE_READER_0_2, a_ms);
    csrw_ss(T_BOUND_READER_0_3, 1);
    csrw_ss(T_STRIDE_READER_0_3, 0);
    csrw_ss(T_BOUND_READER_0_4, 1);
    csrw_ss(T_STRIDE_READER_0_4, 0);
    csrw_ss(T_BOUND_READER_0_5, 1);
    csrw_ss(T_STRIDE_READER_0_5, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0);
    csrw_ss(ENABLED_CHANNEL_READER_0, 0xFFFFFFFFu);  // shape 0: every A channel
    csrw_ss(S_STRIDE_READER_1_0, 8);
    csrw_ss(T_BOUND_READER_1_0, kt);
    csrw_ss(T_STRIDE_READER_1_0, b_ks);
    csrw_ss(T_BOUND_READER_1_1, N_Q);
    csrw_ss(T_STRIDE_READER_1_1, b_ns);
    csrw_ss(T_BOUND_READER_1_2, m);
    csrw_ss(T_STRIDE_READER_1_2, 0);
    csrw_ss(T_BOUND_READER_1_3, 1);
    csrw_ss(T_STRIDE_READER_1_3, 0);
    csrw_ss(T_BOUND_READER_1_4, 1);
    csrw_ss(T_STRIDE_READER_1_4, 0);
    csrw_ss(T_BOUND_READER_1_5, 1);
    csrw_ss(T_STRIDE_READER_1_5, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0);
    csrw_ss(S_STRIDE_READER_1_1, 0);
    csrw_ss(ENABLED_CHANNEL_READER_1, 0xFFu);  // shape 0: one 4 x 16 block, channels 0..7
    dsv2_b_int4(0u);                           // Q8 and P8 are INT8 operands
    csrw_ss(OVERWRITE_ACCUM, 1);
    csrw_ss(ACCUM_BOUND, kt);
    csrw_ss(OUTPUT_BOUND, m * N_Q);
    csrw_ss(SUBTRACTIONS, 0);
    csrw_ss(ARRAY_SHAPE_CFG, 0);
    csrw_ss(DATA_TYPE_CFG, 0);
}

// QK: S^T [Bc, 16] = K [Bc, 576] . Q^T into the score plane. B(Q^T) is A(Q)'s bytes (meshRow ==
// meshCol), walked n-major.
__attribute__((always_inline)) static inline void qk_arm(void) {
    ab_walk(KT_S, M_S, KT_S * 64u, 64u, KT_S * 64u);
    cd_walk(M_S, 0u, 1u, K_S, SP);
    csrw_ss(COLSCALE_CSR + 1, N_Q);
    csrw_ss(COLSCALE_CSR + 0, 0u);
}

// PV: O^T [512, 16] += V^T [512, Bc] . P^T, P8's B blocks k-major as the interleaving quantiser
// writes them. C carries O^T back in, scaled per query by corr_j (masked at j = 0); the last
// tile's D leaves as FP16 at k_o, into the O16 plane.
__attribute__((always_inline)) static inline void pv_arm(uint32_t first, uint32_t last) {
    ab_walk(KT_P, M_P, KT_P * 64u, 64u, PP);
    cd_walk(M_P, !first, last, K_O, (DV + 2u) * BEAT);
    csrw_ss(COLSCALE_CSR + 1, N_Q);
    csrw_ss(COLSCALE_CSR + 0, first ? 0u : 1u);
}

// corr_j's 16 FP16 factors, two per word: the beat the softmax wrote, copied verbatim.
__attribute__((always_inline)) static inline void pv_factors(const volatile uint32_t *f) {
    csrw_ss(COLSCALE_CSR + 2, f[0]);    csrw_ss(COLSCALE_CSR + 3, f[1]);
    csrw_ss(COLSCALE_CSR + 4, f[2]);    csrw_ss(COLSCALE_CSR + 5, f[3]);
    csrw_ss(COLSCALE_CSR + 6, f[4]);    csrw_ss(COLSCALE_CSR + 7, f[5]);
    csrw_ss(COLSCALE_CSR + 8, f[6]);    csrw_ss(COLSCALE_CSR + 9, f[7]);
}

__attribute__((always_inline)) static inline void gemm_go(const void *a, const void *b,
                                                          void *c, void *d) {
    csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)a);
    csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)b);
    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, (uint32_t)c);
    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, (uint32_t)d);
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(GEMMX_START, 1);
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
}

// ============================================================ the SIMD (hart 1)

static inline void simd_task(const snax_simd_shape_t *in, const snax_simd_shape_t *out) {
    snax_simd_program_fast(in, out);
    snax_simd_fire();
}

// Every task submitted so far has retired. Bounded; latches a timeout.
static inline void simd_drain(volatile uint32_t *sy, const char *what) {
    if (snax_simd_wait_all_checked(what, SIMD_WAIT_BUDGET)) TMO++;
}

// D2: one segment of Q8 from FP16 rows `pitch` apart. A beat is one row per channel, four
// values each (lane stride = pitch); the 16 / C beats of rows 0..15 pack into one A block (C =
// SIMD_SPATIAL_CHAN). `kt` blocks.
#if (DSV2_MR / SIMD_SPATIAL_CHAN) % 2u
#error "an A block's rows must fill whole pairs of SIMD beats: the quantiser packs two into one"
#endif
static void quant_q_segment(const void *rows, uint32_t pitch, void *q8, uint32_t kt,
                            uint32_t inv) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, (void *)rows, 1);
    in.lane_stride = pitch;
    in.bound[0] = DSV2_MR / SIMD_SPATIAL_CHAN;
    in.stride[0] = SIMD_SPATIAL_CHAN * pitch;
    in.bound[1] = kt;
    in.stride[1] = 8u;
    snax_simd_shape_flat(&out, q8, kt * BLK / BEAT);
    snax_simd_use0(SIMD_EXT_FP16TOINT8);
    DSV2_ARM_QUANT(inv);
    simd_task(&in, &out);
}

// ============================================================ checks

static uint32_t chk16(const char *what, const uint16_t *got, const uint16_t *want, uint32_t n,
                      uint32_t max_ulp) {
    n = dsv2_check_terms(n);
    uint32_t worst, bad = dsv2_ulp16(got, want, n, &worst);
    printf(TAG "   %s %-34s %5u/%5u differ, worst %u ULP (allowed %u)\n",
           worst <= max_ulp ? "PASS" : "FAIL", what, bad, n, worst, max_ulp);
    return worst <= max_ulp ? 0u : 1u;
}

static uint32_t chk8(const char *what, const int8_t *got, const int8_t *want, uint32_t n,
                     uint32_t max_lsb) {
    n = dsv2_check_terms(n);
    uint32_t worst, bad = dsv2_lsb8(got, want, n, &worst);
    printf(TAG "   %s %-34s %6u/%6u differ, worst %u LSB (allowed %u)\n",
           worst <= max_lsb ? "PASS" : "FAIL", what, bad, n, worst, max_lsb);
    return worst <= max_lsb ? 0u : 1u;
}

// ============================================================ main

int main() {
    // ---- L1, derived identically on every hart ------------------------------------------
    // Adjacency is part of the layout: a sticky task reads its latch as the beat right in
    // front of its data, and a 2-beat lanewise reduce reads its pair as consecutive beats.
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 63u) & ~63u);
    uint8_t *arena = p;
#define TAKE(n) (p += (((n) + 63u) & ~63u), p - (((n) + 63u) & ~63u))
    volatile uint32_t *sy = (volatile uint32_t *)TAKE(1024);
    int8_t *q8 = (int8_t *)TAKE(BR * DQK);                 // Q^T operand, A-layout of Q
    uint8_t *qt = TAKE(HEADS * QT_PITCH);                  // q~, padded rows
    uint8_t *qp = TAKE(HEADS * QP_PITCH);                  // q_pe, padded rows
    // the score plane of every tile, kept for a check, at 32 mod 64: the keys on 64 B
    uint8_t *sbuf = TAKE(NT * SP + 64u) + 32u;             // tile j's latch at sbuf + j SP
    // the state, a beat each: corr's tile-parity slots below l (task 4's stride is unsigned)
    uint8_t *corr = TAKE(2 * BEAT), *l = TAKE(BEAT), *mseed = TAKE(BEAT), *mfin = TAKE(BEAT);
    uint8_t *p8[2];                                        // [P8 B blocks][rowsum][corr*l]
    p8[0] = TAKE(PP);
    p8[1] = TAKE(PP);
    uint8_t *rsc = TAKE(BEAT);                             // c / l, for a check
    // O16 behind its latch (c / l), 32 mod 64 like the score plane; o~^T; then O^T in INT32
    uint8_t *o16 = TAKE((DV + 2u) * BEAT + 64u) + 32u;
    uint16_t *ot16 = (uint16_t *)TAKE(DV * BEAT);
    int32_t *oacc = (int32_t *)TAKE(DV * BR * 4);
    uint16_t *xt = (uint16_t *)oacc;                       // o~, [16, 512], once O^T is dead
    int8_t *kbuf[2], *vbuf[2];
    kbuf[0] = (int8_t *)TAKE(KTILE);
    kbuf[1] = (int8_t *)TAKE(KTILE);
    vbuf[0] = (int8_t *)TAKE(VTILE);
    vbuf[1] = (int8_t *)TAKE(VTILE);
    int8_t *auv = kbuf[0];                                 // W_UV's A operands, 128 KiB, after
    // round 1's goldens; round 2's, the W_UV operands, get their own 128 KiB
    uint8_t *gold = TAKE(BR * DQK + NT * BC * BEAT + 2 * BEAT + BC * DSV2_NU + DV * BEAT +
                         HEADS * DV * 2);
    int8_t *g_auv_l = (int8_t *)TAKE(HEADS * DSV2_MR * DV);
    dsv2_trace_t *tr = (dsv2_trace_t *)TAKE(sizeof(dsv2_trace_t));
    int8_t *g_q8_l = (int8_t *)gold;
    uint16_t *g_s_l = (uint16_t *)(gold + BR * DQK);
    uint16_t *g_m_l = g_s_l + NT * BC * BR;
    uint16_t *g_l_l = g_m_l + BR;
    int8_t *g_p8_l = (int8_t *)(g_l_l + BR);
    uint16_t *g_o16_l = (uint16_t *)(g_p8_l + BC * DSV2_NU);
    uint16_t *g_ot_l = g_o16_l + DV * BR;
    const uint32_t arena_beats = (uint32_t)(p - arena) / DSV2_XBEAT;
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
    if (isG) T_ORG = snrt_mcycle();
    if (isS) snax_perf_arm();  // the census covers the kernel
    snrt_cluster_hw_barrier();
    uint32_t org = T_ORG;
    while (org == 0u) org = T_ORG;  // the barrier does not order the store

    if (isX) {
        // ============================== the xDMA: seed m, later transpose o~^T
        uint32_t x0 = snrt_mcycle();
        dsv2_xdma_fill(mseed, 1u, 0xFBFFFBFFu);
        dsv2_span(tr, DSV2_TR_XDMA, "9 m = -65504", x0, snrt_mcycle());
        M_INIT = 1u;
        SPIN(N_OUT, 1u);
        // o~^T is [512, 16] FP16 (32 B rows), the token's 16 heads
        x0 = snrt_mcycle();
        dsv2_xdma_transpose16(ot16, 2u * BR, xt, 2u * DV, DV, HEADS);
        dsv2_span(tr, DSV2_TR_XDMA, "10 o~ transpose", x0, snrt_mcycle());
        X_OUT = 1u;
        return 0;
    }

    if (isD) {
        // ============================== the iDMA: Q, the K/V stream, the goldens
        uint32_t d0 = snrt_mcycle();
        snrt_dma_start_2d(qt, qt16, 2u * DV, QT_PITCH, 2u * DV, HEADS);
        snrt_dma_start_2d(qp, qpe16, 2u * 64u, QP_PITCH, 2u * 64u, HEADS);
        snrt_dma_wait_all();
        dsv2_span(tr, DSV2_TR_IDMA, "6 q~, q_pe", d0, snrt_mcycle());
        sy[17] = 1u;  // Q inputs landed
        for (uint32_t j = 0; j < NT; j++) {
            if (j >= 2) SPIN(K_FREE, j - 1u);
            d0 = snrt_mcycle();
            snrt_dma_start_1d(kbuf[j & 1u], key + j * KTILE, KTILE);
            snrt_dma_wait_all();
            dsv2_span(tr, DSV2_TR_IDMA, "8 key tile", d0, snrt_mcycle());
            K_IN = j + 1u;
            if (j >= 2) SPIN(V_FREE, j - 1u);
            d0 = snrt_mcycle();
            snrt_dma_start_2d(vbuf[j & 1u], val + j * (16u * BC), 16u * BC, 16u * BC,
                              16u * CAP, DV / DSV2_MR);
            snrt_dma_wait_all();
            dsv2_span(tr, DSV2_TR_IDMA, "10 value tile", d0, snrt_mcycle());
            V_IN = j + 1u;
        }
        SPIN(KDONE, 1u);
        snrt_dma_start_1d(g_ot_l, g_ot, HEADS * DV * 2);
#if DSV2_STAGE_CHECKS
        snrt_dma_start_1d(g_q8_l, g_q8, BR * DQK);
        snrt_dma_start_1d(g_s_l, g_s, NT * BC * BEAT);
        snrt_dma_start_1d(g_m_l, g_m, BEAT);
        snrt_dma_start_1d(g_l_l, g_l, BEAT);
        snrt_dma_start_1d(g_p8_l, g_p8, BC * DSV2_NU);
        snrt_dma_start_1d(g_o16_l, g_o16, DV * BEAT);
#endif
        snrt_dma_wait_all();
        G1 = 1u;
        SPIN(DONE1, 1u);
        snrt_dma_start_1d(g_auv_l, g_auv, HEADS * DSV2_MR * DV);
        snrt_dma_wait_all();
        G2 = 1u;
        return 0;
    }

    if (isG) {
        // ============================== the GEMM: QK(j), then PV(j-1)
        SPIN(Q_OUT, 1u);
        uint32_t id = csrr_ss(GEMMX_FINISHED_TASK);
        for (uint32_t j = 0; j <= NT; j++) {
            if (j < NT) {
                SPIN(K_IN, j + 1u);
                const uint32_t g0 = snrt_mcycle();
                qk_arm();
                gemm_go(kbuf[j & 1u], q8, oacc, sbuf + j * SP + BEAT);
                TMO += dsv2_gemm_wait(++id);
                T_QK(j) = snrt_mcycle() - org;
                dsv2_span(tr, DSV2_TR_GEMM, "8 QK", g0, snrt_mcycle());
                S_OUT = j + 1u;
                K_FREE = j + 1u;
            }
            if (j > 0) {
                const uint32_t t = j - 1u, last = (t == NT - 1u);
                SPIN(P_OUT, t + 1u);
                SPIN(V_IN, t + 1u);
                const uint32_t g0 = snrt_mcycle();
                pv_arm(t == 0u, last);
                if (t) pv_factors((const volatile uint32_t *)(corr + (t & 1u) * BEAT));
                gemm_go(vbuf[t & 1u], p8[t & 1u], oacc, last ? (void *)(o16 + BEAT) : (void *)oacc);
                TMO += dsv2_gemm_wait(++id);
                T_PV(t) = snrt_mcycle() - org;
                dsv2_span(tr, DSV2_TR_GEMM, "10 PV", g0, snrt_mcycle());
                V_FREE = t + 1u;
            }
        }
        csrw_ss(COLSCALE_CSR + 0, 0u);  // no dispatch inherits an armed scaler
        O_OUT = 1u;
        return 0;
    }

    // ============================== the SIMD
    SPIN(sy[17], 1u);
    const uint32_t busy0 = snax_simd_busy_cycles();
    uint32_t t_q0 = snrt_mcycle();
    quant_q_segment(qt, QT_PITCH, q8, DV / DSV2_KU, INV_QT);
    quant_q_segment(qp, QP_PITCH, q8 + (DV / DSV2_KU) * BLK, 64u / DSV2_KU, INV_QPE);
    simd_drain(sy, "Q8");
    uint32_t c_q = snrt_mcycle() - t_q0;
    dsv2_span(tr, DSV2_TR_SIMD, "D2 Q8", t_q0, snrt_mcycle());
    Q_OUT = 1u;

    snax_simd_shape_t in, out;
    // m = -65504 into tile 0's mold beat, once the xDMA has filled the seed
    SPIN(M_INIT, 1u);
    snax_simd_shape_flat(&in, mseed, 1);
    snax_simd_shape_flat(&out, sbuf + S_MOLD, 1);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0u);
    simd_task(&in, &out);
    for (uint32_t j = 0; j < NT; j++) {
        uint8_t *sb = sbuf + j * SP;  // the tile's latch; its keys one beat on
        uint8_t *cj = corr + (j & 1u) * BEAT, *pb = p8[j & 1u];
        SPIN(S_OUT, j + 1u);
        const uint32_t s0 = snrt_mcycle();
        // 1 m_new = max(m_old, rowmax): a lanewise MAX over [keys][mold], into the latch
        snax_simd_shape_flat(&in, sb + BEAT, BC + 1u);
        snax_simd_shape_flat(&out, sb, 1);
        snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, BC + 1u,
                       SIMD_RED_MAX | SIMD_RED_LANEWISE);
        simd_task(&in, &out);
        // 2 corr = exp(a' (m_old - m_new)): EW0 SUB over the pair [latch][mold] -> Map EXP
        snax_simd_shape_2d(&in, sb, 2u, S_MOLD, 1u, 0u);
        snax_simd_shape_flat(&out, cj, 1);
        snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                                         (1u << SIMD_EXT_STREAMMAP));
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 0, 2u);
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 1, SIMD_EW_SUB);
        snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 0, A_EXP);
        snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 1, 0u);
        snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 2, SIMD_FUNC_EXP);
        simd_task(&in, &out);
        // 3 P8 and the rowsum in one sweep over [latch][keys]
        snax_simd_shape_flat(&in, sb, 1u + BC);
        snax_simd_shape_flat(&out, pb, BC / 2u + 1u);
        snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                                         (1u << SIMD_EXT_STREAMMAP) |
                                                         (1u << SIMD_EXT_STREAMREDUCE) |
                                                         (1u << SIMD_EXT_FP16TOINT8));
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 0, 1u);
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 1, SIMD_EW_SUB | SIMD_EW_STICKY_B);
        snax_simd_set_op_csr(SIMD_EXT_STREAMREDUCE_CSR, 0, BC);
        snax_simd_set_op_csr(SIMD_EXT_STREAMREDUCE_CSR, 1,
                             SIMD_RED_ADD | SIMD_RED_LANEWISE | SIMD_RED_TAP);
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 0, P8_INV);
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 1, SIMD_QUANT_TAIL(BC) | SIMD_QUANT_ILV4);
        simd_task(&in, &out);
        simd_drain(sy, "softmax epilogue");
        T_SM(j) = snrt_mcycle() - org;
        dsv2_span(tr, DSV2_TR_SIMD, "9 softmax", s0, snrt_mcycle());
        P_OUT = j + 1u;
        // 4 corr * l_old: EW1 MUL over the pair [corr_j][l], behind the rowsum
        snax_simd_shape_2d(&in, cj, 2u, (uint32_t)(l - cj), 1u, 0u);
        snax_simd_shape_flat(&out, pb + P_RS + BEAT, 1);
        snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 2u,
                       SIMD_EW_MUL);
        simd_task(&in, &out);
        // 5 l = rowsum + corr * l_old, over l
        snax_simd_shape_flat(&in, pb + P_RS, 2);
        snax_simd_shape_flat(&out, l, 1);
        snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                       SIMD_RED_ADD | SIMD_RED_LANEWISE);
        simd_task(&in, &out);
        // 6 m_new into the next tile's mold, or after the last tile into mfin: a copy
        snax_simd_shape_flat(&in, sb, 1);
        snax_simd_shape_flat(&out, j + 1u < NT ? sb + SP + S_MOLD : mfin, 1);
        snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0u);
        simd_task(&in, &out);
    }
    simd_drain(sy, "softmax state");

    // ---- o~ = O16 (.) c / l per query ---------------------------------------------------
    SPIN(O_OUT, 1u);
    uint32_t t_n0 = snrt_mcycle();
    // c / l = RSQRT(l / c)^2: l read twice, both through the map, the pair multiplied, into
    // O16's latch
    snax_simd_shape_broadcast(&in, l, 2);
    snax_simd_shape_flat(&out, o16, 1);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMMAP) |
                                                     (1u << SIMD_EXT_STREAMELEMENTWISE_1));
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 0, A_N);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 1, 0u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 2, SIMD_FUNC_RSQRT);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 0, 2u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1, SIMD_EW_MUL);
    simd_task(&in, &out);
    // o~^T = [c/l][O16 ...]: sticky MUL, lanes = queries
    snax_simd_shape_flat(&in, o16, 1 + DV);
    snax_simd_shape_flat(&out, ot16, DV);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1,
                   SIMD_EW_MUL | SIMD_EW_STICKY_B);
    simd_task(&in, &out);
    simd_drain(sy, "normalise");
    // c / l, kept for a check; volatile, or this becomes a memcpy()
    for (uint32_t i = 0; i < BEAT / 4u; i++)
        ((volatile uint32_t *)rsc)[i] = ((const volatile uint32_t *)o16)[i];
    dsv2_span(tr, DSV2_TR_SIMD, "10 O / l", t_n0, snrt_mcycle());
    N_OUT = 1u;
    SPIN(X_OUT, 1u);
    const uint32_t t_a0 = snrt_mcycle();
    // W_UV's 16 A operands in one task: per head and k, every channel reads the same four values
    // (lane stride 0), and the 16 / C beats of one block pack into its 16 rows
    snax_simd_shape_flat(&in, xt, 1);
    in.lane_stride = 0u;
    in.bound[0] = DSV2_MR / SIMD_SPATIAL_CHAN;
    in.stride[0] = 0u;
    in.bound[1] = DV / DSV2_KU;
    in.stride[1] = 8u;
    in.bound[2] = HEADS;
    in.stride[2] = 2u * DV;
    snax_simd_shape_flat(&out, auv, HEADS * (DV / DSV2_KU) * BLK / BEAT);
    snax_simd_use0(SIMD_EXT_FP16TOINT8);
    DSV2_ARM_QUANT(INV_OT);
    simd_task(&in, &out);
    simd_drain(sy, "W_UV operands");
    uint32_t c_norm = snrt_mcycle() - t_n0;
    dsv2_span(tr, DSV2_TR_SIMD, "11 W_UV operands", t_a0, snrt_mcycle());
    const uint32_t simd_busy = snax_simd_busy_cycles() - busy0;
    snax_perf_snapshot_t census;
    snax_perf_read(&census);
    KDONE = 1u;

    // ---- report and check ---------------------------------------------------------------
    uint32_t fails = 0;
    printf(TAG " %u keys in %u tiles of %u; Q8 %u cc; pipeline %u cc (last PV retired); "
               "normalise + W_UV operands %u cc\n",
           NT * BC, NT, BC, c_q, T_PV(NT - 1), c_norm);
    for (uint32_t j = 0; j < NT; j++)
        printf(TAG "   tile %u: QK retired %6u  softmax published %6u  PV retired %6u\n", j,
               T_QK(j), T_SM(j), T_PV(j));
    SPIN(G1, 1u);
#if DSV2_STAGE_CHECKS
    printf(TAG " D2 query assembly\n");
    fails += chk8("Q8 (A-layout of [16 x 576])", q8, g_q8_l, BR * DQK, 0);
    printf(TAG " A1 scores and softmax\n");
    uint32_t sbad = 0, sworst = 0;
    for (uint32_t j = 0; j < NT; j++) {
        uint32_t w, b = dsv2_ulp16((uint16_t *)(sbuf + j * SP + BEAT), g_s_l + j * BC * BR,
                                   BC * BR, &w);
        sbad += b;
        if (w > sworst) sworst = w;
    }
    printf(TAG "   %s %-34s %5u/%5u differ, worst %u ULP (allowed 0)\n",
           sworst ? "FAIL" : "PASS", "S^T, every tile", sbad, NT * dsv2_check_terms(BC * BR),
           sworst);
    fails += sworst ? 1u : 0u;
    fails += chk16("m", (uint16_t *)mfin, g_m_l, BR, 0);
    fails += chk16("l", (uint16_t *)l, g_l_l, BR, 0);
    fails += chk8("P8, last tile (interleaved)", (int8_t *)p8[(NT - 1) & 1u], g_p8_l,
                  BC * DSV2_NU, 0);
    printf(TAG " A1 output\n");
    fails += chk16("O16 = RNE(O * 2^-k_o)", (const uint16_t *)(o16 + BEAT), g_o16_l, DV * BR, 0);
    {
        const uint16_t *r = (const uint16_t *)rsc;
        for (uint32_t q = 0; q < BR; q++)
            if (r[q] != g_rsc[q])
                printf(TAG "     c/l lane %2u: l = %04x  got %04x  want %04x\n", q,
                       ((const uint16_t *)l)[q], r[q], g_rsc[q]);
    }
    fails += chk16("c / l per query", (const uint16_t *)rsc, g_rsc, BR, 0);
#else
    printf(TAG " A1 output\n");
#endif
    fails += chk16("o~ (16 heads x 512)", xt, g_ot_l, HEADS * DV, 0);
    DONE1 = 1u;
    SPIN(G2, 1u);
    fails += chk8("W_UV A operands", auv, g_auv_l, HEADS * DSV2_MR * DV, 0);
    if (TMO) {
        printf(TAG " FAIL: %u bounded wait(s) ran out\n", TMO);
        fails++;
    }
    printf(fails ? TAG " FAIL (%u checks)\n" : TAG " PASS\n", fails);
    printf("[ORIGIN] %u\n", org);
    dsv2_trace_print(tr);
    dsv2_util_print(&census, simd_busy);
    return (int)fails;
}
