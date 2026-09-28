// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// DeepSeek-V2-Lite layer 1's mixture of experts for NTOK = 1 or 2 tokens, h -> out = h + MoE(h),
// on the four-engine split cluster:
//
//     hn  = rmsnorm(h)                                   V1, then V2 into the GEMV A operand
//     ids, w = top6(softmax(hn . W_r))                   G1 + G3, V5
//     s   = down_s(swiglu(hn . GU_s))  the shared experts
//     e_i = down_i(swiglu(hn . GU_i))  for the slots      G1 + G3, V6 + V2, G1 + G3
//     out = h + s + sum_i w_i e_i                        V7
//
// The app includes data.h first, for HID, N_EXP, TOP_K, I_EXP, I_SH, the shifts K_X, K_ED,
// K_SD, the scale INV_H, and the router (wr, its factors wr_s) and the expert table; and NTOK
// when a pass holds two tokens (1 by default).
//
// ======================================================================================
// EXPERT SLOTS, NOT BRANCHES (D3)
// ======================================================================================
//
// The SIMD hart reads the top-6 ids at run time and, per slot, looks the expert up in a
// 64-entry table (data.h `experts`): its gate|up and down weights in DRAM, their dequant
// factors, its SwiGLU output scale. It then POSTS the slot's two GEMVs as jobs. Nothing is
// skipped and nothing branches on the id, so the kernel is the same for every token. An ELF
// carries only the experts its tokens pick; any other id finds a zero entry, which the block
// refuses loudly (it posts the slot's jobs as empty and counts a refusal).
//
// TWO TOKENS. The slots are the union of the tokens' top-6 (hwmodel.union_order: token 0's in
// its order, then token 1's new ones), and every slot's GEMVs carry both tokens (snax-dsv2.h,
// two tokens per GEMV): an expert both tokens pick streams once. A token adds the slots it
// picked, in slot order.
//
// THE JOB QUEUE. Every GEMV of the block is one job {weight, K, N, k, A, y, token pitch of y}.
// Unit 0 is the shared experts, unit u > 0 the slot u - 1; G_u is a unit's gate|up, D_u its
// down. A down runs after the NEXT unit's gate|up, so the SIMD's SwiGLU of a unit (the down's
// A operand) hides behind a whole GEMV:
//     router, G_0, G_1, D_0, G_2, D_1, ..., G_n, D_n-1, D_n        (moe_jg, moe_jd)
// Units alternate between two gate|up outputs, two down A operands and two down outputs, by
// unit parity. The iDMA streams each job's weight in chunks into two L1 buffers, running ahead
// across job boundaries; the GEMM runs one task per chunk once the job's A operand is ready;
// the SIMD handles the jobs' outputs in the same order: a gate|up's dequant and SwiGLU, which
// release the unit's down and the gate|up two units on (the buffers they reuse are then
// free), and a down's dequant and combine. Handoffs are monotonic counters: jobs posted, A
// operands ready, chunks loaded / freed, jobs done, dequantisations done. The router and G_0
// are posted at once; the rest once the top-6 are known, with the job count.
//
// Chunks are 32 output columns (K x 32 bytes, 64 KiB at K = 2,048), 16 for the shared down
// (K = 2,816; MOE_CW): the n-major B-layout makes any multiple of 16 columns a byte slice of
// the packed weight. Three 64 KiB buffers take chunks in turn: the iDMA issues a chunk before
// it waits for the one before, so a transfer is always in flight, and runs up to two chunks
// ahead of the GEMM, so the handoff of a freed buffer is off its path.
//
// TWO WAYS IN (MOE_DUAL: DSV2_DUAL_LOAD, sw/apps/dsv2/dsv2.mk, with one token). The iDMA loads each
// chunk's first half and the xDMA hart its second, at once (snax-xdma-lib.h): the xDMA hart
// walks the same chunks into the same buffers once the GEMM has freed them, and publishes
// XLOADED; a task waits for both.
//
// THE COMBINE (V7). out = h + s once the shared experts are done, then per slot, as soon as
// its down output is dequantised, out += w_i e_i: Map(a = w_i) scales e_i (w_i widened to FP32
// by the core, the product narrowed once to FP16), EW ADD accumulates -- the device model's
// order of summation.
//
// USE. The L1 the block carves (dsv2_moe_carve) must be zero when a pass starts, and its sync
// words (MOE_SY_BYTES) fresh. The input h is the caller's: NTOK rows MOE_HP apart, each with a
// free beat right below it (the norm's seed). With one token the block keeps every slot's
// dequantised gate|up and down output for a check.

#pragma once

#include "snax-dsv2.h"
#include "snax-dsv2-trace.h"

#if !SNAX_HAS_GEMM_CORE || !SNAX_HAS_SIMD_CORE || !SNAX_HAS_XDMA_CORE || !SNAX_HAS_IDMA_CORE
#error "the MoE block needs the four-engine cluster"
#endif
#ifndef NTOK
#define NTOK 1
#endif
#if NTOK != 1 && NTOK != 2
#error "a pass holds one or two tokens"
#endif

// Whether the chunks load in two parts (DSV2_DUAL_LOAD): with one token only. Two tokens run the
// (16, 4, 16) shape, 64 weight bytes a cycle, which the iDMA's stream already feeds: a second
// writer would only take TCDM banks from the array.
#if DSV2_DUAL_LOAD && NTOK == 1
#define MOE_DUAL 1
#else
#define MOE_DUAL 0
#endif

#define MOE_SY_BYTES 2048u
#define MOE_NBUF 3u                       // weight buffers
#define MOE_BUF (HID * 32u)               // a weight buffer: the largest chunk, 64 KiB
#define MOE_CW(K) ((K) * 32u <= MOE_BUF ? 32u : 16u)  // chunk: output columns per GEMM task
#define MOE_NB 2u                         // n-blocks per chunk, at most
#define MOE_NSLOT_MAX (NTOK * TOP_K)      // the union of the tokens' top-6
#define MOE_NJOB_MAX (3 + 2 * MOE_NSLOT_MAX)
#define MOE_EXP_BEATS (N_EXP / 32)
#define MOE_HP (DSV2_BEAT + 2u * HID)     // the caller's h, and hn: a free beat below each row
#define MOE_LGP (DSV2_BEAT + 2u * N_EXP)  // logits, a latch below
#define MOE_ELP (DSV2_BEAT + 2u * N_EXP + DSV2_BEAT)  // [latch][e][sum]
#define MOE_TB (6u * I_SH)                // a token's [gate|up | silu(gate)], FP16
// A GEMV output of `bytes` per token: one token's buffer ends in the packed output's 15 slots of
// spill; two tokens' outputs sit a pitch apart, 4 slots each (snax-dsv2.h, two tokens per
// GEMV).
#if NTOK == 1
#define MOE_YP(bytes) 0u
#define MOE_YBUF(bytes) ((bytes) + DSV2_SPILL(MOE_NB))
#else
#define MOE_YP(bytes) ((((bytes) + DSV2_SPILL2(MOE_NB)) + 63u) & ~63u)
#define MOE_YBUF(bytes) (2u * MOE_YP(bytes))
#endif
#define MOE_YP_R MOE_YP(2u * N_EXP)
#define MOE_YP_GU(b) ((b) ? MOE_YP(2u * 2u * I_EXP) : MOE_YP(2u * 2u * I_SH))  // ygu[b]
#define MOE_YP_DN MOE_YP(2u * HID)

typedef struct {
    uint32_t w, K, N, k, a, y, r;  // DRAM weight (0 = none), shape, D shift, L1 A and y, y's token pitch
} dsv2_moe_job_t;

// ---- the sync words (MOE_SY_BYTES, fresh per pass) --------------------------------------------
#define MOE_POSTED(m) (m)->sy[0]  // jobs posted                  (SIMD -> iDMA, GEMM)
#define MOE_A_OK(m) (m)->sy[1]    // jobs whose A operand is in L1 (SIMD -> GEMM)
#define MOE_LOADED(m) (m)->sy[2]  // weight chunks in L1          (iDMA -> GEMM)
#define MOE_FREED(m) (m)->sy[3]   // chunks retired               (GEMM -> iDMA)
#define MOE_Y_DONE(m) (m)->sy[4]  // jobs whose y is in L1        (GEMM -> SIMD)
#define MOE_S_IN(m) (m)->sy[5]    // jobs whose dequant factors are in L1 (iDMA -> SIMD)
#define MOE_DQ(m) (m)->sy[6]      // jobs the SIMD dequantised    (SIMD -> iDMA: a factor slot is free)
#define MOE_READY(m) (m)->sy[7]   // h is in L1                   (iDMA -> SIMD)
#define MOE_NJOBS(m) (m)->sy[8]   // the pass's job count, once the top-6 are known (SIMD -> iDMA, GEMM)
#define MOE_DONE(m) (m)->sy[9]    // out computed                 (SIMD -> the caller)
#define MOE_TMO(m) (m)->sy[10]    // bounded waits that ran out
#define MOE_REFUSED(m) (m)->sy[11]  // slots whose expert has no weights in the ELF
#define MOE_T_ORG(m) (m)->sy[12]  // the pass's cycle origin
#define MOE_T_SIMD(m) (m)->sy[13] // SIMD busy cycles in the experts (dequant + SwiGLU + dequant + combine)
#define MOE_XLOADED(m) (m)->sy[14] // chunks whose xDMA half is in L1 (xDMA hart -> GEMM)
#define MOE_NSLOT(m) (m)->sy[15]  // slots: the union of the tokens' top-6
#define MOE_HALF(m, t) (m)->sy[16 + (t)]  // which half of token t's ping-pong holds its out
#define MOE_FIN(m, r) (m)->sy[18 + (r)]  // role r (0 iDMA, 1 GEMM, 2 xDMA hart) wrote its profile words
#define MOE_T_JOB(m, j) (m)->sy[32 + (j)]   // job j done, cycles from the origin
#define MOE_FACT(m, j) (m)->sy[64 + (j)]    // job j's dequant factors (DRAM)
#define MOE_IDS(m, t, i) (m)->sy[96 + 8 * (t) + (i)]  // token t's top-6 ids, in order
#define MOE_SLOT_ID(m, s) (m)->sy[112 + (s)]          // slot s's expert
#define MOE_JOBS(m) ((volatile dsv2_moe_job_t *)((m)->sy + 128))
#define MOE_SPIN(m, w, v) dsv2_spin_ge(&(w), (v), &MOE_TMO(m))
// Where each engine's time went (dsv2_moe_report)
#define MOE_P(m, i) (m)->sy[320 + (i)]
#define MOE_P_GLOAD 0    // GEMM core: waiting for a weight chunk
#define MOE_P_GAOK 1     // GEMM core: waiting for a job or its A operand (the SIMD)
#define MOE_P_GARRAY 2   // GEMM core: waiting for the array
#define MOE_P_DXFER 3    // iDMA: issuing transfers and waiting for them to land
#define MOE_P_DFREE 4    // iDMA: waiting for a free weight buffer
#define MOE_P_DPOST 5    // iDMA: waiting for a job, or for a factor slot (the SIMD)
#define MOE_P_SGEMM 6    // SIMD core: waiting for a job's output or factors
#define MOE_P_SDRAIN 7   // SIMD core: waiting for its own tasks to retire
#define MOE_P_SBUSY 8    // the SIMD datapath busy (SIMD_BUSY_CYCLES)
#define MOE_P_SROUTE 9   // SIMD core: the top-6 and the slot table (core code)
#define MOE_P_XXFER 10   // xDMA hart: its halves in flight
#define MOE_P_XFREE 11   // xDMA hart: waiting for a free weight buffer
#define MOE_P_XPOST 12   // xDMA hart: waiting for a job
#define MOE_XDMA_SPIN 200000u  // CSR polls per xDMA start or wait, about 5 cycles each

// The bytes of a chunk of cb the xDMA carries: its second half, beat-aligned; none without
// MOE_DUAL.
#if MOE_DUAL
#define MOE_XBYTES(cb) (((cb) / 2u) & ~(DSV2_BEAT - 1u))
#else
#define MOE_XBYTES(cb) 0u
#endif
#define MOE_W(m, w, v, acc) dsv2_wait_acc(&(w), (v), &MOE_TMO(m), (acc))

typedef struct {
    volatile uint32_t *sy;
    dsv2_trace_t *tr;  // spans for a Gantt chart, or 0 (snax-dsv2-trace.h)
    uint16_t *h;       // the input (the caller's), MOE_HP per token
    int8_t *bbuf[MOE_NBUF];
    uint16_t *sbuf[2];
    uint16_t *sh;      // the shared output e_s, per token
    uint16_t *hn, *yr, *lg, *pr, *w, *gb, *es, *tw, *outb;
    uint16_t *ygu[2], *ydn[2];  // by unit parity: a gate|up's and a down's raw output
    uint16_t *gu, *sgu;  // one token: every slot's dequantised gate|up, and the shared one
    uint8_t *ssq, *tmp, *el;
    int8_t *ha, *a8[2];  // hn's A operand; by unit parity, a down's
} dsv2_moe_t;

// The job indices of unit u's gate|up and down, n slots (the job queue above).
static inline uint32_t moe_jg(uint32_t u) { return u ? 2u * u : 1u; }
static inline uint32_t moe_jd(uint32_t u, uint32_t n) { return u < n ? 2u * u + 3u : 2u * n + 2u; }

// Job j's stage, for a span label; n jobs in the pass.
static inline const char *moe_job_label(uint32_t j, uint32_t n) {
    if (j == 0u) return "15 router";
    if (j == 1u) return "20 shared gate|up";
    if (j == 3u) return "22 shared down";
    return (j & 1u) || j + 1u == n ? "19 down" : "17 gate|up";
}

// token t's copy of a per-token buffer
#define MOE_TOK(p, t, pitch) ((uint16_t *)((uint8_t *)(p) + (t) * (pitch)))

// Carve the block's L1 from `p` (64-byte aligned); h is the caller's. Returns the end.
static inline uint8_t *dsv2_moe_carve(dsv2_moe_t *m, uint8_t *p, volatile uint32_t *sy,
                                      uint16_t *h) {
#define MOE_TAKE(n) (p += (((n) + 63u) & ~63u), p - (((n) + 63u) & ~63u))
    m->sy = sy;
    m->tr = (dsv2_trace_t *)0;
    m->h = h;
    for (uint32_t b = 0; b < MOE_NBUF; b++) m->bbuf[b] = (int8_t *)MOE_TAKE(MOE_BUF);
    m->sbuf[0] = (uint16_t *)MOE_TAKE(2 * 2 * I_SH);         // two factor slots, by job parity
    m->sbuf[1] = (uint16_t *)MOE_TAKE(2 * 2 * I_SH);
    m->sh = (uint16_t *)MOE_TAKE(NTOK * 2 * HID);            // the shared output, per token
    m->hn = (uint16_t *)(MOE_TAKE(NTOK * MOE_HP) + DSV2_BEAT);  // [latch][hn] per token
    m->ssq = MOE_TAKE(NTOK * DSV2_BEAT);
    m->ha = (int8_t *)MOE_TAKE(DSV2_MR * HID);               // hn's A operand, both tokens
    m->a8[0] = (int8_t *)MOE_TAKE(DSV2_MR * I_SH);           // a down's A: even units (the shared's)
    m->a8[1] = (int8_t *)MOE_TAKE(DSV2_MR * I_EXP);          // ... odd units (slots)
    m->yr = (uint16_t *)MOE_TAKE(MOE_YBUF(2 * N_EXP));       // the router's raw logits
    m->ygu[0] = (uint16_t *)MOE_TAKE(MOE_YBUF(2 * 2 * I_SH));   // a gate|up's raw output, even units
    m->ygu[1] = (uint16_t *)MOE_TAKE(MOE_YBUF(2 * 2 * I_EXP));  // ... odd units
    m->ydn[0] = (uint16_t *)MOE_TAKE(MOE_YBUF(2 * HID));     // a down's raw output, by unit parity
    m->ydn[1] = (uint16_t *)MOE_TAKE(MOE_YBUF(2 * HID));
    m->lg = (uint16_t *)(MOE_TAKE(NTOK * MOE_LGP) + DSV2_BEAT);  // [latch][logits] per token
    m->tmp = MOE_TAKE(NTOK * DSV2_BEAT);
    m->el = MOE_TAKE(NTOK * MOE_ELP);                        // [latch][e][sum] per token
    m->pr = (uint16_t *)MOE_TAKE(NTOK * 2 * N_EXP);
    m->w = (uint16_t *)MOE_TAKE(NTOK * DSV2_BEAT);           // the top-6 weights, a beat per token
    m->gb = (uint16_t *)MOE_TAKE(NTOK * MOE_TB);             // per token [gate|up | silu(gate)]
#if NTOK == 1
    m->gu = (uint16_t *)MOE_TAKE(TOP_K * 2 * 2 * I_EXP);     // every slot's [gate | up], dequantised
    m->sgu = (uint16_t *)MOE_TAKE(2 * 2 * I_SH);             // the shared [gate | up]
    m->es = (uint16_t *)MOE_TAKE(TOP_K * 2 * HID);           // every slot's e_i
#else
    m->gu = m->sgu = (uint16_t *)0;
    m->es = (uint16_t *)MOE_TAKE(NTOK * 2 * HID);            // the slot's e_i, per token
#endif
    m->tw = (uint16_t *)MOE_TAKE(2 * HID);                   // w_i e_i
    m->outb = (uint16_t *)MOE_TAKE(NTOK * 2 * 2 * HID);      // per token, the running sum ping-pong
    return p;
#undef MOE_TAKE
}

static inline void moe_post(dsv2_moe_t *m, uint32_t j, const void *w, uint32_t K, uint32_t N,
                            uint32_t k, const void *a, const void *y, uint32_t r) {
    volatile dsv2_moe_job_t *jb = &MOE_JOBS(m)[j];
    jb->w = (uint32_t)w;
    jb->K = K;
    jb->N = N;
    jb->k = k;
    jb->a = (uint32_t)a;
    jb->y = (uint32_t)y;
    jb->r = r;
}

// Jobs 0-1 are always there; the rest once the SIMD has posted the job count.
static inline uint32_t moe_njobs(dsv2_moe_t *m, uint32_t j) {
    if (j < 2u) return 2u + 1u;  // at least one more to come
    MOE_SPIN(m, MOE_NJOBS(m), 1u);
    return MOE_NJOBS(m);
}

// ============================================================ the iDMA (hart 3)

// h16: the NTOK inputs from DRAM ([NTOK, HID]), or 0 when h is already in L1.
static void dsv2_moe_idma(dsv2_moe_t *m, const uint16_t *h16) {
    if (h16) {
        const uint32_t i0 = snrt_mcycle();
        for (uint32_t t = 0; t < NTOK; t++)
            snrt_dma_start_1d(MOE_TOK(m->h, t, MOE_HP), h16 + t * HID, 2 * HID);
        snrt_dma_wait_all();
        dsv2_span(m->tr, DSV2_TR_IDMA, "0 inputs", i0, snrt_mcycle());
    }
    MOE_READY(m) = 1u;
    uint32_t g = 0, xfer = 0, wfree = 0, wpost = 0;  // g: chunks issued
    for (uint32_t j = 0; j < moe_njobs(m, j); j++) {
        MOE_W(m, MOE_POSTED(m), j + 1u, &wpost);
        const uint32_t w = MOE_JOBS(m)[j].w, K = MOE_JOBS(m)[j].K, N = MOE_JOBS(m)[j].N;
        // the job's dequant factors, into its parity slot once job j-2 is dequantised
        if (j >= 2) MOE_W(m, MOE_DQ(m), j - 1u, &wpost);
        uint32_t x0 = snrt_mcycle();
        const uint32_t j0 = x0;
        if (w) snrt_dma_start_1d(m->sbuf[j & 1u], (const void *)MOE_FACT(m, j), 2u * N);
        const uint32_t base = dsv2_dma_quiet();  // the factors, and the chunk still in flight
        xfer += snrt_mcycle() - x0;
        MOE_LOADED(m) = g;
        MOE_S_IN(m) = j + 1u;
        if (!w) continue;
        // chunk c is transfer base + c + 1: issue it, then publish chunk c - 1
        const uint32_t cw = MOE_CW(K), nch = N / cw, cb = K * cw;
        for (uint32_t c = 0; c < nch; c++, g++) {
            if (g >= MOE_NBUF) MOE_W(m, MOE_FREED(m), g - MOE_NBUF + 1u, &wfree);
            x0 = snrt_mcycle();
            snrt_dma_start_1d(m->bbuf[g % MOE_NBUF], (const int8_t *)w + c * cb,
                              cb - MOE_XBYTES(cb));
            if (c) {
                while (dsv2_dma_retired() - base < c) {
                }
                MOE_LOADED(m) = g;
            }
            xfer += snrt_mcycle() - x0;
        }
        dsv2_span(m->tr, DSV2_TR_IDMA, moe_job_label(j, moe_njobs(m, j)), j0, snrt_mcycle());
    }
    (void)dsv2_dma_quiet();
    MOE_LOADED(m) = g;
    MOE_P(m, MOE_P_DXFER) = xfer;
    MOE_P(m, MOE_P_DFREE) = wfree;
    MOE_P(m, MOE_P_DPOST) = wpost;
    MOE_FIN(m, 0) = 1u;
}

// ============================================================ the xDMA hart (hart 2)

#if MOE_DUAL
// The second half of every chunk, into the buffer the iDMA fills with the first, once the GEMM
// has freed it; published on XLOADED. The descriptor is armed per job and re-pointed per chunk.
static void dsv2_moe_xdma(dsv2_moe_t *m) {
    uint32_t g = 0, xfer = 0, wfree = 0, wpost = 0;  // g: chunks issued
    for (uint32_t j = 0; j < moe_njobs(m, j); j++) {
        MOE_W(m, MOE_POSTED(m), j + 1u, &wpost);
        const uint32_t w = MOE_JOBS(m)[j].w, K = MOE_JOBS(m)[j].K, N = MOE_JOBS(m)[j].N;
        if (!w) continue;
        const uint32_t cw = MOE_CW(K), nch = N / cw, cb = K * cw;
        const uint32_t xb = MOE_XBYTES(cb), off = cb - xb, j0 = snrt_mcycle();
        snax_xdma_read_arm((uint32_t)(m->bbuf[g % MOE_NBUF] + off), w + off, xb / DSV2_BEAT);
        for (uint32_t c = 0; c < nch; c++, g++) {
            if (g >= MOE_NBUF) MOE_W(m, MOE_FREED(m), g - MOE_NBUF + 1u, &wfree);
            const uint32_t x0 = snrt_mcycle();
            if (c)
                snax_xdma_read_retask((uint32_t)(m->bbuf[g % MOE_NBUF] + off), w + c * cb + off,
                                      xb / DSV2_BEAT);
            int to = 0;
            const snax_xdma_task_t tk = snax_xdma_start_bounded(MOE_XDMA_SPIN, &to);
            if (!to) to = snax_xdma_wait_bounded(tk, MOE_XDMA_SPIN);
            MOE_TMO(m) += (uint32_t)to;
            MOE_XLOADED(m) = g + 1u;
            xfer += snrt_mcycle() - x0;
        }
        dsv2_span(m->tr, DSV2_TR_XDMA, moe_job_label(j, moe_njobs(m, j)), j0, snrt_mcycle());
    }
    MOE_P(m, MOE_P_XXFER) = xfer;
    MOE_P(m, MOE_P_XFREE) = wfree;
    MOE_P(m, MOE_P_XPOST) = wpost;
    MOE_FIN(m, 2) = 1u;
}
#endif

// ============================================================ the GEMM (hart 0)

static void dsv2_moe_gemm(dsv2_moe_t *m) {
    const uint32_t org = MOE_T_ORG(m);
    uint32_t cbase = 0, id = csrr_ss(GEMMX_FINISHED_TASK), wload = 0, waok = 0, warr = 0, t0;
    for (uint32_t j = 0; j < moe_njobs(m, j); j++) {
        MOE_W(m, MOE_POSTED(m), j + 1u, &waok);
        MOE_W(m, MOE_A_OK(m), j + 1u, &waok);
        const uint32_t s0 = snrt_mcycle();
        const uint32_t w = MOE_JOBS(m)[j].w, K = MOE_JOBS(m)[j].K, N = MOE_JOBS(m)[j].N;
        const uint32_t k = MOE_JOBS(m)[j].k, a = MOE_JOBS(m)[j].a, y = MOE_JOBS(m)[j].y;
        if (w) {
            const uint32_t cw = MOE_CW(K), nb = cw / DSV2_NU, nch = N / cw;
            dsv2_gemv_arm_ntok(K / DSV2_KU, nb, 1u, 0u, k, NTOK);
            if (NTOK == 2) dsv2_gemv_two_tokens(nb, MOE_JOBS(m)[j].r);
            for (uint32_t c = 0; c < nch; c++) {
                const uint32_t g = cbase + c;
                MOE_W(m, MOE_LOADED(m), g + 1u, &wload);
#if MOE_DUAL
                MOE_W(m, MOE_XLOADED(m), g + 1u, &wload);
#endif
                dsv2_gemv_fire((const void *)a, m->bbuf[g % MOE_NBUF],
                               (uint8_t *)y + c * DSV2_SLOT(nb));
                ++id;
                if (c) {
                    t0 = snrt_mcycle();
                    MOE_TMO(m) += dsv2_gemm_wait(id - 1u);
                    warr += snrt_mcycle() - t0;
                    MOE_FREED(m) = g;
                }
            }
            t0 = snrt_mcycle();
            MOE_TMO(m) += dsv2_gemm_wait(id);
            warr += snrt_mcycle() - t0;
            cbase += nch;
            MOE_FREED(m) = cbase;
        }
        MOE_T_JOB(m, j) = snrt_mcycle() - org;
        MOE_Y_DONE(m) = j + 1u;
        dsv2_span(m->tr, DSV2_TR_GEMM, moe_job_label(j, moe_njobs(m, j)), s0, snrt_mcycle());
    }
    MOE_P(m, MOE_P_GLOAD) = wload;
    MOE_P(m, MOE_P_GAOK) = waok;
    MOE_P(m, MOE_P_GARRAY) = warr;
    MOE_FIN(m, 1) = 1u;
}

// ============================================================ the SIMD (hart 1)

static inline void moe_drain(dsv2_moe_t *m, const char *what) {
    const uint32_t t0 = snrt_mcycle();
    if (snax_simd_wait_all_checked(what, SIMD_WAIT_BUDGET)) MOE_TMO(m)++;
    MOE_P(m, MOE_P_SDRAIN) += snrt_mcycle() - t0;
}

// Unit u's gate|up outputs (job j), both tokens: dequantise, then the SwiGLU into its down's
// A operand.
static void moe_gate_up(dsv2_moe_t *m, uint32_t u, uint32_t j, uint32_t inter, uint16_t *g0,
                        uint32_t inv_a) {
    for (uint32_t t = 0; t < NTOK; t++) {
        dsv2_dequant_prep(MOE_TOK(m->ygu[u & 1u], t, MOE_YP_GU(u & 1u)), m->sbuf[j & 1u],
                          MOE_TOK(g0, t, MOE_TB), 2 * inter);
        snax_simd_fire();
    }
    // silu(gate) at the back of the token's block; the gate|up at g0 (a slot's own copy with
    // one token)
    dsv2_swiglu(g0, (uint8_t *)m->gb + 4u * I_SH, (void *)0, m->a8[u & 1u], inter, inv_a,
                (NTOK - 1u) * MOE_TB);
}

// out_t += w e_t: Map(a = w) scales, EW ADD accumulates into the token's other ping-pong half.
static void moe_accumulate(dsv2_moe_t *m, uint32_t t, uint16_t w, const uint16_t *e,
                           uint32_t *half) {
    snax_simd_shape_t in, out;
    uint16_t *acc = m->outb + (2u * t + half[t]) * HID, *nxt = m->outb + (2u * t + (half[t] ^ 1u)) * HID;
    snax_simd_shape_flat(&in, (void *)e, HID / 32u);
    snax_simd_shape_flat(&out, m->tw, HID / 32u);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, snax_simd_f32_from_f16(w), 0u,
                   SIMD_FUNC_LINEAR);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    dsv2_shape_pair(&in, acc, m->tw, HID / 32u);  // ADD commutes: either operand first
    snax_simd_shape_flat(&out, nxt, HID / 32u);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_0, SIMD_EXT_STREAMELEMENTWISE_0_CSR, 2u, SIMD_EW_ADD);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    half[t] ^= 1u;
}

static void dsv2_moe_simd(dsv2_moe_t *m) {
    const uint32_t busy0 = snax_simd_busy_cycles();
    uint32_t wg = 0;  // waited for a job's output or factors
    MOE_SPIN(m, MOE_READY(m), 1u);
    uint32_t s0 = snrt_mcycle();  // a span's start
#define MOE_SPAN(label) dsv2_span(m->tr, DSV2_TR_SIMD, (label), s0, snrt_mcycle())
    // the router and the shared gate|up need only hn's A operand: post them first
    MOE_FACT(m, 0) = (uint32_t)wr_s;
    MOE_FACT(m, 1) = (uint32_t)shared_expert.gu_s;
    moe_post(m, 0, wr, HID, N_EXP, K_X, m->ha, m->yr, MOE_YP_R);
    moe_post(m, 1, shared_expert.gu, HID, 2 * I_SH, K_X, m->ha, m->ygu[0], MOE_YP_GU(0));
    MOE_POSTED(m) = 2u;
    for (uint32_t t = 0; t < NTOK; t++)
        dsv2_rmsnorm_row(MOE_TOK(m->h, t, MOE_HP), m->ssq + t * DSV2_BEAT,
                         MOE_TOK(m->hn, t, MOE_HP), 11);
    dsv2_quant_a2(m->hn, MOE_TOK(m->hn, NTOK - 1u, MOE_HP), m->ha, HID, INV_H);
    moe_drain(m, "norm");
    MOE_A_OK(m) = 2u;  // the router's and the shared gate|up's A operand is hn's
    MOE_SPAN("14 norm, A operand");
    // router: dequant, softmax, top 6 per token
    MOE_W(m, MOE_S_IN(m), 1u, &wg);
    MOE_W(m, MOE_Y_DONE(m), 1u, &wg);
    s0 = snrt_mcycle();
    for (uint32_t t = 0; t < NTOK; t++) {
        dsv2_dequant_prep(MOE_TOK(m->yr, t, MOE_YP_R), m->sbuf[0], MOE_TOK(m->lg, t, MOE_LGP), N_EXP);
        snax_simd_fire();
        dsv2_softmax_row(MOE_TOK(m->lg, t, MOE_LGP), m->tmp + t * DSV2_BEAT,
                         m->el + t * MOE_ELP + DSV2_BEAT, MOE_TOK(m->pr, t, 2 * N_EXP),
                         MOE_EXP_BEATS);
    }
    moe_drain(m, "router");
    MOE_DQ(m) = 1u;
    MOE_SPAN("16 router dequant, softmax");
    const uint32_t tr0 = snrt_mcycle();
    // the slots: token 0's experts in order, then token 1's new ones
    uint32_t ids[NTOK][TOP_K], nslot = 0, slot_id[MOE_NSLOT_MAX];
    for (uint32_t t = 0; t < NTOK; t++) {
        const uint16_t *pr = MOE_TOK(m->pr, t, 2 * N_EXP);
        dsv2_top_k16(pr, N_EXP, TOP_K, ids[t]);
        for (uint32_t i = 0; i < TOP_K; i++) {
            MOE_TOK(m->w, t, DSV2_BEAT)[i] = pr[ids[t][i]];
            MOE_IDS(m, t, i) = ids[t][i];
            uint32_t s = 0;
            while (s < nslot && slot_id[s] != ids[t][i]) s++;
            if (s == nslot) slot_id[nslot++] = ids[t][i];
        }
    }
    uint32_t refused = 0;
    for (uint32_t u = 0; u <= nslot; u++) {
        const expert_t *ex = u ? &experts[slot_id[u - 1u]] : &shared_expert;
        if (u) {
            if (!ex->gu) {
                printf("[MOE] FAIL: expert %u has no weights in this ELF; slot %u refused\n",
                       slot_id[u - 1u], u - 1u);
                refused++;
            }
            MOE_SLOT_ID(m, u - 1u) = slot_id[u - 1u];
            MOE_FACT(m, moe_jg(u)) = (uint32_t)ex->gu_s;
            moe_post(m, moe_jg(u), ex->gu, HID, 2 * I_EXP, K_X, m->ha, m->ygu[u & 1u],
                     MOE_YP_GU(u & 1u));
        }
        MOE_FACT(m, moe_jd(u, nslot)) = (uint32_t)ex->dn_s;
        moe_post(m, moe_jd(u, nslot), ex->dn, u ? I_EXP : I_SH, HID, u ? K_ED : K_SD,
                 m->a8[u & 1u], m->ydn[u & 1u], MOE_YP_DN);
    }
    MOE_REFUSED(m) = refused;
    MOE_NSLOT(m) = nslot;
    MOE_NJOBS(m) = 3u + 2u * nslot;
    MOE_POSTED(m) = 3u + 2u * nslot;
    MOE_A_OK(m) = 3u;  // G_1: hn's A operand
    MOE_P(m, MOE_P_SROUTE) = snrt_mcycle() - tr0;
    s0 = tr0;
    MOE_SPAN("16 top-6, slots (core)");

    // the units, in the job order: G_0, then per u G_u and D_u-1, then D_n. A gate|up's SwiGLU
    // releases its unit's down and the gate|up two units on; the downs combine in slot order,
    // after out = h + s.
    uint32_t t_simd = 0, t0, half[NTOK];
    snax_simd_shape_t in, out;
    for (uint32_t u = 0; u <= nslot + 1u; u++) {
        if (u <= nslot) {
            const uint32_t jg = moe_jg(u);
            const expert_t *ex = u ? &experts[slot_id[u - 1u]] : &shared_expert;
            MOE_W(m, MOE_S_IN(m), jg + 1u, &wg);
            MOE_W(m, MOE_Y_DONE(m), jg + 1u, &wg);
            t0 = s0 = snrt_mcycle();
            if (!refused || !u) {
                uint16_t *g0 = NTOK == 1 ? (u ? m->gu + (u - 1u) * 2 * I_EXP : m->sgu) : m->gb;
                moe_gate_up(m, u, jg, u ? I_EXP : I_SH, g0, ex->inv_a);
                moe_drain(m, u ? "gate|up + swiglu" : "shared gate|up + swiglu");
            }
            MOE_DQ(m) = jg + 1u;
            MOE_A_OK(m) = u + 2u <= nslot ? moe_jg(u + 2u) + 1u : moe_jd(u, nslot) + 1u;
            t_simd += snrt_mcycle() - t0;
            MOE_SPAN(u ? "18 dequant, SwiGLU" : "21 dequant, SwiGLU");
        }
        if (u == 0u) continue;
        // the down of unit v = u - 1
        const uint32_t v = u - 1u, jd = moe_jd(v, nslot);
        MOE_W(m, MOE_S_IN(m), jd + 1u, &wg);
        MOE_W(m, MOE_Y_DONE(m), jd + 1u, &wg);
        t0 = s0 = snrt_mcycle();
        if (v == 0u) {  // the shared experts: out = h + s per token
            for (uint32_t t = 0; t < NTOK; t++) {
                dsv2_dequant_prep(MOE_TOK(m->ydn[0], t, MOE_YP_DN), m->sbuf[jd & 1u],
                                  MOE_TOK(m->sh, t, 2 * HID), HID);
                snax_simd_fire();
                dsv2_shape_pair(&in, MOE_TOK(m->h, t, MOE_HP), MOE_TOK(m->sh, t, 2 * HID),
                                HID / 32u);
                snax_simd_shape_flat(&out, m->outb + 2u * t * HID, HID / 32u);
                snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_0, SIMD_EXT_STREAMELEMENTWISE_0_CSR, 2u,
                               SIMD_EW_ADD);
                snax_simd_program_fast(&in, &out);
                snax_simd_fire();
                half[t] = 0u;
            }
            moe_drain(m, "shared down");
        } else if (!refused) {  // slot v - 1: add w e to every token that picked the expert
            const uint32_t s = v - 1u;
            uint16_t *e0 = NTOK == 1 ? m->es + s * HID : m->es;
            for (uint32_t t = 0; t < NTOK; t++) {
                dsv2_dequant_prep(MOE_TOK(m->ydn[v & 1u], t, MOE_YP_DN), m->sbuf[jd & 1u],
                                  MOE_TOK(e0, t, 2 * HID), HID);
                snax_simd_fire();
            }
            for (uint32_t t = 0; t < NTOK; t++)
                for (uint32_t i = 0; i < TOP_K; i++)
                    if (ids[t][i] == slot_id[s])
                        moe_accumulate(m, t, MOE_TOK(m->w, t, DSV2_BEAT)[i], MOE_TOK(e0, t, 2 * HID),
                                       half);
            moe_drain(m, "down + combine");
        }
        MOE_DQ(m) = jd + 1u;
        t_simd += snrt_mcycle() - t0;
        MOE_SPAN(v ? "23 dequant, w e added" : "23 dequant, h + s");
    }
    MOE_T_SIMD(m) = t_simd;
    MOE_P(m, MOE_P_SGEMM) = wg;
    MOE_P(m, MOE_P_SBUSY) = snax_simd_busy_cycles() - busy0;
    // each token's sum ends in one half of its ping-pong; the caller reads out_t there
    for (uint32_t t = 0; t < NTOK; t++) MOE_HALF(m, t) = half[t];
    MOE_DONE(m) = 1u;
#undef MOE_SPAN
}

// Token t's output, once the block is done.
static inline const uint16_t *dsv2_moe_out(dsv2_moe_t *m, uint32_t t) {
    return m->outb + (2u * t + MOE_HALF(m, t)) * HID;
}

// The cycle report, from the SIMD hart once the block is done: it waits for the GEMM's and the
// iDMA's profile words.
static void dsv2_moe_report(dsv2_moe_t *m, const char *tag) {
    for (uint32_t r = 0; r < (MOE_DUAL ? 3u : 2u); r++) MOE_SPIN(m, MOE_FIN(m, r), 1u);
    printf("%s MoE, %u token(s):", tag, NTOK);
    for (uint32_t t = 0; t < NTOK; t++) {
        printf(" top-6");
        for (uint32_t i = 0; i < TOP_K; i++) printf(" %u", MOE_IDS(m, t, i));
        printf(";");
    }
    printf(" %u slots\n%s   jobs done at", MOE_NSLOT(m), tag);
    for (uint32_t j = 0; j < MOE_NJOBS(m); j++) printf(" %u", MOE_T_JOB(m, j));
    printf(" cc\n%s   SIMD busy in the experts (dequant + SwiGLU + dequant + combine) %u cc\n", tag,
           MOE_T_SIMD(m));
    printf("%s   GEMM core waited: for weights %u, for a job or its A operand %u, for the array "
           "%u cc\n", tag, MOE_P(m, MOE_P_GLOAD), MOE_P(m, MOE_P_GAOK), MOE_P(m, MOE_P_GARRAY));
    printf("%s   iDMA: issuing and landing transfers %u, waiting for a free buffer %u, for a job "
           "or factor slot %u cc\n", tag, MOE_P(m, MOE_P_DXFER), MOE_P(m, MOE_P_DFREE),
           MOE_P(m, MOE_P_DPOST));
    printf("%s   SIMD: datapath busy %u; core waited for the GEMM %u, for its tasks %u; top-6 and "
           "slots %u cc\n", tag, MOE_P(m, MOE_P_SBUSY), MOE_P(m, MOE_P_SGEMM),
           MOE_P(m, MOE_P_SDRAIN), MOE_P(m, MOE_P_SROUTE));
#if MOE_DUAL
    printf("%s   xDMA hart: its halves in flight %u, waiting for a free buffer %u, for a job %u cc\n",
           tag, MOE_P(m, MOE_P_XXFER), MOE_P(m, MOE_P_XFREE), MOE_P(m, MOE_P_XPOST));
#endif
}
