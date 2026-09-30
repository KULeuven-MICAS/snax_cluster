// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The GEMV with streamed weights, for DeepSeek-V2-Lite layer 1 at its real shapes:
//
//     y  (T x N, FP16) = RNE(x . W * 2^-k)        x: T x K INT8, W: K x N INT8 or INT4 in DRAM
//     yd (T x N, FP16) = y (.) s                   s[n] = s_x * s_w[n] * 2^k, FP16
//
// for T = DSV2_GEMV_NTOK tokens (sw/apps/dsv2/dsv2.mk): 1, or 2 or 4 through the (4, 4, 32)
// array shape, which reads every weight once for all of them.
//
// for the eight shapes of the layer: W_Q 2048 x 3072, W_DKV 2048 x 576, W_O 2048 x 2048,
// the router 2048 x 64, an expert's gate|up 2048 x 2816 and down 1408 x 2048, and the shared
// experts' gate|up 2048 x 5632 and down 2816 x 2048 -- 36 MiB of weights, each shape run on
// the activation that really reaches it in the layer (the golden pack, sw/apps/dsv2/util).
// With DSV2_WBITS = 4 every weight but the router's is INT4 (18 MiB), read through B's converter
// (snax-dsv2.h, WEIGHT WIDTH): a chunk is half the bytes, the array passes are the same.
//
// ======================================================================================
// THE DATAFLOW
// ======================================================================================
//
//   iDMA (hart 3)   streams W in CHUNKS of 64 output columns (K x 64 bytes, B-layout; K x 32 at INT4) into
//                   two L1 buffers, each as soon as the GEMM has retired the task that last
//                   read it. With DSV2_DUAL_LOAD it loads a chunk's first half and the xDMA
//                   hart its second, at once (snax-xdma-lib.h).
//   GEMM (hart 0)   one task per chunk: K, N = 64, no C input, in the array shape DSV2_GEMV
//                   picks (snax-dsv2.h): (1, 4, 32) by default, 128 weight bytes a pass; for
//                   four tokens (4, 4, 32), the same passes. A chunk yields 64 FINISHED
//                   outputs a token, so nothing accumulates across tasks.
//   SIMD (hart 1)   one task per token and shape once every chunk has retired: EW0 MUL over
//                   interleaved y and s beats, in place (G3, the per-column dequantisation).
//   xDMA (hart 2)   zeroes the whole arena first: TCDM has no reset, and on RTL an
//                   unwritten word reads X; then, with DSV2_DUAL_LOAD, the chunks' halves.
//
// Handoffs are monotonic counters in L1 (chunks loaded, chunks freed, shapes done), never a
// write-barrier-read: a barrier does not wait for a posted store.
//
// THE A OPERAND. One token: the input is x alone, K bytes, and the iDMA places it in row 0 of the
// A buffer's 16-row blocks, where the quantiser writes it in the layer: one 2-D transfer of K / 4
// runs of 4 bytes, 64 apart. Several tokens: the operand comes laid out (snax-dsv2.h, SEVERAL
// TOKENS; util/layout.py to_a4), 4 K bytes in one transfer. The GEMV reads rows 0..T-1 alone.
//
// THE OUTPUT needs no gather: task j writes token t's 64 outputs from y + YP t + 128 j, so each
// token's outputs are contiguous, and the D port writes them into the buffer the dequantisation
// overwrites. YP is 2 N, plus the spill two tokens leave behind each output.
//
// THE D-PORT SHIFT (G2). k comes from the worst case, 127^2 * K <= 65,504 * 2^k: 9 at
// K = 1,408 and 2,048, 10 at 2,816. It is set with set_versacore_d_shift(), which
// set_versacore_streamer_csr() never writes. With the stage checks on, the router's chunk is
// rerun at k = 0 as the negative control: the sums pass 65,504, and the output must hold
// the +-Inf the golden predicts, bit for bit.
//
// ======================================================================================
// CHECKS (on the SIMD hart; main returns the number that failed)
// ======================================================================================
//
// The final results, once every shape has run (DSV2_STAGE_CHECKS = 0, the default):
//
//   per shape   yd == golden for every token, bit for bit (EW0 MUL: exact FP32 product, one
//               RNE), no Inf
//
// which holds only if the D port applied the shift. Four tokens' outputs of every shape do not
// fit L1, so with four tokens each shape's are checked as it finishes, between the shapes. With
// DSV2_STAGE_CHECKS = 1 every shape is, with its intermediate and the negative control:
//
//   per shape   y  == golden, bit for bit (the array and the D port), no Inf
//   router      the router's chunk again at k = 0: y == golden, bit for bit, and it holds at
//               least one Inf
//
// Every shape keeps its yd; the goldens are staged into the chunk buffers once the tasks that
// read them have retired. Per shape the report gives the cycles per chunk against the array's
// floor (K/4 x 4 passes), the GEMM's utilisation, and the iDMA's bytes per cycle. The engines'
// spans follow ([SPAN], snax-dsv2-trace.h), one per chunk on the GEMM (a task, from when it
// runs to its retirement), the iDMA (the chunk's head) and the xDMA hart (its tail).

#include "data.h"
#include "snax-core-roles.h"
#include "snax-dsv2.h"
// One span per chunk on the GEMM, the iDMA and the xDMA hart: 286 chunks over the eight shapes,
// plus the operand loads and, with the stage checks, the goldens and the control.
#define DSV2_TR_CAP_GEMM 320u
#define DSV2_TR_CAP_XDMA 320u
#define DSV2_TR_CAP_IDMA 320u
#include "snax-dsv2-trace.h"

#if !SNAX_HAS_GEMM_CORE || !SNAX_HAS_SIMD_CORE || !SNAX_HAS_XDMA_CORE || !SNAX_HAS_IDMA_CORE
#error "this kernel needs the four-engine cluster"
#endif
#if GEMV_CHUNK % DSV2_NU
#error "a chunk must be whole 16-column blocks"
#endif

#define TAG "[GEMV]"
#define NB (GEMV_CHUNK / DSV2_NU)  // n-blocks per chunk = 4
#define YSLOT DSV2_SLOT(NB)         // one chunk's outputs, FP16 = 128 B = the packed row pitch
#define YSPILL DSV2_SPILL(NB)

#ifndef DSV2_GEMV_NTOK
#define DSV2_GEMV_NTOK 1
#endif
#define NTOK DSV2_GEMV_NTOK
#if NTOK != 1 && NTOK != 2 && NTOK != 4
#error "DSV2_GEMV_NTOK is the tokens per GEMV: 1, or 2 or 4 in the (4, 4, 32) shape"
#endif
// bytes between two tokens' outputs of N columns (snax-dsv2.h, SEVERAL TOKENS)
#define YP(N) (2u * (N) + (NTOK == 2 ? DSV2_TOKSPILL : 0u))
#if NTOK > GEMV_DATA_NTOK
#error "data.h holds fewer tokens than DSV2_GEMV_NTOK"
#endif
// Each shape checked as it finishes: with the stage checks, or with four tokens, whose outputs of
// every shape do not fit L1 at once.
#define PER_SHAPE (DSV2_STAGE_CHECKS || NTOK > 1)

// ---- shared words ----------------------------------------------------------------------
#define LOADED sy[0]  // weight chunks in L1, over all shapes      (iDMA -> GEMM)
#define FREED sy[1]   // chunks whose task retired: buffer free     (GEMM -> iDMA)
#define OPS sy[2]     // shapes whose A operand and s are in L1     (iDMA -> GEMM)
#define GDONE sy[3]   // shapes whose GEMV and control retired      (GEMM -> SIMD, iDMA)
#define GOLD sy[4]    // shapes whose goldens are staged            (iDMA -> SIMD)
#define TMO sy[5]     // bounded waits that ran out
#define XREQ sy[6]    // chunks whose buffer is free for the xDMA's part   (iDMA -> xDMA hart)
#define XLOADED sy[7] // chunks whose xDMA part landed                    (xDMA hart -> iDMA)
#define T_FIRST(s) sy[16 + 8 * (s) + 0]  // GEMM: first chunk fired
#define T_LAST(s) sy[16 + 8 * (s) + 1]   // GEMM: last chunk retired
#define D_BUSY(s) sy[16 + 8 * (s) + 2]   // iDMA: cycles inside chunk transfers
#define D_WAIT(s) sy[16 + 8 * (s) + 3]   // iDMA: cycles waiting for a free buffer
#define T_DQ(s) sy[16 + 8 * (s) + 4]     // SIMD: the dequantisation task
#define X_BUSY(s) sy[16 + 8 * (s) + 5]   // xDMA hart: cycles inside its parts' transfers

// With DSV2_DUAL_LOAD a chunk loads in two parts at once: its head on the iDMA and its last
// XBYTES on the xDMA (snax-xdma-lib.h), GEMV_XFRAC sixteenths of it, beat-aligned.
#ifndef GEMV_XFRAC
#define GEMV_XFRAC 8u
#endif
#define XBYTES(cbytes) (((cbytes) / 16u * GEMV_XFRAC) & ~(DSV2_BEAT - 1u))
#define XSPIN 200000u  // CSR polls per xDMA start or wait
#define SPIN(w, v) dsv2_spin_ge(&(w), (v), &TMO)

// ============================================================ the report and the checks

// Shape s's cycles against the array's floor, the weight rate, and the loads' rates.
static void shape_report(volatile uint32_t *sy, uint32_t s) {
    const gemv_shape_t *sh = &gemv_shapes[s];
    const uint32_t K = sh->K, N = sh->N, k = sh->k, nch = N / GEMV_CHUNK;
    const uint32_t cyc = T_LAST(s) - T_FIRST(s);
    const uint32_t shape = NTOK > 1 ? DSV2_SHAPE_GEMV4 : dsv2_gemv_shape();
    // array passes, one per cycle at best: shape 1 takes two column blocks a pass
    const uint32_t bpp = NTOK > 1 ? dsv2_pair_blocks(NB) : dsv2_gemv_blocks_per_pass(NB);
    const uint32_t floor = nch * (K / DSV2_KU) * (NB / bpp);
    const uint32_t cb = DSV2_WB(K * GEMV_CHUNK, sh->wbits == 4u);
    const uint32_t xb = DSV2_DUAL_LOAD ? XBYTES(cb) : 0u;
    const uint32_t dbusy = D_BUSY(s), xbusy = X_BUSY(s);
    printf(TAG " %-15s %4u x %4u  k=%2u  %2u chunks: %7u cc (%5u/chunk, floor %5u) "
               "GEMM %3u%%  %3u B/cc  iDMA %2u B/cc, waited %u cc",
           sh->name, K, N, k, nch, cyc, cyc / nch, floor / nch,
           cyc ? (100u * (floor / 16u)) / (cyc / 16u) : 0u, cyc ? (nch * cb) / cyc : 0u,
           dbusy ? (nch * (cb - xb)) / dbusy : 0u, D_WAIT(s));
    if (xb) printf("  xDMA %2u B/cc", xbusy ? (nch * xb) / xbusy : 0u);
    printf("  dequant %u cc  [shape %u, INT%u, %u token%s]\n", T_DQ(s), shape, sh->wbits, NTOK,
           NTOK > 1 ? "s" : "");
}

// Token by token, n values YP(n) bytes apart against the golden's [NTOK, n]: the values that
// differ, and the infinities.
static uint32_t cmp_tok(const char *what, const uint16_t *got, const uint16_t *want, uint32_t n,
                        uint32_t *ninf) {
    uint32_t bad = 0;
    *ninf = 0;
    for (uint32_t t = 0; t < NTOK; t++) {
        const uint16_t *g = (const uint16_t *)((const uint8_t *)got + t * YP(n));
        bad += dsv2_cmp16(TAG, what, g, want + t * n, n);
        *ninf += dsv2_count_inf16(g, n);
    }
    return bad;
}

// A shape's final result, every token's: yd bit for bit, and no Inf.
static uint32_t check_yd(uint32_t s, const uint16_t *yd, const uint16_t *gyd) {
    const gemv_shape_t *sh = &gemv_shapes[s];
    uint32_t ninf, fails = dsv2_check(TAG, !cmp_tok("yd", yd, gyd, sh->N, &ninf), sh->name,
                                      "yd = y (.) s, bit-exact");
    return fails + dsv2_check(TAG, !ninf, sh->name, "no Inf in yd");
}

#if GEMV_NEG_SHAPE >= 0 && DSV2_STAGE_CHECKS
// The negative control: the router at k = 0, bit for bit, and overflowing.
static uint32_t check_control(const uint16_t *y0, const uint16_t *gy0) {
    const char *name = gemv_shapes[GEMV_NEG_SHAPE].name;
    uint32_t ninf;
    const uint32_t bad = cmp_tok("y(k=0)", y0, gy0, GEMV_CHUNK, &ninf);
    printf(TAG "   negative control: k = 0 gives %u Inf of %u\n", ninf, NTOK * GEMV_CHUNK);
    uint32_t fails = dsv2_check(TAG, !bad, name,
                                "k = 0 output matches the golden, Infs included");
    return fails + dsv2_check(TAG, ninf > 0, name, "k = 0 overflows (the shift is what prevents it)");
}
#endif

// ============================================================ main

int main() {
    // ---- L1, derived identically on every hart -----------------------------------------
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 63u) & ~63u);
    uint8_t *arena = p;
    uint32_t ntot = 0;  // every shape's outputs
    for (uint32_t s = 0; s < GEMV_NSHAPES; s++) ntot += gemv_shapes[s].N;
#define TAKE(n) (p += (((n) + 63u) & ~63u), p - (((n) + 63u) & ~63u))
    volatile uint32_t *sy = (volatile uint32_t *)TAKE(512);
    dsv2_trace_t *tr = (dsv2_trace_t *)TAKE(sizeof(dsv2_trace_t));
    int8_t *abuf = (int8_t *)TAKE(DSV2_MR * GEMV_KMAX);  // the A buffer, token t in row t
    uint16_t *sbuf = (uint16_t *)TAKE(2u * GEMV_NMAX);    // the dequant factors
    // y, then yd in its place: every shape's in order, or with PER_SHAPE the current shape's;
    // then the spill of a shape-0 GEMV
    uint16_t *ydall = (uint16_t *)TAKE(NTOK * YP(PER_SHAPE ? GEMV_NMAX : ntot) + YSPILL);
    uint16_t *y0buf = (uint16_t *)TAKE(NTOK * YP(GEMV_CHUNK) + YSPILL);  // the k = 0 control
    int8_t *bbuf[2];
    bbuf[0] = (int8_t *)TAKE(GEMV_KMAX * GEMV_CHUNK);
    bbuf[1] = (int8_t *)TAKE(GEMV_KMAX * GEMV_CHUNK);
    uint16_t *gold = (uint16_t *)bbuf[0];  // goldens, once the tasks reading bbuf[0] retired
    const uint32_t arena_beats = (uint32_t)(p - arena) / DSV2_BEAT;
    (void)y0buf;  // the control's output, with the stage checks only; both builds share one L1 map
#undef TAKE

    const int isG = snax_is_gemm_core(), isS = snax_is_simd_core();
    const int isX = snax_is_xdma_core(), isD = snax_is_idma_core();

    if (isS) dsv2_print_l1((uint32_t)(p - arena), DSV2_L1_TOP - (uint32_t)arena);
    if ((uint32_t)p > DSV2_L1_TOP) {
        if (isS) printf(TAG " FAIL: the arena ends at 0x%08x, past the stacks at 0x%08x\n",
                        (uint32_t)p, DSV2_L1_TOP);
        return isS ? 1 : 0;
    }
    if (isS) dsv2_tcdm_prio_install(DSV2_TCDM_PRIO);
    const uint32_t f0 = snrt_mcycle();
    if (isX) dsv2_xdma_fill_zero(arena, arena_beats);
    if (isX) dsv2_span(tr, DSV2_TR_XDMA, "0 zero the arena", f0, snrt_mcycle());
    snrt_cluster_hw_barrier();

    if (isD) {
        // ============================== the iDMA: operands, chunks, goldens
        uint32_t cbase = 0;
        for (uint32_t s = 0; s < GEMV_NSHAPES; s++) {
            const gemv_shape_t *sh = &gemv_shapes[s];
            const uint32_t K = sh->K, N = sh->N, nch = N / GEMV_CHUNK;
            const uint32_t cbytes = DSV2_WB(K * GEMV_CHUNK, sh->wbits == 4u);
            const int8_t *w = sh->w;
            uint32_t o0 = snrt_mcycle();
            if (NTOK > 1)
                snrt_dma_start_1d(abuf, sh->a4, DSV2_AROWS(NTOK) * K);
            else
                snrt_dma_start_2d(abuf, sh->a, DSV2_KU, DSV2_MR * DSV2_KU, DSV2_KU, K / DSV2_KU);
            snrt_dma_start_1d(sbuf, sh->s, 2u * N);
            snrt_dma_wait_all();
            OPS = s + 1u;
            dsv2_span(tr, DSV2_TR_IDMA, "0 operands", o0, snrt_mcycle());
            uint32_t busy = 0, waited = 0;
            const uint32_t xb = DSV2_DUAL_LOAD ? XBYTES(cbytes) : 0u;
            for (uint32_t j = 0; j < nch; j++) {
                uint32_t t = snrt_mcycle();
                if (j >= 2) SPIN(FREED, cbase + j - 1u);  // chunk j-2 retired
                uint32_t t1 = snrt_mcycle();
                if (xb) XREQ = cbase + j + 1u;             // the xDMA's part may start
                snrt_dma_start_1d(bbuf[j & 1u], w + j * cbytes, cbytes - xb);
                snrt_dma_wait_all();
                uint32_t t2 = snrt_mcycle();
                if (xb) SPIN(XLOADED, cbase + j + 1u);
                waited += t1 - t;
                busy += t2 - t1;
                LOADED = cbase + j + 1u;
                dsv2_span(tr, DSV2_TR_IDMA, sh->name, t1, t2);
            }
            D_BUSY(s) = busy;
            D_WAIT(s) = waited;
#if PER_SHAPE
            // this shape's goldens: yd, and with the stage checks y and the control's y at k = 0
            SPIN(GDONE, s + 1u);
            o0 = snrt_mcycle();
            snrt_dma_start_1d(gold, sh->yd, 2u * NTOK * N);
#if DSV2_STAGE_CHECKS
            snrt_dma_start_1d(gold + NTOK * N, sh->y, 2u * NTOK * N);
#if GEMV_NEG_SHAPE >= 0
            if ((int)s == GEMV_NEG_SHAPE)
                snrt_dma_start_1d(gold + 2u * NTOK * N, router_y_k0, NTOK * YSLOT);
#endif
#endif
            snrt_dma_wait_all();
            GOLD = s + 1u;
            dsv2_span(tr, DSV2_TR_IDMA, "0 goldens", o0, snrt_mcycle());
#endif
            cbase += nch;
            snrt_cluster_hw_barrier();
        }
#if !PER_SHAPE
        // after the kernel: every shape's yd golden, in order
        uint32_t off = 0;
        for (uint32_t s = 0; s < GEMV_NSHAPES; s++) {
            snrt_dma_start_1d(gold + off, gemv_shapes[s].yd, 2u * gemv_shapes[s].N);
            off += gemv_shapes[s].N;
        }
        snrt_dma_wait_all();
        GOLD = GEMV_NSHAPES;
#endif
        return 0;
    }

    if (isG) {
        // ============================== the GEMM: one task per chunk
        uint32_t cbase = 0, yoff = 0;
        for (uint32_t s = 0; s < GEMV_NSHAPES; s++) {
            const gemv_shape_t *sh = &gemv_shapes[s];
            const uint32_t K = sh->K, N = sh->N, k = sh->k, nch = N / GEMV_CHUNK;
            uint8_t *ybase = (uint8_t *)ydall + (PER_SHAPE ? 0u : 2u * yoff);
            SPIN(OPS, s + 1u);
            // token t's outputs YP(N) bytes after token t-1's
            dsv2_gemv_arm_ntok(K / DSV2_KU, NB, 1u, 0u, k, NTOK, sh->wbits == 4u, YP(N));
            uint32_t id = csrr_ss(GEMMX_FINISHED_TASK), run = 0;
            for (uint32_t j = 0; j < nch; j++) {
                SPIN(LOADED, cbase + j + 1u);
                dsv2_gemv_fire(abuf, bbuf[j & 1u], ybase + j * YSLOT);  // outputs from column 64 j
                const uint32_t fired = snrt_mcycle();
                if (j == 0) T_FIRST(s) = run = fired;
                ++id;
                // Task j is queued behind task j-1; once j-1 retires, its buffer is free, and
                // task j runs from then (or from its issue, had j-1 already retired).
                if (j) {
                    TMO += dsv2_gemm_wait(id - 1u);
                    const uint32_t retired = snrt_mcycle();
                    FREED = cbase + j;
                    dsv2_span(tr, DSV2_TR_GEMM, sh->name, run, retired);
                    run = retired > fired ? retired : fired;
                }
            }
            TMO += dsv2_gemm_wait(id);
            T_LAST(s) = snrt_mcycle();
            dsv2_span(tr, DSV2_TR_GEMM, sh->name, run, T_LAST(s));
#if GEMV_NEG_SHAPE >= 0 && DSV2_STAGE_CHECKS
            if ((int)s == GEMV_NEG_SHAPE) {
                // The negative control: the router's one chunk, still in buffer 0, at k = 0.
                const uint32_t c0 = snrt_mcycle();
                dsv2_gemv_arm_ntok(K / DSV2_KU, NB, 1u, 0u, 0u, NTOK, sh->wbits == 4u,
                                   YP(GEMV_CHUNK));
                dsv2_gemv_fire(abuf, bbuf[0], y0buf);
                TMO += dsv2_gemm_wait(++id);
                dsv2_span(tr, DSV2_TR_GEMM, "15 k = 0 control", c0, snrt_mcycle());
            }
#endif
            FREED = cbase + nch;
            GDONE = s + 1u;
            cbase += nch;
            yoff += N;
            snrt_cluster_hw_barrier();
        }
        return 0;
    }

    if (isX) {
        // ============================== the xDMA hart: each chunk's tail, with DSV2_DUAL_LOAD
        uint32_t cbase = 0;
        for (uint32_t s = 0; s < GEMV_NSHAPES; s++) {
            const gemv_shape_t *sh = &gemv_shapes[s];
            const uint32_t cbytes = DSV2_WB(sh->K * GEMV_CHUNK, sh->wbits == 4u);
            const uint32_t nch = sh->N / GEMV_CHUNK;
#if DSV2_DUAL_LOAD
            const uint32_t xb = XBYTES(cbytes), off = cbytes - xb;
            uint32_t busy = 0;
            snax_xdma_read_arm((uint32_t)(bbuf[0] + off), (uint32_t)(sh->w + off), xb / DSV2_BEAT);
            for (uint32_t j = 0; j < nch; j++) {
                SPIN(XREQ, cbase + j + 1u);
                const uint32_t x0 = snrt_mcycle();
                if (j)
                    snax_xdma_read_retask((uint32_t)(bbuf[j & 1u] + off),
                                          (uint32_t)(sh->w + j * cbytes + off), xb / DSV2_BEAT);
                int to = 0;
                const snax_xdma_task_t tk = snax_xdma_start_bounded(XSPIN, &to);
                if (!to) to = snax_xdma_wait_bounded(tk, XSPIN);
                TMO += (uint32_t)to;
                XLOADED = cbase + j + 1u;
                const uint32_t x1 = snrt_mcycle();
                busy += x1 - x0;
                dsv2_span(tr, DSV2_TR_XDMA, sh->name, x0, x1);
            }
            X_BUSY(s) = busy;
#else
            (void)cbytes;
#endif
            cbase += nch;
            snrt_cluster_hw_barrier();
        }
        return 0;
    }

    // ============================== the SIMD: dequantise; the checks
    uint32_t fails = 0, hung = 0, yoff = 0;
    printf(TAG " %u shapes, %u token%s, weights streamed in %u-column chunks; %s; TCDM "
               "arbitration: %s (control 0x%x)\n",
           GEMV_NSHAPES, NTOK, NTOK > 1 ? "s" : "", GEMV_CHUNK,
           DSV2_STAGE_CHECKS ? "y and yd checked after every shape"
           : PER_SHAPE       ? "yd checked after every shape"
                             : "every yd checked once all have run",
           snax_tcdm_policy_name(DSV2_TCDM_PRIO), snax_tcdm_arb_ctrl_read());
    for (uint32_t s = 0; s < GEMV_NSHAPES; s++) {
        const gemv_shape_t *sh = &gemv_shapes[s];
        const uint32_t N = sh->N;
        uint16_t *y = ydall + (PER_SHAPE ? 0u : yoff);  // token t YP(N) bytes after token t-1
        // token 0's task, programmed while the GEMV runs; in place, s the same for every token
        dsv2_dequant_prep(y, sbuf, y, N);
        SPIN(GDONE, s + 1u);
#if DSV2_STAGE_CHECKS
        // y, before the dequantisation overwrites it
        SPIN(GOLD, s + 1u);
        uint32_t c0 = snrt_mcycle();
        uint32_t ninf;
        fails += dsv2_check(TAG, !cmp_tok("y", y, gold + NTOK * N, N, &ninf), sh->name,
                            "y = RNE(x.W * 2^-k), bit-exact");
        fails += dsv2_check(TAG, !ninf, sh->name, "no Inf at the chosen k");
        dsv2_span(tr, DSV2_TR_SIMD, "check (core)", c0, snrt_mcycle());
#endif
        {
            const uint32_t t0 = snrt_mcycle();
            snax_simd_fire();
            for (uint32_t t = 1; t < NTOK; t++) {
                uint16_t *yt = (uint16_t *)((uint8_t *)y + t * YP(N));
                dsv2_dequant_prep(yt, sbuf, yt, N);
                snax_simd_fire();
            }
            hung += snax_simd_wait_all_checked("dequant", SIMD_WAIT_BUDGET);
            T_DQ(s) = snrt_mcycle() - t0;
            dsv2_span(tr, DSV2_TR_SIMD, "G3 dequant", t0, snrt_mcycle());
        }
#if PER_SHAPE
        SPIN(GOLD, s + 1u);
        {
            const uint32_t c1 = snrt_mcycle();
            shape_report(sy, s);
            fails += check_yd(s, y, gold);
#if GEMV_NEG_SHAPE >= 0 && DSV2_STAGE_CHECKS
            if ((int)s == GEMV_NEG_SHAPE) fails += check_control(y0buf, gold + 2u * NTOK * N);
#endif
            dsv2_span(tr, DSV2_TR_SIMD, "check (core)", c1, snrt_mcycle());
        }
#endif
        yoff += N;
        snrt_cluster_hw_barrier();
    }
#if !PER_SHAPE
    // the final results, once every shape has run
    SPIN(GOLD, GEMV_NSHAPES);
    yoff = 0;
    for (uint32_t s = 0; s < GEMV_NSHAPES; s++) {
        shape_report(sy, s);
        fails += check_yd(s, ydall + yoff, gold + yoff);
        yoff += gemv_shapes[s].N;
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
