// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Engine activity for a Gantt chart, and the utilisation that goes with it.
//
// SPANS. Each hart appends spans -- (label, start, end), in mcycle -- to its own track in L1,
// so no two harts write the same words. A span is one unit of an engine's work: a GEMM phase
// or task, a load, a batch of SIMD tasks, a stretch of core code. Its label is a string
// literal that starts with the layer stage it serves (docs/dsv2_layer1_plan.md, section 1):
// "2 W_Q", "9 softmax". After the run one hart prints every track,
//
//     [SPAN] <track> <start> <end> <label>
//
// tracks 0 GEMM, 1 SIMD, 2 xDMA hart, 3 iDMA; times are raw mcycle, which every hart shares.
//
// UTILISATION (dsv2_util_print), one line over the same window:
//
//     [UTIL] cycles <n> gemm <req> <denied> simd <req> <denied> xdma <req> <denied>
//            snitch <req> <denied> banks <bank-cycles served> wide <DMA beats> simd_busy <n>
//
// the cluster census (snax-perf-census.h, pass 0: TCDM requests per engine, the banks served,
// the wide DMA's 64-byte beats) and the SIMD datapath's busy-cycle counter.
//
// A trace lives in a zeroed arena; dsv2_trace_clear empties it between the blocks of one run.

#pragma once

#include "snax-perf-census.h"

#define DSV2_TR_GEMM 0u
#define DSV2_TR_SIMD 1u
#define DSV2_TR_XDMA 2u
#define DSV2_TR_IDMA 3u

// spans per track (the MLA's iDMA makes one per load, about 230 at L = 511 and 250 with four
// tokens, whose attention loads every key tile twice; with DSV2_DUAL_LOAD its xDMA hart makes one
// per streamed chunk, 178), and where each track starts in the span words. An app that records
// more spans on a track defines its DSV2_TR_CAP_* before including this.
#ifndef DSV2_TR_CAP_GEMM
#define DSV2_TR_CAP_GEMM 128u
#endif
#ifndef DSV2_TR_CAP_SIMD
#define DSV2_TR_CAP_SIMD 128u
#endif
#ifndef DSV2_TR_CAP_XDMA
#if defined(DSV2_DUAL_LOAD) && DSV2_DUAL_LOAD
#define DSV2_TR_CAP_XDMA 192u
#else
#define DSV2_TR_CAP_XDMA 32u
#endif
#endif
#ifndef DSV2_TR_CAP_IDMA
#define DSV2_TR_CAP_IDMA 256u
#endif
#define DSV2_TR_CAP(k)                                                   \
    ((k) == 0u   ? DSV2_TR_CAP_GEMM                                      \
     : (k) == 1u ? DSV2_TR_CAP_SIMD                                      \
     : (k) == 2u ? DSV2_TR_CAP_XDMA                                      \
                 : DSV2_TR_CAP_IDMA)
#define DSV2_TR_BASE(k)                                                  \
    ((k) == 0u   ? 0u                                                    \
     : (k) == 1u ? 3u * DSV2_TR_CAP_GEMM                                 \
     : (k) == 2u ? 3u * (DSV2_TR_CAP_GEMM + DSV2_TR_CAP_SIMD)            \
                 : 3u * (DSV2_TR_CAP_GEMM + DSV2_TR_CAP_SIMD + DSV2_TR_CAP_XDMA))
#define DSV2_TR_WORDS \
    (3u * (DSV2_TR_CAP_GEMM + DSV2_TR_CAP_SIMD + DSV2_TR_CAP_XDMA + DSV2_TR_CAP_IDMA))

typedef struct {
    uint32_t n[16];  // spans per track (64 bytes, so the spans start on a beat)
    uint32_t s[DSV2_TR_WORDS];
} dsv2_trace_t;

static inline void dsv2_span(dsv2_trace_t *tr, uint32_t track, const char *label, uint32_t t0,
                             uint32_t t1) {
    if (!tr) return;
    const uint32_t i = ((volatile uint32_t *)tr->n)[track];
    if (i >= DSV2_TR_CAP(track)) return;
    volatile uint32_t *w = &tr->s[DSV2_TR_BASE(track) + 3u * i];
    w[0] = (uint32_t)label;
    w[1] = t0;
    w[2] = t1;
    ((volatile uint32_t *)tr->n)[track] = i + 1u;
}

// A span line goes straight to putchar, not through printf. printf reads its format string out
// of main memory a character at a time and turns each digit into two divisions by a run-time
// base, which this core's iterative divider takes about 35 cycles each: some 2,600 cycles a
// line, against a few hundred here. The prefix is immediates; only the label is read from
// main memory.
extern void snrt_putchar(char character);

static inline void dsv2_put_u32(uint32_t v) {
    char d[10];
    uint32_t n = 0;
    do {
        d[n++] = (char)('0' + v % 10u);  // a constant divisor: a multiply, not a division
        v /= 10u;
    } while (v);
    while (n) snrt_putchar(d[--n]);
}

static inline void dsv2_put_span(uint32_t k, uint32_t t0, uint32_t t1, const char *label) {
    snrt_putchar('[');
    snrt_putchar('S');
    snrt_putchar('P');
    snrt_putchar('A');
    snrt_putchar('N');
    snrt_putchar(']');
    snrt_putchar(' ');
    dsv2_put_u32(k);
    snrt_putchar(' ');
    dsv2_put_u32(t0);
    snrt_putchar(' ');
    dsv2_put_u32(t1);
    snrt_putchar(' ');
    while (*label) snrt_putchar(*label++);
    snrt_putchar('\n');
}

// Print every track, one `[SPAN] <track> <start> <end> <label>` line a span. Call once every
// hart has written its last span of the window.
static void dsv2_trace_print(const dsv2_trace_t *tr) {
    if (!tr) return;
    for (uint32_t k = 0; k < 4u; k++) {
        const uint32_t n = ((const volatile uint32_t *)tr->n)[k];
        const volatile uint32_t *w = &tr->s[DSV2_TR_BASE(k)];
        for (uint32_t i = 0; i < n; i++)
            dsv2_put_span(k, w[3u * i + 1u], w[3u * i + 2u], (const char *)w[3u * i]);
        if (n >= DSV2_TR_CAP(k)) printf("[SPAN] %u overflow\n", k);
    }
}

static inline void dsv2_trace_clear(dsv2_trace_t *tr) {
    if (!tr) return;
    for (uint32_t k = 0; k < 4u; k++) ((volatile uint32_t *)tr->n)[k] = 0u;
}

static void dsv2_util_print(const snax_perf_snapshot_t *c, uint32_t simd_busy) {
    printf("[UTIL] cycles %u gemm %u %u simd %u %u xdma %u %u snitch %u %u banks %u wide %u "
           "simd_busy %u\n",
           c->v[SNAX_PERF_C_CYCLE], c->v[SNAX_PERF_C_GEMM_REQ], c->v[SNAX_PERF_C_GEMM_STALL],
           c->v[SNAX_PERF_C_SIMD_REQ], c->v[SNAX_PERF_C_SIMD_STALL], c->v[SNAX_PERF_C_XDMA_REQ],
           c->v[SNAX_PERF_C_XDMA_STALL], c->v[SNAX_PERF_C_SNITCH_REQ],
           c->v[SNAX_PERF_C_SNITCH_STALL], c->v[SNAX_PERF_C_BANK_SERVED],
           c->v[SNAX_PERF_C_WIDE_REQ], simd_busy);
}
