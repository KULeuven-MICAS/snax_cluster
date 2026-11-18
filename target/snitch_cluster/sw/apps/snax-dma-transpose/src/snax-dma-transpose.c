// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// The DMA alone transposes an N x N matrix A into T, one 2-D transfer per
// column of A:
//
//   snrt_dma_start_2d(T + p * k, A + b * k, b, b, p_A, N)
//
// copies column k of A (stride p_A, its row pitch) to row k of T (stride b,
// the element's bytes) as N bursts of b bytes, one per element. Each case
// runs twice and keeps the second run, so instruction cache refills stay out
// of the numbers. Core 0 prints one line per case and returns the elements
// of T that differ from A's transpose.

#include "snrt.h"

#include "data.h"

// One DMA beat (512 bit).
#define BEAT 64

// The largest row pitch (int64, 64 elements) times N: the L1 bytes of A and
// of T each.
#define MAX_PITCH (8 * N)
#define MAX_BYTES (MAX_PITCH * N)

// Counters 0 to NB_CNT-1, in the order they are printed.
#define NB_CNT 6
static const enum snrt_perf_cnt_type cnt_type[NB_CNT] = {
    SNRT_PERF_CNT_DMA_BUSY,      // cycles the iDMA's busy_o is high
    SNRT_PERF_CNT_DMA_R_DONE,    // read beats
    SNRT_PERF_CNT_DMA_W_DONE,    // write beats
    SNRT_PERF_CNT_DMA_AR_STALL,  // read requests waiting for the memory
    SNRT_PERF_CNT_DMA_W_STALL,   // memory takes writes, no data there
    SNRT_PERF_CNT_DMA_R_STALL,   // read data waiting on a full buffer
};

typedef struct {
    const char *name;
    uint32_t elem;     // b: bytes per element, 1 (int8) or 8 (int64)
    uint32_t cols;     // columns of A transposed: 1 or N
    uint32_t pitch;    // the row pitch of A and T in L1, in bytes
    uint32_t from_l2;  // read A from data.h's array (pitch N * b) in DRAM
} case_t;

static const case_t cases[] = {
    {"i8 col l1l1 p64", 1, 1, 64, 0},  // one column: 64 bursts
    {"i8 l1l1 p64", 1, N, 64, 0},      // contiguous rows
    {"i8 l1l1 p72", 1, N, 72, 0},      // padded_rows: 9 words a row
    {"i64 l1l1 p512", 8, N, 512, 0},   // int64, 8-byte bursts
    {"i8 l2l1 p64", 1, N, 64, 1},      // from DRAM, for the record
};
#define NB_CASES (sizeof(cases) / sizeof(cases[0]))

typedef struct {
    uint32_t issue;  // mcycle from before the first start to after the last
    uint32_t done;   // mcycle from before the first start to after the wait
    uint32_t cnt[NB_CNT];
    uint32_t errors;
} result_t;

static inline uintptr_t align_up(uintptr_t x, uintptr_t a) {
    return (x + a - 1) & ~(a - 1);
}

static void clear(uint8_t *dst, uint32_t bytes) {
    uint32_t *d = (uint32_t *)dst;
    for (uint32_t i = 0; i < bytes / 4; i++) d[i] = 0;
    // The DMA must not overtake these stores.
    asm volatile("fence" ::: "memory");
}

// A from DRAM into L1 at the row pitch `pitch` (not measured).
static void load(uint8_t *a, const uint8_t *src, uint32_t row, uint32_t pitch) {
    if (pitch == row)
        snrt_dma_start_1d(a, src, N * row);
    else
        snrt_dma_start_2d(a, src, row, pitch, row, N);
    snrt_dma_wait_all();
}

// The elements of T's first `cols` rows that differ from A's columns, both
// in L1 at the row pitch `pitch`.
static uint32_t check(const uint8_t *t, const uint8_t *a, uint32_t elem,
                      uint32_t cols, uint32_t pitch) {
    uint32_t errors = 0;
    for (uint32_t k = 0; k < cols; k++) {
        for (uint32_t r = 0; r < N; r++) {
            const uint8_t *x = t + k * pitch + r * elem;  // T[k][r]
            const uint8_t *y = a + r * pitch + k * elem;  // A[r][k]
            uint32_t bad = 0;
            for (uint32_t i = 0; i < elem; i++) bad |= x[i] != y[i];
            errors += bad;
        }
    }
    return errors;
}

static void run_case(const case_t *c, const uint8_t *l2, uint8_t *a, uint8_t *t,
                     result_t *r, int checked) {
    // Read the case before the first mark: `cases` is in DRAM, and
    // snrt_mcycle's memory clobber would make the compiler reload it inside
    // the measured window.
    const uint32_t elem = c->elem;
    const uint32_t cols = c->cols;
    const uint32_t pitch = c->pitch;
    const uint8_t *src = c->from_l2 ? l2 : a;
    const uint32_t src_pitch = c->from_l2 ? N * elem : pitch;

    clear(t, N * pitch);

    for (int i = 0; i < NB_CNT; i++) snrt_reset_perf_counter(i);
    for (int i = 0; i < NB_CNT; i++)
        snrt_start_perf_counter(i, cnt_type[i], snrt_hartid());
    // The enables are posted stores: read one back so they have landed
    // before the DMA turns busy. The `mv` uses the loaded value, so the core
    // waits for the load's response (a load into x0 would not stall it).
    uint32_t en = snrt_perf_counters()->enable[0].value;
    asm volatile("mv %0, %0" : "+r"(en));

    uint32_t t0 = snrt_mcycle();
    for (uint32_t k = 0; k < cols; k++)  // column k of A -> row k of T
        snrt_dma_start_2d(t + k * pitch, src + k * elem, elem, elem, src_pitch,
                          N);
    uint32_t t1 = snrt_mcycle();
    snrt_dma_wait_all();
    uint32_t t2 = snrt_mcycle();

    for (int i = 0; i < NB_CNT; i++) snrt_stop_perf_counter(i);

    r->issue = t1 - t0;
    r->done = t2 - t0;
    for (int i = 0; i < NB_CNT; i++) r->cnt[i] = snrt_get_perf_counter(i);
    r->errors = checked ? check(t, a, elem, cols, pitch) : 0;
}

int main() {
    // Every core finds the same L1 buffers and result table: A, T, results.
    uint8_t *a = (uint8_t *)align_up((uintptr_t)snrt_l1_next(), BEAT);
    uint8_t *t = a + MAX_BYTES;
    result_t *results = (result_t *)(t + MAX_BYTES);

    if (snrt_is_dm_core()) {
        for (uint32_t i = 0; i < NB_CASES; i++) {
            const case_t *c = &cases[i];
            const uint8_t *l2 = c->elem == 1 ? a8_l2 : (const uint8_t *)a64_l2;
            // A in L1 also for the case from DRAM: the check reads it there.
            load(a, l2, N * c->elem, c->pitch);
            // The first run fills the instruction cache; the second counts.
            run_case(c, l2, a, t, &results[i], 0);
            run_case(c, l2, a, t, &results[i], 1);
        }
    }

    snrt_cluster_hw_barrier();

    if (snrt_cluster_core_idx() == 0) {
        uint32_t errors = 0;
        for (uint32_t i = 0; i < NB_CASES; i++) {
            const case_t *c = &cases[i];
            const result_t *r = &results[i];
            const uint32_t bursts = c->cols * N;
            const uint32_t per100 = r->cnt[0] * 100 / bursts;
            printf(
                "dma-transpose %s: bursts %u issue %u done %u busy %u r %u "
                "w %u ar_stall %u w_stall %u r_stall %u busy/burst %u.%02u "
                "err %u\n",
                c->name, bursts, r->issue, r->done, r->cnt[0], r->cnt[1],
                r->cnt[2], r->cnt[3], r->cnt[4], r->cnt[5], per100 / 100,
                per100 % 100, r->errors);
            errors += r->errors;
        }
        printf("Number of errors: %u\n", errors);
        return errors;
    }

    return 0;
}
