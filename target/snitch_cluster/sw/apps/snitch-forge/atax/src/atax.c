// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// atax on the Snitch cores: y = A^T (A x), each product and sum at acc_t. The
// run section has two halves with a barrier of every core between them, as
// every column of the second needs all of the first's output:
//
// - t = A x into L1: worker c takes the rows c, c + P, ..., each from
//   sf_stagger's column round to it, as matvec.
// - y = A^T t: worker c takes A's 64-bit column words c, c + P, ...
//   (SF_PER_WORD columns each, summed side by side) and walks down them from
//   row 4c round to it, as mvt's second half.
//
// A worker's compute cycles hold both halves and its wait at the barrier.

#include "snitch_forge.h"

int main() {
    sf_t sf = sf_init();
    in_t *a = (in_t *)sf_l1(&sf, sizeof(A));
    in_t *xl = (in_t *)sf_l1(&sf, sizeof(x));
    acc_t *tl = (acc_t *)sf_l1(&sf, SF_M * sizeof(acc_t));
    acc_t *yl = (acc_t *)sf_l1(&sf, sizeof(y));

    sf_section(&sf, SF_DMA_IN);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(a, A, sizeof(A));
        snrt_dma_start_1d(xl, x, sizeof(x));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_RUN);
    uint32_t t0 = sf_tic();
    if (sf_is_worker(&sf)) {
        // t = A x, by rows
        uint32_t start = sf_stagger(&sf, SF_N);
        for (uint32_t i = sf.core; i < SF_M; i += sf.workers) {
            const in_t *row = a + i * SF_N;
            acc_t t = 0;
            for (uint32_t j = start; j < SF_N; j++)
                t += (acc_t)row[j] * (acc_t)xl[j];
            for (uint32_t j = 0; j < start; j++)
                t += (acc_t)row[j] * (acc_t)xl[j];
            tl[i] = t;
        }
    }
    // every core: the second half reads all of t
    snrt_cluster_hw_barrier();
    if (sf_is_worker(&sf)) {
        // y = A^T t, by column words
        uint32_t i0 = (sf.core * 4) % SF_M;
        for (uint32_t w = sf.core; w < SF_N / SF_PER_WORD; w += sf.workers) {
            const uint32_t j0 = w * SF_PER_WORD;
            acc_t s[SF_PER_WORD] = {0};
            for (uint32_t i = i0; i < SF_M; i++) {
                const in_t *aw = a + i * SF_N + j0;
                const acc_t ti = tl[i];
                for (uint32_t e = 0; e < SF_PER_WORD; e++)
                    s[e] += (acc_t)aw[e] * ti;
            }
            for (uint32_t i = 0; i < i0; i++) {
                const in_t *aw = a + i * SF_N + j0;
                const acc_t ti = tl[i];
                for (uint32_t e = 0; e < SF_PER_WORD; e++)
                    s[e] += (acc_t)aw[e] * ti;
            }
            for (uint32_t e = 0; e < SF_PER_WORD; e++) yl[j0 + e] = s[e];
        }
        sf_toc(&sf, t0);
    }

    sf_section(&sf, SF_DMA_OUT);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(y, yl, sizeof(y));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_CHECK);
    if (sf.core != 0) return 0;
    int errors = sf_check(y, y_golden, SF_N);
    sf_report(&sf, "atax", errors);
    return errors;
}
