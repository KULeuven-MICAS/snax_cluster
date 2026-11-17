// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// mvt on the Snitch cores: x1 += A y1 and x2 += A^T y2 in place, each product
// and sum at acc_t. The x come in by DMA, are updated in L1 and go back out.
// One run section holds both halves, which share no output, so a worker goes
// from the first to the second with no barrier:
//
// - x1: worker c takes the rows c, c + P, ..., each from sf_stagger's column
//   round to it, as matvec.
// - x2: worker c takes A's 64-bit column words c, c + P, ... (SF_PER_WORD
//   columns each, summed side by side) and walks down them from row 4c round
//   to it, so the workers read A and y2 in different banks.

#include "snitch_forge.h"

int main() {
    sf_t sf = sf_init();
    in_t *a = (in_t *)sf_l1(&sf, sizeof(A));
    in_t *y1l = (in_t *)sf_l1(&sf, sizeof(y1_l2));
    in_t *y2l = (in_t *)sf_l1(&sf, sizeof(y2));
    acc_t *x1l = (acc_t *)sf_l1(&sf, sizeof(x1));
    acc_t *x2l = (acc_t *)sf_l1(&sf, sizeof(x2));

    sf_section(&sf, SF_DMA_IN);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(a, A, sizeof(A));
        snrt_dma_start_1d(y1l, y1_l2, sizeof(y1_l2));
        snrt_dma_start_1d(y2l, y2, sizeof(y2));
        snrt_dma_start_1d(x1l, x1, sizeof(x1));
        snrt_dma_start_1d(x2l, x2, sizeof(x2));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_RUN);
    if (sf_is_worker(&sf)) {
        uint32_t t0 = sf_tic();
        // x1 += A y1, by rows
        uint32_t start = sf_stagger(&sf, SF_N);
        for (uint32_t r = sf.core; r < SF_M; r += sf.workers) {
            const in_t *row = a + r * SF_N;
            acc_t s = x1l[r];
            for (uint32_t j = start; j < SF_N; j++)
                s += (acc_t)row[j] * (acc_t)y1l[j];
            for (uint32_t j = 0; j < start; j++)
                s += (acc_t)row[j] * (acc_t)y1l[j];
            x1l[r] = s;
        }
        // x2 += A^T y2, by column words
        uint32_t i0 = (sf.core * 4) % SF_M;
        for (uint32_t w = sf.core; w < SF_N / SF_PER_WORD; w += sf.workers) {
            const uint32_t j0 = w * SF_PER_WORD;
            acc_t s[SF_PER_WORD];
            for (uint32_t e = 0; e < SF_PER_WORD; e++) s[e] = x2l[j0 + e];
            for (uint32_t i = i0; i < SF_M; i++) {
                const in_t *aw = a + i * SF_N + j0;
                const acc_t yi = (acc_t)y2l[i];
                for (uint32_t e = 0; e < SF_PER_WORD; e++)
                    s[e] += (acc_t)aw[e] * yi;
            }
            for (uint32_t i = 0; i < i0; i++) {
                const in_t *aw = a + i * SF_N + j0;
                const acc_t yi = (acc_t)y2l[i];
                for (uint32_t e = 0; e < SF_PER_WORD; e++)
                    s[e] += (acc_t)aw[e] * yi;
            }
            for (uint32_t e = 0; e < SF_PER_WORD; e++) x2l[j0 + e] = s[e];
        }
        sf_toc(&sf, t0);
    }

    sf_section(&sf, SF_DMA_OUT);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(x1, x1l, sizeof(x1));
        snrt_dma_start_1d(x2, x2l, sizeof(x2));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_CHECK);
    if (sf.core != 0) return 0;
    int errors = sf_check(x1, x1_golden, SF_M) + sf_check(x2, x2_golden, SF_N);
    sf_report(&sf, "mvt", errors);
    return errors;
}
