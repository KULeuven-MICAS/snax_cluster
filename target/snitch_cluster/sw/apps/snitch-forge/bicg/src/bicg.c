// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// bicg on the Snitch cores: q = A p and s = A^T r, each product and sum at
// acc_t. One run section holds both halves, which share no output, so a worker
// goes from the first to the second with no barrier:
//
// - q: worker c takes the rows c, c + P, ..., each from sf_stagger's column
//   round to it, as matvec.
// - s: worker c takes A's 64-bit column words c, c + P, ... (SF_PER_WORD
//   columns each, summed side by side) and walks down them from row 4c round
//   to it, as mvt's second half, so the workers read A and r in different
//   banks.

#include "snitch_forge.h"

int main() {
    sf_t sf = sf_init();
    in_t *a = (in_t *)sf_l1(&sf, sizeof(A));
    in_t *pl = (in_t *)sf_l1(&sf, sizeof(p));
    in_t *rl = (in_t *)sf_l1(&sf, sizeof(r));
    acc_t *ql = (acc_t *)sf_l1(&sf, sizeof(q));
    acc_t *sl = (acc_t *)sf_l1(&sf, sizeof(s));

    sf_section(&sf, SF_DMA_IN);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(a, A, sizeof(A));
        snrt_dma_start_1d(pl, p, sizeof(p));
        snrt_dma_start_1d(rl, r, sizeof(r));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_RUN);
    if (sf_is_worker(&sf)) {
        uint32_t t0 = sf_tic();
        // q = A p, by rows
        uint32_t start = sf_stagger(&sf, SF_N);
        for (uint32_t i = sf.core; i < SF_M; i += sf.workers) {
            const in_t *row = a + i * SF_N;
            acc_t t = 0;
            for (uint32_t j = start; j < SF_N; j++)
                t += (acc_t)row[j] * (acc_t)pl[j];
            for (uint32_t j = 0; j < start; j++)
                t += (acc_t)row[j] * (acc_t)pl[j];
            ql[i] = t;
        }
        // s = A^T r, by column words
        uint32_t i0 = (sf.core * 4) % SF_M;
        for (uint32_t w = sf.core; w < SF_N / SF_PER_WORD; w += sf.workers) {
            const uint32_t j0 = w * SF_PER_WORD;
            acc_t t[SF_PER_WORD] = {0};
            for (uint32_t i = i0; i < SF_M; i++) {
                const in_t *aw = a + i * SF_N + j0;
                const acc_t ri = (acc_t)rl[i];
                for (uint32_t e = 0; e < SF_PER_WORD; e++)
                    t[e] += (acc_t)aw[e] * ri;
            }
            for (uint32_t i = 0; i < i0; i++) {
                const in_t *aw = a + i * SF_N + j0;
                const acc_t ri = (acc_t)rl[i];
                for (uint32_t e = 0; e < SF_PER_WORD; e++)
                    t[e] += (acc_t)aw[e] * ri;
            }
            for (uint32_t e = 0; e < SF_PER_WORD; e++) sl[j0 + e] = t[e];
        }
        sf_toc(&sf, t0);
    }

    sf_section(&sf, SF_DMA_OUT);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(q, ql, sizeof(q));
        snrt_dma_start_1d(s, sl, sizeof(s));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_CHECK);
    if (sf.core != 0) return 0;
    int errors = sf_check(q, q_golden, SF_M) + sf_check(s, s_golden, SF_N);
    sf_report(&sf, "bicg", errors);
    return errors;
}
