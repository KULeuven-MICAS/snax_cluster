// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// rowsum on the Snitch cores: y = the sum of each row of A, at acc_t. Worker c
// sums the rows c, c + P, ..., each from sf_stagger's column round to it, so
// the rows need no combine.

#include "snitch_forge.h"

int main() {
    sf_t sf = sf_init();
    in_t *a = (in_t *)sf_l1(&sf, sizeof(A));
    acc_t *yl = (acc_t *)sf_l1(&sf, sizeof(y));

    sf_section(&sf, SF_DMA_IN);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(a, A, sizeof(A));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_RUN);
    if (sf_is_worker(&sf)) {
        uint32_t t0 = sf_tic();
        uint32_t start = sf_stagger(&sf, SF_N);
        for (uint32_t r = sf.core; r < SF_M; r += sf.workers) {
            const in_t *row = a + r * SF_N;
            acc_t s = 0;
            for (uint32_t j = start; j < SF_N; j++) s += (acc_t)row[j];
            for (uint32_t j = 0; j < start; j++) s += (acc_t)row[j];
            yl[r] = s;
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
    int errors = sf_check(y, y_golden, SF_M);
    sf_report(&sf, "rowsum", errors);
    return errors;
}
