// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// relu on the Snitch cores: B = max(A, 0) at acc_t.

#include "snitch_forge.h"

int main() {
    sf_t sf = sf_init();
    in_t *a = (in_t *)sf_l1(&sf, sizeof(A));
    acc_t *b = (acc_t *)sf_l1(&sf, sizeof(B));

    sf_section(&sf, SF_DMA_IN);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(a, A, sizeof(A));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_RUN);
    if (sf_is_worker(&sf)) {
        uint32_t t0 = sf_tic();
        uint32_t step = sf.workers * SF_PER_WORD;
        for (uint32_t i = sf.core * SF_PER_WORD; i < SF_N; i += step) {
            for (uint32_t e = 0; e < SF_PER_WORD; e++) {
                acc_t v = (acc_t)a[i + e];
                b[i + e] = v > 0 ? v : 0;
            }
        }
        sf_toc(&sf, t0);
    }

    sf_section(&sf, SF_DMA_OUT);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(B, b, sizeof(B));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_CHECK);
    if (sf.core != 0) return 0;
    int errors = sf_check(B, B_golden, SF_N);
    sf_report(&sf, "relu", errors);
    return errors;
}
