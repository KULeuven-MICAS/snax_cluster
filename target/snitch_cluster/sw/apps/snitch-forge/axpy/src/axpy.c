// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// axpy on the Snitch cores: y += alpha * x in place, at acc_t. y comes in by
// DMA, is updated in L1 and goes back out. alpha is read at run time, as
// forge's axpy reads it from a CSR, so the product is a real multiply.

#include "snitch_forge.h"

int main() {
    sf_t sf = sf_init();
    in_t *xl = (in_t *)sf_l1(&sf, sizeof(x));
    acc_t *yl = (acc_t *)sf_l1(&sf, sizeof(y));

    sf_section(&sf, SF_DMA_IN);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(xl, x, sizeof(x));
        snrt_dma_start_1d(yl, y, sizeof(y));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_RUN);
    if (sf_is_worker(&sf)) {
        uint32_t t0 = sf_tic();
        const acc_t al = alpha;
        uint32_t step = sf.workers * SF_PER_WORD;
        for (uint32_t i = sf.core * SF_PER_WORD; i < SF_N; i += step) {
            for (uint32_t e = 0; e < SF_PER_WORD; e++) {
                yl[i + e] += al * (acc_t)xl[i + e];
            }
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
    sf_report(&sf, "axpy", errors);
    return errors;
}
