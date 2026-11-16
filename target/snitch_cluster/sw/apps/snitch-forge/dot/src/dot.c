// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// dot on the Snitch cores: out = the sum of A * B, both at acc_t. Each worker
// multiplies and accumulates its share in one pass into a partial in L1; after
// a barrier core 0 adds the partials.

#include "snitch_forge.h"

int main() {
    sf_t sf = sf_init();
    in_t *a = (in_t *)sf_l1(&sf, sizeof(A));
    in_t *b = (in_t *)sf_l1(&sf, sizeof(B));
    acc_t *part = (acc_t *)sf_l1(&sf, sf.workers * sizeof(acc_t));
    acc_t *o = (acc_t *)sf_l1(&sf, sizeof(out));

    sf_section(&sf, SF_DMA_IN);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(a, A, sizeof(A));
        snrt_dma_start_1d(b, B, sizeof(B));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_RUN);
    if (sf_is_worker(&sf)) {
        uint32_t t0 = sf_tic();
        uint32_t step = sf.workers * SF_PER_WORD;
        acc_t s = 0;
        for (uint32_t i = sf.core * SF_PER_WORD; i < SF_N; i += step) {
            for (uint32_t e = 0; e < SF_PER_WORD; e++) {
                s += (acc_t)a[i + e] * (acc_t)b[i + e];
            }
        }
        part[sf.core] = s;
        sf_toc(&sf, t0);
    }
    snrt_cluster_hw_barrier();
    if (sf.core == 0) {
        acc_t s = 0;
        for (uint32_t c = 0; c < sf.workers; c++) s += part[c];
        o[0] = s;
    }

    sf_section(&sf, SF_DMA_OUT);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(out, o, sizeof(out));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_CHECK);
    if (sf.core != 0) return 0;
    int errors = sf_check(out, out_golden, 1);
    sf_report(&sf, "dot", errors);
    return errors;
}
