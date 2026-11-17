// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// gesummv on the Snitch cores: out = alpha (A x) + beta (B x), each product
// and sum at acc_t. Worker c takes the rows c, c + P, ..., each from
// sf_stagger's column round to it, as matvec, and sums a row of A and a row of
// B against x in one pass. alpha and beta are read at run time, as forge's
// accelerators read them from a CSR, so their products are real multiplies.

#include "snitch_forge.h"

int main() {
    sf_t sf = sf_init();
    in_t *al = (in_t *)sf_l1(&sf, sizeof(A));
    in_t *bl = (in_t *)sf_l1(&sf, sizeof(B));
    in_t *xl = (in_t *)sf_l1(&sf, sizeof(x));
    acc_t *ol = (acc_t *)sf_l1(&sf, sizeof(out));

    sf_section(&sf, SF_DMA_IN);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(al, A, sizeof(A));
        snrt_dma_start_1d(bl, B, sizeof(B));
        snrt_dma_start_1d(xl, x, sizeof(x));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_RUN);
    if (sf_is_worker(&sf)) {
        uint32_t t0 = sf_tic();
        const acc_t a_s = alpha, b_s = beta;
        uint32_t start = sf_stagger(&sf, SF_N);
        for (uint32_t i = sf.core; i < SF_N; i += sf.workers) {
            const in_t *ra = al + i * SF_N;
            const in_t *rb = bl + i * SF_N;
            acc_t ta = 0, tb = 0;
            for (uint32_t j = start; j < SF_N; j++) {
                const acc_t xj = (acc_t)xl[j];
                ta += (acc_t)ra[j] * xj;
                tb += (acc_t)rb[j] * xj;
            }
            for (uint32_t j = 0; j < start; j++) {
                const acc_t xj = (acc_t)xl[j];
                ta += (acc_t)ra[j] * xj;
                tb += (acc_t)rb[j] * xj;
            }
            ol[i] = a_s * ta + b_s * tb;
        }
        sf_toc(&sf, t0);
    }

    sf_section(&sf, SF_DMA_OUT);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(out, ol, sizeof(out));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_CHECK);
    if (sf.core != 0) return 0;
    int errors = sf_check(out, out_golden, SF_N);
    sf_report(&sf, "gesummv", errors);
    return errors;
}
