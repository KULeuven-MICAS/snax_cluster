// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// The DMA alone copies A (M x N, row by row) into L1's A_T = A^T (N x M), as
// snax-forge's transpose_A_T does (D182): one 2-D transfer per column of A,
//
//   snrt_dma_start_2d(at + M * b * k, a + b * k, b, b, N * b, M)
//
// moving column k of A (stride N * b, its row pitch) to row k of A_T (stride
// b, the element's bytes) as M bursts of b bytes, one per element. The run
// section is the copy alone, from its first start to the DMA's wait, so its
// window is the copy's energy on the gate-level cluster; A comes in and A_T
// goes out in the DMA sections around it. Run on the one-core cluster, whose
// core is the DMA core and the worker; its compute line is the copy's cycles.

#include "snitch_forge.h"

_Static_assert(sizeof(in_t) == sizeof(acc_t),
               "a copy moves in_t: TYPE int8 or int64");

int main() {
    sf_t sf = sf_init();
    in_t *a = (in_t *)sf_l1(&sf, sizeof(A));
    acc_t *at = (acc_t *)sf_l1(&sf, sizeof(A_T));

    sf_section(&sf, SF_DMA_IN);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(a, A, sizeof(A));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_RUN);
    if (snrt_is_dm_core()) {
        uint32_t t0 = sf_tic();
        const uint32_t b = sizeof(in_t);
        for (uint32_t k = 0; k < SF_N; k++)  // column k of A -> row k of A_T
            snrt_dma_start_2d((uint8_t *)at + SF_M * b * k,
                              (uint8_t *)a + b * k, b, b, SF_N * b, SF_M);
        snrt_dma_wait_all();
        if (sf_is_worker(&sf)) sf_toc(&sf, t0);
    }

    sf_section(&sf, SF_DMA_OUT);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(A_T, at, sizeof(A_T));
        snrt_dma_wait_all();
    }

    sf_section(&sf, SF_CHECK);
    if (sf.core != 0) return 0;
    int errors = sf_check(A_T, A_T_golden, SF_N * SF_M);
    sf_report(&sf, "dma_transpose", errors);
    return errors;
}
