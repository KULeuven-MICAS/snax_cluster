// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Robin Geens <robin.geens@esat.kuleuven.be>

#include "data.h"
#include "snax-simbacore-lib.h"

// This test only test on the output stationary dataflow
int main() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr = snrt_l1_next();
    uint16_t* local_a = (uint16_t*)(tcdm_base_ptr + delta_local_a);
    uint16_t* local_b = (uint16_t*)(tcdm_base_ptr + delta_local_b);
    uint16_t* local_c = (uint16_t*)(tcdm_base_ptr + delta_local_c);
    uint16_t* local_d = (uint16_t*)(tcdm_base_ptr + delta_local_d);

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, data_length_a);
        snrt_dma_start_1d(local_b, B, data_length_b);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("Setting up Streamer and SimbaCore...\n");

        set_simbacore_osgemm_streamer_csr((uint32_t)local_a, Aslstride, Atlbound, Atlstride, channel_en_A,   // A
                                          (uint32_t)local_b, Bslstride, Btlbound, Btlstride, channel_en_B,   // B
                                          (uint32_t)local_d, Dslstride, Dtlbound, Dtlstride, channel_en_D);  // D

        set_simbacore_csr(mode, seqLen, dModel, dInner, 1);
        set_simbacore_streamer_start();
        set_simbacore_start();

        // Poll until streamer and accelerator finish
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_OSGeMM_result_sample(local_d, D, test_sample_indices, test_sample_count);
        // err += check_OSGeMM_result_all(local_d, D, data_length_d);

        printf("Test SimbaCore: M = %d, K = %d, N = %d. %s: %u/%d errors.\n", M, K, N, err ? "FAIL" : "PASS", err,
               test_sample_count);
    }
    return err;
}
