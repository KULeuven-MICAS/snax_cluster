// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Robin Geens <robin.geens@esat.kuleuven.be>

#include "data.h"

#include "snax-simbacore-lib.h"

// This test only test on the output stationary dataflow
int main() {
    // Set err value for checking
    int err = 0;

    // Allocate space in TCDM
    uint16_t *local_a, *local_b, *local_c, *local_d;
    local_a = (uint16_t*)(snrt_l1_next() + delta_local_a);
    local_b = (uint16_t*)(snrt_l1_next() + delta_local_b);
    local_c = (uint16_t*)(snrt_l1_next() + delta_local_c);
    local_d = (uint16_t*)(snrt_l1_next() + delta_local_d);

    // printf("local_d[0]: %u @ %p\n", local_d[0], (void*)local_d);

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, data_length_a);
        snrt_dma_start_1d(local_b, B, data_length_b);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    printf("local_a[0]: %u (0x%04x) @ %p\n", local_a[0], local_a[0], (void*)local_a);

    // NOTE no C for now
    // if (snrt_is_dm_core()) {
    //     snrt_dma_start_1d(local_c, C, data_length_c);
    //     snrt_dma_wait_all();
    // }
    // snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("Setting up Streamer and SimbaCore...\n");
        // Set the CSR for the Streamer
        int32_t Aslstride[] = {Aslstride0};
        int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2, Atlbound3};
        int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2, Atlstride3};
        int32_t Bslstride[] = {Bslstride0};
        int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
        int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};

        int32_t Cslstride[] = {Cslstride0};
        int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2, Ctlbound3};
        int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};

        int32_t Dslstride[] = {Dslstride0};
        int32_t Dtlbound[] = {Dtlbound0, Dtlbound1, Dtlbound2, Dtlbound3};
        int32_t Dtlstride[] = {Dtlstride0, Dtlstride1, Dtlstride2, Dtlstride3};

        // Set Streamer configuration CSR
        set_simbacore_oscore_streamer_csr(
            (uint32_t)local_a, Aslstride, Atlbound, Atlstride, set_addr_remap_index_A, channel_en_A,

            (uint32_t)local_b, Bslstride, Btlbound, Btlstride, set_addr_remap_index_B, channel_en_B,

            (uint32_t)local_d, Dslstride, Dtlbound, Dtlstride, set_addr_remap_index_D, channel_en_D);

        // Set CSR
        set_simbacore_csr(1, M, K, N, 1);
        set_simbacore_streamer_start();
        set_simbacore_start();

        // Poll until Streamer and GEMM accelerator finish
        printf("CRSs set. Waiting for simbacore to finish...\n");
        wait_simbacore_and_streamer();

        // Result check
        printf("Simbacore finished. Checking results...\n");
        printf("local_d: %p\n", local_d);
        printf("local_d[0]: %u (0x%04x) @ %p\n", local_d[0], local_d[0], (void*)local_d);
        printf("D[0]: %u (0x%04x) @ %p\n", D[0], D[0], (void*)D);
        err += check_simbacore_result_D(local_d, D, data_length_d, false);

        printf("Test SimbaCore: M = %d, K = %d, N = %d, Error: %d.\n", M, K, N, err ? "FAIL" : "PASS", err);
    }
    return err;
}
