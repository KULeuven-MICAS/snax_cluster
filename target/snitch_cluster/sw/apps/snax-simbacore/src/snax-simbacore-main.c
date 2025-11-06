// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Robin Geens <robin.geens@esat.kuleuven.be>

#include "data.h"
#include "snax-simbacore-lib.h"

int main() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr           = snrt_l1_next();
    uint16_t* local_oscore_in     = (uint16_t*)(tcdm_base_ptr + delta_oscore_in);
    uint16_t* local_oscore_weight = (uint16_t*)(tcdm_base_ptr + delta_oscore_weight);
    uint16_t* local_conv_weight   = (uint16_t*)(tcdm_base_ptr + delta_conv_weight);
    uint16_t* local_conv_bias     = (uint16_t*)(tcdm_base_ptr + delta_conv_bias);
    uint16_t* local_conv_out      = (uint16_t*)(tcdm_base_ptr + delta_conv_out);
    uint16_t* local_iscore_weight = (uint16_t*)(tcdm_base_ptr + delta_iscore_weight);
    uint16_t* local_iscore_out    = (uint16_t*)(tcdm_base_ptr + delta_iscore_out);  // holds the psums

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_oscore_in, oscore_in, length_oscore_in);
        snrt_dma_start_1d(local_oscore_weight, oscore_weight, length_oscore_weight);
        snrt_dma_start_1d(local_conv_weight, conv_weight, length_conv_weight);
        snrt_dma_start_1d(local_conv_bias, conv_bias, length_conv_bias);
        snrt_dma_start_1d(local_iscore_weight, iscore_weight, length_iscore_weight);
        // Input and output psums use the same address
        snrt_dma_start_1d(local_iscore_out, iscore_bias, length_iscore_bias);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("Setting up Streamer and SimbaCore...\n");

        set_streamer_csr((uint32_t)local_oscore_in, R0slstride, R0tlbound, R0tlstride, channel_en,         //
                         (uint32_t)local_oscore_weight, R1slstride, R1tlbound, R1tlstride, channel_en,     //
                         (uint32_t)local_conv_weight, R3slstride, R3tlbound, R3tlstride, channel_en,       //
                         (uint32_t)local_conv_bias, R4slstride, R4tlbound, R4tlstride, channel_en,         //
                         (uint32_t)local_iscore_weight, R12slstride, R12tlbound, R12tlstride, channel_en,  //
                         (uint32_t)local_iscore_out, R13slstride, R13tlbound, R13tlstride, channel_en,     // psums
                         (uint32_t)local_conv_out, W1slstride, W1tlbound, W1tlstride, channel_en,          //
                         (uint32_t)local_iscore_out, W3slstride, W3tlbound, W3tlstride, channel_en         //
        );

        set_simbacore_csr(mode, seqLen, dModel, dInner, 1);
        set_simbacore_streamer_start();
        set_simbacore_start();

        // Poll until streamer and accelerator finish
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err +=
            check_result_sample(local_conv_out, conv_out, test_sample_indices_conv_out, test_sample_count, "conv_out");
        // err += check_result_all(local_conv_out, conv_out, length_conv_out);
        check_result_sample(local_iscore_out, iscore_out, test_sample_indices_iscore_out, test_sample_count,
                            "iscore_out");

        printf("Test SimbaCore: seqLen=%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               test_sample_count);
    }
    return err;
}

// This test only test on the output stationary dataflow
// TODO the usage of naming is not scalable for multiple tests
// TODO the programming should be the same anyways (only for this case )
int test_osgemm() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr = snrt_l1_next();
    uint16_t* local_a   = (uint16_t*)(tcdm_base_ptr + delta_local_a);
    uint16_t* local_b   = (uint16_t*)(tcdm_base_ptr + delta_local_b);
    uint16_t* local_c   = (uint16_t*)(tcdm_base_ptr + delta_local_c);
    uint16_t* local_d   = (uint16_t*)(tcdm_base_ptr + delta_local_d);

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

        set_simbacore_osgemm_streamer_csr((uint32_t)local_a, Aslstride, Atlbound, Atlstride, channel_en,   // A
                                          (uint32_t)local_b, Bslstride, Btlbound, Btlstride, channel_en,   // B
                                          (uint32_t)local_d, Dslstride, Dtlbound, Dtlstride, channel_en);  // D

        set_simbacore_csr(mode, seqLen, dModel, dInner, 1);
        set_simbacore_streamer_start();
        set_simbacore_start();

        // Poll until streamer and accelerator finish
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(local_d, D, test_sample_indices_D, test_sample_count, "out");
        // err += check_OSGeMM_result_all(local_d, D, data_length_d);

        printf("Test SimbaCore: seqLen%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               test_sample_count);
    }
    return err;
}
