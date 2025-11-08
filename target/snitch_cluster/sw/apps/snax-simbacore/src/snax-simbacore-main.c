// Copyright 2025 KU Leuven.
// Not released under license.All rights reserved.
//
// Author : Robin Geens < robin.geens@kuleuven.be>

#include "data.h"
#include "snax-simbacore-lib.h"

int test_phase1() {
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
        snrt_dma_start_1d(local_iscore_out, iscore_bias, length_iscore_out);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("Setting up Streamer and SimbaCore for Phase1...\n");

        set_streamer_csr(

            (uint32_t)local_oscore_in, M0_R0_ss, M0_R0_tb, M0_R0_ts, M0_R0_en,          //
            (uint32_t)local_oscore_weight, M0_R1_ss, M0_R1_tb, M0_R1_ts, M0_R1_en,      //
            (uint32_t)local_conv_weight, M0_R3_ss, M0_R3_tb, M0_R3_ts, M0_R3_en,        //
            (uint32_t)local_conv_bias, M0_R4_ss, M0_R4_tb, M0_R4_ts, M0_R4_en,          //
            (uint32_t)local_iscore_weight, M0_R12_ss, M0_R12_tb, M0_R12_ts, M0_R12_en,  //
            (uint32_t)local_iscore_out, M0_R13_ss, M0_R13_tb, M0_R13_ts, M0_R13_en,     // psums
            (uint32_t)0, 0, 0, 0, 0,                                                    // disable
            (uint32_t)local_conv_out, M0_W1_ss, M0_W1_tb, M0_W1_ts, M0_W1_en,           //
            (uint32_t)local_iscore_out, M0_W3_ss, M0_W3_tb, M0_W3_ts, M0_W3_en          //
        );

        set_simbacore_csr(M0_PHASE1, seqLen, dModel, dInner, 1);
        set_simbacore_streamer_start();
        set_simbacore_start();

        // Poll until streamer and accelerator finish
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(local_conv_out, conv_out, test_samples_conv_out, nb_test_samples, "conv_out");
        // err += check_result_all(local_conv_out, conv_out, length_conv_out);
        err +=
            check_result_sample(local_iscore_out, iscore_out, test_samples_iscore_out, nb_test_samples, "iscore_out");

        printf("Test Phase1: seqLen=%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int test_osgemm() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr = snrt_l1_next();
    uint16_t* local_a   = (uint16_t*)(tcdm_base_ptr + delta_a);
    uint16_t* local_b   = (uint16_t*)(tcdm_base_ptr + delta_b);
    uint16_t* local_d   = (uint16_t*)(tcdm_base_ptr + delta_d);

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, length_a);
        snrt_dma_start_1d(local_b, B, length_b);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("Setting up Streamer and SimbaCore for OSGeMM...\n");

        set_simbacore_osgemm_streamer_csr((uint32_t)local_a, M2_R0_ss, M2_R0_tb, M2_R0_ts,   // A
                                          (uint32_t)local_b, M2_R1_ss, M2_R1_tb, M2_R1_ts,   // B
                                          (uint32_t)local_d, M2_W0_ss, M2_W0_tb, M2_W0_ts);  // D

        set_simbacore_csr(M2_OSGEMM, seqLen, dModel, dInner, 1);
        set_simbacore_streamer_start();
        set_simbacore_start();

        // Poll until streamer and accelerator finish
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(local_d, D, test_samples_D, nb_test_samples, "out");
        // err += check_OSGeMM_result_all(local_d, D, length_d);

        printf("Test OSGeMM: seqLen%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() {
    int err = 0;
    err += test_phase1();
    err += test_osgemm();
    return err;
}
