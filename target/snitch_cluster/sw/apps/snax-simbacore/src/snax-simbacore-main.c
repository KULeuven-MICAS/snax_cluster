// Copyright 2025 KU Leuven.
// Not released under license.All rights reserved.
//
// Author : Robin Geens < robin.geens@kuleuven.be>

#include "data.h"
#include "snax-simbacore-lib.h"

int test_phase1() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr         = snrt_l1_next();
    uint16_t* ptr_oscore_in     = (uint16_t*)(tcdm_base_ptr + M0_addr_oscore_in);
    uint16_t* ptr_oscore_weight = (uint16_t*)(tcdm_base_ptr + M0_addr_oscore_weight);
    uint16_t* ptr_conv_weight   = (uint16_t*)(tcdm_base_ptr + M0_addr_conv_weight);
    uint16_t* ptr_conv_bias     = (uint16_t*)(tcdm_base_ptr + M0_addr_conv_bias);
    uint16_t* ptr_conv_out      = (uint16_t*)(tcdm_base_ptr + M0_addr_conv_out);
    uint16_t* ptr_iscore_weight = (uint16_t*)(tcdm_base_ptr + M0_addr_iscore_weight);
    uint16_t* ptr_iscore_out    = (uint16_t*)(tcdm_base_ptr + M0_addr_iscore_out);  // holds the psums

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_in, M0_oscore_in, M0_length_oscore_in);
        snrt_dma_start_1d(ptr_oscore_weight, M0_oscore_weight, M0_length_oscore_weight);
        snrt_dma_start_1d(ptr_conv_weight, M0_conv_weight, M0_length_conv_weight);
        snrt_dma_start_1d(ptr_conv_bias, M0_conv_bias, M0_length_conv_bias);
        snrt_dma_start_1d(ptr_iscore_weight, M0_iscore_weight, M0_length_iscore_weight);
        // Input and output psums use the same address
        snrt_dma_start_1d(ptr_iscore_out, M0_iscore_bias, M0_length_iscore_out);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("Setting up Streamer and SimbaCore for Phase1...\n");

        set_streamer_csr(

            (uint32_t)ptr_oscore_in, M0_R0_ss, M0_R0_tb, M0_R0_ts, M0_R0_en,          //
            (uint32_t)ptr_oscore_weight, M0_R1_ss, M0_R1_tb, M0_R1_ts, M0_R1_en,      //
            (uint32_t)0, 0, 0, 0, M0_R2_en,                                           // disable
            (uint32_t)ptr_conv_weight, M0_R3_ss, M0_R3_tb, M0_R3_ts, M0_R3_en,        //
            (uint32_t)ptr_conv_bias, M0_R4_ss, M0_R4_tb, M0_R4_ts, M0_R4_en,          //
            (uint32_t)0, 0, 0, 0, M0_R5_en,                                           // disable
            (uint32_t)0, 0, 0, 0, M0_R6_en,                                           // disable
            (uint32_t)0, 0, 0, 0, M0_R7_en,                                           // disable
            (uint32_t)0, 0, 0, 0, M0_R8_en,                                           // disable
            (uint32_t)0, 0, 0, 0, M0_R9_en,                                           // disable
            (uint32_t)0, 0, 0, 0, M0_R10_en,                                          // disable
            (uint32_t)0, 0, 0, 0, M0_R11_en,                                          // disable
            (uint32_t)ptr_iscore_weight, M0_R12_ss, M0_R12_tb, M0_R12_ts, M0_R12_en,  //
            (uint32_t)ptr_iscore_out, M0_R13_ss, M0_R13_tb, M0_R13_ts, M0_R13_en,     // psums
            (uint32_t)0, 0, 0, 0, M0_W0_en,                                           // disable
            (uint32_t)ptr_conv_out, M0_W1_ss, M0_W1_tb, M0_W1_ts, M0_W1_en,           //
            (uint32_t)0, 0, 0, 0, M0_W2_en,                                           // disable
            (uint32_t)ptr_iscore_out, M0_W3_ss, M0_W3_tb, M0_W3_ts, M0_W3_en          //
        );

        set_simbacore_csr(M0_PHASE1, seqLen, dModel, dInner, dtRank);
        set_simbacore_start();
        set_simbacore_streamer_start(M0_R10_en, 0, M0_R11_en, 0);
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(ptr_conv_out, M0_conv_out, M0_test_samples_conv_out, nb_test_samples, "conv_out");

        err += check_result_sample(ptr_iscore_out, M0_iscore_out, M0_test_samples_iscore_out, nb_test_samples,
                                   "iscore_out");

        printf("Test Phase1: seqLen=%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int test_phase2() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr         = snrt_l1_next();
    uint16_t* ptr_oscore_in     = (uint16_t*)(tcdm_base_ptr + M1_addr_oscore_in);
    uint16_t* ptr_oscore_weight = (uint16_t*)(tcdm_base_ptr + M1_addr_oscore_weight);
    uint16_t* ptr_z             = (uint16_t*)(tcdm_base_ptr + M1_addr_z);  // osCore out
    uint16_t* ptr_dt_in         = (uint16_t*)(tcdm_base_ptr + M1_addr_dt_in);
    uint16_t* ptr_dt_weight_1   = (uint16_t*)(tcdm_base_ptr + M1_addr_dt_weight_1);
    uint16_t* ptr_dt_weight_2   = (uint16_t*)(tcdm_base_ptr + M1_addr_dt_weight_2);
    uint16_t* ptr_dt_bias       = (uint16_t*)(tcdm_base_ptr + M1_addr_dt_bias);
    uint16_t* ptr_x             = (uint16_t*)(tcdm_base_ptr + M1_addr_x);  // from Phase1
    uint16_t* ptr_A             = (uint16_t*)(tcdm_base_ptr + M1_addr_A);
    uint16_t* ptr_BC            = (uint16_t*)(tcdm_base_ptr + M1_addr_BC);
    uint16_t* ptr_D             = (uint16_t*)(tcdm_base_ptr + M1_addr_D);
    uint16_t* ptr_y             = (uint16_t*)(tcdm_base_ptr + M1_addr_y);  // SUC out
    uint16_t* ptr_iscore_weight = (uint16_t*)(tcdm_base_ptr + M1_addr_iscore_weight);
    uint16_t* ptr_iscore_out    = (uint16_t*)(tcdm_base_ptr + M1_addr_iscore_out);

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_in, M1_oscore_in, M1_length_oscore_in);
        snrt_dma_start_1d(ptr_oscore_weight, M1_oscore_weight, M1_length_oscore_weight);
        snrt_dma_start_1d(ptr_dt_in, M1_dt_in, M1_length_dt_in);
        snrt_dma_start_1d(ptr_dt_weight_1, M1_dt_weight_1, M1_length_dt_weight_1);
        snrt_dma_start_1d(ptr_dt_weight_2, M1_dt_weight_2, M1_length_dt_weight_2);
        snrt_dma_start_1d(ptr_dt_bias, M1_dt_bias, M1_length_dt_bias);
        snrt_dma_start_1d(ptr_x, M1_suc_x, M1_length_x);
        snrt_dma_start_1d(ptr_A, M1_suc_A, M1_length_A);
        snrt_dma_start_1d(ptr_BC, M1_suc_BC, M1_length_BC);
        snrt_dma_start_1d(ptr_D, M1_suc_D, M1_length_D);
        snrt_dma_start_1d(ptr_iscore_weight, M1_iscore_weight, M1_length_iscore_weight);
        snrt_dma_start_1d(ptr_iscore_out, M1_iscore_bias, M1_length_iscore_out);  // Load bias in psums
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("Setting up Streamer and SimbaCore for Phase2...\n");

        set_streamer_csr(

            (uint32_t)ptr_oscore_in, M1_R0_ss, M1_R0_tb, M1_R0_ts, M1_R0_en,          // osCore in
            (uint32_t)ptr_oscore_weight, M1_R1_ss, M1_R1_tb, M1_R1_ts, M1_R1_en,      // oscore weight
            (uint32_t)ptr_dt_in, M1_R2_ss, M1_R2_tb, M1_R2_ts, M1_R2_en,              // switchCore in
            (uint32_t)ptr_dt_weight_1, M1_R3_ss, M1_R3_tb, M1_R3_ts, M1_R3_en,        // switchCore weight
            (uint32_t)ptr_dt_bias, M1_R4_ss, M1_R4_tb, M1_R4_ts, M1_R4_en,            // switchCore bias
            (uint32_t)ptr_dt_weight_2, M1_R5_ss, M1_R5_tb, M1_R5_ts, M1_R5_en,        // switchCore  matmul weight
            (uint32_t)ptr_A, M1_R6_ss, M1_R6_tb, M1_R6_ts, M1_R6_en,                  //  SUC A
            (uint32_t)ptr_BC, M1_R7_ss, M1_R7_tb, M1_R7_ts, M1_R7_en,                 // SUC BC
            (uint32_t)ptr_D, M1_R8_ss, M1_R8_tb, M1_R8_ts, M1_R8_en,                  // SUC  D
            (uint32_t)ptr_x, M1_R9_ss, M1_R9_tb, M1_R9_ts, M1_R9_en,                  // SUC x
            (uint32_t)ptr_z, M1_R10_ss, M1_R10_tb, M1_R10_ts, M1_R10_en,              // SUC z = osCore out
            (uint32_t)ptr_y, M1_R11_ss, M1_R11_tb, M1_R11_ts, M1_R11_en,              // iscore in = SUC y
            (uint32_t)ptr_iscore_weight, M1_R12_ss, M1_R12_tb, M1_R12_ts, M1_R12_en,  // isCore weight
            (uint32_t)ptr_iscore_out, M1_R13_ss, M1_R13_tb, M1_R13_ts, M1_R13_en,     // isCore psum

            (uint32_t)ptr_z, M1_W0_ss, M1_W0_tb, M1_W0_ts, M1_W0_en,          //
            (uint32_t)0, 0, 0, 0, M1_W1_en,                                   // disable
            (uint32_t)ptr_y, M1_W2_ss, M1_W2_tb, M1_W2_ts, M1_W2_en,          // SUC y
            (uint32_t)ptr_iscore_out, M1_W3_ss, M1_W3_tb, M1_W3_ts, M1_W3_en  // isCore out

        );

        set_simbacore_csr(M1_PHASE2, seqLen, dModel, dInner, dtRank);
        set_simbacore_start();  // Start SimbaCore first: otherwise delayed streamer cannot start
        set_simbacore_streamer_start(M1_R10_en, M1_R10_start_cnt, M1_R11_en, M1_R11_start_cnt);
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(ptr_z, M1_oscore_expected, M1_test_samples_z, nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y, M1_suc_expected, M1_test_samples_y, nb_test_samples, "SUC y");

        err += check_result_sample(ptr_iscore_out, M1_iscore_expected, M1_test_samples_iscore_out, nb_test_samples,
                                   "iscore_out");

        printf("Test Phase2: seqLen=%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               3 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int test_osgemm() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr = snrt_l1_next();
    uint16_t* ptr_a     = (uint16_t*)(tcdm_base_ptr + M2_addr_a);
    uint16_t* ptr_b     = (uint16_t*)(tcdm_base_ptr + M2_addr_b);
    uint16_t* ptr_d     = (uint16_t*)(tcdm_base_ptr + M2_addr_d);

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a, M2_A, M2_length_a);
        snrt_dma_start_1d(ptr_b, M2_B, M2_length_b);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("Setting up Streamer and SimbaCore for OSGeMM...\n");

        set_simbacore_osgemm_streamer_csr((uint32_t)ptr_a, M2_R0_ss, M2_R0_tb, M2_R0_ts,   // A
                                          (uint32_t)ptr_b, M2_R1_ss, M2_R1_tb, M2_R1_ts,   // B
                                          (uint32_t)ptr_d, M2_W0_ss, M2_W0_tb, M2_W0_ts);  // D

        set_simbacore_csr(M2_OSGEMM, seqLen, dModel, dInner, 1);
        set_simbacore_streamer_start(M2_R10_en, 0, M2_R11_en, 0);
        set_simbacore_start();
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(ptr_d, M2_D, M2_test_samples_D, nb_test_samples, "out");
        // err += check_OSGeMM_result_all(ptr_d, D, length_d);

        printf("Test OSGeMM: seqLen%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() {
    int err = 0;
    err += test_phase1();
    err += test_phase2();
    err += test_osgemm();
    return err;
}
