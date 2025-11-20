// Copyright 2025 KU Leuven.
// Not released under license.All rights reserved.
//
// Author : Robin Geens < robin.geens@kuleuven.be>

#include "snax-simbacore-helper.c"
#include "snax-simbacore-lib.h"

int test_phase1() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr        = snrt_l1_next();
    uint8_t* ptr_oscore_in     = (uint8_t*)(tcdm_base_ptr + M0_addr_oscore_in);
    uint8_t* ptr_oscore_weight = (uint8_t*)(tcdm_base_ptr + M0_addr_oscore_weight);
    uint8_t* ptr_conv_weight   = (uint8_t*)(tcdm_base_ptr + M0_addr_conv_weight);
    uint8_t* ptr_conv_bias     = (uint8_t*)(tcdm_base_ptr + M0_addr_conv_bias);
    uint8_t* ptr_conv_out      = (uint8_t*)(tcdm_base_ptr + M0_addr_conv_out);
    uint8_t* ptr_iscore_weight = (uint8_t*)(tcdm_base_ptr + M0_addr_iscore_weight);
    uint16_t* ptr_iscore_out   = (uint16_t*)(tcdm_base_ptr + M0_addr_iscore_out);  // holds the psums

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
        set_streamer_phase1((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight,   //
                            (uint32_t)ptr_conv_weight, (uint32_t)ptr_conv_bias,     //
                            (uint32_t)ptr_iscore_weight, (uint32_t)ptr_iscore_out,  //
                            (uint32_t)ptr_conv_out);

        set_simbacore_csr(M0_PHASE1, seqLen, dModel, dInner, dtRank, xProjDim);
        start_simbacore_and_streamers(M0_R10_en, 0, M0_R11_en, 0);
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(ptr_conv_out, M0_conv_out, M0_test_samples_conv_out,  //
                                   nb_test_samples, "conv_out");

        // Outputs are packed as FP8
        err += check_result_sample((uint8_t*)ptr_iscore_out, M0_iscore_out, M0_test_samples_iscore_out,  //
                                   nb_test_samples, "iscore_out");

        printf("Test Phase1: seqLen=%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int test_phase2() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr        = snrt_l1_next();
    uint8_t* ptr_oscore_in     = (uint8_t*)(tcdm_base_ptr + M1_addr_oscore_in);
    uint8_t* ptr_oscore_weight = (uint8_t*)(tcdm_base_ptr + M1_addr_oscore_weight);
    uint8_t* ptr_z             = (uint8_t*)(tcdm_base_ptr + M1_addr_z);  // osCore out
    uint8_t* ptr_dt_in         = (uint8_t*)(tcdm_base_ptr + M1_addr_dt_BC);
    uint8_t* ptr_BC            = (uint8_t*)(tcdm_base_ptr + M1_addr_dt_BC + M1_dt_to_BC_offset);  //
    // TODO test of dit hetzelfde is
    // uint8_t* ptr_BC            = (void*)ptr_dt_in + M1_dt_to_BC_offset);  //
    uint8_t* ptr_dt_weight_1   = (uint8_t*)(tcdm_base_ptr + M1_addr_dt_weight_1);
    uint8_t* ptr_dt_weight_2   = (uint8_t*)(tcdm_base_ptr + M1_addr_dt_weight_2);
    uint8_t* ptr_dt_bias       = (uint8_t*)(tcdm_base_ptr + M1_addr_dt_bias);
    uint8_t* ptr_x             = (uint8_t*)(tcdm_base_ptr + M1_addr_x);  // from Phase1
    uint8_t* ptr_A             = (uint8_t*)(tcdm_base_ptr + M1_addr_A);
    uint8_t* ptr_D             = (uint8_t*)(tcdm_base_ptr + M1_addr_D);
    uint8_t* ptr_y             = (uint8_t*)(tcdm_base_ptr + M1_addr_y);  // SUC out
    uint8_t* ptr_iscore_weight = (uint8_t*)(tcdm_base_ptr + M1_addr_iscore_weight);
    uint16_t* ptr_iscore_out   = (uint16_t*)(tcdm_base_ptr + M1_addr_iscore_out);

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_in, M1_oscore_in, M1_length_oscore_in);
        snrt_dma_start_1d(ptr_oscore_weight, M1_oscore_weight, M1_length_oscore_weight);
        snrt_dma_start_1d(ptr_dt_in, M1_dt_BC, M1_length_dt_BC);
        snrt_dma_start_1d(ptr_dt_weight_1, M1_dt_weight_1, M1_length_dt_weight_1);
        snrt_dma_start_1d(ptr_dt_weight_2, M1_dt_weight_2, M1_length_dt_weight_2);
        snrt_dma_start_1d(ptr_dt_bias, M1_dt_bias, M1_length_dt_bias);
        snrt_dma_start_1d(ptr_x, M1_suc_x, M1_length_x);
        snrt_dma_start_1d(ptr_A, M1_suc_A, M1_length_A);
        snrt_dma_start_1d(ptr_D, M1_suc_D, M1_length_D);
        snrt_dma_start_1d(ptr_iscore_weight, M1_iscore_weight, M1_length_iscore_weight);
        snrt_dma_start_1d(ptr_iscore_out, M1_iscore_bias, M1_length_iscore_out);  // Load bias in psums
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        set_streamer_phase2((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight,                         //
                            (uint32_t)ptr_z, (uint32_t)ptr_dt_in,                                         //
                            (uint32_t)ptr_dt_weight_1, (uint32_t)ptr_dt_weight_2, (uint32_t)ptr_dt_bias,  //
                            (uint32_t)ptr_x, (uint32_t)ptr_A, (uint32_t)ptr_BC, (uint32_t)ptr_D,          //
                            (uint32_t)ptr_y, (uint32_t)ptr_iscore_weight, (uint32_t)ptr_iscore_out);

        set_simbacore_csr(M1_PHASE2, seqLen, dModel, dInner, dtRank, dModel);
        start_simbacore_and_streamers(M1_R10_en, M1_R10_start_cnt, M1_R11_en, M1_R11_start_cnt);
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(ptr_z, M1_oscore_expected, M1_test_samples_z,  //
                                   nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y, M1_suc_expected, M1_test_samples_y,  //
                                   nb_test_samples, "SUC y");
        err += check_result_sample((uint8_t*)ptr_iscore_out, M1_iscore_expected,  //
                                   M1_test_samples_iscore_out, nb_test_samples, "iscore_out");

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
    uint8_t* ptr_a      = (uint8_t*)(tcdm_base_ptr + M2_addr_a);
    uint8_t* ptr_b      = (uint8_t*)(tcdm_base_ptr + M2_addr_b);
    uint8_t* ptr_d      = (uint8_t*)(tcdm_base_ptr + M2_addr_d);

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

        set_simbacore_csr(M2_OSGEMM, seqLen, dModel, dInner, dtRank, 1);
        start_simbacore_and_streamers(M2_R10_en, 0, M2_R11_en, 0);
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(ptr_d, M2_D, M2_test_samples_D,  //
                                   nb_test_samples, "out");

        printf("Test OSGeMM: seqLen%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int test_phase1_and_2() {
    int err = 0;

    // Allocation. Let's start by naively allocating space for each individual tensor.
    void* tcdm_base_ptr = snrt_l1_next();
    // Phase 1
    uint8_t* ptr_oscore_in        = (uint8_t*)(tcdm_base_ptr + M0_addr_oscore_in);
    uint8_t* ptr_oscore_weight_P1 = (uint8_t*)(tcdm_base_ptr + M0_addr_oscore_weight);
    uint8_t* ptr_conv_weight      = (uint8_t*)(tcdm_base_ptr + M0_addr_conv_weight);
    uint8_t* ptr_conv_bias        = (uint8_t*)(tcdm_base_ptr + M0_addr_conv_bias);
    uint8_t* ptr_conv_out         = (uint8_t*)(tcdm_base_ptr + M0_addr_conv_out);
    uint8_t* ptr_iscore_weight_P1 = (uint8_t*)(tcdm_base_ptr + M0_addr_iscore_weight);
    uint16_t* ptr_iscore_out_P1   = (uint16_t*)(tcdm_base_ptr + M0_addr_iscore_out);  // holds the psums

    // Phase 2
    void* phase2_base_ptr         = ((void*)ptr_iscore_out_P1 + M0_length_iscore_out);
    uint8_t* ptr_oscore_weight_P2 = (uint8_t*)(phase2_base_ptr + M1_addr_oscore_weight);
    uint8_t* ptr_z                = (uint8_t*)(phase2_base_ptr + M1_addr_z);  // osCore out
    uint8_t* ptr_dt_in            = (uint8_t*)ptr_iscore_out_P1;
    uint8_t* ptr_BC               = (void*)ptr_dt_in + M1_dt_to_BC_offset;
    uint8_t* ptr_dt_weight_1      = (uint8_t*)(phase2_base_ptr + M1_addr_dt_weight_1);
    uint8_t* ptr_dt_weight_2      = (uint8_t*)(phase2_base_ptr + M1_addr_dt_weight_2);
    uint8_t* ptr_dt_bias          = (uint8_t*)(phase2_base_ptr + M1_addr_dt_bias);
    uint8_t* ptr_x                = ptr_conv_out;
    uint8_t* ptr_A                = (uint8_t*)(phase2_base_ptr + M1_addr_A);
    uint8_t* ptr_D                = (uint8_t*)(phase2_base_ptr + M1_addr_D);
    uint8_t* ptr_y                = (uint8_t*)(phase2_base_ptr + M1_addr_y);  // SUC out
    uint8_t* ptr_iscore_weight_P2 = (uint8_t*)(phase2_base_ptr + M1_addr_iscore_weight);
    uint16_t* ptr_iscore_out_P2   = (uint16_t*)(phase2_base_ptr + M1_addr_iscore_out);

    // Transfer Phase1 data
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_in, M0_oscore_in, M0_length_oscore_in);
        snrt_dma_start_1d(ptr_oscore_weight_P1, M0_oscore_weight, M0_length_oscore_weight);
        snrt_dma_start_1d(ptr_conv_weight, M0_conv_weight, M0_length_conv_weight);
        snrt_dma_start_1d(ptr_conv_bias, M0_conv_bias, M0_length_conv_bias);
        snrt_dma_start_1d(ptr_iscore_weight_P1, M0_iscore_weight, M0_length_iscore_weight);
        // Input and output psums use the same address
        snrt_dma_start_1d(ptr_iscore_out_P1, M0_iscore_bias, M0_length_iscore_out);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        set_streamer_phase1((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P1,      //
                            (uint32_t)ptr_conv_weight, (uint32_t)ptr_conv_bias,           //
                            (uint32_t)ptr_iscore_weight_P1, (uint32_t)ptr_iscore_out_P1,  //
                            (uint32_t)ptr_conv_out);

        set_simbacore_csr(M0_PHASE1, seqLen, dModel, dInner, dtRank, xProjDim);
        start_simbacore_and_streamers(M0_R10_en, 0, M0_R11_en, 0);
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(ptr_conv_out, M0_conv_out, M0_test_samples_conv_out,  //
                                   nb_test_samples, "conv_out");

        err += check_result_sample((uint8_t*)ptr_iscore_out_P1, M0_iscore_out, M0_test_samples_iscore_out,  //
                                   nb_test_samples, "iscore_out");
    }

    snrt_cluster_hw_barrier();

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_weight_P2, M1_oscore_weight, M1_length_oscore_weight);
        snrt_dma_start_1d(ptr_dt_weight_1, M1_dt_weight_1, M1_length_dt_weight_1);
        snrt_dma_start_1d(ptr_dt_weight_2, M1_dt_weight_2, M1_length_dt_weight_2);
        snrt_dma_start_1d(ptr_dt_bias, M1_dt_bias, M1_length_dt_bias);
        snrt_dma_start_1d(ptr_A, M1_suc_A, M1_length_A);
        snrt_dma_start_1d(ptr_D, M1_suc_D, M1_length_D);
        snrt_dma_start_1d(ptr_iscore_weight_P2, M1_iscore_weight, M1_length_iscore_weight);
        snrt_dma_start_1d(ptr_iscore_out_P2, M1_iscore_bias, M1_length_iscore_out);  // Load bias in psums
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        set_streamer_phase2((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P2,                      //
                            (uint32_t)ptr_z, (uint32_t)ptr_dt_in,                                         //
                            (uint32_t)ptr_dt_weight_1, (uint32_t)ptr_dt_weight_2, (uint32_t)ptr_dt_bias,  //
                            (uint32_t)ptr_x, (uint32_t)ptr_A, (uint32_t)ptr_BC, (uint32_t)ptr_D,          //
                            (uint32_t)ptr_y, (uint32_t)ptr_iscore_weight_P2, (uint32_t)ptr_iscore_out_P2);

        set_simbacore_csr(M1_PHASE2, seqLen, dModel, dInner, dtRank, dModel);
        start_simbacore_and_streamers(M1_R10_en, M1_R10_start_cnt, M1_R11_en, M1_R11_start_cnt);
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(ptr_z, M1_oscore_expected, M1_test_samples_z,  //
                                   nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y, M1_suc_expected, M1_test_samples_y,  //
                                   nb_test_samples, "SUC y");
        err += check_result_sample((uint8_t*)ptr_iscore_out_P2, M1_iscore_expected,  //
                                   M1_test_samples_iscore_out, nb_test_samples, "iscore_out");

        printf("Test Phase2: seqLen=%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               5 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() {
    int err = 0;
    err += test_phase1_and_2();
    // err += test_phase2();
    // err += test_osgemm();
    // err += test_phase1();
    return err;
}
