// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "data.h"
#include "snax-simbacore-lib.h"

int test_isgemm() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr = snrt_l1_next();
    uint8_t* ptr_a      = (uint8_t*)(tcdm_base_ptr + M4_addr_a);
    uint8_t* ptr_b      = (uint8_t*)(tcdm_base_ptr + M4_addr_b);
    uint16_t* ptr_cd    = (uint16_t*)(tcdm_base_ptr + M4_addr_cd);

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a, M4_A, M4_length_a);
        snrt_dma_start_1d(ptr_b, M4_B, M4_length_b);
        snrt_dma_start_1d(ptr_cd, M4_C, M4_length_cd);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("Setting up Streamer and SimbaCore for ISGEMM...\n");

        set_isgemm_streamer_csr((uint32_t)ptr_a, M4_R11_ss, M4_R11_tb, M4_R11_ts,  // A
                                (uint32_t)ptr_b, M4_R12_ss, M4_R12_tb, M4_R12_ts,  // B
                                (uint32_t)ptr_cd, M4_W3_ss, M4_W3_tb, M4_W3_ts);   // C/D

        set_simbacore_csr(M4_ISGEMM, seqLen, dModel, dInner, dtRank, dModel);
        start_simbacore_and_streamers(M4_R10_en, 0, M4_R11_en, 0);
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        // check_result_all((uint8_t*)ptr_cd, M4_D, M4_length_cd);
        err += check_result_sample((uint8_t*)ptr_cd, M4_D, M4_test_samples_D,  //
                                   nb_test_samples, "out");

        printf("Test isgemm: seqLen%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_isgemm(); }
