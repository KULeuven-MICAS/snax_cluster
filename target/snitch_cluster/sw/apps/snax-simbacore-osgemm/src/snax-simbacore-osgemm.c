// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author : Robin Geens <robin.geens@kuleuven.be>

#include "data.h"
#include "snax-simbacore-lib.h"

int test_osgemm() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr = snrt_l1_next();
    uint8_t* ptr_a      = (uint8_t*)(tcdm_base_ptr + M3_addr_a);
    uint8_t* ptr_b      = (uint8_t*)(tcdm_base_ptr + M3_addr_b);
    uint8_t* ptr_d      = (uint8_t*)(tcdm_base_ptr + M3_addr_d);

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a, M3_A, M3_length_a);
        snrt_dma_start_1d(ptr_b, M3_B, M3_length_b);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("Setting up Streamer and SimbaCore for OSGeMM...\n");

        set_simbacore_osgemm_streamer_csr((uint32_t)ptr_a, M3_R0_ss, M3_R0_tb, M3_R0_ts,   // A
                                          (uint32_t)ptr_b, M3_R1_ss, M3_R1_tb, M3_R1_ts,   // B
                                          (uint32_t)ptr_d, M3_W0_ss, M3_W0_tb, M3_W0_ts);  // D

        set_simbacore_csr(M3_OSGEMM, seqLen, dModel, dInner, dtRank, 1);
        start_simbacore_and_streamers(M3_R10_en, 0, M3_R11_en, 0);
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample(ptr_d, M3_D, M3_test_samples_D,  //
                                   nb_test_samples, "out");

        printf("Test OSGeMM: seqLen%d, dModel=%d. %s: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err,
               nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_osgemm(); }
