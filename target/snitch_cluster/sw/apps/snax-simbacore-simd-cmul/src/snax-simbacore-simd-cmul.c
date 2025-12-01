// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "data.h"
#include "snax-simbacore-lib.h"

int test_simd_cmul() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr = snrt_l1_next();
    uint16_t* ptr_a     = (uint16_t*)(tcdm_base_ptr + M5_addr_cmul_a);
    uint16_t* ptr_b     = (uint16_t*)(tcdm_base_ptr + M5_addr_cmul_b);
    uint16_t* ptr_out   = (uint16_t*)(tcdm_base_ptr + M5_addr_cmul_out);

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a, M5_cmul_a, M5_length_cmul_a);
        snrt_dma_start_1d(ptr_b, M5_cmul_b, M5_length_cmul_b);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        set_simd_streamer_csr((uint32_t)ptr_a, M5_R7_ss, M5_R7_tb, M5_R7_ts,     // SUC BC
                              (uint32_t)ptr_b, M5_R13_ss, M5_R13_tb, M5_R13_ts,  // isCore psum
                              (uint32_t)ptr_out, M5_W3_ss, M5_W3_tb, M5_W3_ts    // isCore out
        );
        set_simbacore_csr(M5_SIMD_CMUL, seqLen, dModel, dInner, dtRank, 1);
        start_simbacore_and_streamers(M5_R10_en, 0, M5_R11_en, 0);
        wait_simbacore_and_streamer();
        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample_u16(ptr_out, M5_cmul_out, M5_test_samples_cmul_out,  //
                                       nb_test_samples, "out");

        printf("Test SIMD_CMUL: %u/%d errors.\n", seqLen, dModel, err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_simd_cmul(); }
