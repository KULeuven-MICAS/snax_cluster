// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "data.h"
#include "snax-simbacore-lib.h"
#include "streamer_csr_addr_map.h"

int test_simd() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr    = snrt_l1_next();
    uint16_t* ptr_a        = (uint16_t*)(tcdm_base_ptr + M5_addr_in_a);
    uint16_t* ptr_b        = (uint16_t*)(tcdm_base_ptr + M5_addr_in_b);
    uint16_t* ptr_out_add  = (uint16_t*)(tcdm_base_ptr + M5_addr_add_out);
    uint16_t* ptr_out_sub  = (uint16_t*)(tcdm_base_ptr + M5_addr_sub_out);
    uint16_t* ptr_out_mul  = (uint16_t*)(tcdm_base_ptr + M5_addr_mul_out);
    uint16_t* ptr_out_cmul = (uint16_t*)(tcdm_base_ptr + M5_addr_cmul_out);

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a, M5_simd_a, M5_length_in_a);
        snrt_dma_start_1d(ptr_b, M5_simd_b, M5_length_in_b);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        // CMUL
        set_simd_streamer_csr((uint32_t)ptr_a, M5_R7_ss, M5_R7_tb, M5_R7_ts,        // SUC BC
                              (uint32_t)ptr_b, M5_R13_ss, M5_R13_tb, M5_R13_ts,     // isCore psum
                              (uint32_t)ptr_out_cmul, M5_W3_ss, M5_W3_tb, M5_W3_ts  // isCore out
        );

        set_simbacore_csr(M8_SIMD_CMUL, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(M5_R10_en, 0, M5_R11_en, 0);
        wait_simbacore_and_streamer();

        // ADD
        // only change the streamer that has changed
        write_csr(BASE_PTR_WRITER_3_LOW, ptr_out_add);
        set_simbacore_simd_csr(M5_SIMD_ADD);
        start_simbacore_and_streamers(M5_R10_en, 0, M5_R11_en, 0);
        wait_simbacore_and_streamer();

        // SUB
        write_csr(BASE_PTR_WRITER_3_LOW, ptr_out_sub);
        set_simbacore_simd_csr(M6_SIMD_SUB);
        start_simbacore_and_streamers(M5_R10_en, 0, M5_R11_en, 0);
        wait_simbacore_and_streamer();

        // MUL
        write_csr(BASE_PTR_WRITER_3_LOW, ptr_out_mul);
        set_simbacore_simd_csr(M7_SIMD_MUL);
        start_simbacore_and_streamers(M5_R10_en, 0, M5_R11_en, 0);
        wait_simbacore_and_streamer();

        printf("SimbaCore took %u cycles\n", read_simbacore_perf_counter());

        err += check_result_sample_u16(ptr_out_cmul, M5_cmul_out, M5_test_samples_out,  //
                                       nb_test_samples, "CMUL");
        err += check_result_sample_u16(ptr_out_add, M5_add_out, M5_test_samples_out,  //
                                       nb_test_samples, "ADD");
        err += check_result_sample_u16(ptr_out_sub, M5_sub_out, M5_test_samples_out,  //
                                       nb_test_samples, "SUB");
        err += check_result_sample_u16(ptr_out_mul, M5_mul_out, M5_test_samples_out,  //
                                       nb_test_samples, "MUL");

        printf("Test SIMD: %s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 4 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_simd(); }
