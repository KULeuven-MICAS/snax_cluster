// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include "data.h"

#include "snax-gemmx-params.h"

#include "snax-gemmx-lib.h"

// This is the main function for the SNAX GEMM for Conv2d
// We use several nested loops to iterate over the input data and weights,
// achieving implicit im2col
int main() {
    // Set err value for checking
    int err = 0;

    // Prepare addresses in TCDM
    int8_t *local_a, *local_b;
    int32_t *local_c, *local_d32;
    int8_t *local_d8;

    // Allocate space in TCDM
    local_a = (int8_t *)(snrt_l1_next() + delta_local_a);
    local_b = (int8_t *)(snrt_l1_next() + delta_local_b);
    local_c = (int32_t *)(snrt_l1_next() + delta_local_c);
    local_d32 = (int32_t *)(snrt_l1_next() + delta_local_d32);
    local_d8 = (int8_t *)(snrt_l1_next() + delta_local_d8);

    // Transfer data from L3 to L1
    // Using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A,
                          M * K * meshRow * tileSize * sizeof(int8_t));
        snrt_dma_start_1d(local_b, B,
                          N * K * tileSize * meshCol * sizeof(int8_t));

        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_c, C,
                          M * N * meshRow * meshCol * sizeof(int32_t));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    int32_t Aslstride[] = {Aslstride0};
    int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2,
                          Atlbound3, Atlbound4, Atlbound5};
    int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2,
                           Atlstride3, Atlstride4, Atlstride5};
    int32_t Bslstride[] = {Bslstride0};
    int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
    int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};
    int32_t D8slstride[] = {D8slstride0};
    int32_t D8tlbound[] = {D8tlbound0, D8tlbound1, D8tlbound2, D8tlbound3};
    int32_t D8tlstride[] = {D8tlstride0, D8tlstride1, D8tlstride2, D8tlstride3};
    int32_t Cslstride[] = {Cslstride0};
    int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2, Ctlbound3};
    int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};
    int32_t D32slstride[] = {D32slstride0};
    int32_t D32tlbound[] = {D32tlbound0, D32tlbound1, D32tlbound2, D32tlbound3};
    int32_t D32tlstride[] = {D32tlstride0, D32tlstride1, D32tlstride2,
                             D32tlstride3};

    if (snrt_global_core_idx() == 0) {
        // Set Streamer configuration CSR for conv2d
        set_gemmx_streamer_csr(
            Aslstride, Atlbound, Atlstride, set_addr_remap_index_A,

            Bslstride, Btlbound, Btlstride, set_addr_remap_index_B,

            D8slstride, D8tlbound, D8tlstride, set_addr_remap_index_D8,

            Cslstride, Ctlbound, Ctlstride, set_addr_remap_index_C,

            D32slstride, D32tlbound, D32tlstride, set_addr_remap_index_D32,

            delta_local_a, delta_local_b, delta_local_d8, delta_local_c,
            delta_local_d32, bypassSIMD, transposed_A, transposed_B,
            channel_en_C, broadcast_C);

        // Set GEMMX configuration CSR
        uint32_t subtraction_setting =
            gen_subtraction_config(subtraction_a, subtraction_b);

        uint32_t csr0 =
            gen_csr0_config(input_zp_i, output_zp_i, max_int_i, min_int_i);
        uint32_t csr1 = gen_csr1_config(double_round_i);

        set_gemmx_csr(K, N, M, subtraction_setting, csr0, csr1,
                      shared_bitpacked_shift, shared_multiplier, M * N,
                      bypassSIMD);

        // Set CSR to start Streamer for conv2d
        set_gemmx_streamer_start();

        // Set CSR to start GEMM
        set_gemmx_start();

        // Poll until Streamer and GEMM accelerator finish
        wait_gemmx_and_streamer();

        // check the result of the implicit im2col convolution
        if (!bypassSIMD) {
            err += check_gemmx_result_D8(local_d8, D8, Batch, M, N, false);
        } else {
            err += check_gemmx_result_D32(local_d32, D32, Batch, M, N, false);
        }
        int32_t gemmx_cycles = read_gemmx_perf_counter();
        int32_t gemmx_streamer_cycles = read_gemmx_streamer_perf_counter();
        printf("Workload size: M = %d, N = %d, K = %d\n", M, N, K);
        printf("SNAX GEMM Ideal cycles: %d\n", M * K * N);
        printf("SNAX GEMM cycles: %d\n", gemmx_cycles);
        printf("SNAX GEMM Streamer cycles: %d\n", gemmx_streamer_cycles);
        printf("SNAX GEMM Matmul: %s, Error: %d . bypassSIMD = %d .\n",
               err ? "FAIL" : "PASS", err, bypassSIMD);
    };

    return err;
}
