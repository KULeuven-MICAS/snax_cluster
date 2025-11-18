// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include "data.h"

#include "snax-versacore-to-lib.h"

int main() {
    // Set err value for checking
    int err = 0;

    // Prepare addresses in TCDM
    int8_t *local_a, *local_b;
    int32_t *local_c, *local_d;

    // Allocate space in TCDM
    local_a = (int8_t *)(snrt_l1_next() + delta_local_a);
    local_b = (int8_t *)(snrt_l1_next() + delta_local_b);
    local_c = (int32_t *)(snrt_l1_next() + delta_local_c);
    local_d = (int32_t *)(snrt_l1_next() + delta_local_d);

    // Transfer data from L3 to L1
    // Using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, a_data_length);
        snrt_dma_start_1d(local_b, B, b_data_length);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_c, C, c_data_length);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Set the CSR for the Streamer
    int32_t Aslstride[] = {Aslstride0};
    int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2,
                          Atlbound3, Atlbound4, Atlbound5};
    int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2,
                           Atlstride3, Atlstride4, Atlstride5};
    int32_t Bslstride[] = {Bslstride0};
    int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
    int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};

    int32_t Cslstride[] = {Cslstride0};
    int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2, Ctlbound3};
    int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};

    int32_t D32slstride[] = {D32slstride0};
    int32_t D32tlbound[] = {D32tlbound0, D32tlbound1, D32tlbound2, D32tlbound3};
    int32_t D32tlstride[] = {D32tlstride0, D32tlstride1, D32tlstride2,
                             D32tlstride3};

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        // Set Streamer configuration CSR
        set_versacore_streamer_csr(
            delta_local_a, Aslstride, Atlbound, Atlstride,
            set_addr_remap_index_A, transposed_A, channel_en_A,

            delta_local_b, Bslstride, Btlbound, Btlstride,
            set_addr_remap_index_B, transposed_B, channel_en_B,

            delta_local_c, Cslstride, Ctlbound, Ctlstride,
            set_addr_remap_index_C, channel_en_C,

            delta_local_d, D32slstride, D32tlbound, D32tlstride,
            set_addr_remap_index_D32, channel_en_D, array_shape,

            quantization_enable, shift_i, multiplier_i, input_zp_i, output_zp_i,
            int32tofp16_enable);

        // Set GEMMX configuration CSR
        uint32_t subtraction_setting =
            gen_subtraction_config(subtraction_a, subtraction_b);

        if (stationary == 0) {
            // Set CSR for output-stationary
            set_versacore_csr(1, K, N * M, subtraction_setting, array_shape,
                              data_type);
        } else {
            // Set CSR for weight-stationary or input-stationary
            set_versacore_csr(1, 1, N * K * M, subtraction_setting, array_shape,
                              data_type);
        }

        // Set CSR to start Streamer
        set_versacore_streamer_start();

        // Set CSR to start GEMM
        set_versacore_start();

        // Poll until Streamer and GEMM accelerator finish
        wait_versacore_and_streamer();

        // Result check
        if (quantization_enable == 0 && int32tofp16_enable == 0)
            err += check_versacore_result_D32((int32_t *)local_d, (int32_t *)D,
                                              d_data_length, false);
        else if (quantization_enable == 1 && int32tofp16_enable == 0) {
            err += check_versacore_result_D32(
                (int8_t *)local_d, (int8_t *)D_quantized, d_data_length, false);
        } else if (int32tofp16_enable == 1) {
            err += check_versacore_result_D32((int8_t *)local_d,
                                              (int8_t *)D_int32tofp16,
                                              d_data_length, false);
        }

        printf(
            "Array shape: %d, meshRow %d, tileSize %d, meshCol %d, stationary: "
            "%d, SNAX GEMM Matmul: %s, Error: %d.\n",
            array_shape, meshRow, tileSize, meshCol, stationary,
            err ? "FAIL" : "PASS", err);

        int32_t gemmx_cycles = read_versacore_perf_counter();
        int32_t gemmx_streamer_cycles = read_versacore_streamer_perf_counter();
        printf("Workload size: M = %d, N = %d, K = %d\n", M, N, K);
        printf("SNAX GEMM Ideal cycles: %d\n", M * K * N);
        printf("SNAX GEMM cycles: %d\n", gemmx_cycles);
        printf("SNAX GEMM Streamer cycles: %d\n", gemmx_streamer_cycles);
    };

    return err;
}
