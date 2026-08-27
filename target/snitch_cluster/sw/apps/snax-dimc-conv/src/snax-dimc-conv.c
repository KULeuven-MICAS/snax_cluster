// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "data_conv.h"
#include "snax-dimc-csr.h"
#include "snax-dimc-lib.h"
#include "snrt.h"

static int err = 0;

// Enable me to have checker
// #define CHECK_EN

int main() {
    // printf("Starting the DIMC CONV test\n");

    // Set err value for checking
    err = 0;

    uint64_t kernel_length = 256;
    uint64_t qkv_activation_length = Q_LENGTH;
    uint64_t qkt_activation_length = Q1K1T_LENGTH;

    uint8_t* local_kernel;
    uint64_t *local_activation_qkv, *local_activation_qkt, *local_final_res;

    local_kernel = (uint8_t*)snrt_l1_next();
    local_activation_qkv = (uint64_t*)(local_kernel + kernel_length);
    local_activation_qkt =
        (uint64_t*)(local_activation_qkv + qkv_activation_length);
    local_final_res = (uint64_t*)(local_activation_qkt + qkt_activation_length);

    if (snrt_is_dm_core()) {
        // This measures the start of cycle count
        // for preloading the data to the L1 memory
        uint32_t start_dma_load = snrt_mcycle();
        /**********************************************************************/
        // initialize TCDM by DMA
        /**********************************************************************/

        // read matrix Q from data.h
        int kernel_size = kernel_length * sizeof(uint8_t);
        int activation_size_qkv = qkv_activation_length * sizeof(uint64_t);
        int activation_size_qkt = qkt_activation_length * sizeof(uint64_t);

        snrt_dma_start_1d(local_kernel, kernel, kernel_size);
        snrt_dma_start_1d(local_activation_qkv, Q, activation_size_qkv);
        snrt_dma_start_1d(local_activation_qkt, Q1K1T_G, activation_size_qkt);

        // Measure the end of the transfer process
        snrt_dma_wait_all();

        uint32_t end_dma_load = snrt_mcycle();
        printf("DMA %d\n", end_dma_load - start_dma_load);
    }

    // Wait until DMA transfer is done
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()) {
        uint32_t start_cfg = snrt_mcycle();
        configure_accelerator();
        dimc_set_streamer_dim_w((64 * 16), 1, 64, 0, 8,
                                (uint32_t)(local_final_res));
        dimc_set_streamer_dim_r0(145, 1, 256, 0, 8, (uint32_t)(local_kernel));
        dimc_set_streamer_dim_r1(145, 1, 256, 0, 8,
                                 (uint32_t)(local_kernel + 64));
        dimc_set_streamer_dim_r2(145, 1, 256, 0, 8,
                                 (uint32_t)(local_kernel + 128));
        dimc_set_streamer_dim_r3(145, 1, 256, 0, 8,
                                 (uint32_t)(local_kernel + 192));
        dimc_start_streamer();
        uint32_t end_cfg = snrt_mcycle();
        printf("CFG %d\n", end_cfg - start_cfg);

        dimc_start_conv();
        uint32_t start_exe = snrt_mcycle();
        write_csr_obs(0x00f);

        // This literally takes long like 16k cycles
        while (dimc_is_streamer_busy()) {
        }
        write_csr_obs(0x000);
        uint32_t end_exe = snrt_mcycle();
        printf("EXE %d\n", end_exe - start_exe);

        int err = 0;

#ifdef CHECK_EN
        // const int iterations = 8192 / 8;
        for (int i = 0; i < (8192 / 8); i++) {
            // Pointers to the current chunks
            uint64_t* res_chunk = &local_final_res[i * 8];
            uint8_t* gold_chunk = &gold_conv[i * 36];

            // Buffer to hold bits from local_final_res
            uint8_t res_bits[64];  // 512 bits / 8 bits per byte = 64 bytes

            // Convert local_final_res chunk to bytes
            for (int j = 0; j < 8; j++) {
                uint64_t val = res_chunk[j];
                for (int k = 0; k < 8; ++k) {
                    res_bits[j * 8 + k] =
                        (val >> (k * 8)) & 0xFF;  // Little-endian
                }
            }

            // Buffer to hold bits from gold_conv with padding

            volatile uint8_t gold_bits[64];

            for (int j = 0; j < 36; j++) {
                gold_bits[j] = gold_chunk[j];
            }

            for (int j = 36; j < 64; j++) {
                gold_bits[j] = 0;
            }

            // Compare the two byte arrays
            bool match = true;
            for (int j = 0; j < 64; j++) {
                if (res_bits[j] != gold_bits[j]) {
                    match = false;
                    err += 1;
                    break;
                }
            }
        }
        printf("MISMATCHES: %d\n", err);
#endif
    }

    snrt_cluster_hw_barrier();

    return err;
}
