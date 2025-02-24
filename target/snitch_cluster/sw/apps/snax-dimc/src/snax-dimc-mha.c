// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snax-dimc-csr.h"
#include "snax-dimc-lib.h"
#include "snrt.h"
#include "data.h"

int main() {

    printf("Starting the DIMC test\n");

    // printf("matrix_Q initialized\n");

    // Set err value for checking
    int err = 0;

    // Allocates space in TCDM for matrix Q
    uint64_t *local_q, *local_wq, 
             *local_k, *local_wk, 
             *local_v, *local_wv,
             *local_q1k1t, 
             *local_final_res;

    local_q = (uint64_t *)snrt_l1_next();
    local_wq = local_q + Q_LENGTH;
    local_k = local_wq + Q_LENGTH;
    local_wk = local_k + Q_LENGTH;
    local_wv = local_wk + Q_LENGTH;
    local_v = local_wv + Q_LENGTH;
    local_q1k1t = local_v + Q_LENGTH;
    local_final_res = local_q1k1t + Q1K1T_LENGTH;

    if(snrt_is_dm_core()) {
        printf("DMA core will be configured\n");

        // This measures the start of cycle count
        // for preloading the data to the L1 memory
        uint32_t start_dma_load = snrt_mcycle();
        /**********************************************************************/
        // initialize TCDM by DMA
        /**********************************************************************/
        printf("INITIALIZING TCDM\n");

        // read matrix Q from data.h
        size_t vector_size = Q_LENGTH * sizeof(uint64_t);

        size_t vector_size_q1k1t = Q1K1T_LENGTH * sizeof(uint64_t);

        snrt_dma_start_1d(local_q,     Q,     vector_size);
        snrt_dma_start_1d(local_wq,    WQ,    vector_size);
        snrt_dma_start_1d(local_k,     K,     vector_size);
        snrt_dma_start_1d(local_wk,    WK,    vector_size);
        snrt_dma_start_1d(local_wv,    WV,    vector_size);
        snrt_dma_start_1d(local_v,     V,     vector_size);
        // snrt_dma_start_1d(local_q1k1t, Q1K1T, vector_size_q1k1t);

        // Measure the end of the transfer process
        snrt_dma_wait_all();

        uint32_t end_dma_load = snrt_mcycle();
        
        printf("DMA core exits\n"); 
    }

    // Wait until DMA transfer is done
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()){
        printf("COMPUTE CORE\n");

        /**********************************************************************/
        // configure the accelerator
        /**********************************************************************/
        printf("ENTERING MHA MODE\n");

        uint32_t busy = dimc_query_busy();

        printf("QUERYING BUSY SUCCEEDED\n");
        printf("%d: busy\n", busy);

        configure_accelerator();
        
        printf("CONFIGURING ACCELERATOR SUCCEEDED\n");

        uint32_t read_zp_qkv = read_zp();

        printf("READING ZP SUCCEEDED\n");
        printf("%d: read_zp_qkv\n", read_zp_qkv);

        /**********************************************************************/
        // configure the streamer
        /**********************************************************************/

        // LOAD WK
        printf("CONFIGURING STREAMERS for WK\n");
        dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(local_wk));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(local_wk + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(local_wk + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(local_wk + 24));
        printf("STREAMER CONFIGURED FOR WK\n");
        /**********************************************************************/
        // configure the accelerator
        /**********************************************************************/

        dimc_start_mha();

        printf("STARTING MHA SUCCEEDED\n");

        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
            printf("STREAMER BUSY\n");
        }

        // LOAD K
        printf("CONFIGURING STREAMERS for K\n");
        dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(local_k));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(local_k + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(local_k + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(local_k + 24));
        printf("STREAMER CONFIGURED FOR K\n");

        dimc_start_streamer();

        printf("STREAMER STARTED FOR K1\n");

        while (dimc_is_streamer_busy()) {
            // printf("STREAMER BUSY\n");
        }

        // LOAD WQ
        printf("CONFIGURING STREAMERS for WQ\n");
        dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(local_wq));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(local_wq + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(local_wq + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(local_wq + 24));
        printf("STREAMER CONFIGURED for WQ\n");

        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
            // printf("STREAMER BUSY\n");
        }

        // LOAD Q
        printf("CONFIGURING STREAMERS for Q\n");
        dimc_set_streamer_dim_w(64, 1, 64, 0, 8, (uint32_t)(local_q1k1t));
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(local_q));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(local_q + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(local_q + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(local_q + 24));
        printf("STREAMER CONFIGURED for Q\n");

        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
            // printf("STREAMER BUSY\n");
        }


        // LOAD V
        printf("CONFIGURING STREAMERS for V\n");
        dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(local_v));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(local_v + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(local_v + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(local_v + 24));
        printf("STREAMER CONFIGURED for V\n");

        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
            // printf("STREAMER BUSY\n");
        }

        // LOAD WV
        printf("CONFIGURING STREAMERS for WV\n");
        dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(local_wv));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(local_wv + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(local_wv + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(local_wv + 24));
        printf("STREAMER CONFIGURED for WV\n");

        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
            // printf("STREAMER BUSY\n");
        }

        // LOAD Q1K1T
        printf("CONFIGURING STREAMERS for Q1K1T\n");
        dimc_set_streamer_dim_w(64, 1, 64, 0, 8,   (uint32_t)(local_final_res));
        dimc_set_streamer_dim_r0(16, 1, 256, 0, 8, (uint32_t)(local_q1k1t));
        dimc_set_streamer_dim_r1(16, 1, 256, 0, 8, (uint32_t)(local_q1k1t + 8));
        dimc_set_streamer_dim_r2(16, 1, 256, 0, 8, (uint32_t)(local_q1k1t + 16));
        dimc_set_streamer_dim_r3(16, 1, 256, 0, 8, (uint32_t)(local_q1k1t + 24));
        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
            // printf("STREAMER BUSY\n");
        }
        printf("GO CHECK FINAL RESULT\n");

        uint32_t busy_0 = dimc_query_busy();
        printf("BUSY 0: %d\n", busy_0);

        uint32_t busy_1 = dimc_query_busy();
        printf("BUSY 1: %d\n", busy_1);

        uint32_t busy_2 = dimc_query_busy();
        printf("BUSY 2: %d\n", busy_2);

        uint32_t busy_3 = dimc_query_busy();
        printf("BUSY 3: %d\n", busy_3);

        uint32_t busy_4 = dimc_query_busy();
        printf("BUSY 4: %d\n", busy_4);
        
        // Prepare a 2D array to store the 64x64 uint8_t values
        uint8_t matrix[4096];

        // Convert 512 uint64_t elements to 4096 uint8_t elements
        for (int i = 0; i < 512; ++i) {
            uint64_t value = local_final_res[i];

            uint64_t index = i * 8;

            // Split each uint64_t element into 8 uint8_t elements
            for (int j = 0; j < 8; ++j) {
                matrix[index + j] = (uint8_t)((value >> (j * 8)) & 0xFF);
                if(matrix[index + j] != gold[index + j]) {
                    printf("MISMATCH\n");
                }
            }
        }
    };

    return err;
}
