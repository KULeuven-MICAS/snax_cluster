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

    uint64_t kernel_length = 256;
    uint64_t activation_length = Q_LENGTH;
    // uint64_t result_length = ?; // TODO: fix

    uint8_t *local_kernel;
    uint64_t *local_activation, *local_final_res;
    
    local_kernel = (uint8_t *)snrt_l1_next();
    local_activation = (uint64_t *)(local_kernel + kernel_length);
    local_final_res = (uint64_t *)(local_activation + activation_length);

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
        size_t kernel_size = kernel_length * sizeof(uint8_t);
        size_t activation_size = activation_length * sizeof(uint64_t);

        snrt_dma_start_1d(local_kernel,     kernel, kernel_size);
        snrt_dma_start_1d(local_activation, Q,      activation_size);

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

        dimc_query_busy();

        printf("QUERYING BUSY SUCCEEDED\n");

        dimc_set_alpha_qkv(128);

        printf("SETTING ALPHA QKV SUCCEEDED\n");

        dimc_set_alpha_qkt(256);

        printf("SETTING ALPHA QKT SUCCEEDED\n");

        dimc_set_alpha_conv(512);

        printf("SETTING ALPHA CONV SUCCEEDED\n");    
        /**********************************************************************/
        // configure the accelerator
        /**********************************************************************/

        // load convolution kernels
        dimc_load_kernel();
        
        printf("STARTING LOADING KERNEL SUCCEEDED\n");

        /**********************************************************************/
        // configure the streamer
        /**********************************************************************/

        // LOAD KERNELS
        printf("CONFIGURING STREAMERS for KERNEL\n");
        dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
        dimc_set_streamer_dim_r0(1, 1, 256, 0, 8, (uint32_t)(local_kernel));
        dimc_set_streamer_dim_r1(1, 1, 256, 0, 8, (uint32_t)(local_kernel + 64));
        dimc_set_streamer_dim_r2(1, 1, 256, 0, 8, (uint32_t)(local_kernel + 128));
        dimc_set_streamer_dim_r3(1, 1, 256, 0, 8, (uint32_t)(local_kernel + 192));
        printf("STREAMER CONFIGURED for WQ\n");

        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
            // printf("STREAMER BUSY\n");
        }

        /**********************************************************************/
        // configure the accelerator
        /**********************************************************************/
        dimc_load_act_conv();

        printf("STARTING LOADING ACTIVATION SUCCEEDED\n");

        /**********************************************************************/
        // configure the streamer
        /**********************************************************************/

        // LOAD ACTIVATION
        printf("CONFIGURING STREAMERS for ACTIVATION\n");
        dimc_set_streamer_dim_w((64*16), 1, 64, 0, 8, (uint32_t)(local_final_res));
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(local_activation));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(local_activation + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8, (uint32_t)(local_activation + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8, (uint32_t)(local_activation + 24));
        printf("STREAMER CONFIGURED for ACTIVATION\n");

        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
            // printf("STREAMER BUSY\n");
        }
        
        /**********************************************************************/
        // configure the accelerator
        /**********************************************************************/

        dimc_start_conv();
        
        printf("GO CHECK FINAL RESULT\n");

        dimc_query_busy();
        dimc_query_busy();
        dimc_query_busy();
        dimc_query_busy();
        dimc_query_busy();
    };

    return err;
}
