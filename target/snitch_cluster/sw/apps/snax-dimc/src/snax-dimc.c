// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snax-dimc-csr.h"
#include "snax-dimc-lib.h"
#include "snrt.h"
#include "data.h"

int main() {

    printf("Starting the DIMC test\n ");

    // printf("matrix_Q initialized\n");

    // Set err value for checking
    int err = 0;

    // Allocates space in TCDM for matrix Q
    uint64_t *local_q, *local_wq, *local_k, *local_wk, *local_wv, *local_v, *local_q1k1t;
    local_q = (uint64_t *)snrt_l1_next();
    local_wq = local_q + Q_LENGTH;
    local_k = local_wq + Q_LENGTH;
    local_wk = local_k + Q_LENGTH;
    local_wv = local_wk + Q_LENGTH;
    local_v = local_wv + Q_LENGTH;
    local_q1k1t = local_v + 1;

    if(snrt_is_dm_core()) {
        printf("DMA core will be configured\n ");

        // This measures the start of cycle count
        // for preloading the data to the L1 memory
        uint32_t start_dma_load = snrt_mcycle();

        /**********************************************************************/
        // initialize TCDM by DMA
        /**********************************************************************/
        printf("INITIALIZING TCDM\n ");

        // read matrix Q from data.h
        size_t vector_size = Q_LENGTH * sizeof(uint64_t);
        size_t vector_size_q1k1t = 512 * sizeof(uint64_t);

        // snrt_dma_start_1d(local_q, matrix_Q, vector_size);
        // snrt_dma_start_1d(local_wq, matrix_WQ, vector_size);
        // snrt_dma_start_1d(local_k, matrix_K, vector_size);
        snrt_dma_start_1d(local_wk, matrix_WK, vector_size);
        // snrt_dma_start_1d(local_wv, matrix_WV, vector_size);
        // snrt_dma_start_1d(local_v, matrix_V, vector_size);
        // snrt_dma_start_1d(local_q1k1t, matrix_Q1K1T, vector_size_q1k1t);

        // Measure the end of the transfer process
        uint32_t end_dma_load = snrt_mcycle();
        
        // printf("DMA core exits\n ");
    }

    // Wait until DMA transfer is done
    snrt_cluster_hw_barrier();

    if(snrt_is_dm_core()) delay_cycles(1000000);
    printf("DMA core exits\n  ");

    if (snrt_is_compute_core()){
        
        /**********************************************************************/
        // simple testing code for DMA bahaivour
        // uint32_t test_val;
        
        // csrw_ss(993, 0); // addr 0: cehcking busy state, this should be a read only register

        // TOOD: correct this to Read only register
        // csrw_ss(994, 0); // addr 1: cehcking busy state, this should be a read only register
        
        // csrw_ss(999, 128); // addr 6: setting a alpha parameter for mha quantization of QKV gen

        // csrw_ss(998, 0); // addr 5: start mha calculation, this will cause the accelerator to stay busy since there is no data
        // csrw_ss(994, 0);

        // csrw_ss(997, 0); // addr 4: start loading kernels, this will cause the accelerator to stay busy since there is no data
        // csrw_ss(994, 0);

        // csrw_ss(995, 0); // addr 2: start loading conv activations, this will cause the accelerator to stay busy since there is no data
        // csrw_ss(994, 0);

        // csrw_ss(996, 0); // addr 3: start calculating convolution, this will cause the accelerator to stay busy since there is no data
        // csrw_ss(994, 0);
        

        //test_val = csrr_ss(994); // read reg for cehcking busy state
        // printf("Query Busy Returns %d \n ", test_val);        
        
        
        /**********************************************************************/
        // configure the accelerator
        /**********************************************************************/
        printf("ENTERING MHA MODE\n ");

        dimc_query_busy();

        printf("QUERYING BUSY SUCCEEDED\n ");

        dimc_set_alpha_qkv(128);

        printf("SETTING ALPHA QKV SUCCEEDED\n ");

        dimc_set_alpha_qkt(128);

        printf("SETTING ALPHA QKT SUCCEEDED\n ");

        dimc_start_mha();

        printf("STARTING MHA SUCCEEDED\n ");

        /**********************************************************************/
        // configure the streamer
        /**********************************************************************/
        
        printf("CONFIGURING STREAMER\n ");

        // LOAD WK
        // dimc_set_streamer_dim_r0(128, 0, 256, 0, 0, (local_wk + 0));
        // dimc_set_streamer_dim_r1(128, 0, 256, 0, 0, (local_wk + 1));
        // dimc_set_streamer_dim_r2(128, 0, 256, 0, 0, (local_wk + 2));
        // dimc_set_streamer_dim_r3(128, 0, 256, 0, 0, (local_wk + 3));
        
        // dimc_start_streamer_r();
        /**********************************************************************/
        // TODO: set boundary for the transactions to end
        /**********************************************************************/

        // LOAD K
        // dimc_set_streamer_dim_r0(128, 0, 256, 0, 0, (local_k + 0));
        // dimc_set_streamer_dim_r1(128, 0, 256, 0, 0, (local_k + 64));
        // dimc_set_streamer_dim_r2(128, 0, 256, 0, 0, (local_k + 128));
        // dimc_set_streamer_dim_r3(128, 0, 256, 0, 0, (local_k + 192));
        
        /**********************************************************************/
        // TODO: set boundary for the transactions to end
        /**********************************************************************/

        // LOAD WQ
        // dimc_set_streamer_dim_r0(128, 0, 256, 0, 0, (local_wq + 0));
        // dimc_set_streamer_dim_r1(128, 0, 256, 0, 0, (local_wq + 64));
        // dimc_set_streamer_dim_r2(128, 0, 256, 0, 0, (local_wq + 128));
        // dimc_set_streamer_dim_r3(128, 0, 256, 0, 0, (local_wq + 192));

        /**********************************************************************/
        // TODO: set boundary for the transactions to end
        /**********************************************************************/

        // LOAD Q
        // dimc_set_streamer_dim_r0(128, 0, 256, 0, 0, (local_q + 0));
        // dimc_set_streamer_dim_r1(128, 0, 256, 0, 0, (local_q + 64));
        // dimc_set_streamer_dim_r2(128, 0, 256, 0, 0, (local_q + 128));
        // dimc_set_streamer_dim_r3(128, 0, 256, 0, 0, (local_q + 192));
        
        // WRITE Q1K1T
        // dimc_set_streamer_dim_w(64, 0, 64, 0, 0, local_q1k1t);

        /**********************************************************************/
        // TODO: set boundary for the transactions to end
        /**********************************************************************/

        // LOAD V   
        // dimc_set_streamer_dim_r0(128, 0, 256, 0, 0, (local_v + 0));
        // dimc_set_streamer_dim_r1(128, 0, 256, 0, 0, (local_v + 64));
        // dimc_set_streamer_dim_r2(128, 0, 256, 0, 0, (local_v + 128));
        // dimc_set_streamer_dim_r3(128, 0, 256, 0, 0, (local_v + 192));

        /**********************************************************************/
        // TODO: set boundary for the transactions to end
        /**********************************************************************/

        // LOAD WV
        // dimc_set_streamer_dim_r0(128, 0, 256, 0, 0, (local_wv + 0));
        // dimc_set_streamer_dim_r1(128, 0, 256, 0, 0, (local_wv + 64));
        // dimc_set_streamer_dim_r2(128, 0, 256, 0, 0, (local_wv + 128));
        // dimc_set_streamer_dim_r3(128, 0, 256, 0, 0, (local_wv + 192));

        /**********************************************************************/
        // TODO: set boundary for the transactions to end
        /**********************************************************************/

        // LOAD Q1K1T
        // dimc_set_streamer_dim_r0(16, 0, 256, 0, 0, (local_q1k1t + 0));
        // dimc_set_streamer_dim_r1(16, 0, 256, 0, 0, (local_q1k1t + 64));
        // dimc_set_streamer_dim_r2(16, 0, 256, 0, 0, (local_q1k1t + 128));
        // dimc_set_streamer_dim_r3(16, 0, 256, 0, 0, (local_q1k1t + 192));

        // uint32_t base_ptr_final_result = 200702;
        // dimc_set_streamer_dim_w(64, 0, 64, 0, 0, (base_ptr_Q1K1T + 256));
        /**********************************************************************/
        // TODO: set boundary for the transactions to end
        /**********************************************************************/

    };

    return err;
}
