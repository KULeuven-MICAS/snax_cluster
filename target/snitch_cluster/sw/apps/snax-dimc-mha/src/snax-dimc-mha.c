// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// this version of the code will only allocate 64KB of TCDM for the matrix
// Q/WQ/K/WK/V/WV since the constraints of a limited TCDM (128-KB) and the
// existence of a memory stack

#include "data_mha.h"
#include "snax-dimc-csr.h"
#include "snax-dimc-lib.h"
#include "snrt.h"

#define PRINT_STATS

int main() {
    // set error value for checking
    int err = 0;

    // printf("Starting the DIMC test\n");

    // allocate 32+32+32KB in TCDM for activation and weight pair
    uint64_t *activation_ptr, *weight_ptr, *output_ptr;

    activation_ptr = (uint64_t*)snrt_l1_next();
    weight_ptr = activation_ptr + Q_LENGTH;
    output_ptr = weight_ptr + Q_LENGTH;

    // allocate 4KB in TCDM for Q1K1T and final result
    uint64_t* buffer_ptr = output_ptr + Q_LENGTH;

    // alias for output_ptr, also holding Q
    uint64_t* activation_ptr_i = output_ptr;

    // stage 1:
    // load WK, K, Q to TCDM
    if (snrt_is_dm_core()) {
        // measure the start of cycle count for preloading data to TCDM
        // read weight WK and ativation K from data.h
        size_t vector_size = Q_LENGTH * sizeof(uint64_t);

        uint32_t start_dma_load = snrt_mcycle();
        snrt_dma_start_1d(activation_ptr, K, vector_size);
        snrt_dma_start_1d(activation_ptr_i, Q, vector_size);
        snrt_dma_start_1d(weight_ptr, WK, vector_size);
        snrt_dma_wait_all();
        uint32_t end_dma_load = snrt_mcycle();

        // Intermediate printing
#ifdef PRINT_STATS
        printf("S1 DMA %d \n", end_dma_load - start_dma_load);
#endif
    }

    /**************************************************************************/
    // wait for the DMA to finish loading WK and K to TCDM
    snrt_cluster_hw_barrier();
    /**************************************************************************/
    // stage 2: all three regions are occupied with WK, K, and Q
    // send WK from TCDM to DIMC
    /**************************************************************************/

    if (snrt_is_compute_core()) {
        uint32_t start_cfg_acc = snrt_mcycle();
        configure_accelerator();
        uint32_t end_cfg_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S2 CFG-ACC %d \n", end_cfg_acc - start_cfg_acc);
#endif

        // send WK
        start_cfg_acc = snrt_mcycle();
        dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(weight_ptr));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8,
                                 (uint32_t)(weight_ptr + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8,
                                 (uint32_t)(weight_ptr + 24));

        // configure the accelerator to start MHA computation
        dimc_start_mha();
        end_cfg_acc = snrt_mcycle();

#ifdef PRINT_STATS
        printf("S2 CFG-STRM %d \n", end_cfg_acc - start_cfg_acc);
#endif

        // start streamer data transfer
        // CYCLE_MEASURE: this is the start of the cycle count for LOADING
        uint32_t start_exe_acc = snrt_mcycle();
        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
        }
        uint32_t end_exe_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S2 EXE-LOAD %d \n", end_exe_acc - start_exe_acc);
#endif
        // CYCLE_MEASURE: this is the end of the cycle count for LOADING
    }

    /**************************************************************************/
    // wait for streamer to finish sending WK to DIMC
    snrt_cluster_hw_barrier();
    /**************************************************************************/
    // stage 3: weight_ptr is free, has K and Q in TCDM
    // load WQ to TCDM;
    // send K from TCDM to DIMC; kick start K1 generation;
    /**************************************************************************/

    // BIG NOTE THIS PART CAN BE PARALLELIZED
    if (snrt_is_dm_core()) {
        // read weight WQ from data.h
        size_t vector_size = Q_LENGTH * sizeof(uint64_t);

        uint32_t start_dma_load = snrt_mcycle();
        snrt_dma_start_1d(weight_ptr, WQ, vector_size);

        snrt_dma_wait_all();
        uint32_t end_dma_load = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S3 DMA* %d \n", end_dma_load - start_dma_load);
#endif
    }

    // Takenote that we can take this out
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()) {
        // send K
        uint32_t start_cfg_acc = snrt_mcycle();
        dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(activation_ptr));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8,
                                 (uint32_t)(activation_ptr + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8,
                                 (uint32_t)(activation_ptr + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8,
                                 (uint32_t)(activation_ptr + 24));
        uint32_t end_cfg_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S3 CFG-STRM %d \n", end_cfg_acc - start_cfg_acc);
#endif

        // CYCLE_MEASURE: this is the start of the cycle count for COMPUTING
        uint32_t start_exe_k = snrt_mcycle();
        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
        }
        uint32_t end_exe_k = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S3 EXE-COMP %d \n", end_exe_k - start_exe_k);
#endif
        // CYCLE_MEASURE: this is the end of the cycle count for COMPUTING
    }

    /**************************************************************************/
    // wait for the DMA to finish loading WQ to TCDM and K1 generation
    snrt_cluster_hw_barrier();
    /**************************************************************************/
    // stage 4: activation_ptr is free, has WQ and Q in TCDM
    // streamer sends WQ to DIMC
    /**************************************************************************/

    if (snrt_is_compute_core()) {
        uint32_t start_cfg_acc = snrt_mcycle();
        dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(weight_ptr));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8,
                                 (uint32_t)(weight_ptr + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8,
                                 (uint32_t)(weight_ptr + 24));
        uint32_t end_cfg_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S4 CFG-STRM %d \n", end_cfg_acc - start_cfg_acc);
#endif

        // CYCLE_MEASURE: this is the start of the cycle count for COMPUTING
        uint32_t start_exe_acc = snrt_mcycle();
        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
        }
        uint32_t end_exe_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S4 EXE-COMP %d \n", end_exe_acc - start_exe_acc);
#endif
        // CYCLE_MEASURE: this is the end of the cycle count for COMPUTING
    }

    /**************************************************************************/
    // wait for the streamer to finish sending WQ to DIMC
    snrt_cluster_hw_barrier();
    /**************************************************************************/
    // stage 5: weight_ptr and actication_ptr are free, has Q in TCDM
    // load WV and V to TCDM;
    // streamer sends Q to DIMC & kick start Q1K1T generation;
    /**************************************************************************/

    // BIG NOTE THIS PART CAN BE PARALLELIZED
    if (snrt_is_dm_core()) {
        // read weight WK and ativation K from data.h
        size_t vector_size = Q_LENGTH * sizeof(uint64_t);

        uint32_t start_dma_load = snrt_mcycle();
        snrt_dma_start_1d(activation_ptr, V, vector_size);
        snrt_dma_start_1d(weight_ptr, WV, vector_size);
        snrt_dma_wait_all();
        uint32_t end_dma_load = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S5 DMA* %d \n", end_dma_load - start_dma_load);
#endif
    }

    // This can be taken out
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()) {
        // send Q
        uint32_t start_cfg_acc = snrt_mcycle();
        dimc_set_streamer_dim_w(64, 1, 64, 0, 8, (uint32_t)(buffer_ptr));
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8,
                                 (uint32_t)(activation_ptr_i));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8,
                                 (uint32_t)(activation_ptr_i + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8,
                                 (uint32_t)(activation_ptr_i + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8,
                                 (uint32_t)(activation_ptr_i + 24));
        uint32_t end_cfg_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S5 CFG-STRM %d \n", end_cfg_acc - start_cfg_acc);
#endif

        // CYCLE_MEASURE: this is the start of the cycle count for COMPUTING
        uint32_t start_exe_acc = snrt_mcycle();
        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
        }
        uint32_t end_exe_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S5 EXE-COMP %d \n", end_exe_acc - start_exe_acc);
#endif
        // CYCLE_MEASURE: this is the end of the cycle count for COMPUTING
    }

    /**************************************************************************/
    // wait for the streamer to finish receiving Q1K1T
    snrt_cluster_hw_barrier();
    /**************************************************************************/
    // stage 6: activation_ptr_i is free, has V and WV in TCDM
    // send V to DIMC
    /**************************************************************************/

    if (snrt_is_compute_core()) {
        // send V
        uint32_t start_cfg_acc = snrt_mcycle();
        dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(activation_ptr));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8,
                                 (uint32_t)(activation_ptr + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8,
                                 (uint32_t)(activation_ptr + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8,
                                 (uint32_t)(activation_ptr + 24));
        uint32_t end_cfg_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S6 CFG-STRM %d \n", end_cfg_acc - start_cfg_acc);
#endif

        // CYCLE_MEASURE: this is the start of the cycle count for LOADING
        uint32_t start_exe_acc = snrt_mcycle();
        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
        }
        uint32_t end_exe_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S6 EXE-LOAD %d \n", end_exe_acc - start_exe_acc);
#endif
        // CYCLE_MEASURE: this is the end of the cycle count for LOADING
    }

    /**************************************************************************/
    // wait for the streamer to finish sending V to DIMC
    snrt_cluster_hw_barrier();
    /**************************************************************************/
    // stage 7: activation_ptr, activation_ptr_i are free, has WV in TCDM
    // send WV to DIMC
    /**************************************************************************/

    if (snrt_is_compute_core()) {
        // send WV
        uint32_t start_cfg_acc = snrt_mcycle();
        dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
        dimc_set_streamer_dim_r0(128, 1, 256, 0, 8, (uint32_t)(weight_ptr));
        dimc_set_streamer_dim_r1(128, 1, 256, 0, 8, (uint32_t)(weight_ptr + 8));
        dimc_set_streamer_dim_r2(128, 1, 256, 0, 8,
                                 (uint32_t)(weight_ptr + 16));
        dimc_set_streamer_dim_r3(128, 1, 256, 0, 8,
                                 (uint32_t)(weight_ptr + 24));
        uint32_t end_cfg_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S7 CFG-STRM %d \n", end_cfg_acc - start_cfg_acc);
#endif

        // CYCLE_MEASURE: this is the start of the cycle count for COMPUTING
        uint32_t start_exe_acc = snrt_mcycle();
        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
        }
        uint32_t end_exe_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S7 EXE-COMP %d \n", end_exe_acc - start_exe_acc);
#endif
        // CYCLE_MEASURE: this is the end of the cycle count for COMPUTING
    }

    /**************************************************************************/
    // wait for the streamer to finish sending WV to DIMC and V1 generation
    snrt_cluster_hw_barrier();
    /**************************************************************************/
    // stage 8: activation_ptr, activation_ptr_i, weight_ptr are free
    // has Q1K1T in TCDM
    // send Q1K1T to DIMC, saving final result in
    /**************************************************************************/

    if (snrt_is_compute_core()) {
        // send Q1K1T
        uint32_t start_cfg_acc = snrt_mcycle();
        dimc_set_streamer_dim_w(64, 1, 64, 0, 8, (uint32_t)(output_ptr));
        dimc_set_streamer_dim_r0(16, 1, 256, 0, 8, (uint32_t)(buffer_ptr));
        dimc_set_streamer_dim_r1(16, 1, 256, 0, 8, (uint32_t)(buffer_ptr + 8));
        dimc_set_streamer_dim_r2(16, 1, 256, 0, 8, (uint32_t)(buffer_ptr + 16));
        dimc_set_streamer_dim_r3(16, 1, 256, 0, 8, (uint32_t)(buffer_ptr + 24));
        uint32_t end_cfg_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S8 CFG-STRM %d \n", end_cfg_acc - start_cfg_acc);
#endif

        // CYCLE_MEASURE: this is the start of the cycle count for COMPUTING
        uint32_t start_exe_acc = snrt_mcycle();
        dimc_start_streamer();

        while (dimc_is_streamer_busy()) {
        }
        uint32_t end_exe_acc = snrt_mcycle();
#ifdef PRINT_STATS
        printf("S8 EXE-COMP %d \n", end_exe_acc - start_exe_acc);
#endif
        // CYCLE_MEASURE: this is the end of the cycle count for COMPUTING

#ifdef PRINT_STATS
        printf("CHECK FINAL RESULT\n");
#endif

        // check the final result
        for (int i = 0; i < 512; ++i) {
            uint64_t value = output_ptr[i];

            uint64_t index = i * 8;

            // Split each uint64_t element into 8 uint8_t elements
            for (int j = 0; j < 8; ++j) {
                uint8_t tmp_res = (uint8_t)((value >> (j * 8)) & 0xFF);
                // printf("%d ", tmp_res);
                if (tmp_res != gold[index + j]) {
                    printf("MISMATCH at %d, res:%d, gold:%d\n", (index + j),
                           tmp_res, gold[index + j]);
                    err += 1;
                }
            }
        }
    }

    snrt_cluster_hw_barrier();
    return err;
}
