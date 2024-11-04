// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// #include "snax-dimc-lib.h"
#include "snrt.h"
#include "data.h"

//#include "data.h"

int main() {
    // Set err value for checking
    int err = 0;

    // Allocates space in TCDM
    uint64_t *local_a, *local_b, *local_o;

    local_a = (uint64_t *)snrt_l1_next();
    local_b = local_a + DATA_LEN;
    local_o = local_b + DATA_LEN;

    if (snrt_is_dm_core()) {
        // This measures the start of cycle count
        // for preloading the data to the L1 memory
        uint32_t start_dma_load = snrt_mcycle();

        // The DATA_LEN is found in data.h
        size_t vector_size = DATA_LEN * sizeof(uint64_t);
        snrt_dma_start_1d(local_a, A, vector_size);
        snrt_dma_start_1d(local_b, B, vector_size);

        // Measure the end of the transfer process
        uint32_t end_dma_load = snrt_mcycle();
    }

    // Synchronize cores by setting up a
    // fence barrier for the DMA and accelerator core
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()){
        // This marks the start of the
        // setting of CSRs for the accelerator
        uint32_t start_csr_setup = snrt_mcycle();
        
        for (size_t i = 0; i < DATA_LEN; i++) {
            printf("A[%d] = %d, ", i, *(local_a + i));
        }
        printf("\n");

        for (size_t i = 0; i < DATA_LEN; i++) {
            printf("B[%d] = %d, ", i, *(local_b + i));
        }
        printf("\n");
    };

    return err;
}
