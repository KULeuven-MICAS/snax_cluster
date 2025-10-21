// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include <snrt.h>
#include "data.h"

int main() {
    // Set err value for checking
    int err = 0;

    // Transfer data A from L3 to L1
    // Using DMA only
    if (snrt_is_dm_core()) {
        int8_t *local_seperate_l1_a;
        local_seperate_l1_a = (int8_t *)(snrt_l1_next() + delta_seperate_l1_a);
        int8_t *local_shared_l1_a;
        local_shared_l1_a = (int8_t *)(snrt_l1_next() + delta_shared_l1_a);

        // if seperate l1, use 2d dma
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                snrt_hartid());
        if (if_separate_l1) {
            // 512 wide
            snrt_dma_start_2d(local_seperate_l1_a, A, 64 * sizeof(int8_t), 256,
                              64,
                              M * K * meshRow * tileSize * sizeof(int8_t) / 64);
        } else
        // if shared l1, use 1d dma
        {
            snrt_dma_start_1d(local_shared_l1_a, A,
                              M * K * meshRow * tileSize * sizeof(int8_t));
        }
        snrt_dma_wait_all();
        printf("[Tensor-A]-DMA transfer cycle from DMA hardware counter: %d\n",
               snrt_get_perf_counter(SNRT_PERF_CNT0));
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Transfer data B from L3 to L1
    if (snrt_is_dm_core()) {
        int8_t *local_seperate_l1_b;
        local_seperate_l1_b = (int8_t *)(snrt_l1_next() + delta_seperate_l1_b);
        int8_t *local_shared_l1_b;
        local_shared_l1_b = (int8_t *)(snrt_l1_next() + delta_shared_l1_b);
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                snrt_hartid());
        if (if_separate_l1) {
            // 512 wide
            snrt_dma_start_2d(local_seperate_l1_b, B, 64 * sizeof(int8_t), 256,
                              64,
                              N * K * tileSize * meshCol * sizeof(int8_t) / 64);
        } else {
            snrt_dma_start_1d(local_shared_l1_b, B,
                              N * K * tileSize * meshCol * sizeof(int8_t));
        }
        snrt_dma_wait_all();
        printf("[Tensor-B]-DMA transfer cycle from DMA hardware counter: %d\n",
               snrt_get_perf_counter(SNRT_PERF_CNT0));
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
    }

    snrt_cluster_hw_barrier();

    // Transfer data C from L3 to L1
    if (snrt_is_dm_core()) {
        int8_t *local_seperate_l1_c;
        local_seperate_l1_c = (int8_t *)(snrt_l1_next() + delta_seperate_l1_c);
        int8_t *local_shared_l1_c;
        local_shared_l1_c = (int8_t *)(snrt_l1_next() + delta_shared_l1_c);
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                snrt_hartid());
        if (if_separate_l1) {
            // 1024 wide
            snrt_dma_start_2d(
                local_seperate_l1_c, C, 16 * sizeof(int32_t) * 2, 256,
                16 * sizeof(int32_t) * 2,
                M * N * meshRow * meshCol * sizeof(int32_t) / 64 / 2);
        } else {
            snrt_dma_start_1d(local_shared_l1_c, C,
                              M * N * meshRow * meshCol * sizeof(int32_t));
        }
        snrt_dma_wait_all();
        printf("[Tensor-C]-DMA transfer cycle from DMA hardware counter: %d\n",
               snrt_get_perf_counter(SNRT_PERF_CNT0));
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
    }

    snrt_cluster_hw_barrier();

    // Move partial sum data from l1 to l3
    if (snrt_is_dm_core()) {
        int32_t *local_seperate_l1_d32;
        local_seperate_l1_d32 =
            (int32_t *)(snrt_l1_next() + delta_seperate_l1_d32);
        int32_t *local_shared_l1_d32;
        local_shared_l1_d32 = (int32_t *)(snrt_l1_next() + delta_shared_l1_d32);
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                snrt_hartid());
        if (if_separate_l1) {
            // 1024 wide
            snrt_dma_start_2d(
                D32, local_seperate_l1_d32, 16 * sizeof(int32_t) * 2,
                16 * sizeof(int32_t) * 2, 256,
                M * N * meshRow * meshCol * sizeof(int32_t) / 64 / 2);
        } else {
            snrt_dma_start_1d(D32, local_shared_l1_d32,
                              M * N * meshRow * meshCol * sizeof(int32_t));
        }
        snrt_dma_wait_all();
        printf(
            "[Tensor-D32]-DMA transfer cycle from DMA hardware counter: %d\n",
            snrt_get_perf_counter(SNRT_PERF_CNT0));
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
    }

    snrt_cluster_hw_barrier();

    // Move final output data from l1 to l3
    if (snrt_is_dm_core()) {
        int8_t *local_seperate_l1_d8;
        local_seperate_l1_d8 =
            (int8_t *)(snrt_l1_next() + delta_seperate_l1_d8);
        // int32_t data_move_loop =
        //     M * N * meshRow * meshCol * sizeof(int8_t) / 64 / 2;
        int8_t *local_shared_l1_d8;
        local_shared_l1_d8 = (int8_t *)(snrt_l1_next() + delta_shared_l1_d8);
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                snrt_hartid());
        if (if_separate_l1) {
            // 1024 wide
            // if (data_move_loop_d8 == 0) {
            //     data_move_loop_d8 = 1;
            // }
            snrt_dma_start_2d(D8, local_seperate_l1_d8, 64 * sizeof(int8_t),
                              64 * sizeof(int8_t), 256, data_move_loop_d8);
        } else {
            snrt_dma_start_1d(D8, local_shared_l1_d8,
                              M * N * meshRow * meshCol * sizeof(int8_t));
        }
        snrt_dma_wait_all();
        printf("[Tensor-D8]-DMA transfer cycle from DMA hardware counter: %d\n",
               snrt_get_perf_counter(SNRT_PERF_CNT0));
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
    }

    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        printf("Data move with %s L1 successful!\n",
               if_separate_l1 ? "separate" : "shared");
        printf("M=%d, N=%d, K=%d, meshRow=%d, meshCol=%d, tileSize=%d\n", M, N,
               K, meshRow, meshCol, tileSize);
    }

    snrt_cluster_hw_barrier();

    return err;
}
