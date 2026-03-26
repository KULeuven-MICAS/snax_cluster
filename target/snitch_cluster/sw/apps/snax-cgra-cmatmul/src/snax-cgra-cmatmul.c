// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// JYIN

#include "snax-cgra-lib.h"
#include "streamer_csr_addr_map.h"

#include "data.h"
#include "snax-cgra-workload.b32.hpp"
#include "snrt.h"
#include "stdint.h"

#define debug_mode 1

int main() {

    // Set err value for checking
    // int err = 0;
    int err_counter;

    int32_t *local_config_data;
    int16_t *local_d16;

    local_config_data = (int32_t *)(snrt_l1_next() + delta_config_data);
    local_d16 = (int16_t *)(snrt_l1_next() + delta_store_data);

    // Using DMA only
    if (snrt_is_dm_core()) {
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                snrt_hartid());
        local_config_data = (int32_t *)(snrt_l1_next() + delta_config_lut);
        snrt_dma_start_1d(local_config_data, CONFIG_LUT,
                            CONFIG_SIZE_LUT * sizeof(uint32_t));
        snrt_dma_wait_all();

        local_config_data = (int32_t *)(snrt_l1_next() + delta_config_data);
        snrt_dma_start_1d(local_config_data, CONFIG_CONST,
                            CONFIG_SIZE_DATA * sizeof(uint32_t));
        snrt_dma_wait_all();

        if (CONFIG_SIZE_CMD_0 != 0) {
            local_config_data =
                (int32_t *)(snrt_l1_next() + delta_config_cmd_0);
            snrt_dma_start_1d(local_config_data, CONFIG_CMD,
                                CONFIG_SIZE_CMD_0 * sizeof(uint32_t));
            snrt_dma_wait_all();
        }

        local_config_data =
            (int32_t *)(snrt_l1_next() + delta_config_cmd_ss);
        snrt_dma_start_1d(local_config_data, CONFIG_CMD_SS,
                            CONFIG_SIZE_CMD_SS * sizeof(uint32_t));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    local_config_data = (int32_t *)(snrt_l1_next() + delta_comp_data);
    // Using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_config_data, COMP_DATA,
                            COMP_DATA_SIZE * sizeof(uint32_t));
        snrt_dma_wait_all();
        if (debug_mode) {
            printf("DMA transfer cycle from DMA hardware counter %d \r\n",
                    snrt_get_perf_counter(SNRT_PERF_CNT0));
        }
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
    }
    snrt_cluster_hw_barrier();

    // testing csr -> cgra
    if (snrt_is_compute_core()) {
        // printf ("hello world!\r\n\r\n");
        uint32_t mcycle_timestamps[7];

        // launch_cgra_0(delta_config_data, delta_comp_data,
        // delta_store_data, mcycle_timestamps);

        // launch_cgra_0_compact(delta_config_data, delta_comp_data,
        // delta_store_data, mcycle_timestamps);

        launch_cgra_0_config(delta_config_data, delta_comp_data,
                                delta_store_data, mcycle_timestamps);
        launch_cgra_0_go(delta_config_data, delta_comp_data,
                            delta_store_data, mcycle_timestamps);

        // cgra_hw_barrier(10, 1e5, 1, 1);
        cgra_hw_barrier_fast(10, 1e5);

        cgra_hw_profiler();

        // fast run again, without re-load cfg
        launch_cgra_0_relaunch(mcycle_timestamps);

        // cgra_hw_barrier(10, 1e5, 1, 1);
        cgra_hw_barrier_fast(10, 1e5);

        // cgra_hw_profiler();

        printf("mcycle cgra_init = %d\r\n", mcycle_timestamps[1] - mcycle_timestamps[0]);
        printf("mcycle cgra_config_prep = %d\r\n", mcycle_timestamps[2] - mcycle_timestamps[1]);
        printf("mcycle cgra_data_prep = %d\r\n", mcycle_timestamps[3] - mcycle_timestamps[2]);
        printf("mcycle cgra_launch_init = %d\r\n", mcycle_timestamps[4] - mcycle_timestamps[3]);
        printf("mcycle cgra_relaunch_init = %d\r\n", mcycle_timestamps[5] -  mcycle_timestamps[4]);
        printf("mcycle cgra_relaunch_done = %d\r\n", mcycle_timestamps[6] - mcycle_timestamps[5]);

        return 0;
    } else
        return 0;
}
