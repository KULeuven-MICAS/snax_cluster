// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Jonas Crols <jonas.crols@student.kuleuven.be>
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#ifndef READER_EXT_ELEMENTWISEADDBIT32
#error \
    "READER_EXT_ELEMENTWISEADDBIT32 is missing. Regenerate the XDMA CSR map with HasElementwiseAdd enabled."
#endif

static uint32_t align_up(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

int main() {
    int err = 0;
    uint32_t tcdm_baseaddress = snrt_cluster_base_addrl();
    uint32_t tcdm_input_stride = align_up(input_bytes, XDMA_WIDTH);
    uint8_t* tcdm_in1 = (uint8_t*)tcdm_baseaddress;
    uint8_t* tcdm_in2 = tcdm_in1 + tcdm_input_stride;
    uint8_t* tcdm_out = tcdm_in2 + tcdm_input_stride;

    if (snrt_is_dm_core()) {
        printf(
            "[ElementwiseAdd] M=%u, N=%u, padded_N=%u, elements=%u, "
            "tiles=%u\n",
            matrix_m, matrix_n, matrix_padded_n, matrix_elements, tile_count);

        if (tcdm_input_stride != input_bytes_aligned) {
            printf(
                "[ElementwiseAdd] Generated input stride %u does not match "
                "runtime aligned stride %u\n",
                input_bytes_aligned, tcdm_input_stride);
            err++;
        }

        snrt_dma_start_1d(tcdm_in1, input_matrix1, input_bytes);
        snrt_dma_wait_all();

        snrt_dma_start_1d(tcdm_in2, input_matrix2, input_bytes);
        snrt_dma_wait_all();

        uint32_t ext_param[1] = {2};
        if (snax_xdma_enable_src_ext(READER_EXT_ELEMENTWISEADDBIT32,
                                     ext_param) != 0) {
            printf(
                "[ElementwiseAdd] Failed to enable reader elementwise add\n");
            err++;
        }

        if (err == 0 &&
            snax_xdma_memcpy_nd(tcdm_in1, tcdm_out, spatial_stride_src,
                                spatial_stride_dst, temporal_dimension_src,
                                temporal_strides_src, temporal_bounds_src,
                                temporal_dimension_dst, temporal_strides_dst,
                                temporal_bounds_dst, 0xFFFFFFFF, 0xFFFFFFFF,
                                0xFFFFFFFF) != 0) {
            printf("[ElementwiseAdd] Failed to configure XDMA memcpy\n");
            err++;
        }

        if (err == 0) {
            int task_id = snax_xdma_start();
            snax_xdma_local_wait(task_id);
            printf("[ElementwiseAdd] xdma task %d finished in %d cycles\n",
                   task_id, snax_xdma_last_task_cycle());

            uint32_t* tcdm_result = (uint32_t*)tcdm_out;
            const uint32_t* golden_result =
                (const uint32_t*)golden_output_matrix;
            uint32_t mismatch_count = 0;

            for (uint32_t i = 0; i < matrix_elements; i++) {
                if (tcdm_result[i] != golden_result[i]) {
                    if (mismatch_count < 8) {
                        printf(
                            "[ElementwiseAdd] Mismatch at element %u "
                            "(byte %u): got 0x%08x expected 0x%08x\n",
                            i, i * (uint32_t)sizeof(uint32_t), tcdm_result[i],
                            golden_result[i]);
                    }
                    mismatch_count++;
                }
            }

            if (mismatch_count == 0) {
                printf("[ElementwiseAdd] All values are correct\n");
            } else {
                printf("[ElementwiseAdd] Failed with %u mismatches\n",
                       mismatch_count);
                err++;
            }
        }

        snax_xdma_disable_src_ext(READER_EXT_ELEMENTWISEADDBIT32);
    }

    return err != 0;
}
