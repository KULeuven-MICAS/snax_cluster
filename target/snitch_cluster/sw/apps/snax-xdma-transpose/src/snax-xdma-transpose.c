// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "data.h"
#include "snax-core-roles.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if defined(READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16)
#define READER_TRANSPOSE_EXT_ID READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
#elif defined(READER_EXT_TRANSPOSERROW8_8_8COL8_8_8BIT8_16_32)
#define READER_TRANSPOSE_EXT_ID READER_EXT_TRANSPOSERROW8_8_8COL8_8_8BIT8_16_32
#elif defined(READER_EXT_TRANSPOSERROW8_8_4COL8_8_4BIT8_16_32)
#define READER_TRANSPOSE_EXT_ID READER_EXT_TRANSPOSERROW8_8_4COL8_8_4BIT8_16_32
#endif

static uint32_t align_up(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

// One transpose case, on the xDMA core. Returns 1 if the case failed.
//
// A FUNCTION rather than the loop body it used to be: the body bails out early
// when this build lacks the extension a case needs, and as a `continue` inside
// the loop that early exit would skip the trailing barrier its peers still
// execute -- deadlocking the next iteration. Returning keeps every hart's
// barrier count identical whatever the case does.
static int run_transpose_case(uint32_t case_idx,
                              transpose_test_case_t* test_case,
                              uint8_t* tcdm_in, uint8_t* tcdm_out) {
    int case_err = 0;

    printf(
        "[Transpose] Running case %u (%s), M=%u, N=%u, BIT_WIDTH=%u, "
        "transpose=%u\n",
        case_idx, test_case->name, test_case->M, test_case->N,
        test_case->bit_width, test_case->enable_transpose);

    // --------------------- Configure the Ext / Helper
    // --------------------- //
    if (test_case->use_row_major_transpose_helper) {
        if (snax_xdma_row_major_transpose(tcdm_in, tcdm_out,
                                          test_case->M, test_case->N,
                                          test_case->bit_width) != 0) {
            printf(
                "[Transpose] Failed to configure row-major transpose "
                "helper\n");
            return 1;
        }
    } else {
#ifdef READER_TRANSPOSE_EXT_ID
        if (test_case->enable_transpose) {
            if (snax_xdma_enable_src_ext(READER_TRANSPOSE_EXT_ID,
                                         test_case->transposer_csr) !=
                0) {
                printf(
                    "[Transpose] Failed to enable reader transposer\n");
                return 1;
            }
        } else if (snax_xdma_disable_src_ext(READER_TRANSPOSE_EXT_ID) !=
                   0) {
            printf("[Transpose] Failed to disable reader transposer\n");
            return 1;
        }
#else
        if (test_case->enable_transpose) {
            printf(
                "[Transpose] Reader transposer is not available in "
                "this build\n");
            return 1;
        }
#endif

        // --------------------- Configure the AGU ---------------------
        // //
        if (snax_xdma_memcpy_nd(
                tcdm_in, tcdm_out, test_case->spatial_stride_src,
                test_case->spatial_stride_dst,
                test_case->temporal_dimension_src,
                test_case->temporal_strides_src,
                test_case->temporal_bounds_src,
                test_case->temporal_dimension_dst,
                test_case->temporal_strides_dst,
                test_case->temporal_bounds_dst, 0xFFFFFFFF, 0xFFFFFFFF,
                0xFFFFFFFF) != 0) {
            printf("[Transpose] Failed to configure XDMA memcpy\n");
            return 1;
        }
    }

    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    printf("[Transpose] xdma task %d finished in %d cycles\n", task_id,
           snax_xdma_last_task_cycle());

    // --------------------- Checking the Results ---------------------
    // //
    for (uint32_t byte_idx = 0; byte_idx < test_case->output_bytes;
         byte_idx++) {
        if (tcdm_out[byte_idx] !=
            test_case->golden_output_bytes[byte_idx]) {
            if (case_err < 8) {
                printf(
                    "[Transpose] Mismatch in case %s at byte %u: got "
                    "0x%02x expected 0x%02x\n",
                    test_case->name, byte_idx, tcdm_out[byte_idx],
                    test_case->golden_output_bytes[byte_idx]);
            }
            case_err++;
        }
    }

    if (case_err == 0) {
        printf("[Transpose] Case %s passed\n", test_case->name);
        return 0;
    }
    printf("[Transpose] Case %s failed with %d byte mismatches\n",
           test_case->name, case_err);
    return 1;
}

int main() {
    int err = 0;
    uint32_t tcdm_baseaddress = snrt_cluster_base_addrl();
    uint8_t* tcdm_in = (uint8_t*)tcdm_baseaddress;
    uint8_t* tcdm_out = tcdm_in + align_up(max_case_input_bytes, XDMA_WIDTH);

    for (uint32_t case_idx = 0; case_idx < transpose_test_case_count;
         case_idx++) {
        transpose_test_case_t* test_case = &transpose_test_cases[case_idx];

        // Stage THIS case's input on the core that owns the iDMA.
        if (snax_is_idma_core()) {
            snrt_dma_start_1d(tcdm_in, test_case->input_matrix_bytes,
                              test_case->input_bytes);
            snrt_dma_wait_all();
        }
        // Both barriers sit outside the engine guard, and every hart reaches
        // both. The trailing one is not optional: without it the iDMA core
        // would overwrite tcdm_in with the next case's input while the xDMA
        // core is still reading it for this one.
        snrt_cluster_hw_barrier();

        if (snax_is_xdma_core())
            err += run_transpose_case(case_idx, test_case, tcdm_in, tcdm_out);

        snrt_cluster_hw_barrier();
    }

    if (snax_is_xdma_core()) {
#ifdef READER_TRANSPOSE_EXT_ID
        snax_xdma_disable_src_ext(READER_TRANSPOSE_EXT_ID);
#endif

        if (err == 0) {
            printf("[Transpose] All cases passed\n");
        } else {
            printf("[Transpose] Encountered %d failing cases\n", err);
        }
    }

    return err != 0;
}
