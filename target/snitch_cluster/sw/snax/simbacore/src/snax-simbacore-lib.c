// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
// Robin Geens <robin.geens@esat.kuleuven.be>

#include "snax-simbacore-lib.h"
#include <stdbool.h>
#include "snrt.h"
#include "streamer_csr_addr_map.h"

void set_simbacore_oscore_streamer_csr(uint32_t ptr_a, int32_t* Aslstride, int32_t* Atlbound, int32_t* Atlstride,
                                       int32_t set_addr_remap_index_A, int32_t* channel_en_A,

                                       uint32_t ptr_b, int32_t* Bslstride, int32_t* Btlbound, int32_t* Btlstride,
                                       int32_t set_addr_remap_index_B, int32_t* channel_en_B,

                                       uint32_t ptr_d, int32_t* Dslstride, int32_t* Dtlbound, int32_t* Dtlstride,
                                       int32_t set_addr_remap_index_D, int32_t* channel_en_D) {
    // -------------------
    // A streamer setting
    // -------------------
    // base ptr for A
    csrw_ss(BASE_PTR_READER_0_LOW, ptr_a);

    // spatial strides for A
    for (int i = 0; i < S_STRIDE_NUM_READER_0; i++) {
        csrw_ss(S_STRIDE_BASE_READER_0 + i, Aslstride[i]);
    }

    // loop bounds, from innermost to outermost, for data mover A
    for (int i = 0; i < T_BOUND_NUM_READER_0; i++) {
        csrw_ss(T_BOUND_BASE_READER_0 + i, Atlbound[i]);
    }

    // temporal strides for A
    for (int i = 0; i < T_STRIDE_NUM_READER_0; i++) {
        csrw_ss(T_STRIDE_BASE_READER_0 + i, Atlstride[i]);
    }

    // set the address remap index for A
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, set_addr_remap_index_A);
#endif

    // set the channel enable
#ifdef ENABLED_CHANNEL_READER_0
    for (int i = 0; i < ENABLED_CHANNEL_READER_0_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_0 + i, channel_en_A[i]);
    }
#endif

    // -------------------
    // B streamer setting
    // -------------------
    // base ptr for B
    csrw_ss(BASE_PTR_READER_1_LOW, ptr_b);

    // spatial strides for B
    for (int i = 0; i < S_STRIDE_NUM_READER_1; i++) {
        csrw_ss(S_STRIDE_BASE_READER_1 + i, Bslstride[i]);
    }

    // loop bounds, from innermost to outermost, for data mover B
    for (int i = 0; i < T_BOUND_NUM_READER_1; i++) {
        csrw_ss(T_BOUND_BASE_READER_1 + i, Btlbound[i]);
    }

    // temporal strides for B
    for (int i = 0; i < T_STRIDE_NUM_READER_1; i++) {
        csrw_ss(T_STRIDE_BASE_READER_1 + i, Btlstride[i]);
    }

    // set the address remap index for B
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, set_addr_remap_index_B);
#endif

    // set the channel enable
#ifdef ENABLED_CHANNEL_READER_1
    for (int i = 0; i < ENABLED_CHANNEL_READER_1_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_1 + i, channel_en_B[i]);
    }
#endif

    // -------------------
    // D streamer setting
    // -------------------
    // base ptr for D
    csrw_ss(BASE_PTR_WRITER_0_LOW, ptr_d);

    // spatial strides for D
    for (int i = 0; i < S_STRIDE_NUM_WRITER_0; i++) {
        csrw_ss(S_STRIDE_BASE_WRITER_0 + i, Dslstride[i]);
    }

    // for D, from N to M

    for (int i = 0; i < T_BOUND_NUM_WRITER_0; i++) {
        csrw_ss(T_BOUND_BASE_WRITER_0 + i, Dtlbound[i]);
    }

    // temporal strides for D
    for (int i = 0; i < T_STRIDE_NUM_WRITER_0; i++) {
        csrw_ss(T_STRIDE_BASE_WRITER_0 + i, Dtlstride[i]);
    }

    // set the address remap index for D
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, set_addr_remap_index_D);
#endif

    // set the channel enable
#ifdef ENABLED_CHANNEL_WRITER_0
    for (int i = 0; i < ENABLED_CHANNEL_WRITER_0_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_WRITER_0 + i, channel_en_D[i]);
    }
#endif
}

void set_simbacore_csr(uint32_t mode, uint32_t seqLen, uint32_t dModel, uint32_t dInner, uint32_t dtRank) {
    csrw_ss(MODE, mode);
    csrw_ss(SEQ_LEN, seqLen);
    csrw_ss(D_MODEL, dModel);
    csrw_ss(D_INNER, dInner);
    csrw_ss(DT_RANK, dtRank);
}

// Stall until Streamer and GEMM accelerator finish
void wait_simbacore_and_streamer() {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(SIMBACORE_START, 0);
    while (csrr_ss(SIMBACORE_BUSY));
    while (csrr_ss(STREAMER_BUSY_CSR));
}

void wait_simbacore() {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    while (csrr_ss(SIMBACORE_BUSY));
}

// Read performance counter of the Streamer, a read-only CSR
uint32_t read_simbacore_oscore_streamer_perf_counter() {
    uint32_t perf_counter = csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
    return perf_counter;
}

// Read performance counter of GEMM, a read-only CSR
uint32_t read_simbacore_perf_counter() {
    uint32_t perf_counter = csrr_ss(SIMBACORE_PERFORMANCE_COUNTER);
    return perf_counter;
}

// Check result, word-by-word. data_length in bytes
uint32_t check_simbacore_result_D(uint16_t* output, uint16_t* output_golden, int32_t data_length,
                                  bool banked_data_layout) {
    uint32_t err = 0;
    int32_t num_elements = data_length / sizeof(uint16_t);
    printf("Sanity check 4\n");
    printf("Start checking results. data_length: %d bytes (%d elements)\n", data_length, num_elements);

    if (banked_data_layout) {
        for (int i = 0; i < num_elements / 16; i += 1) {
            for (int j = 0; j < 16; j++) {
                if (*(output + i * (256 / (4 * sizeof(uint16_t))) + j) != output_golden[i * 16 + j]) {
                    err++;
                }
            }
        }
    } else {
        printf("Right before the loop\n");
        for (int i = 0; i < num_elements; i++) {
            printf("Loop iteration: %d\n", i);
            printf("%d\n", output[i]);
            printf("%d\n", output_golden[i]);
            if (output[i] != output_golden[i]) {
                err++;
                printf("Unequals. output[%d] = %d, output_golden[%d] = %d\n", i, output[i], i, output_golden[i]);
            } else {
                printf("pass: output[%d] = %d, output_golden[%d] = %d\n", i, output[i], i, output_golden[i]);
            }
        }
    }

    return err;
}
