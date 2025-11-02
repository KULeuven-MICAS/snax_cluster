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
    write_csr(BASE_PTR_READER_0_LOW, ptr_a);

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
    write_csr(ADDR_REMAP_INDEX_READER_0, set_addr_remap_index_A);
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
    write_csr(BASE_PTR_READER_1_LOW, ptr_b);

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
    write_csr(ADDR_REMAP_INDEX_READER_1, set_addr_remap_index_B);
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
    write_csr(BASE_PTR_WRITER_0_LOW, ptr_d);

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
    write_csr(ADDR_REMAP_INDEX_WRITER_0, set_addr_remap_index_D);
#endif

    // set the channel enable
#ifdef ENABLED_CHANNEL_WRITER_0
    for (int i = 0; i < ENABLED_CHANNEL_WRITER_0_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_WRITER_0 + i, channel_en_D[i]);
    }
#endif
}

void set_simbacore_csr(uint32_t mode, uint32_t seqLen, uint32_t dModel, uint32_t dInner, uint32_t dtRank) {
    write_csr(MODE, mode);
    write_csr(SEQ_LEN, seqLen);
    write_csr(D_MODEL, dModel);
    write_csr(D_INNER, dInner);
    write_csr(DT_RANK, dtRank);
}

// Stall until Streamer and GEMM accelerator finish
void wait_simbacore_and_streamer() {
    printf("Waiting for Streamer and SimbaCore to finish...\n");
    write_csr(STREAMER_START_CSR, 0);
    write_csr(STREAMER_START_CSR, 0);
    write_csr(SIMBACORE_START, 0);
    while (read_csr(SIMBACORE_BUSY));  // 1185 = 0x4a1
    printf("SimbaCore has finished. Polling Streamer...\n");
    while (read_csr(STREAMER_BUSY_CSR));  // 1177 = 0x499
    printf("Streamer and SimbaCore have finished\n");
}

void wait_simbacore() {
    printf("Waiting for SimbaCore to finish...\n");
    write_csr(SIMBACORE_START, 0);
    write_csr(SIMBACORE_START, 0);
    while (read_csr(SIMBACORE_BUSY));
    printf("SimbaCore has finished\n");
}

// Read performance counter of the Streamer, a read-only CSR
uint32_t read_streamer_perf_counter() {
    uint32_t perf_counter = read_csr(STREAMER_PERFORMANCE_COUNTER_CSR);
    return perf_counter;
}

// Read performance counter of GEMM, a read-only CSR
uint32_t read_simbacore_perf_counter() {
    uint32_t perf_counter = read_csr(SIMBACORE_PERFORMANCE_COUNTER);
    return perf_counter;
}

// Check result, word-by-word. data_length in bytes
uint32_t check_simbacore_result_D(uint16_t* output, uint16_t* output_golden, int32_t data_length,
                                  bool banked_data_layout) {
    uint32_t err = 0;
    int32_t num_elements = data_length / sizeof(uint16_t);
    printf("Checking results: %d bytes (%d elements)\n", data_length, num_elements);

    // TODO most likely incorrect
    if (banked_data_layout) {
        for (int i = 0; i < num_elements / 16; i += 1) {
            for (int j = 0; j < 16; j++) {
                if (*(output + i * (256 / (4 * sizeof(uint16_t))) + j) != output_golden[i * 16 + j]) {
                    err++;
                }
            }
        }
    } else {
        for (int i = 0; i < num_elements; i++) {
            if (output[i] != output_golden[i]) {
                err++;
                printf("FAIL out[%d] = %d, ref[%d] = %d\n", i, output[i], i, output_golden[i]);
            } else {
                printf("PASS out[%d] = %d, ref[%d] = %d\n", i, output[i], i, output_golden[i]);
            }
        }
    }

    return err;
}
