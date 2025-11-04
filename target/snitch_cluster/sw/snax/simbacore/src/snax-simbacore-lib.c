// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
// Robin Geens <robin.geens@esat.kuleuven.be>

#include "snax-simbacore-lib.h"
#include <stdbool.h>
#include "streamer_csr_addr_map.h"

void set_simbacore_osgemm_streamer_csr(uint32_t ptr_a, int32_t* Aslstride, int32_t* Atlbound, int32_t* Atlstride,
                                       uint32_t channel_en_A,  //
                                       uint32_t ptr_b, int32_t* Bslstride, int32_t* Btlbound, int32_t* Btlstride,
                                       uint32_t channel_en_B,  //
                                       uint32_t ptr_d, int32_t* Dslstride, int32_t* Dtlbound, int32_t* Dtlstride,
                                       uint32_t channel_en_D) {
    // -------------------
    // A streamer setting
    // -------------------
    write_csr(BASE_PTR_READER_0_LOW, ptr_a);

    for (int i = 0; i < S_STRIDE_NUM_READER_0; i++)  // spatial strides
        csrw_ss(S_STRIDE_BASE_READER_0 + i, Aslstride[i]);

    for (int i = 0; i < T_BOUND_NUM_READER_0; i++)  // temporal bounds
        csrw_ss(T_BOUND_BASE_READER_0 + i, Atlbound[i]);

    for (int i = 0; i < T_STRIDE_NUM_READER_0; i++)  // temporal strides
        csrw_ss(T_STRIDE_BASE_READER_0 + i, Atlstride[i]);

    write_csr(ENABLED_CHANNEL_READER_0, channel_en_A);

    // -------------------
    // B streamer setting
    // -------------------
    write_csr(BASE_PTR_READER_1_LOW, ptr_b);

    for (int i = 0; i < S_STRIDE_NUM_READER_1; i++)  // spatial strides
        csrw_ss(S_STRIDE_BASE_READER_1 + i, Bslstride[i]);

    for (int i = 0; i < T_BOUND_NUM_READER_1; i++)  // temporal bounds
        csrw_ss(T_BOUND_BASE_READER_1 + i, Btlbound[i]);

    for (int i = 0; i < T_STRIDE_NUM_READER_1; i++)  // temporal strides
        csrw_ss(T_STRIDE_BASE_READER_1 + i, Btlstride[i]);

    write_csr(ENABLED_CHANNEL_READER_1, channel_en_B);

    // -------------------
    // D streamer setting
    // -------------------
    write_csr(BASE_PTR_WRITER_0_LOW, ptr_d);

    for (int i = 0; i < S_STRIDE_NUM_WRITER_0; i++)  // spatial strides
        csrw_ss(S_STRIDE_BASE_WRITER_0 + i, Dslstride[i]);

    for (int i = 0; i < T_BOUND_NUM_WRITER_0; i++)  // temporal bounds
        csrw_ss(T_BOUND_BASE_WRITER_0 + i, Dtlbound[i]);

    for (int i = 0; i < T_STRIDE_NUM_WRITER_0; i++)  // temporal strides
        csrw_ss(T_STRIDE_BASE_WRITER_0 + i, Dtlstride[i]);

    write_csr(ENABLED_CHANNEL_WRITER_0, channel_en_D);
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
uint32_t check_OSGeMM_result_all(uint16_t* output, uint16_t* output_golden, int32_t data_length) {
    uint32_t err = 0;
    int32_t num_elements = data_length / sizeof(uint16_t);
    printf("Checking results: %d bytes (%d elements)\n", data_length, num_elements);

    for (int i = 0; i < num_elements; i++) {
        uint16_t output_value = output[i];
        uint16_t golden_value = output_golden[i];
        if (output_value != golden_value) {
            err++;
            printf("FAIL out[%d] = %d,\tref = %d\n", i, output_value, golden_value);
        } else {
            printf("PASS out[%d] = %d,\tref = %d\n", i, output_value, golden_value);
        }
    }
    return err;
}

// Check some samples of ther result to speed up verification
uint32_t check_OSGeMM_result_sample(uint16_t* output, uint16_t* output_golden, int32_t* sample_indices,
                                    int32_t test_sample_count) {
    uint32_t err = 0;
    printf("Checking results: sampling %d elements\n", test_sample_count);

    for (int i = 0; i < test_sample_count; i++) {
        int sample_index = sample_indices[i];
        uint16_t output_value = output[sample_index];
        uint16_t golden_value = output_golden[sample_index];
        if (output_value != golden_value) {
            err++;
            printf("FAIL out[%d] = %d,\tref = %d\n", sample_index, output_value, golden_value);
        } else {
            printf("PASS out[%d] = %d,\tref = %d\n", sample_index, output_value, golden_value);
        }
    }
    return err;
}
