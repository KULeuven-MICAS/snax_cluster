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

void set_streamer_csr(

    uint32_t ptr_R0, int32_t* R0slstride, int32_t* R0tlbound, int32_t* R0tlstride, uint32_t enable_R0,  // osCore in
    uint32_t ptr_R1, int32_t* R1slstride, int32_t* R1tlbound, int32_t* R1tlstride, uint32_t enable_R1,  // oscore weight
    // uint32_t ptr_R2, int32_t* R2slstride, int32_t* R2tlbound, int32_t* R2tlstride, uint32_t enable_R2,  // switchCore
    // in
    uint32_t ptr_R3, int32_t* R3slstride, int32_t* R3tlbound, int32_t* R3tlstride,
    uint32_t enable_R3,  // switchCore weight
    uint32_t ptr_R4, int32_t* R4slstride, int32_t* R4tlbound, int32_t* R4tlstride,
    uint32_t enable_R4,  // switchCore bias
    // uint32_t ptr_R5, int32_t* R5slstride, int32_t* R5tlbound, int32_t* R5tlstride, uint32_t enable_R5,  // switchCore
    // matmul weight
    // uint32_t ptr_R6, int32_t* R6slstride, int32_t* R6tlbound, int32_t* R6tlstride, uint32_t enable_R6,
    // // SUC A
    //  uint32_t ptr_R7, int32_t* R7slstride, int32_t* R7tlbound, int32_t* R7tlstride, uint32_t enable_R7,  //
    // SUC BC uint32_t ptr_R8, int32_t* R8slstride, int32_t* R8tlbound, int32_t* R8tlstride, uint32_t enable_R8,  // SUC
    // D
    // uint32_t ptr_R9, int32_t* R9slstride, int32_t* R9tlbound, int32_t* R9tlstride, uint32_t enable_R9,  // SUC x
    // uint32_t ptr_R10, int32_t* R10slstride, int32_t* R10tlbound, int32_t* R10tlstride, uint32_t enable_R10,  // SUC z
    // uint32_t ptr_R11, int32_t* R11slstride, int32_t* R11tlbound, int32_t* R11tlstride, uint32_t enable_R11,  //
    // iscore in
    uint32_t ptr_R12, int32_t* R12slstride, int32_t* R12tlbound, int32_t* R12tlstride,
    uint32_t enable_R12,  // isCore weight
    uint32_t ptr_R13, int32_t* R13slstride, int32_t* R13tlbound, int32_t* R13tlstride,
    uint32_t enable_R13,  // isCore psum

    // uint32_t ptr_W0, int32_t* W0slstride, int32_t* W0tlbound, int32_t* W0tlstride, uint32_t enable_W0,  // osCore out
    uint32_t ptr_W1, int32_t* W1slstride, int32_t* W1tlbound, int32_t* W1tlstride,
    uint32_t enable_W1,  // switchCore out
    // uint32_t ptr_W2, int32_t* W2slstride, int32_t* W2tlbound, int32_t* W2tlstride, uint32_t enable_W2,  // SUC out
    uint32_t ptr_W3, int32_t* W3slstride, int32_t* W3tlbound, int32_t* W3tlstride, uint32_t enable_W3  // isCore out
) {
    if (enable_R0) {
        write_csr(BASE_PTR_READER_0_LOW, ptr_R0);
        for (int i = 0; i < S_STRIDE_NUM_READER_0; i++) csrw_ss(S_STRIDE_BASE_READER_0 + i, R0slstride[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_0; i++) csrw_ss(T_BOUND_BASE_READER_0 + i, R0tlbound[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_0; i++) csrw_ss(T_STRIDE_BASE_READER_0 + i, R0tlstride[i]);
        write_csr(ENABLED_CHANNEL_READER_0, enable_R0);
    }

    if (enable_R1) {
        write_csr(BASE_PTR_READER_1_LOW, ptr_R1);
        for (int i = 0; i < S_STRIDE_NUM_READER_1; i++) csrw_ss(S_STRIDE_BASE_READER_1 + i, R1slstride[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_1; i++) csrw_ss(T_BOUND_BASE_READER_1 + i, R1tlbound[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_1; i++) csrw_ss(T_STRIDE_BASE_READER_1 + i, R1tlstride[i]);
        write_csr(ENABLED_CHANNEL_READER_1, enable_R1);
    }

    // if (enable_R2) {
    //     write_csr(BASE_PTR_READER_2_LOW, ptr_R2);
    //     for (int i = 0; i < S_STRIDE_NUM_READER_2; i++) csrw_ss(S_STRIDE_BASE_READER_2 + i, R2slstride[i]);
    //     for (int i = 0; i < T_BOUND_NUM_READER_2; i++) csrw_ss(T_BOUND_BASE_READER_2 + i, R2tlbound[i]);
    //     for (int i = 0; i < T_STRIDE_NUM_READER_2; i++) csrw_ss(T_STRIDE_BASE_READER_2 + i, R2tlstride[i]);
    //     write_csr(ENABLED_CHANNEL_READER_2, enable_R2);
    // }

    if (enable_R3) {
        write_csr(BASE_PTR_READER_3_LOW, ptr_R3);
        for (int i = 0; i < S_STRIDE_NUM_READER_3; i++) csrw_ss(S_STRIDE_BASE_READER_3 + i, R3slstride[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_3; i++) csrw_ss(T_BOUND_BASE_READER_3 + i, R3tlbound[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_3; i++) csrw_ss(T_STRIDE_BASE_READER_3 + i, R3tlstride[i]);
        write_csr(ENABLED_CHANNEL_READER_3, enable_R3);
    }

    if (enable_R4) {
        write_csr(BASE_PTR_READER_4_LOW, ptr_R4);
        for (int i = 0; i < S_STRIDE_NUM_READER_4; i++) csrw_ss(S_STRIDE_BASE_READER_4 + i, R4slstride[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_4; i++) csrw_ss(T_BOUND_BASE_READER_4 + i, R4tlbound[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_4; i++) csrw_ss(T_STRIDE_BASE_READER_4 + i, R4tlstride[i]);
        write_csr(ENABLED_CHANNEL_READER_4, enable_R4);
    }

    // if (enable_R5) {
    //     write_csr(BASE_PTR_READER_5_LOW, ptr_R5);
    //     for (int i = 0; i < S_STRIDE_NUM_READER_5; i++) csrw_ss(S_STRIDE_BASE_READER_5 + i, R5slstride[i]);
    //     for (int i = 0; i < T_BOUND_NUM_READER_5; i++) csrw_ss(T_BOUND_BASE_READER_5 + i, R5tlbound[i]);
    //     for (int i = 0; i < T_STRIDE_NUM_READER_5; i++) csrw_ss(T_STRIDE_BASE_READER_5 + i, R5tlstride[i]);
    //     write_csr(ENABLED_CHANNEL_READER_5, enable_R5);
    // }

    // if (enable_R6) {
    //     write_csr(BASE_PTR_READER_6_LOW, ptr_R6);
    //     for (int i = 0; i < S_STRIDE_NUM_READER_6; i++) csrw_ss(S_STRIDE_BASE_READER_6 + i, R6slstride[i]);
    //     for (int i = 0; i < T_BOUND_NUM_READER_6; i++) csrw_ss(T_BOUND_BASE_READER_6 + i, R6tlbound[i]);
    //     for (int i = 0; i < T_STRIDE_NUM_READER_6; i++) csrw_ss(T_STRIDE_BASE_READER_6 + i, R6tlstride[i]);
    //     write_csr(ENABLED_CHANNEL_READER_6, enable_R6);
    // }

    // if (enable_R7) {
    //     write_csr(BASE_PTR_READER_7_LOW, ptr_R7);
    //     for (int i = 0; i < S_STRIDE_NUM_READER_7; i++) csrw_ss(S_STRIDE_BASE_READER_7 + i, R7slstride[i]);
    //     for (int i = 0; i < T_BOUND_NUM_READER_7; i++) csrw_ss(T_BOUND_BASE_READER_7 + i, R7tlbound[i]);
    //     for (int i = 0; i < T_STRIDE_NUM_READER_7; i++) csrw_ss(T_STRIDE_BASE_READER_7 + i, R7tlstride[i]);
    //     write_csr(ENABLED_CHANNEL_READER_7, enable_R7);
    // }

    // if (enable_R8) {
    //     write_csr(BASE_PTR_READER_8_LOW, ptr_R8);
    //     for (int i = 0; i < S_STRIDE_NUM_READER_8; i++) csrw_ss(S_STRIDE_BASE_READER_8 + i, R8slstride[i]);
    //     for (int i = 0; i < T_BOUND_NUM_READER_8; i++) csrw_ss(T_BOUND_BASE_READER_8 + i, R8tlbound[i]);
    //     for (int i = 0; i < T_STRIDE_NUM_READER_8; i++) csrw_ss(T_STRIDE_BASE_READER_8 + i, R8tlstride[i]);
    //     write_csr(ENABLED_CHANNEL_READER_8, enable_R8);
    // }

    // if (enable_R9) {
    //     write_csr(BASE_PTR_READER_9_LOW, ptr_R9);
    //     for (int i = 0; i < S_STRIDE_NUM_READER_9; i++) csrw_ss(S_STRIDE_BASE_READER_9 + i, R9slstride[i]);
    //     for (int i = 0; i < T_BOUND_NUM_READER_9; i++) csrw_ss(T_BOUND_BASE_READER_9 + i, R9tlbound[i]);
    //     for (int i = 0; i < T_STRIDE_NUM_READER_9; i++) csrw_ss(T_STRIDE_BASE_READER_9 + i, R9tlstride[i]);
    //     write_csr(ENABLED_CHANNEL_READER_9, enable_R9);
    // }

    // if (enable_R10) {
    //     write_csr(BASE_PTR_READER_10_LOW, ptr_R10);
    //     for (int i = 0; i < S_STRIDE_NUM_READER_10; i++) csrw_ss(S_STRIDE_BASE_READER_10 + i, R10slstride[i]);
    //     for (int i = 0; i < T_BOUND_NUM_READER_10; i++) csrw_ss(T_BOUND_BASE_READER_10 + i, R10tlbound[i]);
    //     for (int i = 0; i < T_STRIDE_NUM_READER_10; i++) csrw_ss(T_STRIDE_BASE_READER_10 + i, R10tlstride[i]);
    //     write_csr(ENABLED_CHANNEL_READER_10, enable_R10);
    // }

    // if (enable_R11) {
    //     write_csr(BASE_PTR_READER_11_LOW, ptr_R11);
    //     for (int i = 0; i < S_STRIDE_NUM_READER_11; i++) csrw_ss(S_STRIDE_BASE_READER_11 + i, R11slstride[i]);
    //     for (int i = 0; i < T_BOUND_NUM_READER_11; i++) csrw_ss(T_BOUND_BASE_READER_11 + i, R11tlbound[i]);
    //     for (int i = 0; i < T_STRIDE_NUM_READER_11; i++) csrw_ss(T_STRIDE_BASE_READER_11 + i, R11tlstride[i]);
    //     write_csr(ENABLED_CHANNEL_READER_11, enable_R11);
    // }

    if (enable_R12) {
        write_csr(BASE_PTR_READER_12_LOW, ptr_R12);
        for (int i = 0; i < S_STRIDE_NUM_READER_12; i++) csrw_ss(S_STRIDE_BASE_READER_12 + i, R12slstride[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_12; i++) csrw_ss(T_BOUND_BASE_READER_12 + i, R12tlbound[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_12; i++) csrw_ss(T_STRIDE_BASE_READER_12 + i, R12tlstride[i]);
        write_csr(ENABLED_CHANNEL_READER_12, enable_R12);
    }

    if (enable_R13) {
        write_csr(BASE_PTR_READER_13_LOW, ptr_R13);
        for (int i = 0; i < S_STRIDE_NUM_READER_13; i++) csrw_ss(S_STRIDE_BASE_READER_13 + i, R13slstride[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_13; i++) csrw_ss(T_BOUND_BASE_READER_13 + i, R13tlbound[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_13; i++) csrw_ss(T_STRIDE_BASE_READER_13 + i, R13tlstride[i]);
        write_csr(ENABLED_CHANNEL_READER_13, enable_R13);
    }

    // if (enable_W0) {
    //     write_csr(BASE_PTR_WRITER_0_LOW, ptr_W0);
    //     for (int i = 0; i < S_STRIDE_NUM_WRITER_0; i++) csrw_ss(S_STRIDE_BASE_WRITER_0 + i, W0slstride[i]);
    //     for (int i = 0; i < T_BOUND_NUM_WRITER_0; i++) csrw_ss(T_BOUND_BASE_WRITER_0 + i, W0tlbound[i]);
    //     for (int i = 0; i < T_STRIDE_NUM_WRITER_0; i++) csrw_ss(T_STRIDE_BASE_WRITER_0 + i, W0tlstride[i]);
    //     write_csr(ENABLED_CHANNEL_WRITER_0, enable_W0);
    // }

    if (enable_W1) {
        write_csr(BASE_PTR_WRITER_1_LOW, ptr_W1);
        for (int i = 0; i < S_STRIDE_NUM_WRITER_1; i++) csrw_ss(S_STRIDE_BASE_WRITER_1 + i, W1slstride[i]);
        for (int i = 0; i < T_BOUND_NUM_WRITER_1; i++) csrw_ss(T_BOUND_BASE_WRITER_1 + i, W1tlbound[i]);
        for (int i = 0; i < T_STRIDE_NUM_WRITER_1; i++) csrw_ss(T_STRIDE_BASE_WRITER_1 + i, W1tlstride[i]);
        write_csr(ENABLED_CHANNEL_WRITER_1, enable_W1);
    }

    // if (enable_W2) {
    //     write_csr(BASE_PTR_WRITER_2_LOW, ptr_W2);
    //     for (int i = 0; i < S_STRIDE_NUM_WRITER_2; i++) csrw_ss(S_STRIDE_BASE_WRITER_2 + i, W2slstride[i]);
    //     for (int i = 0; i < T_BOUND_NUM_WRITER_2; i++) csrw_ss(T_BOUND_BASE_WRITER_2 + i, W2tlbound[i]);
    //     for (int i = 0; i < T_STRIDE_NUM_WRITER_2; i++) csrw_ss(T_STRIDE_BASE_WRITER_2 + i, W2tlstride[i]);
    //     write_csr(ENABLED_CHANNEL_WRITER_2, enable_W2);
    // }

    if (enable_W3) {
        write_csr(BASE_PTR_WRITER_3_LOW, ptr_W3);
        for (int i = 0; i < S_STRIDE_NUM_WRITER_3; i++) csrw_ss(S_STRIDE_BASE_WRITER_3 + i, W3slstride[i]);
        for (int i = 0; i < T_BOUND_NUM_WRITER_3; i++) csrw_ss(T_BOUND_BASE_WRITER_3 + i, W3tlbound[i]);
        for (int i = 0; i < T_STRIDE_NUM_WRITER_3; i++) csrw_ss(T_STRIDE_BASE_WRITER_3 + i, W3tlstride[i]);
        write_csr(ENABLED_CHANNEL_WRITER_3, enable_W3);
    }
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
    printf("Waiting for Streamers and SimbaCore to finish...\n");
    write_csr(STREAMER_START_CSR, 0);
    write_csr(STREAMER_START_CSR, 0);
    write_csr(SIMBACORE_START, 0);
    while (read_csr(SIMBACORE_BUSY));  // 1185 = 0x4a1
    printf("SimbaCore has finished. Polling Streamers...\n");
    while (read_csr(STREAMER_BUSY_CSR));  // 1177 = 0x499
    printf("Streamers and SimbaCore have finished\n");
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
uint32_t check_result_all(uint16_t* output, uint16_t* output_golden, int32_t data_length) {
    uint32_t err         = 0;
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
uint32_t check_result_sample(uint16_t* output, uint16_t* output_golden, int32_t* sample_indices,
                             int32_t test_sample_count, const char* tensor_name) {
    uint32_t err = 0;
    printf("Checking results: sampling %d elements\n", test_sample_count);

    for (int i = 0; i < test_sample_count; i++) {
        int sample_index      = sample_indices[i];
        uint16_t output_value = output[sample_index];
        uint16_t golden_value = output_golden[sample_index];
        if (output_value != golden_value) {
            err++;
            printf("FAIL %s[%d] = %d,\tref = %d\n", tensor_name, sample_index, output_value, golden_value);
        } else {
            printf("PASS %s[%d] = %d,\tref = %d\n", tensor_name, sample_index, output_value, golden_value);
        }
    }
    return err;
}
