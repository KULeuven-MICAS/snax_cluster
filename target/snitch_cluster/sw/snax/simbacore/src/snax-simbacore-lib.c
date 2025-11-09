// Copyright 2025 KU Leuven.
// Not released under license.All rights reserved.
//
// Author : Robin Geens < robin.geens@kuleuven.be>

#include "snax-simbacore-lib.h"
#include <stdint.h>
#include "streamer_csr_addr_map.h"

// Shorthand function to set only the streamers used in OSGeMM (mode 2)
void set_simbacore_osgemm_streamer_csr(uint32_t A_ptr, int32_t* A_ss, int32_t* A_tb, int32_t* A_ts,  //
                                       uint32_t B_ptr, int32_t* B_ss, int32_t* B_tb, int32_t* B_ts,  //
                                       uint32_t D_ptr, int32_t* D_ss, int32_t* D_tb, int32_t* D_ts) {
    // osCore input R0
    write_csr(BASE_PTR_READER_0_LOW, A_ptr);
    for (int i = 0; i < S_STRIDE_NUM_READER_0; i++) csrw_ss(S_STRIDE_BASE_READER_0 + i, A_ss[i]);
    for (int i = 0; i < T_BOUND_NUM_READER_0; i++) csrw_ss(T_BOUND_BASE_READER_0 + i, A_tb[i]);
    for (int i = 0; i < T_STRIDE_NUM_READER_0; i++) csrw_ss(T_STRIDE_BASE_READER_0 + i, A_ts[i]);

    // osCore weight R1
    write_csr(BASE_PTR_READER_1_LOW, B_ptr);
    for (int i = 0; i < S_STRIDE_NUM_READER_1; i++) csrw_ss(S_STRIDE_BASE_READER_1 + i, B_ss[i]);
    for (int i = 0; i < T_BOUND_NUM_READER_1; i++) csrw_ss(T_BOUND_BASE_READER_1 + i, B_tb[i]);
    for (int i = 0; i < T_STRIDE_NUM_READER_1; i++) csrw_ss(T_STRIDE_BASE_READER_1 + i, B_ts[i]);

    // osCore output W0
    write_csr(BASE_PTR_WRITER_0_LOW, D_ptr);
    for (int i = 0; i < S_STRIDE_NUM_WRITER_0; i++) csrw_ss(S_STRIDE_BASE_WRITER_0 + i, D_ss[i]);
    for (int i = 0; i < T_BOUND_NUM_WRITER_0; i++) csrw_ss(T_BOUND_BASE_WRITER_0 + i, D_tb[i]);
    for (int i = 0; i < T_STRIDE_NUM_WRITER_0; i++) csrw_ss(T_STRIDE_BASE_WRITER_0 + i, D_ts[i]);

    // Disable all other streamers by setting bound to 0
    write_csr(T_BOUND_BASE_READER_2, 0);
    write_csr(T_BOUND_BASE_READER_3, 0);
    write_csr(T_BOUND_BASE_READER_4, 0);
    write_csr(T_BOUND_BASE_READER_5, 0);
    write_csr(T_BOUND_BASE_READER_6, 0);
    write_csr(T_BOUND_BASE_READER_7, 0);
    write_csr(T_BOUND_BASE_READER_8, 0);
    write_csr(T_BOUND_BASE_READER_9, 0);
    write_csr(T_BOUND_BASE_READER_10, 0);
    write_csr(T_BOUND_BASE_READER_11, 0);
    write_csr(T_BOUND_BASE_READER_12, 0);
    write_csr(T_BOUND_BASE_READER_13, 0);
    write_csr(T_BOUND_BASE_WRITER_1, 0);
    write_csr(T_BOUND_BASE_WRITER_2, 0);
    write_csr(T_BOUND_BASE_WRITER_3, 0);
}

void set_streamer_csr(

    uint32_t R0_ptr, int32_t* R0_ss, int32_t* R0_tb, int32_t* R0_ts, uint32_t R0_en,       // osCore in
    uint32_t R1_ptr, int32_t* R1_ss, int32_t* R1_tb, int32_t* R1_ts, uint32_t R1_en,       // oscore weight
    uint32_t R2_ptr, int32_t* R2_ss, int32_t* R2_tb, int32_t* R2_ts, uint32_t R2_en,       // switchCore/ in
    uint32_t R3_ptr, int32_t* R3_ss, int32_t* R3_tb, int32_t* R3_ts, uint32_t R3_en,       // switchCore weight
    uint32_t R4_ptr, int32_t* R4_ss, int32_t* R4_tb, int32_t* R4_ts, uint32_t R4_en,       // switchCore bias
    uint32_t R5_ptr, int32_t* R5_ss, int32_t* R5_tb, int32_t* R5_ts, uint32_t R5_en,       // switchCore  matmul weight
    uint32_t R6_ptr, int32_t* R6_ss, int32_t* R6_tb, int32_t* R6_ts, uint32_t R6_en,       //  SUC A
    uint32_t R7_ptr, int32_t* R7_ss, int32_t* R7_tb, int32_t* R7_ts, uint32_t R7_en,       // SUC BC
    uint32_t R8_ptr, int32_t* R8_ss, int32_t* R8_tb, int32_t* R8_ts, uint32_t R8_en,       // SUC  D
    uint32_t R9_ptr, int32_t* R9_ss, int32_t* R9_tb, int32_t* R9_ts, uint32_t R9_en,       // SUC x
    uint32_t R10_ptr, int32_t* R10_ss, int32_t* R10_tb, int32_t* R10_ts, uint32_t R10_en,  // SUC z
    uint32_t R11_ptr, int32_t* R11_ss, int32_t* R11_tb, int32_t* R11_ts, uint32_t R11_en,  // iscore in
    uint32_t R12_ptr, int32_t* R12_ss, int32_t* R12_tb, int32_t* R12_ts, uint32_t R12_en,  // isCore weight
    uint32_t R13_ptr, int32_t* R13_ss, int32_t* R13_tb, int32_t* R13_ts, uint32_t R13_en,  // isCore psum

    uint32_t W0_ptr, int32_t* W0_ss, int32_t* W0_tb, int32_t* W0_ts, uint32_t W0_en,  // osCore out
    uint32_t W1_ptr, int32_t* W1_ss, int32_t* W1_tb, int32_t* W1_ts, uint32_t W1_en,  // switchCore out
    uint32_t W2_ptr, int32_t* W2_ss, int32_t* W2_tb, int32_t* W2_ts, uint32_t W2_en,  // SUC out
    uint32_t W3_ptr, int32_t* W3_ss, int32_t* W3_tb, int32_t* W3_ts, uint32_t W3_en   // isCore out
) {
    if (R0_en) {
        write_csr(BASE_PTR_READER_0_LOW, R0_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_0; i++) csrw_ss(S_STRIDE_BASE_READER_0 + i, R0_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_0; i++) csrw_ss(T_BOUND_BASE_READER_0 + i, R0_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_0; i++) csrw_ss(T_STRIDE_BASE_READER_0 + i, R0_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_0, 0);
    }

    if (R1_en) {
        write_csr(BASE_PTR_READER_1_LOW, R1_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_1; i++) csrw_ss(S_STRIDE_BASE_READER_1 + i, R1_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_1; i++) csrw_ss(T_BOUND_BASE_READER_1 + i, R1_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_1; i++) csrw_ss(T_STRIDE_BASE_READER_1 + i, R1_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_1, 0);
    }

    if (R2_en) {
        write_csr(BASE_PTR_READER_2_LOW, R2_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_2; i++) csrw_ss(S_STRIDE_BASE_READER_2 + i, R2_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_2; i++) csrw_ss(T_BOUND_BASE_READER_2 + i, R2_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_2; i++) csrw_ss(T_STRIDE_BASE_READER_2 + i, R2_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_2, 0);
    }

    if (R3_en) {
        write_csr(BASE_PTR_READER_3_LOW, R3_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_3; i++) csrw_ss(S_STRIDE_BASE_READER_3 + i, R3_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_3; i++) csrw_ss(T_BOUND_BASE_READER_3 + i, R3_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_3; i++) csrw_ss(T_STRIDE_BASE_READER_3 + i, R3_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_3, 0);
    }

    if (R4_en) {
        write_csr(BASE_PTR_READER_4_LOW, R4_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_4; i++) csrw_ss(S_STRIDE_BASE_READER_4 + i, R4_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_4; i++) csrw_ss(T_BOUND_BASE_READER_4 + i, R4_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_4; i++) csrw_ss(T_STRIDE_BASE_READER_4 + i, R4_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_4, 0);
    }

    if (R5_en) {
        write_csr(BASE_PTR_READER_5_LOW, R5_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_5; i++) csrw_ss(S_STRIDE_BASE_READER_5 + i, R5_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_5; i++) csrw_ss(T_BOUND_BASE_READER_5 + i, R5_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_5; i++) csrw_ss(T_STRIDE_BASE_READER_5 + i, R5_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_5, 0);
    }

    if (R6_en) {
        write_csr(BASE_PTR_READER_6_LOW, R6_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_6; i++) csrw_ss(S_STRIDE_BASE_READER_6 + i, R6_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_6; i++) csrw_ss(T_BOUND_BASE_READER_6 + i, R6_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_6; i++) csrw_ss(T_STRIDE_BASE_READER_6 + i, R6_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_6, 0);
    }

    if (R7_en) {
        write_csr(BASE_PTR_READER_7_LOW, R7_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_7; i++) csrw_ss(S_STRIDE_BASE_READER_7 + i, R7_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_7; i++) csrw_ss(T_BOUND_BASE_READER_7 + i, R7_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_7; i++) csrw_ss(T_STRIDE_BASE_READER_7 + i, R7_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_7, 0);
    }

    if (R8_en) {
        write_csr(BASE_PTR_READER_8_LOW, R8_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_8; i++) csrw_ss(S_STRIDE_BASE_READER_8 + i, R8_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_8; i++) csrw_ss(T_BOUND_BASE_READER_8 + i, R8_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_8; i++) csrw_ss(T_STRIDE_BASE_READER_8 + i, R8_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_8, 0);
    }

    if (R9_en) {
        write_csr(BASE_PTR_READER_9_LOW, R9_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_9; i++) csrw_ss(S_STRIDE_BASE_READER_9 + i, R9_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_9; i++) csrw_ss(T_BOUND_BASE_READER_9 + i, R9_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_9; i++) csrw_ss(T_STRIDE_BASE_READER_9 + i, R9_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_9, 0);
    }

    if (R10_en) {
        write_csr(BASE_PTR_READER_10_LOW, R10_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_10; i++) csrw_ss(S_STRIDE_BASE_READER_10 + i, R10_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_10; i++) csrw_ss(T_BOUND_BASE_READER_10 + i, R10_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_10; i++) csrw_ss(T_STRIDE_BASE_READER_10 + i, R10_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_10, 0);
    }

    if (R11_en) {
        write_csr(BASE_PTR_READER_11_LOW, R11_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_11; i++) csrw_ss(S_STRIDE_BASE_READER_11 + i, R11_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_11; i++) csrw_ss(T_BOUND_BASE_READER_11 + i, R11_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_11; i++) csrw_ss(T_STRIDE_BASE_READER_11 + i, R11_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_11, 0);
    }

    if (R12_en) {
        write_csr(BASE_PTR_READER_12_LOW, R12_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_12; i++) csrw_ss(S_STRIDE_BASE_READER_12 + i, R12_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_12; i++) csrw_ss(T_BOUND_BASE_READER_12 + i, R12_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_12; i++) csrw_ss(T_STRIDE_BASE_READER_12 + i, R12_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_12, 0);
    }

    if (R13_en) {
        write_csr(BASE_PTR_READER_13_LOW, R13_ptr);
        for (int i = 0; i < S_STRIDE_NUM_READER_13; i++) csrw_ss(S_STRIDE_BASE_READER_13 + i, R13_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_READER_13; i++) csrw_ss(T_BOUND_BASE_READER_13 + i, R13_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_READER_13; i++) csrw_ss(T_STRIDE_BASE_READER_13 + i, R13_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_READER_13, 0);
    }

    if (W0_en) {
        write_csr(BASE_PTR_WRITER_0_LOW, W0_ptr);
        for (int i = 0; i < S_STRIDE_NUM_WRITER_0; i++) csrw_ss(S_STRIDE_BASE_WRITER_0 + i, W0_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_WRITER_0; i++) csrw_ss(T_BOUND_BASE_WRITER_0 + i, W0_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_WRITER_0; i++) csrw_ss(T_STRIDE_BASE_WRITER_0 + i, W0_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_WRITER_0, 0);
    }

    if (W1_en) {
        write_csr(BASE_PTR_WRITER_1_LOW, W1_ptr);
        for (int i = 0; i < S_STRIDE_NUM_WRITER_1; i++) csrw_ss(S_STRIDE_BASE_WRITER_1 + i, W1_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_WRITER_1; i++) csrw_ss(T_BOUND_BASE_WRITER_1 + i, W1_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_WRITER_1; i++) csrw_ss(T_STRIDE_BASE_WRITER_1 + i, W1_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_WRITER_1, 0);
    }

    if (W2_en) {
        write_csr(BASE_PTR_WRITER_2_LOW, W2_ptr);
        for (int i = 0; i < S_STRIDE_NUM_WRITER_2; i++) csrw_ss(S_STRIDE_BASE_WRITER_2 + i, W2_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_WRITER_2; i++) csrw_ss(T_BOUND_BASE_WRITER_2 + i, W2_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_WRITER_2; i++) csrw_ss(T_STRIDE_BASE_WRITER_2 + i, W2_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_WRITER_2, 0);
    }

    if (W3_en) {
        write_csr(BASE_PTR_WRITER_3_LOW, W3_ptr);
        for (int i = 0; i < S_STRIDE_NUM_WRITER_3; i++) csrw_ss(S_STRIDE_BASE_WRITER_3 + i, W3_ss[i]);
        for (int i = 0; i < T_BOUND_NUM_WRITER_3; i++) csrw_ss(T_BOUND_BASE_WRITER_3 + i, W3_tb[i]);
        for (int i = 0; i < T_STRIDE_NUM_WRITER_3; i++) csrw_ss(T_STRIDE_BASE_WRITER_3 + i, W3_ts[i]);
    } else {
        write_csr(T_BOUND_BASE_WRITER_3, 0);
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
