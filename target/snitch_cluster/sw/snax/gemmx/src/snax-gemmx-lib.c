// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include "snax-gemmx-lib.h"
#include <stdbool.h>
#include "snax-gemmx-params.h"
#include "snrt.h"
#include "stdint.h"
#include "streamer_csr_addr_map.h"

// Write a vector of CSRs at COMPILE-TIME-CONSTANT addresses.
//
// csrw_ss (snRuntime/src/csr.h) is a switch over the CSR number, because the RISC-V
// csrw instruction takes an immediate. A constant address folds to one `csrw imm`;
// an address computed in a loop becomes a jump-table load out of L2 plus an indirect
// jump -- roughly 20 cycles instead of 1. Configuring the streamer from `for` loops
// therefore cost ~1000 cycles per launch, which on a four-engine cluster showed up as
// the GEMM sitting idle while its core wrote CSRs.
//
// `num` is always a compile-time constant here (S_STRIDE_NUM_*, T_BOUND_NUM_*,
// T_STRIDE_NUM_*), so every test below folds away and only the real writes remain.
#define SNAX_CSR_WRITE_ZEROS(base, num)                                       \
    do {                                                                       \
        if ((num) > 0) csrw_ss((base) + 0, 0);                                 \
        if ((num) > 1) csrw_ss((base) + 1, 0);                                 \
        if ((num) > 2) csrw_ss((base) + 2, 0);                                 \
        if ((num) > 3) csrw_ss((base) + 3, 0);                                 \
        if ((num) > 4) csrw_ss((base) + 4, 0);                                 \
        if ((num) > 5) csrw_ss((base) + 5, 0);                                 \
        if ((num) > 6) csrw_ss((base) + 6, 0);                                 \
        if ((num) > 7) csrw_ss((base) + 7, 0);                                 \
    } while (0)

#define SNAX_CSR_WRITE_VEC(base, arr, num)                                    \
    do {                                                                       \
        if ((num) > 0) csrw_ss((base) + 0, (arr)[0]);                          \
        if ((num) > 1) csrw_ss((base) + 1, (arr)[1]);                          \
        if ((num) > 2) csrw_ss((base) + 2, (arr)[2]);                          \
        if ((num) > 3) csrw_ss((base) + 3, (arr)[3]);                          \
        if ((num) > 4) csrw_ss((base) + 4, (arr)[4]);                          \
        if ((num) > 5) csrw_ss((base) + 5, (arr)[5]);                          \
        if ((num) > 6) csrw_ss((base) + 6, (arr)[6]);                          \
        if ((num) > 7) csrw_ss((base) + 7, (arr)[7]);                          \
    } while (0)


int32_t gen_size_config(uint8_t Batch, uint8_t M, uint8_t K, uint8_t N) {
    return ((int32_t)Batch << 24) | ((int32_t)M << 16) | ((int32_t)K << 8) |
           (int32_t)N;
}

int32_t gen_subtraction_config(int8_t subtraction_a, int8_t subtraction_b) {
    return ((uint8_t)subtraction_b << 8) | (uint8_t)subtraction_a;
}

int32_t gen_csr0_config(uint8_t input_zp_i, uint8_t output_zp_i,
                        uint8_t max_int_i, uint8_t min_int_i) {
    // encode the configuration into a single 32-bit integer
    return ((int32_t)min_int_i << 24) | ((int32_t)max_int_i << 16) |
           ((int32_t)output_zp_i << 8) | (int32_t)input_zp_i;
}

int32_t gen_csr1_config(bool double_round_i) {
    // encode the configuration into a single 32-bit integer
    return (uint32_t)double_round_i;
}

// Set STREAMER configuration CSR
// Re-point a configured GEMM at new buffers: five CSR writes, nothing else.
//
// Two matmuls of the same shape differ only in where their operands live -- in
// FlashAttention, S = Q.K^T and O = P.V are exactly that. Re-issuing
// set_gemmx_streamer_csr() to write ~84 identical numbers costs about a
// thousand cycles, most of it instruction fetch rather than the writes
// themselves. Pass a NULL-equivalent (-1) for any buffer that should keep its
// current base.
//
// The strides, bounds, remap indices and accelerator CSRs are untouched, so the
// caller is asserting the shape is unchanged. If it is not, reconfigure.
void set_gemmx_bases(int32_t delta_local_a, int32_t delta_local_b,
                     int32_t delta_local_d8, int32_t delta_local_c,
                     int32_t delta_local_d32) {
    uint32_t l1 = (uint32_t)snrt_l1_next();
    if (delta_local_a >= 0) csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)delta_local_a + l1);
    if (delta_local_b >= 0) csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)delta_local_b + l1);
    if (delta_local_d8 >= 0) csrw_ss(BASE_PTR_WRITER_0_LOW, (uint32_t)delta_local_d8 + l1);
    if (delta_local_c >= 0)
        csrw_ss(BASE_PTR_READER_WRITER_0_LOW, (uint32_t)delta_local_c + l1);
    if (delta_local_d32 >= 0)
        csrw_ss(BASE_PTR_READER_WRITER_1_LOW, (uint32_t)delta_local_d32 + l1);
}

void set_gemmx_streamer_csr(int32_t* Aslstride, int32_t* Atlbound,
                            int32_t* Atlstride, int32_t set_addr_remap_index_A,

                            int32_t* Bslstride, int32_t* Btlbound,
                            int32_t* Btlstride, int32_t set_addr_remap_index_B,

                            int32_t* D8slstride, int32_t* D8tlbound,
                            int32_t* D8tlstride,
                            int32_t set_addr_remap_index_D8,

                            int32_t* Cslstride, int32_t* Ctlbound,
                            int32_t* Ctlstride, int32_t set_addr_remap_index_C,

                            int32_t* D32slstride, int32_t* D32tlbound,
                            int32_t* D32tlstride,
                            int32_t set_addr_remap_index_D32,

                            int32_t delta_local_a, int32_t delta_local_b,
                            int32_t delta_local_d8, int32_t delta_local_c,
                            int32_t delta_local_d32, int32_t bypassSIMD,
                            int32_t transpose_A, int32_t transpose_B,
                            int32_t* channel_en_C, int32_t broadcast_C) {
    // ----------------------------------A-----------------------------------
    // ----------------------------------A-----------------------------------
    // ----------------------------------A-----------------------------------
    // base ptr for A
    csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(delta_local_a + snrt_l1_next()));

    // spatial strides for A
    SNAX_CSR_WRITE_VEC(S_STRIDE_BASE_READER_0, Aslstride, S_STRIDE_NUM_READER_0);

    // loop bounds, from innermost to outermost, for data mover A
    SNAX_CSR_WRITE_VEC(T_BOUND_BASE_READER_0, Atlbound, T_BOUND_NUM_READER_0);

    // temporal strides for A
    SNAX_CSR_WRITE_VEC(T_STRIDE_BASE_READER_0, Atlstride, T_STRIDE_NUM_READER_0);

    // set the address remap index for A
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, set_addr_remap_index_A);
#endif

    // ----------------------------------B-----------------------------------
    // ----------------------------------B-----------------------------------
    // ----------------------------------B-----------------------------------

    // base ptr for B
    csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(delta_local_b + snrt_l1_next()));

    // spatial strides for B
    SNAX_CSR_WRITE_VEC(S_STRIDE_BASE_READER_1, Bslstride, S_STRIDE_NUM_READER_1);

    // loop bounds, from innermost to outermost, for data mover B
    SNAX_CSR_WRITE_VEC(T_BOUND_BASE_READER_1, Btlbound, T_BOUND_NUM_READER_1);

    // temporal strides for B
    SNAX_CSR_WRITE_VEC(T_STRIDE_BASE_READER_1, Btlstride, T_STRIDE_NUM_READER_1);

    // set the address remap index for B
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, set_addr_remap_index_B);
#endif

    // ----------------------------------D8-----------------------------------
    // ----------------------------------D8-----------------------------------
    // ----------------------------------D8-----------------------------------
    // base ptr for D8
    csrw_ss(BASE_PTR_WRITER_0_LOW, (uint32_t)(delta_local_d8 + snrt_l1_next()));

    // spatial strides for D8
    SNAX_CSR_WRITE_VEC(S_STRIDE_BASE_WRITER_0, D8slstride, S_STRIDE_NUM_WRITER_0);

    // for D8, from N to M
    if (bypassSIMD == 1) {
        SNAX_CSR_WRITE_ZEROS(T_BOUND_BASE_WRITER_0, T_BOUND_NUM_WRITER_0);
    } else {
        SNAX_CSR_WRITE_VEC(T_BOUND_BASE_WRITER_0, D8tlbound, T_BOUND_NUM_WRITER_0);
    }

    // temporal strides for D8
    SNAX_CSR_WRITE_VEC(T_STRIDE_BASE_WRITER_0, D8tlstride, T_STRIDE_NUM_WRITER_0);

    // set the address remap index for D8
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, set_addr_remap_index_D8);
#endif

    // ----------------------------------C-----------------------------------
    // ----------------------------------C-----------------------------------
    // ----------------------------------C-----------------------------------
    // base ptr for C
    csrw_ss(BASE_PTR_READER_WRITER_0_LOW,
            (uint32_t)(delta_local_c + snrt_l1_next()));

    // spatial strides for C
    SNAX_CSR_WRITE_VEC(S_STRIDE_BASE_READER_WRITER_0, Cslstride, S_STRIDE_NUM_READER_WRITER_0);

    // loop bounds, from innermost to outermost, for data mover C
    SNAX_CSR_WRITE_VEC(T_BOUND_BASE_READER_WRITER_0, Ctlbound, T_BOUND_NUM_READER_WRITER_0);

    // temporal strides for C
    SNAX_CSR_WRITE_VEC(T_STRIDE_BASE_READER_WRITER_0, Ctlstride, T_STRIDE_NUM_READER_WRITER_0);

    // set the address remap index for C
#ifdef ADDR_REMAP_INDEX_READER_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, set_addr_remap_index_C);
#endif

    // set the channel enable
#ifdef ENABLED_CHANNEL_READER_WRITER_0
    SNAX_CSR_WRITE_VEC(ENABLED_CHANNEL_READER_WRITER_0, channel_en_C, ENABLED_CHANNEL_READER_WRITER_0_CSR_NUM);
#endif

    // ----------------------------------D32-----------------------------------
    // ----------------------------------D32-----------------------------------
    // ----------------------------------D32-----------------------------------
    // base ptr for D32
    csrw_ss(BASE_PTR_READER_WRITER_1_LOW,
            (uint32_t)(delta_local_d32 + snrt_l1_next()));

    // spatial strides for D32
    SNAX_CSR_WRITE_VEC(S_STRIDE_BASE_READER_WRITER_1, D32slstride, S_STRIDE_NUM_READER_WRITER_1);

    // for D32, from N to M
    if (bypassSIMD == 0) {
        SNAX_CSR_WRITE_ZEROS(T_BOUND_BASE_READER_WRITER_1, T_BOUND_NUM_READER_WRITER_1);
    } else {
        SNAX_CSR_WRITE_VEC(T_BOUND_BASE_READER_WRITER_1, D32tlbound, T_BOUND_NUM_READER_WRITER_1);
    }

    // temporal strides for D32
    SNAX_CSR_WRITE_VEC(T_STRIDE_BASE_READER_WRITER_1, D32tlstride, T_STRIDE_NUM_READER_WRITER_1);

    // set the address remap index for D32
#ifdef ADDR_REMAP_INDEX_READER_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, set_addr_remap_index_D32);
#endif

    // ------------------------- datapath extension ----------------------------
    // ------------------------- datapath extension ----------------------------
    // ------------------------- datapath extension ----------------------------

    // set the transpose
#ifdef READER_EXTENSION_0_CSR_BASE
    csrw_ss(READER_EXTENSION_0_CSR_BASE, transpose_A);
#endif

#ifdef READER_EXTENSION_1_CSR_BASE
    csrw_ss(READER_EXTENSION_1_CSR_BASE, transpose_B);
#endif

#ifdef READER_WRITER_EXTENSION_0_CSR_BASE
    csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE, broadcast_C);
#endif
}

// Set GEMM configuration CSR
void set_gemmx_csr(int32_t tempLoop0, int32_t tempLoop1, int32_t tempLoop2,
                   int32_t subtractions, uint32_t csr0, uint32_t csr1,
                   int32_t* shared_bitpacked_shift, int32_t* shared_multiplier,
                   uint32_t temporal_loop_bound, uint32_t bypassSIMD) {
    // set loop bounds, from innermost to outermost, aka from K to N to M
    csrw_ss(T_BOUND_K, tempLoop0);
    csrw_ss(T_BOUND_N, tempLoop1);
    csrw_ss(T_BOUND_M, tempLoop2);

    // set subtraction a and b
    csrw_ss(SUBTRACTIONS, subtractions);

    // set the constants for the SIMD unit
    csrw_ss(SIMD_CSR0, csr0);
    csrw_ss(SIMD_CSR1, csr1);

    // set the shared bitpacked shift
    for (uint32_t i = 0; i < SIMD_SHARED_BITPACKED_SHIFT_NUM; i++) {
        csrw_ss(SIMD_SHARED_BITPACKED_SHIFT + i, shared_bitpacked_shift[i]);
    }

    // set the shared multipliers
    for (uint32_t i = 0; i < SIMD_SHARED_MULTIPLIER_NUM; i++) {
        csrw_ss(SIMD_SHARED_MULTIPLIER + i, shared_multiplier[i]);
    }

    // set the temporal loop bound
    csrw_ss(TEMPORAL_LOOP_BOUND, temporal_loop_bound);

    csrw_ss(BYPASS_SIMD, bypassSIMD);
}

// Stall until Streamer and GEMM accelerator finish
void wait_gemmx_and_streamer() {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    while (csrr_ss(GEMMX_BUSY)) {
    }
    while (csrr_ss(STREAMER_BUSY_CSR)) {
    }
    csrw_ss(GEMMX_START, 0);
}

// Read performance counter of the Streamer, a read-only CSR
uint32_t read_gemmx_streamer_perf_counter() {
    uint32_t perf_counter = csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
    return perf_counter;
}

// Read performance counter of GEMM, a read-only CSR
uint32_t read_gemmx_perf_counter() {
    uint32_t perf_counter = csrr_ss(GEMMX_PERFORMANCE_COUNTER);
    return perf_counter;
}

uint32_t check_gemmx_result_D8(int8_t* output, int8_t* output_golden,
                               int32_t Batch, int32_t M, int32_t N,
                               bool banked_data_layout) {
    uint32_t err = 0;
    uint32_t size = 0;
    size = Batch * M * N * meshRow * meshCol;

    if (banked_data_layout) {
        for (int i = 0; i < size / 64; i += 1) {
            for (int j = 0; j < 64; j++) {
                if (*(output + i * 256 + j) != output_golden[i * 64 + j]) {
                    err++;
                }
            }
        }
    } else {
        for (int i = 0; i < size; i++) {
            if (output[i] != output_golden[i]) {
                err++;
                printf("output[%d] = %d, output_golden[%d] = %d\n", i,
                       output[i], i, output_golden[i]);
            }
        }
    }

    return err;
}

uint32_t check_gemmx_result_D32(int32_t* output, int32_t* output_golden,
                                int32_t Batch, int32_t M, int32_t N,
                                bool banked_data_layout) {
    uint32_t err = 0;
    uint32_t size = 0;
    size = Batch * M * N * meshRow * meshCol;

    if (banked_data_layout) {
        for (int i = 0; i < size / 16; i += 1) {
            for (int j = 0; j < 16; j++) {
                if (*(output + i * (256 / 4) + j) !=
                    output_golden[i * 16 + j]) {
                    err++;
                }
            }
        }
    } else {
        for (int i = 0; i < size; i++) {
            if (output[i] != output_golden[i]) {
                err++;
                printf("output[%d] = %d, output_golden[%d] = %d\n", i,
                       output[i], i, output_golden[i]);
            }
        }
    }

    return err;
}
