// Copyright 2023 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include "snax-streamer-simd-lib.h"

int32_t gen_csr0_config(uint8_t input_zp_i, uint8_t output_zp_i,
                        uint8_t shift_i, uint8_t max_int_i)
{
    // encode the configuration into a single 32-bit integer
    return ((int32_t)max_int_i << 24) | ((int32_t)shift_i << 16) |
           ((int32_t)output_zp_i << 8) | (int32_t)input_zp_i;
}

int32_t gen_csr1_config(uint8_t min_int_i, bool double_round_i) {
    // encode the configuration into a single 32-bit integer
    return ((uint8_t)double_round_i << 8) | (uint8_t)min_int_i;
}

int32_t gen_csr2_config(uint32_t multiplier_i) { return multiplier_i; }

void set_streamer_simd_csr(int tempLoop0, int tempLoop1, int tempStride0_in,
                       int tempStride1_in, int tempStride0_out,
                       int tempStride1_out, int32_t delta_local_in, int32_t delta_local_out) {
    // loop bounds, from innermost to outermost
    write_csr(0, tempLoop0);
    write_csr(1, tempLoop1);

    // temporal strides
    write_csr(2, tempStride0_in);
    write_csr(3, tempStride1_in);

    write_csr(4, tempStride0_out);
    write_csr(5, tempStride1_out);

    // 6 7 leave for parfor strides

    // base ptr for In
    write_csr(8, (uint32_t)(base_ptr_i + snrt_l1_next()));

    // base ptr for Out
    write_csr(9, (uint32_t)(base_ptr_o + snrt_l1_next()));
}

void start_streamer_simd() { write_csr(10, 1); }

void set_simd_csr(uint32_t csr0, uint32_t csr1, uint32_t csr2) {
    // set the constants for the SIMD unit
    write_csr(11, csr0);
    write_csr(12, csr1);
    write_csr(13, csr2);
}

void start_simd() { write_csr(14, 1); }

void wait_streamer_simd() {
    write_csr(10, 0);
    write_csr(14, 0);
}

void load_simd_test_data(int tempLoop0, int tempLoop1, int tempStride0,
                         int tempStride1, int32_t* base_ptr_local,
                         int32_t* base_ptr_l2) {
    int32_t* addr_in;
    int32_t* addr_In;

    for (int loop1 = 0; loop1 < tempLoop1; loop1++) {
        for (int loop0 = 0; loop0 < tempLoop0; loop0++) {
            addr_in =
                base_ptr_local + (loop1 * tempStride1 + loop0 * tempStride0) / sizeof(int32_t);
            addr_In =
                base_ptr_l2 + loop1 * tempLoop0 * vec_len + loop0 * vec_len;
            snrt_dma_start_1d(addr_in, addr_In, vec_len * sizeof(int32_t));
        }
    }
}

uint32_t check_simd_result(int tempLoop0, int tempLoop1, int tempStride0,
                           int tempStride1, int8_t* base_ptr_local,
                           int8_t* base_ptr_l2) {
    int8_t* addr_out;
    int8_t* addr_Out;
    uint32_t error = 0;

    for (int loop1 = 0; loop1 < tempLoop1; loop1++) {
        for (int loop0 = 0; loop0 < tempLoop0; loop0++) {
            addr_out =
                base_ptr_local + (loop1 * tempStride1 + loop0 * tempStride0) / sizeof(int32_t);
            addr_Out =
                base_ptr_l2 + loop1 * tempLoop0 * vec_len + loop0 * vec_len;
            for (int i = 0; i < vec_len; i++) {
                if (addr_out[i] != addr_Out[i]) {
                    error++;
                }
            }
        }
    }
    return error;
}