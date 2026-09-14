// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include <stdbool.h>
#include "simd_csr_addr_map.h"
#include "snrt.h"
#include "stdint.h"
#include "streamer_csr_addr_map.h"

#pragma once

#define GEMMX_CSR_ADDR_BASE (STREAMER_PERFORMANCE_COUNTER_CSR + 1)
// GeMM CSR = 4
#define T_BOUND_K (GEMMX_CSR_ADDR_BASE)
#define T_BOUND_N (T_BOUND_K + 1)
#define T_BOUND_M (T_BOUND_N + 1)

#define SUBTRACTIONS (T_BOUND_M + 1)

// GeMMX CSR
#define BYPASS_SIMD (TEMPORAL_LOOP_BOUND + 1)
#define GEMMX_START (BYPASS_SIMD + 1)

// GeMMX read-only CSR
#define GEMMX_BUSY (GEMMX_START + 1)
#define GEMMX_PERFORMANCE_COUNTER (GEMMX_BUSY + 1)

// Pack matrix size setting to one CSR
int32_t gen_size_config(uint8_t Batch, uint8_t M, uint8_t K, uint8_t N);

// Pack two subtraction values to one CSR
int32_t gen_subtraction_config(int8_t subtraction_a, int8_t subtraction_b);

// generate the configuration for CSR0
int32_t gen_csr0_config(uint8_t input_zp_i, uint8_t output_zp_i,
                        uint8_t max_int_i, uint8_t min_int_i);

// generate the configuration for CSR1
int32_t gen_csr1_config(bool double_round_i);

// Set STREAMER configuration CSR
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
                            int32_t* channel_en_C, int32_t broadcast_C);

// The CSR helpers below are always_inline so their compile-time-constant address
// propagates into the csrr_ss/csrw_ss switch and constant-folds to a single direct
// `csrr/csrw <imm>`. Out of line the address is opaque, so every access pays a
// jump-table load from L2 plus an indirect jump -- which is measured, not assumed,
// to be the dominant cost of accelerator configuration on this core. The SIMD and
// xDMA libraries were fixed this way; the GEMM's had been missed, and it is the
// engine that configures most often (5 base pointers + 2 counter reads per dispatch).

// Set CSR to start STREAMER
__attribute__((always_inline)) static inline void set_gemmx_streamer_start() {
    csrw_ss(STREAMER_START_CSR, 1);
}

// Set GEMM configuration CSR
void set_gemmx_csr(int32_t tempLoop0, int32_t tempLoop1, int32_t tempLoop2,
                   int32_t subtractions, uint32_t csr0, uint32_t csr1,
                   int32_t* shared_bitpacked_shift, int32_t* shared_multiplier,
                   uint32_t temporal_loop_bound, uint32_t bypassSIMD);

// Set CSR to start GEMM
__attribute__((always_inline)) static inline void set_gemmx_start() {
    csrw_ss(GEMMX_START, 1);
}

// Poll until Streamer and GEMM accelerator finish
// Wait for the ACCELERATOR TO START before waiting for it to finish.
//
// This used to be two redundant `csrw STREAMER_START, 0` writes followed straight by the
// busy poll. Those writes were not redundant in effect: out of line, each CSR access went
// through the csrw_ss jump table, so they cost ~20-30 cycles and that was just enough for
// GEMMX_BUSY to rise before the first poll sampled it. Inlining the helper (which is
// otherwise a large win) collapsed them to 2 cycles, the first poll then read busy=0 on a
// task that had not started yet, and the function returned immediately -- so the caller
// read a partial performance counter and reconfigured the GEMM out from under a running
// matmul.
//
// It was invisible on short tasks and on the standalone matmul, and showed up only as a
// large tile reporting FEWER cycles than its own arithmetic floor. Correctness must not
// depend on how slowly a CSR write happens to execute.
//
// The rise-wait is bounded so that a task which completes before we look cannot hang us:
// if busy never rises within the guard, either it already finished (the fall-waits below
// exit at once, which is correct) or the engine was never started (a configuration bug,
// which the caller's own timeout will catch).
__attribute__((always_inline)) static inline void wait_gemmx_and_streamer() {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    for (uint32_t g = 0; g < 64u; g++) {
        if (csrr_ss(GEMMX_BUSY) || csrr_ss(STREAMER_BUSY_CSR)) break;
    }
    while (csrr_ss(GEMMX_BUSY)) {
    }
    while (csrr_ss(STREAMER_BUSY_CSR)) {
    }
    csrw_ss(GEMMX_START, 0);
}

// Read performance counter of the Streamer, a read-only CSR
__attribute__((always_inline)) static inline uint32_t
read_gemmx_streamer_perf_counter() {
    return csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
}

// Read performance counter of GEMM, a read-only CSR
__attribute__((always_inline)) static inline uint32_t read_gemmx_perf_counter() {
    return csrr_ss(GEMMX_PERFORMANCE_COUNTER);
}

// Re-point an already-configured GEMM at new buffers without touching its shape:
// five csrw against set_gemmx_streamer_csr()'s ~84. Pass -1 to leave a pointer alone.
__attribute__((always_inline)) static inline void set_gemmx_bases(
    int32_t delta_local_a, int32_t delta_local_b, int32_t delta_local_d8,
    int32_t delta_local_c, int32_t delta_local_d32) {
    uint32_t l1 = (uint32_t)snrt_l1_next();
    if (delta_local_a >= 0)
        csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)delta_local_a + l1);
    if (delta_local_b >= 0)
        csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)delta_local_b + l1);
    if (delta_local_d8 >= 0)
        csrw_ss(BASE_PTR_WRITER_0_LOW, (uint32_t)delta_local_d8 + l1);
    if (delta_local_c >= 0)
        csrw_ss(BASE_PTR_READER_WRITER_0_LOW, (uint32_t)delta_local_c + l1);
    if (delta_local_d32 >= 0)
        csrw_ss(BASE_PTR_READER_WRITER_1_LOW, (uint32_t)delta_local_d32 + l1);
}

// Check the result of the implicit im2col convolution
uint32_t check_gemmx_result_D8(int8_t* output, int8_t* output_golden,
                               int32_t Batch, int32_t M, int32_t N,
                               bool banked_data_layout);

uint32_t check_gemmx_result_D32(int32_t* output, int32_t* output_golden,
                                int32_t Batch, int32_t M, int32_t N,
                                bool banked_data_layout);
