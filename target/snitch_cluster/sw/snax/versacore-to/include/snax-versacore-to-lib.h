// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include <stdbool.h>

#include "snrt.h"
#include "stdint.h"
#include "streamer_csr_addr_map.h"

#pragma once

// GeMM CSR = 4
#define GEMMX_CSR_ADDR_BASE (STREAMER_PERFORMANCE_COUNTER_CSR + 1)
#define OVERWRITE_ACCUM (GEMMX_CSR_ADDR_BASE)
#define ACCUM_BOUND (OVERWRITE_ACCUM + 1)
#define OUTPUT_BOUND (ACCUM_BOUND + 1)

#define SUBTRACTIONS (OUTPUT_BOUND + 1)

#define ARRAY_SHAPE_CFG (SUBTRACTIONS + 1)
#define DATA_TYPE_CFG (ARRAY_SHAPE_CFG + 1)

// GEMMX CSR
#define GEMMX_START (DATA_TYPE_CFG + 1)

// GeMMX read-only CSR
#define GEMMX_BUSY (GEMMX_START + 1)
#define GEMMX_PERFORMANCE_COUNTER (GEMMX_BUSY + 1)
// Stall census for the task just run. Every busy cycle is either a pass entering the array or
// one of these three stalls, so the three plus the accepted passes equal the cycle count.
// A and B say the operand feed could not keep up; D says the accelerator refused the pass
// because the accumulator or the D drain was busy.
#define GEMMX_STALL_A (GEMMX_PERFORMANCE_COUNTER + 1)
#define GEMMX_STALL_B (GEMMX_STALL_A + 1)
#define GEMMX_STALL_D (GEMMX_STALL_B + 1)
// Matmuls this array has retired, free-running. The completion signal to wait on when more
// than one configuration can be queued: busy_o cannot separate two back-to-back dispatches,
// and the streamer's counter marks its data movers done, which lands before the array is.
#define GEMMX_FINISHED_TASK (GEMMX_STALL_D + 1)

// Pack two subtraction values to one CSR
int32_t gen_subtraction_config(int8_t subtraction_a, int8_t subtraction_b);

void set_versacore_streamer_csr(
    int32_t delta_local_a, int32_t* Aslstride, int32_t* Atlbound,
    int32_t* Atlstride, int32_t set_addr_remap_index_A, int32_t transpose_A,
    int32_t* channel_en_A,

    int32_t delta_local_b, int32_t* Bslstride, int32_t* Btlbound,
    int32_t* Btlstride, int32_t set_addr_remap_index_B, int32_t transpose_B,
    int32_t* channel_en_B,

    int32_t delta_local_c, int32_t* Cslstride, int32_t* Ctlbound,
    int32_t* Ctlstride, int32_t set_addr_remap_index_C, int32_t* channel_en_C,

    int32_t delta_local_d32, int32_t* D32slstride, int32_t* D32tlbound,
    int32_t* D32tlstride, int32_t set_addr_remap_index_D32,
    int32_t* channel_en_D, int32_t array_shape, uint32_t quantization_enable,
    uint32_t shift_i, uint32_t multiplier_i, int32_t input_zp_i,
    int32_t output_zp_i, int32_t int32tofp16_enable, int32_t int4_a_enable,
    int32_t int4_b_enable);

// Set CSR to start STREAMER
inline void set_versacore_streamer_start() { csrw_ss(STREAMER_START_CSR, 1); }

// Set GEMM configuration CSR
void set_versacore_csr(uint32_t take_in_new_c,
                       uint32_t a_b_input_times_one_output,
                       uint32_t output_times, uint32_t subtractions,
                       uint32_t array_shape, uint32_t data_type);

// Set CSR to start GEMM
inline void set_versacore_start() { csrw_ss(GEMMX_START, 1); }

// Poll until Streamer and GEMM accelerator finish
void wait_versacore_and_streamer();

// ---- task accounting -------------------------------------------------------
//
// The busy flag is a level: it says work is happening, not WHICH work. That is
// enough only while at most one configuration can be in flight. Once the CSR
// manager can stage a second one behind the first (cfgQueueDepth > 1), a poll on
// busy can return between two queued tasks and report the wrong one finished.
//
// These two counters make it exact. Submitting returns an id, and the task is
// complete once the finished counter reaches it. Both are free-running and wrap
// together, so the comparison below stays correct across a wrap.
__attribute__((always_inline)) static inline uint32_t
snax_versacore_submitted_tasks() {
    return csrr_ss(STREAMER_SUBMITTED_TASK_CSR);
}

__attribute__((always_inline)) static inline uint32_t
snax_versacore_finished_tasks() {
    return csrr_ss(STREAMER_FINISHED_TASK_CSR);
}

// Start the configured task and return its id. The trailing zero write leaves the
// start register clear so the next non-zero write is a fresh submission rather
// than a repeat of this one.
__attribute__((always_inline)) static inline uint32_t
snax_versacore_submit() {
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(STREAMER_START_CSR, 0);
    return csrr_ss(STREAMER_SUBMITTED_TASK_CSR);
}

// Wait for one specific task. Subtracting before comparing is what makes this
// wrap-safe: it asks "has the finished counter reached or passed id", not
// "is finished numerically at least id".
__attribute__((always_inline)) static inline void
snax_versacore_wait_task(uint32_t task_id) {
    while ((int32_t)(csrr_ss(STREAMER_FINISHED_TASK_CSR) - task_id) < 0) {
    }
}

void wait_versacore();

// Read performance counter of the Streamer, a read-only CSR
uint32_t read_versacore_streamer_perf_counter();

// Read performance counter of GEMM, a read-only CSR
uint32_t read_versacore_perf_counter();

// Check the result of GEMMX
uint32_t check_versacore_result_D32(int8_t* output, int8_t* output_golden,
                                    int32_t data_length,
                                    bool banked_data_layout);
