// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//-------------------------------
// Mace <ce.ma@kuleuven.be>
//
// Library: Functions for Setting DIMC CSRs
//
// This pre-built library contains functions to set
// the DIMC accelerator's CSRs.
//-------------------------------

#include <stdbool.h>
#include "snax-dimc-csr.h"
#include "snax-dimc-lib.h"
#include "snrt.h"
#include "stdint.h"

//-------------------------------
// Streamer functions
//-------------------------------

void dimc_set_streamer_dim_r0(uint32_t loop_bound_0,
                              uint32_t loop_bound_1,
                              uint32_t temp_stride_0,
                              uint32_t temp_stride_1,
                              uint32_t spat_stride, 
                              uint32_t base_ptr) {

    csrw_ss(DIMC_STREAMER_R_0_LOOP_BOUND_0, loop_bound_0);
    csrw_ss(DIMC_STREAMER_R_0_LOOP_BOUND_1, loop_bound_1);
    csrw_ss(DIMC_STREAMER_R_0_TEMP_STRIDE_0, temp_stride_0);
    csrw_ss(DIMC_STREAMER_R_0_TEMP_STRIDE_1, temp_stride_1);
    csrw_ss(DIMC_STREAMER_R_0_SPAT_STRIDE, spat_stride);
    csrw_ss(DIMC_STREAMER_R_0_BASE_PTR, base_ptr);
}

void dimc_set_streamer_dim_r1(uint32_t loop_bound_0,
                              uint32_t loop_bound_1,
                              uint32_t temp_stride_0,
                              uint32_t temp_stride_1,
                              uint32_t spat_stride, 
                              uint32_t base_ptr) {

    csrw_ss(DIMC_STREAMER_R_1_LOOP_BOUND_0, loop_bound_0);
    csrw_ss(DIMC_STREAMER_R_1_LOOP_BOUND_1, loop_bound_1);
    csrw_ss(DIMC_STREAMER_R_1_TEMP_STRIDE_0, temp_stride_0);
    csrw_ss(DIMC_STREAMER_R_1_TEMP_STRIDE_1, temp_stride_1);
    csrw_ss(DIMC_STREAMER_R_1_SPAT_STRIDE, spat_stride);
    csrw_ss(DIMC_STREAMER_R_1_BASE_PTR, base_ptr);
}

void dimc_set_streamer_dim_r2(uint32_t loop_bound_0,
                              uint32_t loop_bound_1,
                              uint32_t temp_stride_0,
                              uint32_t temp_stride_1,
                              uint32_t spat_stride, 
                              uint32_t base_ptr) {
    csrw_ss(DIMC_STREAMER_R_2_LOOP_BOUND_0, loop_bound_0);
    csrw_ss(DIMC_STREAMER_R_2_LOOP_BOUND_1, loop_bound_1);
    csrw_ss(DIMC_STREAMER_R_2_TEMP_STRIDE_0, temp_stride_0);
    csrw_ss(DIMC_STREAMER_R_2_TEMP_STRIDE_1, temp_stride_1);
    csrw_ss(DIMC_STREAMER_R_2_SPAT_STRIDE, spat_stride);
    csrw_ss(DIMC_STREAMER_R_2_BASE_PTR, base_ptr);
}

void dimc_set_streamer_dim_r3(uint32_t loop_bound_0,
                              uint32_t loop_bound_1,
                              uint32_t temp_stride_0,
                              uint32_t temp_stride_1,
                              uint32_t spat_stride, 
                              uint32_t base_ptr) {
    csrw_ss(DIMC_STREAMER_R_3_LOOP_BOUND_0, loop_bound_0);
    csrw_ss(DIMC_STREAMER_R_3_LOOP_BOUND_1, loop_bound_1);
    csrw_ss(DIMC_STREAMER_R_3_TEMP_STRIDE_0, temp_stride_0);
    csrw_ss(DIMC_STREAMER_R_3_TEMP_STRIDE_1, temp_stride_1);
    csrw_ss(DIMC_STREAMER_R_3_SPAT_STRIDE, spat_stride);
    csrw_ss(DIMC_STREAMER_R_3_BASE_PTR, base_ptr);
}

void dimc_set_streamer_dim_w(uint32_t loop_bound_0,
                             uint32_t loop_bound_1,
                             uint32_t temp_stride_0,
                             uint32_t temp_stride_1,
                             uint32_t spat_stride, 
                             uint32_t base_ptr) {
    csrw_ss(DIMC_STREAMER_W_0_LOOP_BOUND_0, loop_bound_0);
    csrw_ss(DIMC_STREAMER_W_0_LOOP_BOUND_1, loop_bound_1);
    csrw_ss(DIMC_STREAMER_W_0_TEMP_STRIDE_0, temp_stride_0);
    csrw_ss(DIMC_STREAMER_W_0_TEMP_STRIDE_1, temp_stride_1);
    csrw_ss(DIMC_STREAMER_W_0_SPAT_STRIDE, spat_stride);
    csrw_ss(DIMC_STREAMER_W_0_BASE_PTR, base_ptr);
}

void dimc_start_streamer() {
    csrw_ss(DIMC_STREAMER_START, 1);
}

void dimc_read_perf_counter() {
    csrr_ss(DIMC_STREAMER_PERF_COUNTER);
}

// return true if the streamers are busy
bool dimc_is_streamer_busy() {
    uint32_t busy = csrr_ss(DIMC_STREAMER_BUSY);
    if (busy == 1) {
        return true;
    } else {
        return false;
    }
}

//-------------------------------
// DIMC accelerator functions
//-------------------------------

// query if the accelerator is busy
uint32_t dimc_query_busy() {
    // csrw_ss(DIMC_ACC_BUSY);
    return csrr_ss(DIMC_ACC_BUSY);
}

void dimc_load_act_conv() {
    csrw_ss(DIMC_LOAD_CONV_ACTIVATION, 1);
}

void dimc_start_conv() {
    csrw_ss(DIMC_START_CONV, 1);
}

void dimc_load_kernel() {
    csrw_ss(DIMC_LOAD_KERNEL, 1);
}

void dimc_start_mha() {
    csrw_ss(DIMC_START_MHA, 1);
}

/********************************************************/
/********************************************************/
// functions to configure the parameters

void configure_accelerator() {
    csrw_ss(DIMC_CFG_INPUT_ZP_QKV, 0);
    csrw_ss(DIMC_CFG_INPUT_ZP_QKT, 0);
    csrw_ss(DIMC_CFG_INPUT_ZP_CONV, 0);

    csrw_ss(DIMC_CFG_OUTPUT_ZP_QKV, 0);
    csrw_ss(DIMC_CFG_OUTPUT_ZP_QKT, 0);
    csrw_ss(DIMC_CFG_OUTPUT_ZP_CONV, 0);

    csrw_ss(DIMC_CFG_MULTIPLIER_QKV, 10);
    csrw_ss(DIMC_CFG_MULTIPLIER_QKT, 10);
    csrw_ss(DIMC_CFG_MULTIPLIER_CONV, 10);

    csrw_ss(DIMC_CFG_SHIFT_QKV, 8);
    csrw_ss(DIMC_CFG_SHIFT_QKT, 8);
    csrw_ss(DIMC_CFG_SHIFT_CONV, 8);

    csrw_ss(DIMC_CFG_MAX_INT_QKV, 127);
    csrw_ss(DIMC_CFG_MAX_INT_QKT, 127);
    csrw_ss(DIMC_CFG_MAX_INT_CONV, 127);

    csrw_ss(DIMC_CFG_MIN_INT_QKV, 128);
    csrw_ss(DIMC_CFG_MIN_INT_QKT, 128);
    csrw_ss(DIMC_CFG_MIN_INT_CONV, 128);

    csrw_ss(DIMC_CFG_DOUBLE_RND_QKV, 1);
    csrw_ss(DIMC_CFG_DOUBLE_RND_QKT, 1);
    csrw_ss(DIMC_CFG_DOUBLE_RND_CONV, 1);

    csrw_ss(DIMC_CFG_CYCLE_CNT, 1);
}

uint32_t read_zp() {
    return csrr_ss(DIMC_CFG_INPUT_ZP_QKV);
}
