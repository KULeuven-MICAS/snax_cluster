// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//-------------------------------
// Mace <ce.ma@kuleuven.be>
//
// Header file for functions in snax-dimc-lib.c
//-------------------------------

#ifndef SNAX_DIMC_LIB_H
#define SNAX_DIMC_LIB_H

#include <stdbool.h>
#include "snax-dimc-csr.h"
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
    return csrr_ss(DIMC_STREAMER_BUSY);
}

//-------------------------------
// DIMC accelerator functions
//-------------------------------

// query if the accelerator is busy
void dimc_query_busy() {
    csrw_ss(DIMC_ACC_BUSY, 1);
}

// load the activation for convolution
void dimc_load_act_conv() {
    csrw_ss(DIMC_LOAD_CONV_ACTIVATION, 1);
}

// start the convolution
void dimc_start_conv() {
    csrw_ss(DIMC_START_CONV, 1);
}

// load the kernel for convolution
void dimc_load_kernel() {
    csrw_ss(DIMC_LOAD_KERNEL, 1);
}

// start the multi-head attention
void dimc_start_mha() {
    csrw_ss(DIMC_START_MHA, 1);
}

// set the alpha for QKV
void dimc_set_alpha_qkv(uint32_t alpha) {
    csrw_ss(DIMC_SET_ALPHA_QKV, alpha);
}

// set the alpha for QKT
void dimc_set_alpha_qkt(uint32_t alpha) {
    csrw_ss(DIMC_SET_ALPHA_QKT, alpha);
}

// set the alpha for convolution
void dimc_set_alpha_conv(uint32_t alpha) {
    csrw_ss(DIMC_SET_ALPHA_CONV, alpha);
}

#endif // SNAX_DIMC_LIB_H
