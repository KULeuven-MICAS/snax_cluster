// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//-------------------------------
// Mace <ce.ma@kuleuven.be>
//
// Header file for functions in snax-dimc-lib.c
//-------------------------------

#pragma once
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
                              uint32_t base_ptr
);

void dimc_set_streamer_dim_r1(uint32_t loop_bound_0,
                              uint32_t loop_bound_1,
                              uint32_t temp_stride_0,
                              uint32_t temp_stride_1,
                              uint32_t spat_stride, 
                              uint32_t base_ptr
);

void dimc_set_streamer_dim_r2(uint32_t loop_bound_0,
                              uint32_t loop_bound_1,
                              uint32_t temp_stride_0,
                              uint32_t temp_stride_1,
                              uint32_t spat_stride, 
                              uint32_t base_ptr
);

void dimc_set_streamer_dim_r3(uint32_t loop_bound_0,
                              uint32_t loop_bound_1,
                              uint32_t temp_stride_0,
                              uint32_t temp_stride_1,
                              uint32_t spat_stride, 
                              uint32_t base_ptr
);

void dimc_set_streamer_dim_w(uint32_t loop_bound_0,
                             uint32_t loop_bound_1,
                             uint32_t temp_stride_0,
                             uint32_t temp_stride_1,
                             uint32_t spat_stride, 
                             uint32_t base_ptr
);

void dimc_start_streamer();
void dimc_read_perf_counter();

// return true if the streamers are busy
bool dimc_is_streamer_busy();

//-------------------------------
// DIMC accelerator functions
//-------------------------------

// query if the accelerator is busy
uint32_t dimc_query_busy();

// load the activation for convolution
void dimc_load_act_conv();

// start the convolution
void dimc_start_conv();

// load the kernel for convolution
void dimc_load_kernel();

// start the multi-head attention
void dimc_start_mha();

void configure_accelerator();
uint32_t read_zp();