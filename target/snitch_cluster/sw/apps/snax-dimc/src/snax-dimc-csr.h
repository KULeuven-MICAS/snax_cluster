// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//-------------------------------
// Mace <ce.ma@kuleuven.be>
//
// Header: Headerfile for built-in DIMC CSRs
//
// This headerfile contains the addresses of the DIMC
// accelerator's CSRs.
//-------------------------------
#ifndef SNAX_DIMC_CSR_H
#define SNAX_DIMC_CSR_H

// Starting addresses and offsets
#define SNAX_CSR_BASE 960
#define DIMC_STREAMER_OFFSET 34

// Loop bound registers per streamer, first configured, second 0
#define DIMC_STREAMER_R_0_LOOP_BOUND_0 (SNAX_CSR_BASE )
#define DIMC_STREAMER_R_0_LOOP_BOUND_1 (SNAX_CSR_BASE + 1)

#define DIMC_STREAMER_R_1_LOOP_BOUND_0 (SNAX_CSR_BASE + 2)
#define DIMC_STREAMER_R_1_LOOP_BOUND_1 (SNAX_CSR_BASE + 3)

#define DIMC_STREAMER_R_2_LOOP_BOUND_0 (SNAX_CSR_BASE + 4)
#define DIMC_STREAMER_R_2_LOOP_BOUND_1 (SNAX_CSR_BASE + 5)

#define DIMC_STREAMER_R_3_LOOP_BOUND_0 (SNAX_CSR_BASE + 6)
#define DIMC_STREAMER_R_3_LOOP_BOUND_1 (SNAX_CSR_BASE + 7)

#define DIMC_STREAMER_W_0_LOOP_BOUND_0 (SNAX_CSR_BASE + 8)
#define DIMC_STREAMER_W_0_LOOP_BOUND_1 (SNAX_CSR_BASE + 9)

// Temporal stride registers per streamer, first one needs to be configures, second 0
#define DIMC_STREAMER_R_0_TEMP_STRIDE_0 (SNAX_CSR_BASE + 10)
#define DIMC_STREAMER_R_0_TEMP_STRIDE_1 (SNAX_CSR_BASE + 11)

#define DIMC_STREAMER_R_1_TEMP_STRIDE_0 (SNAX_CSR_BASE + 12)
#define DIMC_STREAMER_R_1_TEMP_STRIDE_1 (SNAX_CSR_BASE + 13)

#define DIMC_STREAMER_R_2_TEMP_STRIDE_0 (SNAX_CSR_BASE + 14)
#define DIMC_STREAMER_R_2_TEMP_STRIDE_1 (SNAX_CSR_BASE + 15)

#define DIMC_STREAMER_R_3_TEMP_STRIDE_0 (SNAX_CSR_BASE + 16)
#define DIMC_STREAMER_R_3_TEMP_STRIDE_1 (SNAX_CSR_BASE + 17)

#define DIMC_STREAMER_W_0_TEMP_STRIDE_0 (SNAX_CSR_BASE + 18)
#define DIMC_STREAMER_W_0_TEMP_STRIDE_1 (SNAX_CSR_BASE + 19)

// Spatial stride registers per streamer, these can be 0
#define DIMC_STREAMER_R_0_SPAT_STRIDE (SNAX_CSR_BASE + 20)
#define DIMC_STREAMER_R_1_SPAT_STRIDE (SNAX_CSR_BASE + 21)
#define DIMC_STREAMER_R_2_SPAT_STRIDE (SNAX_CSR_BASE + 22)
#define DIMC_STREAMER_R_3_SPAT_STRIDE (SNAX_CSR_BASE + 23)
#define DIMC_STREAMER_W_0_SPAT_STRIDE (SNAX_CSR_BASE + 24)

// Base pointer registers per streamer, these should be configured
#define DIMC_STREAMER_R_0_BASE_PTR (SNAX_CSR_BASE + 35)
#define DIMC_STREAMER_R_1_BASE_PTR (SNAX_CSR_BASE + 26)
#define DIMC_STREAMER_R_2_BASE_PTR (SNAX_CSR_BASE + 27)
#define DIMC_STREAMER_R_3_BASE_PTR (SNAX_CSR_BASE + 28)
#define DIMC_STREAMER_W_0_BASE_PTR (SNAX_CSR_BASE + 29)

// Other registers
#define DIMC_STREAMER_START (SNAX_CSR_BASE + 30)
#define DIMC_STREAMER_PERF_COUNTER (SNAX_CSR_BASE + 31)
#define DIMC_STREAMER_BUSY (SNAX_CSR_BASE + 32)

// DIMC accelerator registers
#define DIMC_CSR_OFFSET (SNAX_CSR_BASE + DIMC_STREAMER_OFFSET)
#define DIMC_ACC_BUSY (DIMC_CSR_OFFSET + 0)
#define DIMC_LOAD_CONV_ACTIVATION (DIMC_CSR_OFFSET + 1)
#define DIMC_START_CONV (DIMC_CSR_OFFSET + 2)
#define DIMC_LOAD_KERNEL (DIMC_CSR_OFFSET + 3)
#define DIMC_START_MHA (DIMC_CSR_OFFSET + 4)
#define DIMC_SET_ALPHA_QKV (DIMC_CSR_OFFSET + 5)
#define DIMC_SET_ALPHA_QKT (DIMC_CSR_OFFSET + 6)
#define DIMC_SET_ALPHA_CONV (DIMC_CSR_OFFSET + 7)

#endif // SNAX_DIMC_CSR_H
