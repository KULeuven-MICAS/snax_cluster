// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
// Robin Geens <robin.geens@esat.kuleuven.be>

#include <stdbool.h>

#include "snrt.h"
#include "stdint.h"
#include "streamer_csr_addr_map.h"

#pragma once

// SimbaCore CSR: 5
#define SIMBACORE_CSR_ADDR_BASE (STREAMER_PERFORMANCE_COUNTER_CSR + 1)
#define MODE (SIMBACORE_CSR_ADDR_BASE + 0)
#define SEQ_LEN (SIMBACORE_CSR_ADDR_BASE + 1)
#define D_MODEL (SIMBACORE_CSR_ADDR_BASE + 2)
#define DT_RANK (SIMBACORE_CSR_ADDR_BASE + 3)
#define D_INNER (SIMBACORE_CSR_ADDR_BASE + 4)
#define SIMBACORE_START (SIMBACORE_CSR_ADDR_BASE + 5)

// SimbaCore read-only CSR
#define SIMBACORE_BUSY (SIMBACORE_CSR_ADDR_BASE + 6)
#define SIMBACORE_PERFORMANCE_COUNTER (SIMBACORE_CSR_ADDR_BASE + 7)

void set_simbacore_oscore_streamer_csr(uint32_t ptr_a, int32_t* Aslstride, int32_t* Atlbound, int32_t* Atlstride,
                                       int32_t set_addr_remap_index_A, int32_t* channel_en_A,

                                       uint32_t ptr_b, int32_t* Bslstride, int32_t* Btlbound, int32_t* Btlstride,
                                       int32_t set_addr_remap_index_B, int32_t* channel_en_B,

                                       uint32_t ptr_d, int32_t* Dslstride, int32_t* Dtlbound, int32_t* Dtlstride,
                                       int32_t set_addr_remap_index_D, int32_t* channel_en_D);

// Set CSR to start STREAMER
static inline void set_simbacore_streamer_start() { write_csr(STREAMER_START_CSR, 1); }

// Set GEMM configuration CSR
void set_simbacore_csr(uint32_t mode, uint32_t seqLen, uint32_t dModel, uint32_t dInner, uint32_t dtRank);

// Set CSR to start GEMM
static inline void set_simbacore_start() { write_csr(SIMBACORE_START, 1); }

// Poll until Streamer and SimbaCore accelerator finish
void wait_simbacore_and_streamer();

void wait_simbacore();

// Read performance counter of the Streamer, a read-only CSR
uint32_t read_streamer_perf_counter();

// Read performance counter of GEMM, a read-only CSR
uint32_t read_simbacore_perf_counter();

uint32_t check_OSGeMM_result_all(uint16_t* output, uint16_t* output_golden, int32_t data_length);
uint32_t check_OSGeMM_result_sample(uint16_t* output, uint16_t* output_golden, int32_t* sample_indices,
                                    int32_t test_sample_count);
