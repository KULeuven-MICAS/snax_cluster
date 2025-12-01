// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

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
#define D_FINAL (SIMBACORE_CSR_ADDR_BASE + 5)
#define SIMBACORE_START (SIMBACORE_CSR_ADDR_BASE + 6)

// SimbaCore read-only CSR
#define SIMBACORE_BUSY (SIMBACORE_CSR_ADDR_BASE + 7)
#define SIMBACORE_PERFORMANCE_COUNTER (SIMBACORE_CSR_ADDR_BASE + 8)
#define R10_DELAY_GAUGE (SIMBACORE_CSR_ADDR_BASE + 9)   // Number of tiles that have been streamed out of osCore
#define R11_DELAY_GAUGE (SIMBACORE_CSR_ADDR_BASE + 10)  // Number of elements that have been streamed out of SUC

void set_streamer_csr(uint32_t R0_ptr, int32_t* R0_ss, int32_t* R0_tb, int32_t* R0_ts, bool R0_en,       //
                      uint32_t R1_ptr, int32_t* R1_ss, int32_t* R1_tb, int32_t* R1_ts, bool R1_en,       //
                      uint32_t R2_ptr, int32_t* R2_ss, int32_t* R2_tb, int32_t* R2_ts, bool R2_en,       //
                      uint32_t R3_ptr, int32_t* R3_ss, int32_t* R3_tb, int32_t* R3_ts, bool R3_en,       //
                      uint32_t R4_ptr, int32_t* R4_ss, int32_t* R4_tb, int32_t* R4_ts, bool R4_en,       //
                      uint32_t R5_ptr, int32_t* R5_ss, int32_t* R5_tb, int32_t* R5_ts, bool R5_en,       //
                      uint32_t R6_ptr, int32_t* R6_ss, int32_t* R6_tb, int32_t* R6_ts, bool R6_en,       //
                      uint32_t R7_ptr, int32_t* R7_ss, int32_t* R7_tb, int32_t* R7_ts, bool R7_en,       //
                      uint32_t R8_ptr, int32_t* R8_ss, int32_t* R8_tb, int32_t* R8_ts, bool R8_en,       //
                      uint32_t R9_ptr, int32_t* R9_ss, int32_t* R9_tb, int32_t* R9_ts, bool R9_en,       //
                      uint32_t R10_ptr, int32_t* R10_ss, int32_t* R10_tb, int32_t* R10_ts, bool R10_en,  //
                      uint32_t R11_ptr, int32_t* R11_ss, int32_t* R11_tb, int32_t* R11_ts, bool R11_en,  //
                      uint32_t R12_ptr, int32_t* R12_ss, int32_t* R12_tb, int32_t* R12_ts, bool R12_en,  //
                      uint32_t R13_ptr, int32_t* R13_ss, int32_t* R13_tb, int32_t* R13_ts, bool R13_en,  //

                      uint32_t W0_ptr, int32_t* W0_ss, int32_t* W0_tb, int32_t* W0_ts, bool W0_en,  //
                      uint32_t W1_ptr, int32_t* W1_ss, int32_t* W1_tb, int32_t* W1_ts, bool W1_en,  //
                      uint32_t W2_ptr, int32_t* W2_ss, int32_t* W2_tb, int32_t* W2_ts, bool W2_en,  //
                      uint32_t W3_ptr, int32_t* W3_ss, int32_t* W3_tb, int32_t* W3_ts, bool W3_en);

void set_osgemm_streamer_csr(uint32_t A_ptr, int32_t* A_ss, int32_t* A_tb, int32_t* A_ts,  //
                             uint32_t B_ptr, int32_t* B_ss, int32_t* B_tb, int32_t* B_ts,  //
                             uint32_t D_ptr, int32_t* D_ss, int32_t* D_tb, int32_t* D_ts);

void set_isgemm_streamer_csr(uint32_t A_ptr, int32_t* A_ss, int32_t* A_tb, int32_t* A_ts,  //
                             uint32_t B_ptr, int32_t* B_ss, int32_t* B_tb, int32_t* B_ts,  //
                             uint32_t CD_ptr, int32_t* CD_ss, int32_t* CD_tb, int32_t* CD_ts);

void set_simd_streamer_csr(uint32_t A_ptr, int32_t* A_ss, int32_t* A_tb, int32_t* A_ts,  //
                           uint32_t B_ptr, int32_t* B_ss, int32_t* B_tb, int32_t* B_ts,  //
                           uint32_t C_ptr, int32_t* C_ss, int32_t* C_tb, int32_t* C_ts);

// Set GEMM configuration CSR. dFinal is the IScore output dimension (either xProjDim or dModel)
void set_simbacore_csr(uint32_t mode, uint32_t seqLen, uint32_t dModel, uint32_t dInner, uint32_t dtRank,
                       uint32_t dFinal);

// Set CSR to start SimbaCore
static inline void _set_simbacore_start() { write_csr(SIMBACORE_START, 1); }

static inline void _set_streamer_start() { write_csr(STREAMER_START_CSR, 1); }
void start_simbacore_and_streamers(bool R10_en, uint32_t R10_start_cnt, bool R11_en, uint32_t R11_start_cnt);

// Poll until Streamer and SimbaCore accelerator finish
void wait_simbacore_and_streamer();

// Read performance counter of the Streamer, a read-only CSR
uint32_t read_streamer_perf_counter();

// Read performance counter of GEMM, a read-only CSR
uint32_t read_simbacore_perf_counter();

uint32_t check_result_all(uint8_t* output, uint8_t* output_golden, int32_t data_length);
uint32_t check_result_sample(uint8_t* output, uint8_t* output_golden, int32_t* sample_indices,
                             int32_t test_sample_count, const char* tensor_name);
uint32_t check_result_sample_u16(uint16_t* output, uint16_t* output_golden, int32_t* sample_indices,
                                 int32_t test_sample_count, const char* tensor_name);
