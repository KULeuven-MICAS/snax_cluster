// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//-------------------------------
// Author: Ryan Antonio <ryan.antonio@esat.kuleuven.be>
//
// Program: Hypercorex Test CSRs Header File
//
// This program is to test the capabilities
// of the HyperCoreX accelerator's CSRs so the test is
// to check if registers are working as intended.
//
// This includes checking for RW, RO, and WO registers.
//-------------------------------

#include "snax-hypercorex-lib.h"
#include "snrt.h"

//-------------------------------
// Test values
//-------------------------------

// Test values for streamer RW
uint32_t test_streamer_test_val1 = 0xffffffff;
uint32_t test_streamer_test_val2 = 0x1234abcd;
uint32_t test_streamer_test_val3 = 0xdeadbeef;
uint32_t test_streamer_test_val4 = 0x01010101;
uint32_t test_streamer_test_val5 = 0x55551111;
uint32_t test_streamer_test_val6 = 0x7189cdef;
uint32_t test_streamer_test_val7 = 0x1f9d39a0;
uint32_t test_streamer_test_val8 = 0xbabec0de;
uint32_t test_streamer_test_val9 = 0xc0de9111;
uint32_t test_streamer_test_val10 = 0xabcdef01;
uint32_t test_streamer_test_val11 = 0x12345678;

// Need to leave core_config LSB as 0
// Since it starts the core
// All the rest should be okay
uint32_t test_core_config = 0xfffffffe;
uint32_t test_am_num_predict = 0xffffffff;
uint32_t test_am_predict = 0xffffffff;
uint32_t test_inst_ctrl = 0xffffffff;

uint32_t test_inst_write_addr = 0xffffffff;
uint32_t test_inst_write_data = 0xffffffff;
uint32_t test_inst_rddbg_addr = 0x000003ff;
uint32_t test_inst_pc_addr = 0xffffffff;
uint32_t test_inst_inst_at_addr = 0xffffffff;

uint32_t test_inst_loop_ctrl = 0xffffffff;
uint8_t test_inst_loop_jump_addr1 = 0xff;
uint8_t test_inst_loop_jump_addr2 = 0xff;
uint8_t test_inst_loop_jump_addr3 = 0xff;
uint8_t test_inst_loop_jump_addr4 = 0xff;
uint8_t test_inst_loop_end_addr1 = 0xff;
uint8_t test_inst_loop_end_addr2 = 0xff;
uint8_t test_inst_loop_end_addr3 = 0xff;
uint8_t test_inst_loop_end_addr4 = 0xff;
uint16_t test_inst_loop_count1 = 0xffff;
uint16_t test_inst_loop_count2 = 0xffff;
uint16_t test_inst_loop_count3 = 0xffff;
uint16_t test_inst_loop_count4 = 0xffff;

uint32_t test_data_slice_ctrl = 0xffffffff;
uint32_t test_data_slice_num_elem_a = 0xffffffff;
uint32_t test_data_slice_num_elem_b = 0xffffffff;

uint32_t test_auto_counter_start_a = 0xffffffff;
uint32_t test_auto_counter_start_b = 0xffffffff;
uint32_t test_auto_counter_num_a = 0xffffffff;
uint32_t test_auto_counter_num_b = 0xffffffff;

//-------------------------------
// Golden values
//-------------------------------

// Test values for streamer RW
uint32_t golden_streamer_test_val1 = 0xffffffff;
uint32_t golden_streamer_test_val2 = 0x1234abcd;
uint32_t golden_streamer_test_val3 = 0xdeadbeef;
uint32_t golden_streamer_test_val4 = 0x01010101;
uint32_t golden_streamer_test_val5 = 0x55551111;
uint32_t golden_streamer_test_val6 = 0x7189cdef;
uint32_t golden_streamer_test_val7 = 0x1f9d39a0;
uint32_t golden_streamer_test_val8 = 0xbabec0de;
uint32_t golden_streamer_test_val9 = 0xc0de9111;
uint32_t golden_streamer_test_val10 = 0xabcdef01;
uint32_t golden_streamer_test_val11 = 0x12345678;

uint32_t golden_core_config = 0x0000003c;
uint32_t golden_am_num_predict = 0xffffffff;
uint32_t golden_am_predict = 0x00000000;
uint32_t golden_inst_ctrl = 0x00000003;

uint32_t golden_inst_write_addr = 0x00000000;
uint32_t golden_inst_write_data = 0x00000000;
uint32_t golden_inst_rddbg_addr = 0x000003ff;
uint32_t golden_inst_pc_addr = 0x00000000;
uint32_t golden_inst_at_addr = 0xffffffff;

uint32_t golden_inst_loop_ctrl = 0x000003ff;
uint32_t golden_inst_loop_jump_addr = 0xffffffff;
uint32_t golden_inst_loop_end_addr = 0xffffffff;
uint32_t golden_inst_loop_count1 = 0xffffffff;
uint32_t golden_inst_loop_count2 = 0xffffffff;

uint32_t golden_data_slice_ctrl = 0x0000003f;
uint32_t golden_data_slice_num_elem_a = 0xffffffff;
uint32_t golden_data_slice_num_elem_b = 0xffffffff;

uint32_t golden_auto_counter_start_a = 0xffffffff;
uint32_t golden_auto_counter_start_b = 0xffffffff;
uint32_t golden_auto_counter_num_a = 0xffffffff;
uint32_t golden_auto_counter_num_b = 0xffffffff;
