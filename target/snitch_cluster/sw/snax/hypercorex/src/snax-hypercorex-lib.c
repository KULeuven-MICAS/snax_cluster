// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//-------------------------------
// Author: Ryan Antonio <ryan.antonio@esat.kuleuven.be>
//
// Library: Functions for Setting Hypecorex CSRs
//
// This pre-built library contains functions to set
// the HyperCoreX accelerator's CSRs.
//-------------------------------

#include "snax-hypercorex-csr.h"
#include "snrt.h"
#include "stdint.h"

// This is used for writing unto the
// instruction loop jump addresses
uint32_t hypercorex_set_inst_loop_jump_addr(uint8_t config1, uint8_t config2,
                                            uint8_t config3) {
    uint32_t config = (config3 << 16) | (config2 << 8) | config1;

    csrw_ss(HYPERCOREX_INST_LOOP_JUMP_ADDR_REG_ADDR, config);
    return 0;
};

// This is used for writing unto the
// instruction loop end addresses
uint32_t hypercorex_set_inst_loop_end_addr(uint8_t config1, uint8_t config2,
                                           uint8_t config3) {
    uint32_t config = (config3 << 16) | (config2 << 8) | config1;

    csrw_ss(HYPERCOREX_INST_LOOP_END_ADDR_REG_ADDR, config);
    return 0;
};

// This is used for writing unto the
// instruction loop count
uint32_t hypercorex_set_inst_loop_count(uint8_t config1, uint8_t config2,
                                        uint8_t config3) {
    uint32_t config = (config3 << 16) | (config2 << 8) | config1;

    csrw_ss(HYPERCOREX_INST_LOOP_COUNT_REG_ADDR, config);
    return 0;
};

// This is used for writing registers to multiple
// Orothoginal IM seeds
uint32_t hypercorex_set_im_base_seed(uint32_t im_idx, uint32_t config) {
    csrw_ss(HYPERCOREX_IM_BASE_SEED_REG_ADDR + im_idx, config);
    return 0;
};
