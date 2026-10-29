// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// These accessors perform real CSR bus transactions against the Verilated RTL.
// They do not emulate registers or implement the XDMA transfer in software.
uint32_t csrr_ss(uint32_t address);
void csrw_ss(uint32_t address, uint32_t value);
void e2e_mem_write(uint32_t address, uint8_t value);
uint8_t e2e_mem_read(uint32_t address);
static inline uint32_t snrt_cluster_base_addrh(void) { return 0; }

#ifdef __cplusplus
}
#endif
