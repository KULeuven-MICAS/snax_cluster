// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

//===============================================================
// Constants
//===============================================================

#define CLUSTER_CLINT_SET_ADDR  \
    (CLUSTER_PERIPH_BASE_ADDR + \
     SNITCH_CLUSTER_PERIPHERAL_CL_CLINT_SET_REG_OFFSET)
#define CLUSTER_CLINT_CLR_ADDR  \
    (CLUSTER_PERIPH_BASE_ADDR + \
     SNITCH_CLUSTER_PERIPHERAL_CL_CLINT_CLEAR_REG_OFFSET)

#define CLUSTER_HW_BARRIER_ADDR \
    (CLUSTER_PERIPH_BASE_ADDR + SNITCH_CLUSTER_PERIPHERAL_HW_BARRIER_REG_OFFSET)

#define CLUSTER_PERF_COUNTER_ADDR \
    (CLUSTER_PERIPH_BASE_ADDR +   \
     SNITCH_CLUSTER_PERIPHERAL_PERF_COUNTER_ENABLE_0_REG_OFFSET)

#define CLUSTER_TCDM_START_ADDR CLUSTER_TCDM_BASE_ADDR

#define CLUSTER_TCDM_END_ADDR CLUSTER_PERIPH_BASE_ADDR

//===============================================================
// snRuntime interface functions
//===============================================================

// Place an object in the cluster TCDM instead of DRAM.
//
// The default for a `static` or a global is .bss/.data, which this target maps
// to L3 -- correct for cold data, expensive for anything a kernel reads per
// beat. SNRT_L1_DATA moves the object into the .l1 section, which base.ld maps
// to the TCDM.
//
// The section is NOLOAD, so an SNRT_L1_DATA object is NOT zero-initialised and
// cannot carry an initialiser: write it before you read it. It is also shared
// by every hart in the cluster, exactly like any other static -- give each hart
// its own element if they both write.
//
// The alternative, a large automatic, is usually wrong here: SNRT_LOG2_STACK_SIZE
// bounds a hart's stack, and overflowing it runs into the neighbouring hart's
// stack silently.
#define SNRT_L1_DATA __attribute__((section(".l1")))

// End of the .l1 output section (base.ld). Zero-sized when no object is
// declared SNRT_L1_DATA, in which case this is the plain TCDM base.
extern uint32_t __l1_end;

inline uint32_t __attribute__((const)) snrt_l1_start_addr() {
    // The heap starts after whatever .l1 claimed, not at the TCDM base --
    // otherwise the allocator would hand out memory the linker has already
    // given to SNRT_L1_DATA objects.
    return (uint32_t)&__l1_end;
}

inline uint32_t __attribute__((const)) snrt_l1_end_addr() {
    return CLUSTER_TCDM_END_ADDR;
}

inline volatile uint32_t* __attribute__((const)) snrt_cluster_clint_set_ptr() {
    return (uint32_t*)CLUSTER_CLINT_SET_ADDR;
}

inline volatile uint32_t* __attribute__((const)) snrt_cluster_clint_clr_ptr() {
    return (uint32_t*)CLUSTER_CLINT_CLR_ADDR;
}

inline uint32_t __attribute__((const)) snrt_cluster_hw_barrier_addr() {
    return CLUSTER_HW_BARRIER_ADDR;
}

inline uint32_t __attribute__((const)) snrt_cluster_perf_counters_addr() {
    return CLUSTER_PERF_COUNTER_ADDR;
}

inline volatile uint32_t* __attribute__((const)) snrt_zero_memory_ptr() {
    return (uint32_t*)CLUSTER_ZERO_MEM_START_ADDR;
}
