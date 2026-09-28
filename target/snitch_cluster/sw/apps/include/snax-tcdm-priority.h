// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Run-time control of the narrow TCDM interconnect's arbitration.
//
// Every bank serves the highest level among the requests that want it, round robin within that
// level. By default the hardware sets every level:
//
//   0 .. 3  the requester's urgency: how close it is to stalling the engine it serves, graded by
//           each streamer channel from its own FIFO (a reader by the beats it holds, a writer by
//           its free slots); the xDMA reports a fixed 2
//   4       the starvation guard: a request that has waited GUARD_THRESHOLD cycles
//
// Software has the last word. It can switch the urgency or the guard off, change the
// threshold, and pin any input at a level from 0 to 7: 5 to 7 is above everything the hardware
// grades, 0 is below it and never promoted. One hart installs a policy, at any time; an app
// installs the one that suits its kernels, and may change it between phases.
//
// The wide DMA port (the iDMA) is outside this arbitration: it takes its whole super bank
// whenever it is valid. Inputs are numbered as snax-tcdm-ports-defs.h lists them, so a pin is
// built from its SNAX_TCDM_<ENGINE>_<GROUP>_BASE/_PORTS pairs and stays right when the cluster
// configuration moves the ports.

#pragma once

#include <stdint.h>

#include "snax-tcdm-ports-defs.h"
#include "snitch_cluster_peripheral.h"

#define SNAX_TCDM_LEVEL_GUARD 4u
#define SNAX_TCDM_LEVEL_MAX 7u
#define SNAX_TCDM_GUARD_THRESHOLD_RESET 16u
#define SNAX_TCDM_ARB_WORDS SNITCH_CLUSTER_PERIPHERAL_PARAM_NUM_TCDM_ARB_OVERRIDE_REGS
// The override words that hold this cluster's inputs, eight inputs a word.
#define SNAX_TCDM_ARB_USED_WORDS ((SNAX_TCDM_NUM_PORTS + 7u) / 8u)

typedef struct {
    uint32_t urgency_en;
    uint32_t guard_en;
    uint32_t guard_threshold;
    uint32_t pin[SNAX_TCDM_ARB_USED_WORDS];  // four bits an input: bit 3 pins, bits 2..0 level
} snax_tcdm_arb_t;

// The hardware's own arbitration: urgency and the guard on, the reset threshold, nothing
// pinned.
static inline void snax_tcdm_arb_default(snax_tcdm_arb_t *a) {
    a->urgency_en = 1u;
    a->guard_en = 1u;
    a->guard_threshold = SNAX_TCDM_GUARD_THRESHOLD_RESET;
    for (uint32_t w = 0; w < SNAX_TCDM_ARB_USED_WORDS; w++) a->pin[w] = 0u;
}

// Pin inputs base .. base + count - 1 at `level`.
static inline void snax_tcdm_arb_pin(snax_tcdm_arb_t *a, uint32_t base, uint32_t count,
                                     uint32_t level) {
    for (uint32_t p = base; p < base + count && p < SNAX_TCDM_NUM_PORTS; p++) {
        const uint32_t sh = 4u * (p & 7u);
        a->pin[p >> 3] = (a->pin[p >> 3] & ~(0xFu << sh)) | ((0x8u | (level & 7u)) << sh);
    }
}

static inline volatile uint32_t *snax_tcdm_arb_reg(uint32_t offset) {
    return (volatile uint32_t *)(snrt_cluster_perf_counters_addr() + offset);
}

static inline uint32_t snax_tcdm_arb_ctrl_word(const snax_tcdm_arb_t *a) {
    return (a->urgency_en ? 1u : 0u) | (a->guard_en ? 2u : 0u) |
           ((a->guard_threshold & 0xFFu) << 8);
}

// Install: the pins first, then the control word, then a read-back, which returns once the
// peripheral holds the new control word, so whatever the caller starts next runs under it.
static inline void snax_tcdm_arb_install(const snax_tcdm_arb_t *a) {
    for (uint32_t w = 0; w < SNAX_TCDM_ARB_USED_WORDS; w++)
        *snax_tcdm_arb_reg(SNITCH_CLUSTER_PERIPHERAL_TCDM_ARB_OVERRIDE_0_REG_OFFSET + 8u * w) =
            a->pin[w];
    *snax_tcdm_arb_reg(SNITCH_CLUSTER_PERIPHERAL_TCDM_ARB_CTRL_REG_OFFSET) =
        snax_tcdm_arb_ctrl_word(a);
    (void)*snax_tcdm_arb_reg(SNITCH_CLUSTER_PERIPHERAL_TCDM_ARB_CTRL_REG_OFFSET);
}

static inline uint32_t snax_tcdm_arb_ctrl_read(void) {
    return *snax_tcdm_arb_reg(SNITCH_CLUSTER_PERIPHERAL_TCDM_ARB_CTRL_REG_OFFSET);
}

// Named policies, for tuning an app by hand: each is the hardware's own arbitration with one
// change.
enum {
    SNAX_TCDM_POLICY_HW = 0,           // the hardware's own: urgency and the guard
    SNAX_TCDM_POLICY_ROUND_ROBIN = 1,  // urgency and guard off: one round robin
    SNAX_TCDM_POLICY_NO_GUARD = 2,     // urgency alone
    SNAX_TCDM_POLICY_GEMM_FIRST = 3,   // every GEMM port pinned at 5
    SNAX_TCDM_POLICY_XDMA_FIRST = 4,   // every xDMA port pinned at 5
    SNAX_TCDM_POLICY_SIMD_FIRST = 5,   // every SIMD port pinned at 5
    SNAX_TCDM_POLICY_XDMA_LAST = 6,    // every xDMA port pinned at 0: its moves yield to all
    SNAX_TCDM_NUM_POLICIES
};

static inline const char *snax_tcdm_policy_name(uint32_t policy) {
    static const char *const names[SNAX_TCDM_NUM_POLICIES] = {
        "hardware", "round robin", "no guard", "GEMM first", "xDMA first", "SIMD first",
        "xDMA last"};
    return policy < SNAX_TCDM_NUM_POLICIES ? names[policy] : "?";
}

// Install a named policy. guard_threshold: the guard's wait in cycles, 0 for the reset value.
static inline void snax_tcdm_policy_install(uint32_t policy, uint32_t guard_threshold) {
    snax_tcdm_arb_t a;
    snax_tcdm_arb_default(&a);
    if (guard_threshold) a.guard_threshold = guard_threshold;
    switch (policy) {
        case SNAX_TCDM_POLICY_ROUND_ROBIN:
            a.urgency_en = 0u;
            a.guard_en = 0u;
            break;
        case SNAX_TCDM_POLICY_NO_GUARD:
            a.guard_en = 0u;
            break;
#ifdef SNAX_TCDM_GEMM_RW0_BASE
        case SNAX_TCDM_POLICY_GEMM_FIRST:
            snax_tcdm_arb_pin(&a, SNAX_TCDM_GEMM_RD0_BASE,
                              SNAX_TCDM_GEMM_RD0_PORTS + SNAX_TCDM_GEMM_RD1_PORTS +
                                  SNAX_TCDM_GEMM_RW0_PORTS,
                              5u);
            break;
#endif
#ifdef SNAX_TCDM_XDMA_RD_BASE
        case SNAX_TCDM_POLICY_XDMA_FIRST:
            snax_tcdm_arb_pin(&a, SNAX_TCDM_XDMA_RD_BASE,
                              SNAX_TCDM_XDMA_RD_PORTS + SNAX_TCDM_XDMA_WR_PORTS, 5u);
            break;
        case SNAX_TCDM_POLICY_XDMA_LAST:
            snax_tcdm_arb_pin(&a, SNAX_TCDM_XDMA_RD_BASE,
                              SNAX_TCDM_XDMA_RD_PORTS + SNAX_TCDM_XDMA_WR_PORTS, 0u);
            break;
#endif
#ifdef SNAX_TCDM_SIMD_RD_BASE
        case SNAX_TCDM_POLICY_SIMD_FIRST:
            snax_tcdm_arb_pin(&a, SNAX_TCDM_SIMD_RD_BASE,
                              SNAX_TCDM_SIMD_RD_PORTS + SNAX_TCDM_SIMD_WR_PORTS, 5u);
            break;
#endif
        default:
            break;
    }
    snax_tcdm_arb_install(&a);
}
