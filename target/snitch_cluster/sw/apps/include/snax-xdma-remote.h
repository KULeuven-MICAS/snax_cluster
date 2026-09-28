// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// A contiguous main-memory -> L1 read on the xDMA, the second way into the cluster.
//
// The iDMA reads main memory as AXI reads: AR out of the cluster's wide port, the data back
// on R, then into the TCDM through the wide DMA port. The xDMA reads it through the endpoint
// that sits on main memory: the local engine sees a source that is not its own and forwards
// a cfg frame to that endpoint, which reads its memory and pushes the bytes back as AXI
// WRITES (AW/W) into this cluster's xDMA data window; the xDMA's writer then stores them
// through its eight 64-bit TCDM ports. The two transfers use opposite directions of the
// wide link and different TCDM paths, so they can run at the same time
// (sw/apps/snax-xdma-idma-read-bw measures how close to twice one port that gets).
//
// Every CSR address below is a compile-time constant. snax_xdma_memcpy_nd() builds its
// addresses in loops, and csrw_ss() with a computed address becomes a jump-table load out
// of main memory plus an indirect jump, about 92 cycles a write; here each write is one
// csrw. Only the xDMA's own hart can write these CSRs.

#pragma once

#include <stdint.h>
#include "snax-xdma-lib.h"

#define SNAX_XDMA_BEAT 64u

// Arm a read of `beats` 64-byte beats from `src` (main memory) to `dst` (L1), both
// contiguous. The 15 unused multicast destination slots keep their reset value of zero.
__attribute__((always_inline)) static inline void snax_xdma_read_arm(
    uint32_t dst, uint32_t src, uint32_t beats) {
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, src);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_SPATIAL_STRIDE_PTR, 8);
    snax_write_xdma_cfg_reg(XDMA_DST_SPATIAL_STRIDE_PTR, 8);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 0, beats);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 1, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 2, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 0, SNAX_XDMA_BEAT);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 1, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 2, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 4, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 0, beats);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 1, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 2, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 0, SNAX_XDMA_BEAT);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 1, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 2, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 4, 0);
    // Exactly the engine's eight channels on both sides: a reader told it owns 32 waits on
    // twenty-four that never answer, and the task never retires.
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, 0xFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, 0xFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, 0xFFFFFFFFu);
    // No extension and no junction on the way: whatever an earlier task left armed (a
    // memset's writer extension, a junction) would otherwise act on these bytes.
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLE_PTR, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLE_PTR, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_JCT_ENABLE_PTR, 0);
}

// Re-point an armed read. Both base addresses are rewritten, not just the destination: the
// engine consumes its address registers as it walks, so a task issued without restoring
// them reads from wherever the previous one ended.
__attribute__((always_inline)) static inline void snax_xdma_read_retask(
    uint32_t dst, uint32_t src, uint32_t beats) {
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, src);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 0, beats);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 0, beats);
}

// Start the armed task. The hardware decides whether it runs locally or on the peer and
// bumps that commit counter; the handle records which. Bounded like the wait below; on a
// timeout `*timeout` is set and the handle's id can never be reached.
static inline snax_xdma_task_t snax_xdma_start_bounded(uint32_t limit, int *timeout) {
    uint32_t l0 = snax_read_xdma_cfg_reg(XDMA_COMMIT_LOCAL_TASK_PTR);
    uint32_t r0 = snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR);
    snax_write_xdma_cfg_reg(XDMA_START_PTR, 1);
    for (uint32_t s = 0; s < limit; s++) {
        uint32_t l = snax_read_xdma_cfg_reg(XDMA_COMMIT_LOCAL_TASK_PTR);
        if (l != l0) return (snax_xdma_task_t){l, 0};
        uint32_t r = snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR);
        if (r != r0) return (snax_xdma_task_t){r, 1};
    }
    *timeout = 1;
    return (snax_xdma_task_t){0xFFFFFFFFu, 0};
}

// Wait for one issued task on the counter the hardware used for it, bounded: an xDMA that
// never answers otherwise presents as a simulation that never ends. Returns 1 on timeout.
// The task and finish counters are 8 bits and wrap: the test is snax_xdma_task_done's.
static inline int snax_xdma_wait_bounded(snax_xdma_task_t t, uint32_t limit) {
    if (t.remote) {
        for (uint32_t s = 0; s < limit; s++)
            if (snax_xdma_task_done(snax_read_xdma_cfg_reg(XDMA_FINISH_REMOTE_TASK_PTR),
                                    t.task_id))
                return 0;
    } else {
        for (uint32_t s = 0; s < limit; s++)
            if (snax_xdma_task_done(snax_read_xdma_cfg_reg(XDMA_FINISH_LOCAL_TASK_PTR),
                                    t.task_id))
                return 0;
    }
    return 1;
}
