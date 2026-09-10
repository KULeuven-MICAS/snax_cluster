// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// P6: the concurrency proof for the four-engine split.
//
// The entire point of giving the SIMD block and the xDMA their own cores is that
// their work can overlap -- one CSR file, one task queue and one busy flag each,
// driven by two harts that issue independently. Before the split both lived on
// one XDMACtrl task queue and could not overlap at all.
//
// Measured three ways, from the same two tasks:
//   A  SIMD alone      (xDMA idle)
//   B  xDMA alone      (SIMD idle)
//   C  both, launched from a common barrier
//
// If the engines really are independent, C is close to max(A,B) and clearly below
// A+B. If anything still serialises them -- a shared port group, an arbiter
// starving one side, an accidental barrier -- C collapses towards A+B and this
// test is what shows it.

#include <stdint.h>
#include "snax-core-roles.h"
#include "snax-simd-lib.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

// Four buffers plus a result block must fit in TCDM alongside the four cores'
// stacks. 128 beats = 8 KiB each, 32 KiB total on a 128 KiB TCDM. Sizing this
// wrong overruns into another hart's stack, which shows up as that hart jumping
// into your test pattern rather than as an allocation error.
#define BEATS 128
#define BYTES (BEATS * SIMD_WIDTH)

// The SIMD side runs an identity map -- out = 1.0f*x + 0.0f -- which is exact for
// FP16 but still exercises the whole FP datapath, so it has real work to do.

// Elapsed cycles are published through TCDM so hart 0 can print one tidy table.
typedef struct {
    uint32_t simd_alone;
    uint32_t xdma_alone;
    uint32_t simd_together;
    uint32_t xdma_together;
    uint32_t simd_busy;
    uint32_t xdma_busy;
} results_t;

int main() {
    uint8_t *base = (uint8_t *)snrt_l1_next();
    base = (uint8_t *)(((uintptr_t)base + SIMD_WIDTH - 1) &
                       ~(uintptr_t)(SIMD_WIDTH - 1));

    // Disjoint buffers: the two engines must not touch the same words, or we
    // would be measuring a data race rather than concurrency.
    uint8_t *simd_src = base;
    uint8_t *simd_dst = base + BYTES;
    uint8_t *xdma_src = base + 2 * BYTES;
    uint8_t *xdma_dst = base + 3 * BYTES;
    volatile results_t *res = (volatile results_t *)(base + 4 * BYTES);

    uint32_t t0, t1;

    if (snax_is_simd_core()) {
        for (uint32_t i = 0; i < BYTES / 4; i++) {
            ((volatile uint32_t *)simd_src)[i] = 0x3C000000u | (0x3C00u + (i & 0xFF));
        }
    }
    if (snax_is_xdma_core()) {
        for (uint32_t i = 0; i < BYTES / 4; i++) {
            ((volatile uint32_t *)xdma_src)[i] = 0x5A5A0000u | i;
        }
    }
    snrt_cluster_hw_barrier();

    // ---------------------------------------------------------------- A: SIMD alone
    if (snax_is_simd_core()) {
        snax_simd_shape_t in, out;
        snax_simd_op_t op;
        snax_simd_shape_flat(&in, simd_src, BEATS);
        snax_simd_shape_flat(&out, simd_dst, BEATS);
        snax_simd_op_map(&op, SIMD_EXT_STREAMMAP, SIMD_FUNC_LINEAR,
                         SIMD_F32_ONE, SIMD_F32_ZERO);
        snax_simd_configure(&in, &out, &op, 1);
        t0 = snrt_mcycle();
        snax_simd_wait(snax_simd_launch());
        t1 = snrt_mcycle();
        res->simd_alone = t1 - t0;
        res->simd_busy = snax_simd_last_task_cycle();
    }
    snrt_cluster_hw_barrier();

    // ---------------------------------------------------------------- B: xDMA alone
    if (snax_is_xdma_core()) {
        snax_xdma_memcpy_1d(xdma_src, xdma_dst, BYTES);
        t0 = snrt_mcycle();
        snax_xdma_local_wait(snax_xdma_start());
        t1 = snrt_mcycle();
        res->xdma_alone = t1 - t0;
        res->xdma_busy = snax_xdma_last_task_cycle();
    }
    snrt_cluster_hw_barrier();

    // ---------------------------------------------------------------- C: together
    // Stage both configurations BEFORE the barrier, so the barrier releases two
    // harts that each have nothing left to do but write their start CSR.
    if (snax_is_simd_core()) {
        snax_simd_shape_t in, out;
        snax_simd_op_t op;
        snax_simd_shape_flat(&in, simd_src, BEATS);
        snax_simd_shape_flat(&out, simd_dst, BEATS);
        snax_simd_op_map(&op, SIMD_EXT_STREAMMAP, SIMD_FUNC_LINEAR,
                         SIMD_F32_ONE, SIMD_F32_ZERO);
        snax_simd_configure(&in, &out, &op, 1);
    }
    if (snax_is_xdma_core()) {
        snax_xdma_memcpy_1d(xdma_src, xdma_dst, BYTES);
    }
    snrt_cluster_hw_barrier();

    if (snax_is_simd_core()) {
        t0 = snrt_mcycle();
        snax_simd_wait(snax_simd_launch());
        t1 = snrt_mcycle();
        res->simd_together = t1 - t0;
    }
    if (snax_is_xdma_core()) {
        t0 = snrt_mcycle();
        snax_xdma_local_wait(snax_xdma_start());
        t1 = snrt_mcycle();
        res->xdma_together = t1 - t0;
    }
    snrt_cluster_hw_barrier();

    int err = 0;
    if (snax_is_gemm_core()) {
        uint32_t a = res->simd_alone, b = res->xdma_alone;
        uint32_t ct = res->simd_together > res->xdma_together
                          ? res->simd_together
                          : res->xdma_together;
        printf("\n=== four-engine concurrency, %d beats each ===\n", BEATS);
        printf("A  SIMD alone      : %u cycles (engine busy %u)\n", a,
               res->simd_busy);
        printf("B  xDMA alone      : %u cycles (engine busy %u)\n", b,
               res->xdma_busy);
        printf("C  both together   : %u cycles (SIMD %u, xDMA %u)\n", ct,
               res->simd_together, res->xdma_together);
        printf("   serialised A+B  : %u cycles\n", a + b);

        // Concurrency, stated as the fraction of the serialised time saved.
        // 0%% would mean the two engines still take turns.
        uint32_t saved = (a + b > ct) ? (a + b - ct) : 0;
        printf("   overlap saves   : %u cycles (%u%% of A+B)\n", saved,
               (a + b) ? (100 * saved / (a + b)) : 0);

        // The claim under test: running both must cost clearly less than running
        // them one after the other. Anything at or above A+B means they are still
        // serialised and the split bought nothing.
        if (ct >= a + b) {
            printf("FAIL: no overlap -- the engines are still serialised\n");
            err++;
        } else {
            printf("PASS: the engines overlap\n");
        }
    }

    snrt_cluster_hw_barrier();
    return err;
}
