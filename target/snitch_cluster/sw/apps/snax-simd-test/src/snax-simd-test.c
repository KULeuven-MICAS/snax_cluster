// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Bring-up test for the standalone SIMD block on the four-engine split cluster.
//
// This is the first thing that exercises <cluster>_simd inside a real cluster:
// its own hart, its own CSR file, the TCDM interconnect, real memory. The
// chiseltest suite covers the block in isolation; what is new here is the
// integration -- CSR window, TCDM port slice, and the fact that hart 1 drives
// it while the other three harts are doing nothing but waiting at barriers.
//
//   T1  plain copy                reader -> writer, no extension
//   T2  StreamMap identity        out = 1.0f * x + 0.0f, bit-exact for FP16
//   T3  two queued tasks          the 2-deep task queue under real latency
//
// T2 needs no golden data: multiplying an FP16 value by 1.0f in FP32 and
// narrowing back is exact, so the output must equal the input bit for bit. Any
// CSR-map or lane-ordering mistake shows up as a mismatch.

#include <stdint.h>
#include "snax-core-roles.h"
#include "snax-simd-lib.h"
#include "snrt.h"

#define BEATS 16
#define BYTES (BEATS * SIMD_WIDTH)
#define WORDS (BYTES / 4)

static int check(const char *what, volatile uint32_t *got,
                 volatile uint32_t *want, uint32_t words) {
    for (uint32_t i = 0; i < words; i++) {
        if (got[i] != want[i]) {
            printf("%s: word %u mismatch, got 0x%08x want 0x%08x\n", what, i,
                   got[i], want[i]);
            return 1;
        }
    }
    printf("%s: PASS\n", what);
    return 0;
}

int main() {
    int err = 0;

    uint8_t *base = (uint8_t *)snrt_l1_next();
    // Every SIMD buffer must be beat-aligned; the library refuses otherwise.
    base = (uint8_t *)(((uintptr_t)base + SIMD_WIDTH - 1) & ~(uintptr_t)(SIMD_WIDTH - 1));

    volatile uint32_t *src = (volatile uint32_t *)base;
    volatile uint32_t *dst1 = (volatile uint32_t *)(base + BYTES);
    volatile uint32_t *dst2 = (volatile uint32_t *)(base + 2 * BYTES);
    volatile uint32_t *dst3 = (volatile uint32_t *)(base + 3 * BYTES);

    if (snax_is_simd_core()) {
        // A pattern that is a valid pair of small FP16 values in every word, so
        // the same buffer serves the copy test and the FP identity test.
        // 0x3C00 = 1.0h, and successive words step the low half.
        for (uint32_t i = 0; i < WORDS; i++) {
            src[i] = 0x3C000000u | (0x3C00u + (i & 0xFF));
        }
        for (uint32_t i = 0; i < WORDS; i++) {
            dst1[i] = 0xDEADBEEFu;
            dst2[i] = 0xDEADBEEFu;
            dst3[i] = 0xDEADBEEFu;
        }
    }
    snrt_cluster_hw_barrier();

    if (snax_is_simd_core()) {
        printf("SIMD core is hart %u; block has %d extensions, %d RW CSRs\n",
               snrt_cluster_core_idx(), SIMD_EXT_NUM, SIMD_RW_CSR_NUM);

        // ---- T1: plain copy -- the degenerate task, no operator -------------
        if (snax_simd_copy((void *)src, (void *)dst1, BYTES) != 0) {
            printf("T1: configuration failed\n");
            err++;
        } else {
            printf("T1 copy: %u cycles (reader %u, writer %u)\n",
                   snax_simd_last_task_cycle(), snax_simd_last_read_cycle(),
                   snax_simd_last_write_cycle());
            err += check("T1 copy", dst1, src, WORDS);
        }

#ifdef SIMD_EXT_STREAMMAP
        // ---- T2: an operator -- out = 1.0f * x + 0.0f -----------------------
        // Identity through the full FP datapath, so it needs no golden data:
        // multiplying an FP16 value by 1.0f in FP32 and narrowing back is exact.
        if (snax_simd_map(SIMD_EXT_STREAMMAP, (void *)src, (void *)dst2, BEATS,
                          SIMD_FUNC_LINEAR, SIMD_F32_ONE,
                          SIMD_F32_ZERO) != 0) {
            printf("T2: configuration failed\n");
            err++;
        } else {
            printf("T2 map(identity): %u cycles\n",
                   snax_simd_last_task_cycle());
            err += check("T2 map(identity)", dst2, src, WORDS);
        }
#else
        printf("T2 skipped: this SIMD block has no StreamMap\n");
#endif

        // ---- T3: two queued tasks ------------------------------------------
        // Stage both before waiting, so the second sits in the task queue while
        // the first runs. An early retire would let them overlap and corrupt.
        snax_simd_shape_t a_in, a_out, b_in, b_out;
        snax_simd_shape_flat(&a_in, (void *)src, BEATS);
        snax_simd_shape_flat(&a_out, (void *)dst3, BEATS);
        snax_simd_shape_flat(&b_in, (void *)dst3, BEATS);
        snax_simd_shape_flat(&b_out, (void *)dst1, BEATS);
        if (snax_simd_configure(&a_in, &a_out, 0, 0) != 0) {
            printf("T3: first configuration failed\n");
            err++;
        } else {
            uint32_t id1 = snax_simd_launch();
            if (snax_simd_configure(&b_in, &b_out, 0, 0) != 0) {
                printf("T3: second configuration failed\n");
                err++;
            } else {
                uint32_t id2 = snax_simd_launch();
                snax_simd_wait(id2);
                printf("T3 queued tasks: ids %u and %u\n", id1, id2);
                err += check("T3 queued task A", dst3, src, WORDS);
                err += check("T3 queued task B", dst1, src, WORDS);
            }
        }

        if (snax_simd_bad_config()) {
            printf("SIMD reported a bad-config task (status bit 1)\n");
            err++;
        }
        printf("SIMD bring-up finished with %d error(s)\n", err);
    }

    snrt_cluster_hw_barrier();
    return err;
}
