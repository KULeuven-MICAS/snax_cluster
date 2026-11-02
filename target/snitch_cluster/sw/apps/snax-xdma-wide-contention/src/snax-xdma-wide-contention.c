// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "snax-xdma-lib.h"
#include "snrt.h"

_Static_assert(XDMA_WIDTH == 64 && XDMA_SPATIAL_CHAN == 1,
               "Build this test with the native 512-bit XDMA configuration");
_Static_assert(XDMA_DST_ENABLED_BYTE_CSR_NUM == 2,
               "The wide cluster must expose both byte-mask CSRs");

enum { BYTES = 4096, BEAT = 64, GUARD = 0xd3, SENTINEL = 0xa5 };

static uint8_t pattern(uint32_t i, uint32_t seed) { return (i + seed) & 0x7f; }

static int run_case(uint8_t* buffers, uint32_t round) {
    uint8_t* data[4];
    for (uint32_t b = 0; b < 4; b++) {
        volatile uint8_t* guarded = buffers + b * (BYTES + 2 * BEAT);
        for (uint32_t i = 0; i < BYTES + 2 * BEAT; i++) guarded[i] = GUARD;
        data[b] = (uint8_t*)guarded + BEAT;
        volatile uint8_t* bytes = data[b];
        for (uint32_t i = 0; i < BYTES; i++)
            bytes[i] = (b & 1) ? SENTINEL : pattern(i, round + 17 * b);
    }
    volatile uint32_t* cpu =
        (volatile uint32_t*)(buffers + 4 * (BYTES + 2 * BEAT));
    uint32_t masks[2] = {0, UINT32_MAX};
    if (round == 1) {
        masks[0] = UINT32_MAX;
        masks[1] = 0;
    } else if (round == 2) {
        masks[0] = 0xa5a5a5a5;
        masks[1] = 0x5a5a5a5a;
    } else if (round == 3) {
        masks[0] = masks[1] = UINT32_MAX;
    }

    if (snax_xdma_memcpy_1d(data[0], data[1], BYTES)) return 1;
    snax_xdma_set_dst_byte_mask_words(masks);
    asm volatile("fence iorw, iorw" ::: "memory");

    // XDMA read and write use two arbiter inputs. The original iDMA reaches
    // input zero through AXI. Launch both before waiting; all buffers span
    // the same four physical bank groups, at disjoint rows.
    uint32_t task;
    if (round & 1) {
        task = snax_xdma_start();
        snrt_dma_start_1d(data[3], data[2], BYTES);
    } else {
        snrt_dma_start_1d(data[3], data[2], BYTES);
        task = snax_xdma_start();
    }
    // Exercise the unchanged CPU/narrow path while the transfers progress.
    for (uint32_t i = 0; i < 512; i++) {
        cpu[i % 32] = 0x12340000 + i;
        if (cpu[i % 32] != 0x12340000 + i) return 1;
    }
    snax_xdma_local_wait(task);
    snrt_dma_wait_all();
    asm volatile("fence iorw, iorw" ::: "memory");

    for (uint32_t b = 0; b < 4; b++) {
        volatile uint8_t* bytes = data[b];
        for (uint32_t i = 0; i < BYTES; i++) {
            uint8_t expected;
            if (b == 1) {
                uint32_t bit = i % BEAT;
                expected = ((masks[bit / 32] >> (bit % 32)) & 1)
                               ? pattern(i, round)
                               : SENTINEL;
            } else {
                expected = pattern(i, round + 17 * (b & ~1u));
            }
            if (bytes[i] != expected) {
                printf("[WideContention] FAIL round=%u buffer=%u byte=%u\n",
                       round, b, i);
                return 1;
            }
        }
        for (uint32_t i = 0; i < BEAT; i++) {
            if (bytes[(int)i - BEAT] != GUARD || bytes[BYTES + i] != GUARD) {
                printf("[WideContention] FAIL buffer=%u guard\n", b);
                return 1;
            }
        }
    }
    for (uint32_t i = 0; i < 32; i++)
        if (cpu[i] != 0x12340000 + 480 + i) return 1;
    printf("[WideContention] PASS round=%u\n", round);
    return 0;
}

int main() {
    if (!snrt_is_dm_core()) return 0;
    uintptr_t base = (uintptr_t)snrt_l1_next();
    base = ((base + BEAT - 1) / BEAT) * BEAT;
    for (uint32_t round = 0; round < 4; round++)
        if (run_case((uint8_t*)base, round)) return 1;
    printf("[WideContention] PASS XDMA + iDMA + CPU, all four cases\n");
    return 0;
}
