// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "snax-xdma-lib.h"
#include "snrt.h"

// XDMA_WIDTH is the complete beat in bytes. The same byte mask applies to
// each spatial channel, so mask positions repeat once per TCDM bank word.
enum {
    BANK_BYTES = XDMA_WIDTH / XDMA_SPATIAL_CHAN,
    COPY_BYTES = 4 * XDMA_WIDTH,
    SENTINEL = 0xa5,
    GUARD = 0xd3
};

static uint8_t source_byte(uint32_t index) {
    // Never equal SENTINEL, so a missed write cannot accidentally pass.
    return index & 0x7f;
}

static int transfer(const char* name, uint8_t* src, uint8_t* dst,
                    const uint32_t* mask_words) {
    volatile uint8_t* source = src;
    volatile uint8_t* destination = dst;
    volatile uint8_t* guarded = dst - XDMA_WIDTH;
    for (uint32_t i = 0; i < COPY_BYTES; i++) source[i] = source_byte(i);
    for (uint32_t i = 0; i < COPY_BYTES + 2 * XDMA_WIDTH; i++)
        guarded[i] = GUARD;
    for (uint32_t i = 0; i < COPY_BYTES; i++) destination[i] = SENTINEL;

    if (snax_xdma_memcpy_1d(src, dst, COPY_BYTES) != 0) {
        printf("[ByteMask] FAIL %s: copy configuration failed\n", name);
        return 1;
    }
    if (mask_words) snax_xdma_set_dst_byte_mask_words(mask_words);

    // Make CPU initialization visible before XDMA starts accessing TCDM.
    asm volatile("fence iorw, iorw" ::: "memory");
    uint32_t task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    asm volatile("fence iorw, iorw" ::: "memory");

    for (uint32_t i = 0; i < COPY_BYTES; i++) {
        uint32_t bank_byte = i % BANK_BYTES;
        uint32_t enabled =
            !mask_words ||
            ((mask_words[bank_byte / 32] >> (bank_byte % 32)) & 1);
        uint8_t expected = enabled ? source_byte(i) : SENTINEL;
        uint8_t actual = destination[i];
        if (actual != expected) {
            printf("[ByteMask] FAIL %s: byte %u, expected 0x%02x, got 0x%02x\n",
                   name, i, expected, actual);
            return 1;
        }
        if (source[i] != source_byte(i)) {
            printf("[ByteMask] FAIL %s: source byte %u changed\n", name, i);
            return 1;
        }
    }
    for (uint32_t i = 0; i < XDMA_WIDTH; i++) {
        if (guarded[i] != GUARD || destination[COPY_BYTES + i] != GUARD) {
            printf("[ByteMask] FAIL %s: destination guard changed\n", name);
            return 1;
        }
    }
    printf("[ByteMask] PASS %s\n", name);
    return 0;
}

int main() {
    if (!snrt_is_dm_core()) return 0;

    // Use fresh L1 space, aligned to a complete XDMA beat, with separate
    // source, destination and guard regions. Other cores do not access these
    // buffers.
    uintptr_t base = (uintptr_t)snrt_l1_next();
    base = ((base + XDMA_WIDTH - 1) / XDMA_WIDTH) * XDMA_WIDTH;
    uint8_t* src = (uint8_t*)base;
    uint8_t* dst = src + COPY_BYTES + XDMA_WIDTH;
    uint32_t mask_words[XDMA_DST_ENABLED_BYTE_CSR_NUM];

    for (uint32_t i = 0; i < XDMA_SRC_EXT_NUM; i++)
        snax_xdma_disable_src_ext(i);
    for (uint32_t i = 0; i < XDMA_DST_EXT_NUM; i++)
        snax_xdma_disable_dst_ext(i);

    printf("[ByteMask] bank bytes=%u, mask CSRs=%u, copy bytes=%u\n",
           BANK_BYTES, XDMA_DST_ENABLED_BYTE_CSR_NUM, COPY_BYTES);
    if (transfer("scalar-full", src, dst, NULL)) return 1;

    for (uint32_t i = 0; i < XDMA_DST_ENABLED_BYTE_CSR_NUM; i++)
        mask_words[i] = UINT32_MAX;
    if (transfer("word-full", src, dst, mask_words)) return 1;

    // Distinct adjacent words expose word ordering when a bank needs more
    // than one CSR. Narrow banks exercise the low bits of the first word.
    for (uint32_t i = 0; i < XDMA_DST_ENABLED_BYTE_CSR_NUM; i++)
        mask_words[i] = (i & 1) ? 0x5a5a5a5a : 0xa5a5a5a5;
    if (transfer("mixed-words", src, dst, mask_words)) return 1;

    for (uint32_t i = 0; i < XDMA_DST_ENABLED_BYTE_CSR_NUM; i++)
        mask_words[i] = 0;
    mask_words[(BANK_BYTES - 1) / 32] = 1u << ((BANK_BYTES - 1) % 32);
    if (transfer("highest-byte", src, dst, mask_words)) return 1;

    for (uint32_t i = 0; i < XDMA_DST_ENABLED_BYTE_CSR_NUM; i++)
        mask_words[i] = 0;
    if (transfer("clear-mask", src, dst, mask_words)) return 1;

    // The scalar API must restore every mask word after a cleared mask.
    if (transfer("scalar-full-after-clear", src, dst, NULL)) return 1;
    printf("[ByteMask] PASS all six transfers\n");
    return 0;
}
