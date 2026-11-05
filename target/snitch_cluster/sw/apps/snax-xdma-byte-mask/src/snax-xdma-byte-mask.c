// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "snax-xdma-lib.h"
#include "snrt.h"

// XDMA_WIDTH is the complete beat in bytes. Array masks cover each byte of
// each channel independently; scalar APIs retain a repeated per-channel
// pattern.
_Static_assert(XDMA_DST_ENABLED_BYTE_CSR_NUM == (XDMA_WIDTH + 31) / 32,
               "The mask must cover every byte of every channel");
enum {
    CHANNEL_BYTES = XDMA_WIDTH / XDMA_SPATIAL_CHAN,
    COPY_BYTES = 4 * XDMA_WIDTH,
    SENTINEL = 0xa5,
    GUARD = 0xd3
};

static uint8_t source_byte(uint32_t index) {
    // Never equal SENTINEL, so a missed write cannot accidentally pass.
    return index & 0x7f;
}

static uint32_t passed_cases;

static int transfer(const char* name, uint8_t* src, uint8_t* dst,
                    const uint32_t* mask_words, uint32_t scalar_mask) {
    volatile uint8_t* source = src;
    volatile uint8_t* destination = dst;
    volatile uint8_t* guarded = dst - XDMA_WIDTH;
    for (uint32_t i = 0; i < COPY_BYTES; i++) source[i] = source_byte(i);
    for (uint32_t i = 0; i < COPY_BYTES + 2 * XDMA_WIDTH; i++)
        guarded[i] = GUARD;
    for (uint32_t i = 0; i < COPY_BYTES; i++) destination[i] = SENTINEL;

    int32_t result;
    if (scalar_mask == UINT32_MAX) {
        result = snax_xdma_memcpy_1d(src, dst, COPY_BYTES);
    } else {
        uint32_t stride = XDMA_WIDTH;
        uint32_t bound = COPY_BYTES / XDMA_WIDTH;
        result = snax_xdma_memcpy_nd(src, dst, CHANNEL_BYTES, CHANNEL_BYTES, 1,
                                     &stride, &bound, 1, &stride, &bound,
                                     UINT32_MAX, UINT32_MAX, scalar_mask);
    }
    if (result != 0) {
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
        uint32_t beat_byte = i % XDMA_WIDTH;
        uint32_t enabled =
            mask_words ? ((mask_words[beat_byte / 32] >> (beat_byte % 32)) & 1u)
                       : ((scalar_mask >> ((i % CHANNEL_BYTES) % 32)) & 1u);
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
    passed_cases++;
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

    printf("[ByteMask] channel bytes=%u, mask CSRs=%u, copy bytes=%u\n",
           CHANNEL_BYTES, XDMA_DST_ENABLED_BYTE_CSR_NUM, COPY_BYTES);
    if (transfer("scalar-full", src, dst, NULL, UINT32_MAX)) return 1;
    if (transfer("scalar-pattern", src, dst, NULL, 0x0f)) return 1;
    if (transfer("scalar-inverse-pattern", src, dst, NULL, 0xf0)) return 1;

    for (uint32_t i = 0; i < XDMA_DST_ENABLED_BYTE_CSR_NUM; i++)
        mask_words[i] = UINT32_MAX;
    if (transfer("word-full", src, dst, mask_words, UINT32_MAX)) return 1;

    // Keep every channel enabled: these cases must use the second mask CSR,
    // rather than selecting bytes by disabling entire narrow channels.
    if (XDMA_DST_ENABLED_BYTE_CSR_NUM > 1) {
        for (uint32_t i = 0; i < XDMA_DST_ENABLED_BYTE_CSR_NUM; i++)
            mask_words[i] = i == 0 ? 0 : UINT32_MAX;
        if (transfer("upper-words-only", src, dst, mask_words, UINT32_MAX))
            return 1;

        for (uint32_t i = 0; i < XDMA_DST_ENABLED_BYTE_CSR_NUM; i++)
            mask_words[i] = i == 0 ? UINT32_MAX : 0;
        if (transfer("lower-word-only", src, dst, mask_words, UINT32_MAX))
            return 1;
    }

    // Distinct words expose ordering across the complete multi-channel mask.
    for (uint32_t i = 0; i < XDMA_DST_ENABLED_BYTE_CSR_NUM; i++)
        mask_words[i] = (i & 1) ? 0x5a5a5a5a : 0xa5a5a5a5;
    if (transfer("mixed-words", src, dst, mask_words, UINT32_MAX)) return 1;

    // Each channel enables a different byte, checking slices within both CSRs.
    for (uint32_t i = 0; i < XDMA_DST_ENABLED_BYTE_CSR_NUM; i++)
        mask_words[i] = 0;
    for (uint32_t channel = 0; channel < XDMA_SPATIAL_CHAN; channel++) {
        uint32_t bit = channel * CHANNEL_BYTES + channel % CHANNEL_BYTES;
        mask_words[bit / 32] |= 1u << (bit % 32);
    }
    if (transfer("independent-channels", src, dst, mask_words, UINT32_MAX))
        return 1;

    for (uint32_t i = 0; i < XDMA_DST_ENABLED_BYTE_CSR_NUM; i++)
        mask_words[i] = 0;
    mask_words[(XDMA_WIDTH - 1) / 32] = 1u << ((XDMA_WIDTH - 1) % 32);
    if (transfer("highest-byte", src, dst, mask_words, UINT32_MAX)) return 1;

    for (uint32_t i = 0; i < XDMA_DST_ENABLED_BYTE_CSR_NUM; i++)
        mask_words[i] = 0;
    if (transfer("clear-mask", src, dst, mask_words, UINT32_MAX)) return 1;
    if (transfer("scalar-pattern-after-clear", src, dst, NULL, 0x0f)) return 1;

    // The scalar API must restore every mask word after a cleared mask.
    if (transfer("scalar-full-after-clear", src, dst, NULL, UINT32_MAX))
        return 1;
    printf("[ByteMask] PASS all %u transfers\n", passed_cases);
    return 0;
}
