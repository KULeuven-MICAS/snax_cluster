// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>
#include "snax-xdma-lib.h"

enum { SRC = 0x1000, DST = 0x2000, BANK_BYTES = 64, SIZE = 4 * BANK_BYTES };
enum { SENTINEL = 0xa5, GUARD = 0xd3 };

static uint8_t source_byte(unsigned index) { return index & 0x7f; }

static int transfer(const char* name, const uint32_t* mask) {
    for (unsigned i = 0; i < SIZE; i++) e2e_mem_write(SRC + i, source_byte(i));
    for (unsigned i = 0; i < SIZE + 2 * BANK_BYTES; i++)
        e2e_mem_write(DST - BANK_BYTES + i, GUARD);
    for (unsigned i = 0; i < SIZE; i++) e2e_mem_write(DST + i, SENTINEL);

    if (snax_xdma_memcpy_1d_full_addr(SRC, DST, SIZE) != 0) {
        fprintf(stderr, "FAIL %s: copy configuration failed\n", name);
        return 1;
    }
    if (mask) {
#ifdef XDMA_DST_ENABLED_BYTE_CSR_NUM
        // Configure the upper word through the real public driver API.
        snax_xdma_set_dst_byte_mask_words(mask);
#else
        // The unmodified pre-fix driver reaches the memory failure in the first
        // transfer. Keep this test buildable there for a before/after
        // comparison.
        fprintf(stderr, "FAIL %s: multiword masks are not supported\n", name);
        return 1;
#endif
    }
    uint32_t task = snax_xdma_start();
    snax_xdma_local_wait(task);

    for (unsigned i = 0; i < SIZE; i++) {
        unsigned byte = i % BANK_BYTES;
        unsigned enabled = !mask || ((mask[byte / 32] >> (byte % 32)) & 1);
        uint8_t expected = enabled ? source_byte(i) : SENTINEL;
        uint8_t actual = e2e_mem_read(DST + i);
        if (actual != expected) {
            fprintf(stderr,
                    "FAIL %s: byte %u (bank byte %u): expected 0x%02x, got "
                    "0x%02x\n",
                    name, i, byte, expected, actual);
            return 1;
        }
        if (e2e_mem_read(SRC + i) != source_byte(i)) {
            fprintf(stderr, "FAIL %s: source memory changed at byte %u\n", name,
                    i);
            return 1;
        }
    }
    for (unsigned i = 0; i < BANK_BYTES; i++) {
        if (e2e_mem_read(DST - BANK_BYTES + i) != GUARD ||
            e2e_mem_read(DST + SIZE + i) != GUARD) {
            fprintf(stderr, "FAIL %s: destination guard changed\n", name);
            return 1;
        }
    }
    printf(
        "PASS %s: copied enabled bytes; disabled bytes and guards preserved\n",
        name);
    return 0;
}

int byte_mask_e2e_main(void) {
    // A 512-bit bank needs 64 mask bits. The original one-CSR implementation
    // leaves bytes 32..63 unchanged and fails the first real memory comparison.
    if (transfer("full-mask-copy", NULL)) return 1;
    const uint32_t upper_only[2] = {0, UINT32_MAX};
    const uint32_t mixed[2] = {0xa5a5a5a5, 0x5a5a5a5a};
    const uint32_t disabled[2] = {0, 0};
    if (transfer("upper-half-only", upper_only)) return 1;
    if (transfer("distinct-mask-words", mixed)) return 1;
    if (transfer("clear-entire-mask", disabled)) return 1;
    puts("PASS byte-mask end-to-end: 512-bit bank, 4 completed transfers");
    return 0;
}
