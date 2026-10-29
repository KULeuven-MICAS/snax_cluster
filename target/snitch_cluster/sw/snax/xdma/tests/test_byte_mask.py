# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Host-side tests of the real XDMA C driver's byte-mask CSR writes.

Run inside snax-shell:
    python target/snitch_cluster/sw/snax/xdma/tests/test_byte_mask.py

Only the SNRT CSR accessors and hardware address definitions are mocked.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

XDMA = Path(__file__).resolve().parents[1]
HEADER = """
#define XDMA_WIDTH BANK_BYTES
#define XDMA_SPATIAL_CHAN 1
#define XDMA_MAX_DST_COUNT 1
#define XDMA_SRC_ADDR_PTR_LSB 0
#define XDMA_SRC_ADDR_PTR_MSB 1
#define XDMA_DST_ADDR_PTR_LSB 2
#define XDMA_DST_ADDR_PTR_MSB 3
#define XDMA_SRC_SPATIAL_STRIDE_PTR 4
#define XDMA_SRC_TEMP_BOUND_PTR 5
#define XDMA_SRC_TEMP_STRIDE_PTR 6
#define XDMA_SRC_TEMP_DIM 1
#define XDMA_SRC_ENABLED_CHAN_PTR 7
#define XDMA_SRC_ENABLE_PTR 8
#define XDMA_SRC_EXT_CSR_PTR 8
#define XDMA_SRC_EXT_NUM 0
#define XDMA_SRC_EXT_CUSTOM_CSR_NUM {}
#define XDMA_DST_SPATIAL_STRIDE_PTR 8
#define XDMA_DST_TEMP_BOUND_PTR 9
#define XDMA_DST_TEMP_STRIDE_PTR 10
#define XDMA_DST_TEMP_DIM 1
#define XDMA_DST_ENABLED_CHAN_PTR 11
#define XDMA_DST_ENABLED_BYTE_PTR 12
#define XDMA_DST_ENABLED_BYTE_CSR_NUM MASK_WORDS
#define XDMA_DST_ENABLE_PTR (12 + MASK_WORDS)
#define XDMA_DST_EXT_CSR_PTR (12 + MASK_WORDS)
#define XDMA_DST_EXT_NUM 0
#define XDMA_DST_EXT_CUSTOM_CSR_NUM {}
#define XDMA_START_PTR (12 + MASK_WORDS)
#define XDMA_COMMIT_LOCAL_TASK_PTR (13 + MASK_WORDS)
#define XDMA_COMMIT_REMOTE_TASK_PTR (14 + MASK_WORDS)
#define XDMA_FINISH_LOCAL_TASK_PTR (15 + MASK_WORDS)
#define XDMA_FINISH_REMOTE_TASK_PTR (16 + MASK_WORDS)
#define XDMA_PERF_CTR_TASK (17 + MASK_WORDS)
#define XDMA_PERF_CTR_READER (18 + MASK_WORDS)
#define XDMA_PERF_CTR_WRITER (19 + MASK_WORDS)
"""
SNRT = """
#pragma once
#include <stdint.h>
#include <stdio.h>
uint32_t csrr_ss(uint32_t addr);
void csrw_ss(uint32_t addr, uint32_t value);
static inline uint32_t snrt_cluster_base_addrh(void) { return 0; }
"""
HARNESS = """
#include <assert.h>
#include "snax-xdma-lib.h"

static uint32_t regs[128];
static uint32_t writes[MASK_WORDS];
uint32_t csrr_ss(uint32_t addr) { return regs[addr - XDMA_CFG_ADDR]; }
void csrw_ss(uint32_t addr, uint32_t value) {
    uint32_t index = addr - XDMA_CFG_ADDR;
    assert(index < 128);
    regs[index] = value;
    if (index >= XDMA_DST_ENABLED_BYTE_PTR &&
        index < XDMA_DST_ENABLED_BYTE_PTR + MASK_WORDS)
        writes[index - XDMA_DST_ENABLED_BYTE_PTR]++;
}
static void reset_regs(void) {
    for (unsigned i = 0; i < 128; i++) regs[i] = 0xdeadbeef;
    for (unsigned i = 0; i < MASK_WORDS; i++) writes[i] = 0;
}
static void check_mask(const uint32_t* expected) {
    for (unsigned i = 0; i < MASK_WORDS; i++) {
        assert(regs[XDMA_DST_ENABLED_BYTE_PTR + i] == expected[i]);
        assert(writes[i] == 1);
    }
    // The next extension/start register must not be overwritten by mask setup.
    assert(regs[XDMA_DST_ENABLED_BYTE_PTR + MASK_WORDS] == 0xdeadbeef);
}
int main(void) {
    uint32_t patterns[4] = {0x89abcdef, 0x01234567, 0x76543210, 0xfedcba98};
    uint32_t zeros[4] = {0};
    uint32_t ones[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    reset_regs();
    snax_xdma_set_dst_byte_mask_words(patterns);
    check_mask(patterns);
    assert(regs[XDMA_DST_ENABLED_BYTE_PTR - 1] == 0xdeadbeef);
    for (unsigned i = 0; i < MASK_WORDS; i++) writes[i] = 0;
    snax_xdma_set_dst_byte_mask_words(zeros);
    check_mask(zeros);

    uint32_t stride = BANK_BYTES, bound = 1;
    reset_regs();
    assert(snax_xdma_memcpy_nd_full_addr(
        0x1000, 0x2000, BANK_BYTES, BANK_BYTES, 1, &stride, &bound,
        1, &stride, &bound, UINT32_MAX, UINT32_MAX, UINT32_MAX) == 0);
    check_mask(ones);
    reset_regs();
    uint32_t repeated[4] = {0x89abcdef, 0x89abcdef, 0x89abcdef, 0x89abcdef};
    assert(snax_xdma_memcpy_nd_full_addr(
        0x1000, 0x2000, BANK_BYTES, BANK_BYTES, 1, &stride, &bound,
        1, &stride, &bound, UINT32_MAX, UINT32_MAX, repeated[0]) == 0);
    check_mask(repeated);
    reset_regs();
    assert(snax_xdma_memcpy_nd_full_addr_fast(
        0x1000, 0x2000, BANK_BYTES, BANK_BYTES, 1, &stride, &bound,
        1, &stride, &bound, UINT32_MAX, UINT32_MAX, UINT32_MAX) == 0);
    check_mask(ones);
    reset_regs();
    uint64_t dst = 0x2000;
    assert(snax_xdma_multicast_nd_full_address(
        0x1000, &dst, 1, BANK_BYTES, BANK_BYTES, 1, &stride, &bound,
        1, &stride, &bound, UINT32_MAX, UINT32_MAX, UINT32_MAX) == 0);
    check_mask(ones);
    reset_regs();
    assert(snax_xdma_memcpy_1d_full_addr(0x1000, 0x2000, BANK_BYTES) == 0);
    check_mask(ones);
    return 0;
}
"""


class ByteMaskDriverTest(unittest.TestCase):
    def check_bank_width(self, width):
        compiler = shutil.which("clang") or shutil.which("cc")
        self.assertIsNotNone(compiler, "Run inside snax-shell with a host C compiler")
        with tempfile.TemporaryDirectory(prefix="xdma-byte-mask-") as work:
            directory = Path(work)
            # Copy the real public header so generated build headers cannot shadow the mock.
            shutil.copyfile(
                XDMA / "include/snax-xdma-lib.h", directory / "snax-xdma-lib.h"
            )
            (directory / "snrt.h").write_text(SNRT)
            (directory / "snax-xdma-addr.h").write_text(HEADER)
            (directory / "test.c").write_text(HARNESS)
            binary = directory / "test"
            subprocess.run(
                [
                    compiler, "-std=gnu11", "-Werror=implicit-function-declaration",
                    "-DBANK_BYTES=" + str(width // 8),
                    "-DMASK_WORDS=" + str((width // 8 + 31) // 32),
                    "-I" + str(directory),
                    str(XDMA / "src/snax-xdma-lib.c"), str(directory / "test.c"),
                    "-o", str(binary),
                ],
                check=True, capture_output=True, text=True,
            )
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    def test_64_bit_bank(self):
        self.check_bank_width(64)

    def test_512_bit_bank(self):
        self.check_bank_width(512)

    def test_1024_bit_bank(self):
        self.check_bank_width(1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
