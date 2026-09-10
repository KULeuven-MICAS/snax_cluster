// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snax-simd-lib.h"
#include <stdbool.h>
#include "stdint.h"

#define SIMD_DEBUG
#ifdef SIMD_DEBUG
#define SIMD_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define SIMD_DEBUG_PRINT(...)
#endif

#define SIMD_LANE_BYTES (SIMD_WIDTH / SIMD_SPATIAL_CHAN)

int32_t snax_simd_memcpy_nd(void* src, void* dst, uint32_t spatial_stride_src,
                            uint32_t spatial_stride_dst, uint32_t temp_dim_src,
                            uint32_t* temp_stride_src, uint32_t* temp_bound_src,
                            uint32_t temp_dim_dst, uint32_t* temp_stride_dst,
                            uint32_t* temp_bound_dst, uint32_t enabled_chan_src,
                            uint32_t enabled_chan_dst,
                            uint32_t enabled_byte_dst) {
    if (temp_dim_src > SIMD_SRC_TEMP_DIM) {
        SIMD_DEBUG_PRINT("Source dimension is too high for the SIMD block\n");
        return -4;
    }
    if (temp_dim_dst > SIMD_DST_TEMP_DIM) {
        SIMD_DEBUG_PRINT(
            "Destination dimension is too high for the SIMD block\n");
        return -4;
    }

    // The G>1 sparse-interconnect wiring (when enabled) makes a misaligned base
    // unroutable rather than merely slow, and the AGU's spatial sweep assumes a
    // whole beat, so refuse a base that is not beat-aligned.
    if (((uintptr_t)src % SIMD_WIDTH) != 0 ||
        ((uintptr_t)dst % SIMD_WIDTH) != 0) {
        SIMD_DEBUG_PRINT("SIMD buffers must be %d-byte aligned\n", SIMD_WIDTH);
        return -6;
    }

    // Both pointers are local TCDM addresses; the AGU truncates to its own
    // width, so only the low word carries information. The high word is written
    // anyway to keep the CSR sequence identical in shape to the xDMA's.
    uint64_t src_addr = (uint64_t)(uintptr_t)src;
    uint64_t dst_addr = (uint64_t)(uintptr_t)dst;

    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_LSB, (uint32_t)src_addr);
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_MSB, (uint32_t)(src_addr >> 32));

    // Frame counts must agree: the writer drains exactly what the reader (and
    // the extension chain) produce. A mismatch is the classic way to hang a
    // task, so report it rather than launching.
    uint32_t src_size = 1;
    for (uint32_t i = 0; i < temp_dim_src; i++) src_size *= temp_bound_src[i];
    uint32_t dst_size = 1;
    for (uint32_t i = 0; i < temp_dim_dst; i++) dst_size *= temp_bound_dst[i];
    if (src_size == 0 || dst_size == 0) {
        SIMD_DEBUG_PRINT("A temporal bound of 0 makes an empty task\n");
        return -2;
    }

    snax_write_simd_cfg_reg(SIMD_SRC_SPATIAL_STRIDE_PTR, spatial_stride_src);

    for (uint32_t i = 0; i < temp_dim_src; i++) {
        snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + i, temp_bound_src[i]);
        snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + i,
                                temp_stride_src[i]);
    }
    for (uint32_t i = temp_dim_src; i < SIMD_SRC_TEMP_DIM; i++) {
        snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + i, 1);
        snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + i, 0);
    }
    snax_write_simd_cfg_reg(SIMD_SRC_ENABLED_CHAN_PTR, enabled_chan_src);

    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_LSB, (uint32_t)dst_addr);
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_MSB, (uint32_t)(dst_addr >> 32));
    snax_write_simd_cfg_reg(SIMD_DST_SPATIAL_STRIDE_PTR, spatial_stride_dst);

    for (uint32_t i = 0; i < temp_dim_dst; i++) {
        snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + i, temp_bound_dst[i]);
        snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + i,
                                temp_stride_dst[i]);
    }
    for (uint32_t i = temp_dim_dst; i < SIMD_DST_TEMP_DIM; i++) {
        snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + i, 1);
        snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + i, 0);
    }
    snax_write_simd_cfg_reg(SIMD_DST_ENABLED_CHAN_PTR, enabled_chan_dst);
    snax_write_simd_cfg_reg(SIMD_DST_ENABLED_BYTE_PTR, enabled_byte_dst);

    return 0;
}

int32_t snax_simd_memcpy_1d(void* src, void* dst, uint32_t size) {
    if (size % SIMD_WIDTH != 0) {
        SIMD_DEBUG_PRINT("Transfer size must be a multiple of %d bytes\n",
                         SIMD_WIDTH);
        return -1;
    }
    uint32_t beats = size / SIMD_WIDTH;
    uint32_t stride = SIMD_WIDTH;
    return snax_simd_memcpy_nd(src, dst, SIMD_LANE_BYTES, SIMD_LANE_BYTES, 1,
                               &stride, &beats, 1, &stride, &beats, 0xFFFFFFFF,
                               0xFFFFFFFF, 0xFFFFFFFF);
}

int32_t snax_simd_enable_ext(uint8_t ext, uint32_t* csr_value) {
#if SIMD_EXT_NUM == 0
    (void)ext;
    (void)csr_value;
    SIMD_DEBUG_PRINT("This SIMD block has no extensions\n");
    return -1;
#else
    if (ext >= SIMD_EXT_NUM) {
        return -1;
    }
    uint8_t custom_csr_list[SIMD_EXT_NUM] = SIMD_EXT_CUSTOM_CSR_NUM;
    uint32_t csr_offset = SIMD_EXT_CSR_PTR;
    for (uint8_t i = 0; i < ext; i++) {
        csr_offset += custom_csr_list[i];
    }

    snax_write_simd_cfg_reg(
        SIMD_EXT_ENABLE_PTR,
        snax_read_simd_cfg_reg(SIMD_EXT_ENABLE_PTR) | (1 << ext));

    for (uint8_t i = 0; i < custom_csr_list[ext]; i++) {
        snax_write_simd_cfg_reg(csr_offset + i, csr_value[i]);
    }
    return 0;
#endif
}

int32_t snax_simd_disable_ext(uint8_t ext) {
#if SIMD_EXT_NUM == 0
    (void)ext;
    return 0;
#else
    if (ext >= SIMD_EXT_NUM) {
        return 0;
    }
    snax_write_simd_cfg_reg(
        SIMD_EXT_ENABLE_PTR,
        snax_read_simd_cfg_reg(SIMD_EXT_ENABLE_PTR) & ~(1 << ext));
    return 0;
#endif
}

void snax_simd_disable_all_ext(void) {
#if SIMD_EXT_NUM != 0
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0);
#endif
}
