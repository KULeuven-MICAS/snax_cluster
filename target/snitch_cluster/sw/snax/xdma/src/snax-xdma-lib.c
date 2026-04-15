// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "snax-xdma-lib.h"
#include <stdbool.h>
#include "stdint.h"

#define XDMA_DEBUG
#ifdef XDMA_DEBUG
#define XDMA_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define XDMA_DEBUG_PRINT(...)
#endif

#define XDMA_LANE_BYTES (XDMA_WIDTH / XDMA_SPATIAL_CHAN)

#if defined(READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16)
#define XDMA_ROW_MAJOR_TRANSPOSE_EXT_ID READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
#elif defined(READER_EXT_TRANSPOSERROW8_8_8COL8_8_8BIT8_16_32)
#define XDMA_ROW_MAJOR_TRANSPOSE_EXT_ID \
    READER_EXT_TRANSPOSERROW8_8_8COL8_8_8BIT8_16_32
#elif defined(READER_EXT_TRANSPOSERROW8_8_4COL8_8_4BIT8_16_32)
#define XDMA_ROW_MAJOR_TRANSPOSE_EXT_ID \
    READER_EXT_TRANSPOSERROW8_8_4COL8_8_4BIT8_16_32
#endif

typedef struct {
    uint32_t tile_width;
    uint32_t transfer_count;
    uint32_t csr_value[1];
} snax_xdma_row_major_transpose_cfg_t;

static int32_t snax_xdma_get_row_major_transpose_cfg(
    uint32_t element_width_bits, snax_xdma_row_major_transpose_cfg_t* cfg) {
#ifndef XDMA_ROW_MAJOR_TRANSPOSE_EXT_ID
    (void)element_width_bits;
    (void)cfg;
    XDMA_DEBUG_PRINT("Reader transposer extension is not available\n");
    return -5;
#else
    switch (element_width_bits) {
        case 8:
            cfg->tile_width = 8;
            cfg->transfer_count = 1;
            cfg->csr_value[0] = 0;
            return 0;
        case 16:
            cfg->tile_width = 8;
            cfg->transfer_count = 2;
            cfg->csr_value[0] = 1;
            return 0;
        default:
            XDMA_DEBUG_PRINT(
                "Unsupported transpose element width %u bits, expected "
                "8/16\n",
                element_width_bits);
            return -1;
    }
#endif
}

int32_t snax_xdma_row_major_transpose(void* src, void* dst, uint32_t rows,
                                      uint32_t cols,
                                      uint32_t element_width_bits) {
    snax_xdma_row_major_transpose_cfg_t cfg;
    int32_t ret =
        snax_xdma_get_row_major_transpose_cfg(element_width_bits, &cfg);
    if (ret != 0) {
        return ret;
    }

    if (rows == 0 || cols == 0) {
        XDMA_DEBUG_PRINT("Transpose matrix dimensions must be non-zero\n");
        return -2;
    }

    uint32_t bytes_per_element = element_width_bits / 8;
    if (rows % cfg.tile_width != 0 || cols % cfg.tile_width != 0) {
        XDMA_DEBUG_PRINT(
            "Row-major transpose requires rows/cols to be multiples of %u\n",
            cfg.tile_width);
        return -3;
    }

    uint64_t src_spatial_stride = (uint64_t)cols * bytes_per_element;
    uint64_t dst_spatial_stride = (uint64_t)rows * bytes_per_element;
    uint64_t src_outer_stride =
        (uint64_t)cols * cfg.tile_width * bytes_per_element;
    uint64_t dst_mid_stride =
        (uint64_t)rows * cfg.tile_width * bytes_per_element;

    if (src_spatial_stride > UINT32_MAX || dst_spatial_stride > UINT32_MAX ||
        src_outer_stride > UINT32_MAX || dst_mid_stride > UINT32_MAX) {
        XDMA_DEBUG_PRINT(
            "Row-major transpose AGU parameters overflow 32-bit registers\n");
        return -4;
    }

    uint32_t spatial_stride_src;
    uint32_t spatial_stride_dst;
    uint32_t temp_dim_src;
    uint32_t temp_dim_dst;
    uint32_t temp_bound_src[3];
    uint32_t temp_stride_src[3];
    uint32_t temp_bound_dst[3];
    uint32_t temp_stride_dst[3];

    if (cfg.transfer_count == 1 && rows == cfg.tile_width &&
        cols == cfg.tile_width) {
        spatial_stride_src = XDMA_LANE_BYTES;
        spatial_stride_dst = XDMA_LANE_BYTES;
        temp_dim_src = 1;
        temp_dim_dst = 1;
        temp_bound_src[0] = 1;
        temp_stride_src[0] = XDMA_WIDTH;
        temp_bound_dst[0] = 1;
        temp_stride_dst[0] = XDMA_WIDTH;
    } else {
        spatial_stride_src = (uint32_t)src_spatial_stride;
        spatial_stride_dst = (uint32_t)dst_spatial_stride;
        temp_dim_src = 3;
        temp_dim_dst = 3;
        temp_bound_src[0] = cfg.transfer_count;
        temp_bound_src[1] = cols / cfg.tile_width;
        temp_bound_src[2] = rows / cfg.tile_width;
        temp_stride_src[0] = XDMA_LANE_BYTES;
        temp_stride_src[1] = cfg.tile_width * bytes_per_element;
        temp_stride_src[2] = (uint32_t)src_outer_stride;
        temp_bound_dst[0] = cfg.transfer_count;
        temp_bound_dst[1] = cols / cfg.tile_width;
        temp_bound_dst[2] = rows / cfg.tile_width;
        temp_stride_dst[0] = XDMA_LANE_BYTES;
        temp_stride_dst[1] = (uint32_t)dst_mid_stride;
        temp_stride_dst[2] = cfg.tile_width * bytes_per_element;
    }

#ifndef XDMA_ROW_MAJOR_TRANSPOSE_EXT_ID
    (void)src;
    (void)dst;
    return -5;
#else
    ret = snax_xdma_enable_src_ext(XDMA_ROW_MAJOR_TRANSPOSE_EXT_ID,
                                   cfg.csr_value);
    if (ret != 0) {
        XDMA_DEBUG_PRINT("Failed to enable reader transposer extension\n");
        return ret;
    }

    ret = snax_xdma_memcpy_nd(src, dst, spatial_stride_src, spatial_stride_dst,
                              temp_dim_src, temp_stride_src, temp_bound_src,
                              temp_dim_dst, temp_stride_dst, temp_bound_dst,
                              0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    if (ret != 0) {
        snax_xdma_disable_src_ext(XDMA_ROW_MAJOR_TRANSPOSE_EXT_ID);
        return ret;
    }

    return 0;
#endif
}

int32_t snax_xdma_memcpy_nd_full_addr(
    uint64_t src, uint64_t dst, uint32_t spatial_stride_src,
    uint32_t spatial_stride_dst, uint32_t temp_dim_src,
    uint32_t* temp_stride_src, uint32_t* temp_bound_src, uint32_t temp_dim_dst,
    uint32_t* temp_stride_dst, uint32_t* temp_bound_dst,
    uint32_t enabled_chan_src, uint32_t enabled_chan_dst,
    uint32_t enabled_byte_dst) {
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, (uint32_t)src);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, (uint32_t)(src >> 32));

    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, (uint32_t)dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, (uint32_t)(dst >> 32));

    for (uint32_t i = 1; i < XDMA_MAX_DST_COUNT; i++) {
        snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB + i * 2, 0);
        snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB + i * 2, 0);
    }

    // Rule check
    // The enabled spatial bound for input should be equal to the enabled
    // Src frame count and dst frame count should be equal
    uint32_t src_size = 1;
    if (temp_dim_src > 0) {
        for (uint32_t i = 0; i < temp_dim_src; i++) {
            src_size *= temp_bound_src[i];
        }
    }
    uint32_t dst_size = 1;
    if (temp_dim_dst > 0) {
        for (uint32_t i = 0; i < temp_dim_dst; i++) {
            dst_size *= temp_bound_dst[i];
        }
    }
    if (src_size != dst_size) {
        XDMA_DEBUG_PRINT("src loop and dst loop is not equal\n");
        // return -3;
    }
    // Spatial Stride at src
    snax_write_xdma_cfg_reg(XDMA_SRC_SPATIAL_STRIDE_PTR, spatial_stride_src);

    // Spatial Stride at dst
    snax_write_xdma_cfg_reg(XDMA_DST_SPATIAL_STRIDE_PTR, spatial_stride_dst);

    // Temporal Dimension 0 to n at src
    for (uint32_t i = 0; i < temp_dim_src; i++) {
        if (i >= XDMA_SRC_TEMP_DIM) {
            XDMA_DEBUG_PRINT("Source dimension is too high for xdma\n");
            return -4;
        }
        snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + i, temp_bound_src[i]);
        snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + i,
                                temp_stride_src[i]);
    }
    // Dimension n to MAX at src
    for (uint32_t i = temp_dim_src; i < XDMA_SRC_TEMP_DIM; i++) {
        snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + i, 1);
        snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + i, 0);
    }
    // Temporal Dimension 0 to n at dst
    for (uint32_t i = 0; i < temp_dim_dst; i++) {
        if (i >= XDMA_DST_TEMP_DIM) {
            XDMA_DEBUG_PRINT("Destination dimension is too high for xdma\n");
            return -4;
        }
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + i, temp_bound_dst[i]);
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + i,
                                temp_stride_dst[i]);
    }
    // Dimension n to MAX at dst
    for (uint32_t i = temp_dim_dst; i < XDMA_DST_TEMP_DIM; i++) {
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + i, 1);
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + i, 0);
    }
    // Enabled channel at src
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, enabled_chan_src);
    // Enabled channel at dst
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, enabled_chan_dst);
    // Enabled byte at dst
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, enabled_byte_dst);
    return 0;
}

int32_t snax_xdma_memcpy_nd(void* src, void* dst, uint32_t spatial_stride_src,
                            uint32_t spatial_stride_dst, uint32_t temp_dim_src,
                            uint32_t* temp_stride_src, uint32_t* temp_bound_src,
                            uint32_t temp_dim_dst, uint32_t* temp_stride_dst,
                            uint32_t* temp_bound_dst, uint32_t enabled_chan_src,
                            uint32_t enabled_chan_dst,
                            uint32_t enabled_byte_dst) {
    uint64_t cluster_base_address_h = snrt_cluster_base_addrh();
    cluster_base_address_h = cluster_base_address_h << 32;
    return snax_xdma_memcpy_nd_full_addr(
        (uint64_t)src + cluster_base_address_h,
        (uint64_t)dst + cluster_base_address_h, spatial_stride_src,
        spatial_stride_dst, temp_dim_src, temp_stride_src, temp_bound_src,
        temp_dim_dst, temp_stride_dst, temp_bound_dst, enabled_chan_src,
        enabled_chan_dst, enabled_byte_dst);
}

int32_t snax_xdma_memcpy_1d_full_addr(uint64_t src, uint64_t dst,
                                      uint32_t size) {
    if (size % XDMA_WIDTH != 0) {
        XDMA_DEBUG_PRINT("Size is not multiple of XDMA_WIDTH\n");
        return -1;
    }
    uint32_t temporal_stride[1] = {XDMA_WIDTH};
    uint32_t temporal_bound[1] = {size / XDMA_WIDTH};
    return snax_xdma_memcpy_nd_full_addr(
        src, dst, XDMA_WIDTH / XDMA_SPATIAL_CHAN,
        XDMA_WIDTH / XDMA_SPATIAL_CHAN, 1, temporal_stride, temporal_bound, 1,
        temporal_stride, temporal_bound, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
}

int32_t snax_xdma_memcpy_1d(void* src, void* dst, uint32_t size) {
    if (size % XDMA_WIDTH != 0) {
        XDMA_DEBUG_PRINT("Size is not multiple of XDMA_WIDTH\n");
        return -1;
    }
    uint32_t temporal_stride[1] = {XDMA_WIDTH};
    uint32_t temporal_bound[1] = {size / XDMA_WIDTH};
    return snax_xdma_memcpy_nd(
        src, dst, XDMA_WIDTH / XDMA_SPATIAL_CHAN,
        XDMA_WIDTH / XDMA_SPATIAL_CHAN, 1, temporal_stride, temporal_bound, 1,
        temporal_stride, temporal_bound, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
}

int32_t snax_xdma_multicast_nd_full_address(
    uint64_t src, uint64_t* dst, uint32_t dst_num, uint32_t spatial_stride_src,
    uint32_t spatial_stride_dst, uint32_t temp_dim_src,
    uint32_t* temp_stride_src, uint32_t* temp_bound_src, uint32_t temp_dim_dst,
    uint32_t* temp_stride_dst, uint32_t* temp_bound_dst,
    uint32_t enabled_chan_src, uint32_t enabled_chan_dst,
    uint32_t enabled_byte_dst) {
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, (uint32_t)src);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, (uint32_t)(src >> 32));

    if (dst_num > XDMA_MAX_DST_COUNT) {
        XDMA_DEBUG_PRINT("Number of destination exceeds the hardware limit\n");
    }

    // Set the destination address for each destination
    for (uint32_t i = 0; i < dst_num; i++) {
        snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB + i * 2,
                                (uint32_t)dst[i]);
        snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB + i * 2,
                                (uint32_t)(dst[i] >> 32));
    }

    // The remaining is set to 0, and XDMA will stop the chained copy to the
    // next destination
    for (uint32_t i = dst_num; i < XDMA_MAX_DST_COUNT; i++) {
        snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB + i * 2, 0);
        snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB + i * 2, 0);
    }
    // Rule check
    // The enabled spatial bound for input should be equal to the enabled
    // Src frame count and dst frame count should be equal
    uint32_t src_size = 1;
    if (temp_dim_src > 0) {
        for (uint32_t i = 0; i < temp_dim_src; i++) {
            src_size *= temp_bound_src[i];
        }
    }
    uint32_t dst_size = 1;
    if (temp_dim_dst > 0) {
        for (uint32_t i = 0; i < temp_dim_dst; i++) {
            dst_size *= temp_bound_dst[i];
        }
    }
    if (src_size != dst_size) {
        XDMA_DEBUG_PRINT("src loop and dst loop is not equal\n");
        // return -3;
    }

    // Spatial Stride at src
    snax_write_xdma_cfg_reg(XDMA_SRC_SPATIAL_STRIDE_PTR, spatial_stride_src);

    // Spatial Stride at dst
    snax_write_xdma_cfg_reg(XDMA_DST_SPATIAL_STRIDE_PTR, spatial_stride_dst);

    // Temporal Dimension 0 to n at src
    for (uint32_t i = 0; i < temp_dim_src; i++) {
        if (i >= XDMA_SRC_TEMP_DIM) {
            XDMA_DEBUG_PRINT("Source dimension is too high for xdma\n");
            return -4;
        }
        snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + i, temp_bound_src[i]);
        snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + i,
                                temp_stride_src[i]);
    }
    // Dimension n to MAX at src
    for (uint32_t i = temp_dim_src; i < XDMA_SRC_TEMP_DIM; i++) {
        snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + i, 1);
        snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + i, 0);
    }
    // Temporal Dimension 0 to n at dst
    for (uint32_t i = 0; i < temp_dim_dst; i++) {
        if (i >= XDMA_DST_TEMP_DIM) {
            XDMA_DEBUG_PRINT("Destination dimension is too high for xdma\n");
            return -4;
        }
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + i, temp_bound_dst[i]);
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + i,
                                temp_stride_dst[i]);
    }
    // Dimension n to MAX at dst
    for (uint32_t i = temp_dim_dst; i < XDMA_DST_TEMP_DIM; i++) {
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + i, 1);
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + i, 0);
    }
    // Enabled channel at src
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, enabled_chan_src);
    // Enabled channel at dst
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, enabled_chan_dst);
    // Enabled byte at dst
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, enabled_byte_dst);
    return 0;
}

int32_t snax_xdma_multicast_nd(
    void* src, void** dst, uint32_t dst_num, uint32_t spatial_stride_src,
    uint32_t spatial_stride_dst, uint32_t temp_dim_src,
    uint32_t* temp_stride_src, uint32_t* temp_bound_src, uint32_t temp_dim_dst,
    uint32_t* temp_stride_dst, uint32_t* temp_bound_dst,
    uint32_t enabled_chan_src, uint32_t enabled_chan_dst,
    uint32_t enabled_byte_dst) {
    uint64_t cluster_base_address_h = snrt_cluster_base_addrh();
    cluster_base_address_h = cluster_base_address_h << 32;

    uint64_t dst_full_address[XDMA_MAX_DST_COUNT];
    for (uint32_t i = 0; i < dst_num; i++) {
        dst_full_address[i] = (uint64_t)dst[i] + cluster_base_address_h;
    }
    for (uint32_t i = dst_num; i < XDMA_MAX_DST_COUNT; i++) {
        dst_full_address[i] = 0;
    }

    return snax_xdma_multicast_nd_full_address(
        (uint64_t)src + cluster_base_address_h, dst_full_address, dst_num,
        spatial_stride_src, spatial_stride_dst, temp_dim_src, temp_stride_src,
        temp_bound_src, temp_dim_dst, temp_stride_dst, temp_bound_dst,
        enabled_chan_src, enabled_chan_dst, enabled_byte_dst);
}

int32_t snax_xdma_multicast_1d_full_address(uint64_t src, uint64_t* dst,
                                            uint32_t dst_num, uint32_t size) {
    if (size % XDMA_WIDTH != 0) {
        XDMA_DEBUG_PRINT("Size is not multiple of XDMA_WIDTH\n");
        return -1;
    }
    uint32_t temporal_stride[1] = {XDMA_WIDTH};
    uint32_t temporal_bound[1] = {size / XDMA_WIDTH};
    return snax_xdma_multicast_nd_full_address(
        src, dst, dst_num, XDMA_WIDTH / XDMA_SPATIAL_CHAN,
        XDMA_WIDTH / XDMA_SPATIAL_CHAN, 1, temporal_stride, temporal_bound, 1,
        temporal_stride, temporal_bound, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
}

int32_t snax_xdma_multicast_1d(void* src, void** dst, uint32_t dst_num,
                               uint32_t size) {
    if (size % XDMA_WIDTH != 0) {
        XDMA_DEBUG_PRINT("Size is not multiple of XDMA_WIDTH\n");
        return -1;
    }
    uint32_t temporal_stride[1] = {XDMA_WIDTH};
    uint32_t temporal_bound[1] = {size / XDMA_WIDTH};
    return snax_xdma_multicast_nd(
        src, dst, dst_num, XDMA_WIDTH / XDMA_SPATIAL_CHAN,
        XDMA_WIDTH / XDMA_SPATIAL_CHAN, 1, temporal_stride, temporal_bound, 1,
        temporal_stride, temporal_bound, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
}

// xdma extension interface
int32_t snax_xdma_enable_src_ext(uint8_t ext, uint32_t* csr_value) {
    if (ext >= XDMA_SRC_EXT_NUM) {
        return -1;
    }
    uint8_t custom_csr_list[XDMA_SRC_EXT_NUM] = XDMA_SRC_EXT_CUSTOM_CSR_NUM;
    uint32_t csr_offset = XDMA_SRC_EXT_CSR_PTR;
    for (uint8_t i = 0; i < ext; i++) {
        csr_offset += custom_csr_list[i];
    }

    snax_write_xdma_cfg_reg(
        XDMA_SRC_ENABLE_PTR,
        snax_read_xdma_cfg_reg(XDMA_SRC_ENABLE_PTR) | (1 << ext));

    for (uint8_t i = 0; i < custom_csr_list[ext]; i++) {
        snax_write_xdma_cfg_reg(csr_offset + i, csr_value[i]);
    }
    return 0;
}
int32_t snax_xdma_enable_dst_ext(uint8_t ext, uint32_t* csr_value) {
    if (ext >= XDMA_DST_EXT_NUM) {
        return -1;
    }
    uint8_t custom_csr_list[XDMA_DST_EXT_NUM] = XDMA_DST_EXT_CUSTOM_CSR_NUM;
    uint32_t csr_offset = XDMA_DST_EXT_CSR_PTR;
    for (uint8_t i = 0; i < ext; i++) {
        csr_offset += custom_csr_list[i];
    }

    snax_write_xdma_cfg_reg(
        XDMA_DST_ENABLE_PTR,
        snax_read_xdma_cfg_reg(XDMA_DST_ENABLE_PTR) | (1 << ext));
    for (uint8_t i = 0; i < custom_csr_list[ext]; i++) {
        snax_write_xdma_cfg_reg(csr_offset + i, csr_value[i]);
    }
    return 0;
}

int32_t snax_xdma_disable_src_ext(uint8_t ext) {
    if (ext >= XDMA_SRC_EXT_NUM) {
        return 0;
    }
    snax_write_xdma_cfg_reg(
        XDMA_SRC_ENABLE_PTR,
        snax_read_xdma_cfg_reg(XDMA_SRC_ENABLE_PTR) & ~(1 << ext));

    return 0;
}

int32_t snax_xdma_disable_dst_ext(uint8_t ext) {
    if (ext >= XDMA_DST_EXT_NUM) {
        return 0;
    }
    snax_write_xdma_cfg_reg(
        XDMA_DST_ENABLE_PTR,
        snax_read_xdma_cfg_reg(XDMA_DST_ENABLE_PTR) & ~(1 << ext));

    return 0;
}
