// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "snax-xdma-lib.h"
#include <stdbool.h>
#include "snrt.h"
#include "stdint.h"

// Soft switch for CSR to make it support dynamic addressing
// The function can address 32 CSR registers starting from 960

uint32_t read_csr_soft_switch(uint32_t csr_address) {
    uint32_t value;
    switch (csr_address) {
        case 960:
            return read_csr(960);
        case 961:
            return read_csr(961);
        case 962:
            return read_csr(962);
        case 963:
            return read_csr(963);
        case 964:
            return read_csr(964);
        case 965:
            return read_csr(965);
        case 966:
            return read_csr(966);
        case 967:
            return read_csr(967);
        case 968:
            return read_csr(968);
        case 969:
            return read_csr(969);
        case 970:
            return read_csr(970);
            break;
        case 971:
            return read_csr(971);
        case 972:
            return read_csr(972);
        case 973:
            return read_csr(973);
        case 974:
            return read_csr(974);
            break;
        case 975:
            return read_csr(975);
        case 976:
            return read_csr(976);
        case 977:
            return read_csr(977);
        case 978:
            return read_csr(978);
        case 979:
            return read_csr(979);
        case 980:
            return read_csr(980);
        case 981:
            return read_csr(981);
        case 982:
            return read_csr(982);
        case 983:
            return read_csr(983);
        case 984:
            return read_csr(984);
        case 985:
            return read_csr(985);
        case 986:
            return read_csr(986);
        case 987:
            return read_csr(987);
            break;
        case 988:
            return read_csr(988);
        case 989:
            return read_csr(989);
        case 990:
            return read_csr(990);
        case 991:
            return read_csr(991);
    }
    return 0;
}

uint32_t write_csr_soft_switch(uint32_t csr_address, uint32_t value) {
    switch (csr_address) {
        case 960:
            write_csr(960, value);
            break;
        case 961:
            write_csr(961, value);
            break;
        case 962:
            write_csr(962, value);
            break;
        case 963:
            write_csr(963, value);
            break;
        case 964:
            write_csr(964, value);
            break;
        case 965:
            write_csr(965, value);
            break;
        case 966:
            write_csr(966, value);
            break;
        case 967:
            write_csr(967, value);
            break;
        case 968:
            write_csr(968, value);
            break;
        case 969:
            write_csr(969, value);
            break;
        case 970:
            write_csr(970, value);
            break;
        case 971:
            write_csr(971, value);
            break;
        case 972:
            write_csr(972, value);
            break;
        case 973:
            write_csr(973, value);
            break;
        case 974:
            write_csr(974, value);
            break;
        case 975:
            write_csr(975, value);
            break;
        case 976:
            write_csr(976, value);
            break;
        case 977:
            write_csr(977, value);
            break;
        case 978:
            write_csr(978, value);
            break;
        case 979:
            write_csr(979, value);
            break;
        case 980:
            write_csr(980, value);
            break;
        case 981:
            write_csr(981, value);
            break;
        case 982:
            write_csr(982, value);
            break;
        case 983:
            write_csr(983, value);
            break;
        case 984:
            write_csr(984, value);
            break;
        case 985:
            write_csr(985, value);
            break;
        case 986:
            write_csr(986, value);
            break;
        case 987:
            write_csr(987, value);
            break;
        case 988:
            write_csr(988, value);
            break;
        case 989:
            write_csr(989, value);
            break;
        case 990:
            write_csr(990, value);
            break;
        case 991:
            write_csr(991, value);
            break;
    }
}

int xdma_memcpy_nd(uint8_t* src, uint8_t* dst, uint32_t unit_size_src,
                   uint32_t unit_size_dst, uint32_t dim_src, uint32_t dim_dst,
                   uint32_t* stride_src, uint32_t* stride_dst,
                   uint32_t* bound_src, uint32_t* bound_dst) {
    write_csr_soft_switch(XDMA_SRC_ADDR_PTR_LSB, (uint32_t)(uint64_t)src);
    write_csr_soft_switch(XDMA_SRC_ADDR_PTR_MSB,
                          (uint32_t)((uint64_t)src << 32));
    // Rule check
    // unit size only support 8 bytes or n * 64 bytes
    if (unit_size_src % 64 != 0 || unit_size_src != 8) {
        return -1;
    }
    if (unit_size_dst % 64 != 0 || unit_size_dst != 8) {
        return -1;
    }
    // Src size and dst size should be equal
    uint32_t src_size = unit_size_src;
    for (uint32_t i = 0; i < dim_src; i++) {
        src_size *= bound_src[i];
    }
    uint32_t dst_size = unit_size_dst;
    for (uint32_t i = 0; i < dim_dst; i++) {
        dst_size *= bound_dst[i];
    }
    if (src_size != dst_size) {
        return -1;
    }

    // Dimension 1 at src
    uint32_t i = 0;
    if (unit_size_src % 64 == 0) {
        write_csr_soft_switch(XDMA_SRC_STRIDE_PTR + i, 8);
        write_csr_soft_switch(XDMA_SRC_BOUND_PTR + i, unit_size_src >> 3);
        i++;
    }
    // Dimension 2 to n at src
    for (uint32_t j = 0; j < dim_src - 1; j++) {
        if (i + j >= XDMA_SRC_DIM) {
            return -2;
        }
        write_csr_soft_switch(XDMA_SRC_BOUND_PTR + i + j, bound_src[j]);
        write_csr_soft_switch(XDMA_SRC_STRIDE_PTR + i + j, stride_src[j]);
    }

    // Dimension 1 at dst
    i = 0;
    if (unit_size_dst % 64 == 0) {
        write_csr_soft_switch(XDMA_DST_STRIDE_PTR + i, 8);
        write_csr_soft_switch(XDMA_DST_BOUND_PTR + i, unit_size_dst >> 3);
        i++;
    }
    // Dimension 2 to n at dst
    for (uint32_t j = 0; j < dim_dst - 1; j++) {
        if (i + j >= XDMA_DST_DIM) {
            return -2;
        }
        write_csr_soft_switch(XDMA_DST_BOUND_PTR + i + j, bound_dst[j]);
        write_csr_soft_switch(XDMA_DST_STRIDE_PTR + i + j, stride_dst[j]);
    }
    return 0;
}

int xdma_memcpy_1d(uint8_t* src, uint8_t* dst, uint32_t size) {
    return xdma_memcpy_nd(src, dst, size, size, 1, 1, (uint32_t*)NULL,
                          (uint32_t*)NULL, (uint32_t*)NULL, (uint32_t*)NULL);
}

// xdma extension interface
int xdma_enable_src_ext(uint8_t ext, uint32_t* csr_value) {
    if (ext >= XDMA_SRC_EXT_NUM) {
        return -1;
    }
    uint8_t custom_csr_list[XDMA_SRC_EXT_NUM] = XDMA_SRC_EXT_CUSTOM_CSR_NUM;
    uint32_t csr_offset = XDMA_SRC_EXT_CSR_PTR;
    for (uint8_t i = 0; i < ext; i++) {
        csr_offset += custom_csr_list[i] + 1;
    }
    write_csr_soft_switch(csr_offset, 1);
    csr_offset++;
    for (uint8_t i = 0; i < custom_csr_list[ext]; i++) {
        write_csr_soft_switch(csr_offset + i, csr_value[i]);
    }
    return 0;
}
int xdma_enable_dst_ext(uint8_t ext, uint32_t* csr_value) {
    if (ext >= XDMA_DST_EXT_NUM) {
        return -1;
    }
    uint8_t custom_csr_list[XDMA_DST_EXT_NUM] = XDMA_DST_EXT_CUSTOM_CSR_NUM;
    uint32_t csr_offset = XDMA_DST_EXT_CSR_PTR;
    for (uint8_t i = 0; i < ext; i++) {
        csr_offset += custom_csr_list[i] + 1;
    }
    write_csr_soft_switch(csr_offset, 1);
    csr_offset++;
    for (uint8_t i = 0; i < custom_csr_list[ext]; i++) {
        write_csr_soft_switch(csr_offset + i, csr_value[i]);
    }
    return 0;
}

int xdma_disable_src_ext(uint8_t ext) {
    if (ext >= XDMA_SRC_EXT_NUM) {
        return -1;
    }
    uint8_t custom_csr_list[XDMA_SRC_EXT_NUM] = XDMA_SRC_EXT_CUSTOM_CSR_NUM;
    uint32_t csr_offset = XDMA_SRC_EXT_CSR_PTR;
    for (uint8_t i = 0; i < ext; i++) {
        csr_offset += custom_csr_list[i] + 1;
    }
    write_csr_soft_switch(csr_offset, 0);
    return 0;
}

int xdma_disable_dst_ext(uint8_t ext) {
    if (ext >= XDMA_DST_EXT_NUM) {
        return -1;
    }
    uint8_t custom_csr_list[XDMA_DST_EXT_NUM] = XDMA_DST_EXT_CUSTOM_CSR_NUM;
    uint32_t csr_offset = XDMA_DST_EXT_CSR_PTR;
    for (uint8_t i = 0; i < ext; i++) {
        csr_offset += custom_csr_list[i] + 1;
    }
    write_csr_soft_switch(csr_offset, 0);
    return 0;
}

// Start xdma
void xdma_start() { write_csr_soft_switch(XDMA_START_PTR, 1); }

// Check if xdma is finished
bool xdma_is_finished() {
    return read_csr_soft_switch(XDMA_FINISH_TASK_PTR) ==
           read_csr_soft_switch(XDMA_COMMIT_TASK_PTR);
}

void xdma_wait() {
    while (!xdma_is_finished()) {
        // Wait for xdma to finish
    }
}
