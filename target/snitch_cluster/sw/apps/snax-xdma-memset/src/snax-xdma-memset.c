
// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

int main() {
    // Set err value for checking
    int err = 0;

    // Obtain the start address of the TCDM memory
    uint8_t *tcdm_baseaddress = (uint8_t *)snrt_l1_next();
    uint8_t *tcdm_0 = tcdm_baseaddress;
    uint8_t *tcdm_16 = tcdm_baseaddress + 0x4000 * sizeof(uint8_t);
    uint8_t *tcdm_32 = tcdm_baseaddress + 0x8000 * sizeof(uint8_t);
    uint8_t *tcdm_48 = tcdm_baseaddress + 0xc000 * sizeof(uint8_t);
    uint8_t *tcdm_64 = tcdm_baseaddress + 0x10000 * sizeof(uint8_t);
    uint8_t *tcdm_80 = tcdm_baseaddress + 0x14000 * sizeof(uint8_t);
    uint8_t *tcdm_96 = tcdm_baseaddress + 0x18000 * sizeof(uint8_t);
    uint8_t *tcdm_112 = tcdm_baseaddress + 0x1c000 * sizeof(uint8_t);

    // Transfer data from L3 to L1
    // Using xdma core only
    if (snrt_cluster_core_idx() == snrt_cluster_compute_core_num()) {
        // The xdma core is the last compute core in the cluster
        if (xdma_memcpy_1d(tcdm_0, tcdm_0, 0x4000 * sizeof(uint8_t)) != 0) {
            printf("Error in xdma agu configuration\n");
        }

        uint32_t ext_param[1] = {0xFFFFFFFF};
        if (xdma_enable_dst_ext(0, ext_param) != 0) {
            printf("Error in enableing xdma extension 0\n");
        }

        if (xdma_disable_dst_ext(1) != 0) {
            printf("Error in disabling xdma extension 1\n");
        }

        if (xdma_disable_dst_ext(2) != 0) {
            printf("Error in disabling xdma extension 2\n");
        }

        xdma_start();
        printf("The xdma is started, setting memory region to 0xFF\n");
        xdma_wait();

        printf("The xdma is finished\n");
        // Check the data
        for (int i = 0; i < 0x4000; i++) {
            if (tcdm_0[i] != 0xFF) {
                printf("Error in memset\n");
                err = 1;
                break;
            }
        }
    } else {
        snrt_wfi();
    }

    return err;
}
