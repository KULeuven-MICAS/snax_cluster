// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Which hart carries which engine, and how to place a kernel on it.
//
// The role map is NOT written here. snax-core-roles-defs.h is generated from
// the active cluster cfg by util/snaxgen/core_roles.py, which derives it from
// the accelerator blocks the cfg puts on each core: snax_acc_cfg is the matmul
// array, snax_simd_cfg the SIMD operator bank, snax_xdma_cfg the SNAX transfer
// engine, and the `xdma` boolean the Snitch DMA ISA. Software reads the
// placement instead of assuming it, so an app no longer needs to know which
// cluster shape it was built for and there is no -D flag to get wrong.

#pragma once
#include "snax-core-roles-defs.h"
#include "snrt.h"

// Two generators, one cfg. If these disagree, the role indices below are being
// compared against a different cluster's core count.
_Static_assert(SNAX_CLUSTER_NUM_CORES == CFG_CLUSTER_NR_CORES,
               "snax-core-roles-defs.h and snitch_cluster_cfg.h were generated "
               "from different cluster cfgs");

// A cluster that lacks an engine defines SNAX_HAS_<ROLE>_CORE as 0 and does not
// define the predicate at all, so a kernel that needs it fails to build rather
// than running on hart 0. An app can say so itself with, e.g.
//
//     #if !SNAX_HAS_SIMD_CORE
//     #error "this kernel needs a SIMD block"
//     #endif

#if SNAX_HAS_GEMM_CORE
static inline int snax_is_gemm_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_GEMM;
}
#endif

#if SNAX_HAS_SIMD_CORE
static inline int snax_is_simd_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_SIMD;
}
#endif

#if SNAX_HAS_XDMA_CORE
static inline int snax_is_xdma_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_XDMA;
}
#endif

#if SNAX_HAS_IDMA_CORE
static inline int snax_is_idma_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_IDMA;
}
#endif
