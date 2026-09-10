// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Core roles for the four-engine split cluster.
//
//   hart 0  GEMM   snax_streamer_gemmX
//   hart 1  SIMD   <cluster>_simd, the reader-side operator bank
//   hart 2  xDMA   transfer engine: writer extensions + junctions + AXI
//   hart 3  DM     the classic iDMA, and the cluster's singleton-init core
//
// snRuntime's own notion of roles stays as it is: SNRT_CLUSTER_DM_CORE_NUM
// remains 1, so snrt_is_dm_core() still selects exactly hart 3. That matters
// because snrt uses it for singleton work -- the L1/L3 allocator init in
// alloc.h and global-barrier participation in sync.h -- which must happen on
// one core and only one. Harts 1 and 2 are "compute cores" to snRuntime; these
// helpers are what distinguishes them.

#pragma once
#include "snrt.h"

#define SNAX_CORE_GEMM 0
#define SNAX_CORE_SIMD 1
#define SNAX_CORE_XDMA 2

// SNAX_SPLIT_ENGINES is set by apps/common.mk for the split cfg. Without it these
// helpers describe the ORIGINAL topology, where the DM core owns both the iDMA and
// the xDMA and there is no separate SIMD engine. That is what makes porting an app
// a one-line change that still builds for every other cluster: replace
// snrt_is_dm_core() with snax_is_xdma_core() (or snax_is_simd_core()) and the same
// source works on both shapes.
#ifdef SNAX_SPLIT_ENGINES

static inline int snax_is_gemm_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_GEMM;
}

static inline int snax_is_simd_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_SIMD;
}

static inline int snax_is_xdma_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_XDMA;
}

#else

// Unsplit cluster: the DM core owns the xDMA, and the reader extensions live on it,
// so the "SIMD core" is that same core.
static inline int snax_is_gemm_core(void) { return snrt_is_compute_core(); }
static inline int snax_is_simd_core(void) { return snrt_is_dm_core(); }
static inline int snax_is_xdma_core(void) { return snrt_is_dm_core(); }

#endif
