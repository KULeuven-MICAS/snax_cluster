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

#ifdef SNAX_SPLIT_ENGINES

// ---------------------------------------------------------------- staging
//
// These kernels stage their input from L3 into TCDM with snrt_dma_start_1d/2d
// from inside their engine block. On the unsplit cluster that block runs on the
// DM core, which owns the iDMA, so the call is legal. On the split cluster the
// block runs on hart 1, which has NO DMA ISA -- only hart 3 does, because
// snitch_cluster.sv asserts $onehot0(Xdma). Issuing a dm* instruction there
// traps with an illegal instruction, which is exactly how this was found.
//
// snax_stage_*() keeps the kernels single-core by copying with the core's own
// loads and stores on the split cluster. That is slower than the iDMA, but these
// kernels report ENGINE cycles from the perf CSRs, which this does not affect.
//
// A production kernel should NOT do this: it should stage on the DM core, hit a
// barrier, and let the SIMD core compute -- which also overlaps the staging of
// the next tile with the current one. The FlashAttention loop skeleton in the
// plan (§9.3) is written that way.
// Byte-accurate on purpose. A word-only copy silently transfers NOTHING for the
// sub-word moves these kernels do -- RoPE's adjacent-halfword swap is a 2-byte
// strided copy, and `bytes / 4` rounds that to zero.
static inline void snax_stage_1d(void* dst, const void* src, uint32_t bytes) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    if ((((uintptr_t)d | (uintptr_t)s) & 3u) == 0) {
        uint32_t words = bytes >> 2;
        for (uint32_t i = 0; i < words; i++) {
            ((uint32_t*)d)[i] = ((const uint32_t*)s)[i];
        }
        d += words << 2;
        s += words << 2;
        bytes &= 3u;
    }
    for (uint32_t i = 0; i < bytes; i++) d[i] = s[i];
}

static inline void snax_stage_2d(void* dst, const void* src, uint32_t bytes,
                                 uint32_t dst_stride, uint32_t src_stride,
                                 uint32_t repeat) {
    for (uint32_t r = 0; r < repeat; r++) {
        snax_stage_1d((uint8_t*)dst + r * dst_stride,
                      (const uint8_t*)src + r * src_stride, bytes);
    }
}

#else

// Unsplit cluster: the engine block already runs on the DM core, which owns the
// iDMA, so stage with it.
static inline void snax_stage_1d(void* dst, const void* src, uint32_t bytes) {
    snrt_dma_start_1d(dst, src, bytes);
    snrt_dma_wait_all();
}

static inline void snax_stage_2d(void* dst, const void* src, uint32_t bytes,
                                 uint32_t dst_stride, uint32_t src_stride,
                                 uint32_t repeat) {
    snrt_dma_start_2d(dst, src, bytes, dst_stride, src_stride, repeat);
    snrt_dma_wait_all();
}

#endif
