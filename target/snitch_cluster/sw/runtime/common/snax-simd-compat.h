// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// One header that lets a reader-extension ("SIMD") kernel build unchanged on
// both cluster shapes.
//
//   unsplit cluster  -- the reader extensions live on the xDMA, driven from the
//                       DM core. Include snax-xdma-lib.h and change nothing.
//   split cluster    -- the reader extensions live on <cluster>_simd on hart 1.
//                       Redirect the reader-side xDMA API onto the SIMD library.
//
// The redirection goes in the xDMA -> SIMD direction on purpose: kernels keep
// their existing `snax_xdma_*_src_ext` calls and `READER_EXT_*` ids, so porting
// a kernel is two lines -- swap this header in for snax-xdma-lib.h, and swap
// snrt_is_dm_core() for snax_is_simd_core() (both from snax-core-roles.h).
//
// Only the READER side is mapped. Writer extensions and the junctions stayed on
// the xDMA and are not reachable from the SIMD core; see the disable_dst_ext
// note below.

#pragma once
#include "snax-core-roles.h"

#ifdef SNAX_SPLIT_ENGINES

#include "snax-simd-lib.h"

// Geometry
#define XDMA_WIDTH SIMD_WIDTH
#define XDMA_SPATIAL_CHAN SIMD_SPATIAL_CHAN
#define XDMA_SRC_TEMP_DIM SIMD_SRC_TEMP_DIM
#define XDMA_DST_TEMP_DIM SIMD_DST_TEMP_DIM

// Task setup and run control
#define snax_xdma_memcpy_nd snax_simd_memcpy_nd
#define snax_xdma_memcpy_nd_fast snax_simd_memcpy_nd_fast
#define snax_xdma_memcpy_1d snax_simd_memcpy_1d
#define snax_xdma_retask_1d snax_simd_retask_1d
#define snax_xdma_row_major_transpose snax_simd_row_major_transpose
#define snax_xdma_enable_src_ext snax_simd_enable_ext
#define snax_xdma_disable_src_ext snax_simd_disable_ext
#define snax_xdma_start snax_simd_start
#define snax_xdma_local_wait snax_simd_wait
#define snax_xdma_last_task_cycle snax_simd_last_task_cycle
#define snax_xdma_last_read_cycle snax_simd_last_read_cycle
#define snax_xdma_last_write_cycle snax_simd_last_write_cycle

// The kernels call disable_dst_ext defensively to clear writer state they never
// set. The SIMD block has no writer extensions, so there is nothing to clear and
// this is a no-op returning success. A kernel that genuinely needs a writer
// extension must run on the xDMA core instead -- it cannot be shimmed, because
// the hardware is on the other hart.
static inline int32_t snax_simd_compat_disable_dst_ext(int ext) {
    (void)ext;
    return 0;
}
#define snax_xdma_disable_dst_ext snax_simd_compat_disable_dst_ext

// Extension ids. The generated SIMD header names them SIMD_EXT_*; kernels use
// READER_EXT_*. Map each one that this build actually has.
#ifdef SIMD_EXT_STREAMMAP
#define READER_EXT_STREAMMAP SIMD_EXT_STREAMMAP
#endif
#ifdef SIMD_EXT_STREAMREDUCE
#define READER_EXT_STREAMREDUCE SIMD_EXT_STREAMREDUCE
#endif
#ifdef SIMD_EXT_STREAMELEMENTWISE
#define READER_EXT_STREAMELEMENTWISE SIMD_EXT_STREAMELEMENTWISE
#endif
#ifdef SIMD_EXT_FP16TOINT8
#define READER_EXT_FP16TOINT8 SIMD_EXT_FP16TOINT8
#endif
#ifdef SIMD_EXT_ELEMENTWISEADDBIT32
#define READER_EXT_ELEMENTWISEADDBIT32 SIMD_EXT_ELEMENTWISEADDBIT32
#endif
#ifdef SIMD_EXT_MAXPOOLBIT8
#define READER_EXT_MAXPOOLBIT8 SIMD_EXT_MAXPOOLBIT8
#endif
#ifdef SIMD_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
#define READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16 \
    SIMD_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
#endif

#else  // !SNAX_SPLIT_ENGINES

#include "snax-xdma-lib.h"

#endif
