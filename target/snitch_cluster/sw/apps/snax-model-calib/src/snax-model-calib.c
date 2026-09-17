// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Per-engine calibration for the four-engine split cluster.
//
// The performance model in `bingo` needs a handful of machine constants, and every one of
// them has to come from a measurement of the MACHINE rather than from a fit to a workload
// -- otherwise the model reproduces the workload it was fitted to and nothing else. This
// app is that measurement: one ELF, four sweeps, one parseable table.
//
//   1  iDMA    cycles vs transfer size        -> port B/cc and per-transfer start cost
//   2  xDMA    cycles vs memset beats         -> writer beat rate and start cost
//   3a SIMD    cycles vs number of 1-beat tasks -> the per-task start/drain cost
//   3b SIMD    cycles vs beats in one task    -> the per-beat initiation interval and fill
//
// Each point is run TWICE and the second value reported: the first pays the instruction
// cache, and a cold number would be fitted into a steady-state constant.
//
// Output lines are `CALIB <bench> <x> <wall> <busy>` so a script can read them without
// parsing prose.

#include <stdint.h>
#include "snax-core-roles.h"
#include "snax-simd-lib.h"
#include "snax-xdma-addr.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#define BEAT SIMD_BEAT_BYTES

// ---- sweep points -------------------------------------------------------------
#define N_COPY 9
static const uint32_t copy_bytes[N_COPY] = {512,  1024,  2048,  4096, 8192,
                                            16384, 32768, 65536, 131072};
#define N_MEMSET 8
static const uint32_t memset_beats[N_MEMSET] = {1, 2, 4, 16, 64, 256, 1024, 2048};
#define N_TASKS 7
static const uint32_t task_counts[N_TASKS] = {1, 2, 4, 8, 16, 32, 64};
#define N_BEATS 8
static const uint32_t beat_counts[N_BEATS] = {1, 2, 4, 8, 32, 128, 256, 512};

#define MAX_COPY 131072u
#define MAX_MEMSET_BYTES (2048u * BEAT)
#define SIMD_BUF_BYTES (512u * BEAT)

// The iDMA source. .bss maps to main memory on this target, which is what makes this a
// DRAM->TCDM transfer rather than a TCDM->TCDM one.
static int8_t dram_src[MAX_COPY];

// Results, published through TCDM so one hart prints a single tidy table.
typedef struct {
    uint32_t copy_wall[N_COPY];
    uint32_t memset_wall[N_MEMSET];
    uint32_t memset_busy[N_MEMSET];
    uint32_t tasks_wall[N_TASKS];
    uint32_t tasks_busy[N_TASKS];
    uint32_t tasks_cold[N_TASKS];
    uint32_t seq_cold, seq_warm;      // a 9-task sequence of DIFFERENT shapes, first vs second
    uint32_t beats_wall[N_BEATS];
    uint32_t beats_busy[N_BEATS];
} results_t;

// The xDMA writer-side memset, armed with CONSTANT CSR addresses. csrw_ss switches on the
// address, so a compile-time-constant one folds to a single `csrw imm` while a computed one
// costs a jump-table load from main memory plus an indirect jump -- which would otherwise
// dominate what this sweep is trying to measure.
__attribute__((always_inline)) static inline void xdma_fill_arm(uint32_t dst, uint32_t stride,
                                                                uint32_t bound,
                                                                uint32_t pattern) {
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, dst);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_SPATIAL_STRIDE_PTR, 8);
    snax_write_xdma_cfg_reg(XDMA_DST_SPATIAL_STRIDE_PTR, 8);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 0, bound);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 1, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 2, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 0, stride);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 1, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 2, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 4, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 0, bound);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 1, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 2, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 0, stride);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 1, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 2, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 4, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, 0xFFFFFFFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, 0xFFFFFFFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLE_PTR, 1u << WRITER_EXT_VERILOGMEMSET);
    snax_write_xdma_cfg_reg(XDMA_DST_EXT_CSR_PTR, pattern);
}

int main() {
    uint8_t *l1 = (uint8_t *)snrt_l1_next();
    uint32_t top = 0;

    uint32_t copy_dst = top;    top += MAX_COPY;
    uint32_t mset_dst = top;    top += MAX_MEMSET_BYTES;
    uint32_t simd_src = top;    top += SIMD_BUF_BYTES;
    uint32_t simd_dst = top;    top += SIMD_BUF_BYTES;
    uint32_t res_off  = top;    top += sizeof(results_t);
    top = (top + BEAT - 1u) & ~(BEAT - 1u);

    volatile results_t *res = (volatile results_t *)(l1 + res_off);

    // TCDM is SRAM with no reset: a location never written reads X on RTL, and X
    // propagates. Define the whole arena before anyone reads it -- the same rule the
    // FlashAttention kernels follow.
    if (snax_is_xdma_core()) {
        xdma_fill_arm((uint32_t)l1, BEAT, (top + BEAT - 1u) / BEAT, 0u);
        snax_xdma_local_wait(snax_xdma_start());
    }
    snrt_cluster_hw_barrier();

    if (snax_is_gemm_core()) {
        printf("=== split-cluster engine calibration ===\n");
        printf("  arena %lu B, beat %u B\n", (unsigned long)top, (unsigned)BEAT);
    }
    snrt_cluster_hw_barrier();

    // ---- 1  iDMA: cycles against transfer size ---------------------------------
    if (snrt_is_dm_core()) {
        for (uint32_t i = 0; i < N_COPY; i++) {
            for (uint32_t rep = 0; rep < 2; rep++) {
                uint32_t t0 = snrt_mcycle();
                snrt_dma_start_1d(l1 + copy_dst, dram_src, copy_bytes[i]);
                snrt_dma_wait_all();
                uint32_t t1 = snrt_mcycle();
                res->copy_wall[i] = t1 - t0;   // the second pass overwrites the first
            }
        }
    }
    snrt_cluster_hw_barrier();

    // ---- 2  xDMA: cycles against memset beats ----------------------------------
    if (snax_is_xdma_core()) {
        for (uint32_t i = 0; i < N_MEMSET; i++) {
            for (uint32_t rep = 0; rep < 2; rep++) {
                uint32_t t0 = snrt_mcycle();
                xdma_fill_arm((uint32_t)(l1 + mset_dst), BEAT, memset_beats[i], 0u);
                snax_xdma_local_wait(snax_xdma_start());
                uint32_t t1 = snrt_mcycle();
                res->memset_wall[i] = t1 - t0;
                res->memset_busy[i] = snax_xdma_last_task_cycle();
            }
        }
    }
    snrt_cluster_hw_barrier();

    // ---- 3  SIMD -----------------------------------------------------------------
    // An identity map (out = 1.0*x + 0.0) -- exact in FP16, but it still runs the whole
    // FP datapath, so the timing is a real one.
    if (snax_is_simd_core()) {
        for (uint32_t i = 0; i < SIMD_BUF_BYTES / 4; i++)
            ((volatile uint32_t *)(l1 + simd_src))[i] = 0x3C003C00u;

        snax_simd_shape_t in, out;
        snax_simd_op_t op;
        snax_simd_op_map(&op, SIMD_EXT_STREAMMAP, SIMD_FUNC_LINEAR, SIMD_F32_ONE,
                         SIMD_F32_ZERO);

        // 3a  N one-beat tasks back to back. The engine's busy counter over N tasks
        //     divided by N is the per-task start/drain cost the state update pays.
        snax_simd_shape_flat(&in, l1 + simd_src, 1);
        snax_simd_shape_flat(&out, l1 + simd_dst, 1);
        snax_simd_configure(&in, &out, &op, 1);
        for (uint32_t i = 0; i < N_TASKS; i++) {
            for (uint32_t rep = 0; rep < 2; rep++) {
                uint32_t b0 = snax_simd_busy_cycles();
                uint32_t t0 = snrt_mcycle();
                for (uint32_t k = 0; k < task_counts[i]; k++) snax_simd_fire();
                snax_simd_wait_all();
                uint32_t t1 = snrt_mcycle();
                if (rep == 0) res->tasks_cold[i] = t1 - t0;
                res->tasks_wall[i] = t1 - t0;
                res->tasks_busy[i] = snax_simd_busy_cycles() - b0;
            }
        }

        // 3b  one task of B beats: the slope is the per-beat initiation interval, the
        //     intercept the pipeline fill.
        for (uint32_t i = 0; i < N_BEATS; i++) {
            snax_simd_shape_flat(&in, l1 + simd_src, beat_counts[i]);
            snax_simd_shape_flat(&out, l1 + simd_dst, beat_counts[i]);
            snax_simd_configure(&in, &out, &op, 1);
            for (uint32_t rep = 0; rep < 2; rep++) {
                uint32_t b0 = snax_simd_busy_cycles();
                uint32_t t0 = snrt_mcycle();
                snax_simd_wait(snax_simd_launch());
                uint32_t t1 = snrt_mcycle();
                res->beats_wall[i] = t1 - t0;
                res->beats_busy[i] = snax_simd_busy_cycles() - b0;
            }
        }

        // 3c  A SEQUENCE OF DIFFERENT SHAPES, which is what a real kernel issues.
        //
        // The sweeps above re-fire ONE configured task, so they price the fire path and nothing
        // else. A kernel's tile is nine tasks with nine different shapes, each re-programmed
        // before it is fired, and the first pass through that pays for nine distinct code paths
        // in a cold instruction cache. That one-time cost lands on the lane, not on the engine:
        // it delays when the tile's result is published without adding a cycle of engine busy,
        // which is exactly the kind of term a busy-counter model cannot see.
        static const uint32_t seq_beats[9] = {512, 2, 2, 2, 1, 513, 2, 2, 2};
        for (uint32_t rep = 0; rep < 2; rep++) {
            uint32_t t0 = snrt_mcycle();
            for (uint32_t i = 0; i < 9; i++) {
                snax_simd_shape_flat(&in, l1 + simd_src, seq_beats[i]);
                snax_simd_shape_flat(&out, l1 + simd_dst, seq_beats[i]);
                snax_simd_configure(&in, &out, &op, 1);
                snax_simd_fire();
            }
            snax_simd_wait_all();
            uint32_t t1 = snrt_mcycle();
            if (rep == 0) res->seq_cold = t1 - t0; else res->seq_warm = t1 - t0;
        }
    }
    snrt_cluster_hw_barrier();

    if (snax_is_gemm_core()) {
        printf("CALIB simd_seq 9 %lu %lu\n", (unsigned long)res->seq_cold,
               (unsigned long)res->seq_warm);
        for (uint32_t i = 0; i < N_COPY; i++)
            printf("CALIB idma_copy %lu %lu 0\n", (unsigned long)copy_bytes[i],
                   (unsigned long)res->copy_wall[i]);
        for (uint32_t i = 0; i < N_MEMSET; i++)
            printf("CALIB xdma_memset %lu %lu %lu\n", (unsigned long)memset_beats[i],
                   (unsigned long)res->memset_wall[i], (unsigned long)res->memset_busy[i]);
        for (uint32_t i = 0; i < N_TASKS; i++)
            printf("CALIB simd_tasks %lu %lu %lu\n", (unsigned long)task_counts[i],
                   (unsigned long)res->tasks_wall[i], (unsigned long)res->tasks_busy[i]);
        // The cold pass of the same burst: wall on the FIRST time through the issue loop.
        for (uint32_t i = 0; i < N_TASKS; i++)
            printf("CALIB simd_cold %lu %lu %lu\n", (unsigned long)task_counts[i],
                   (unsigned long)res->tasks_cold[i], (unsigned long)res->tasks_wall[i]);
        for (uint32_t i = 0; i < N_BEATS; i++)
            printf("CALIB simd_beats %lu %lu %lu\n", (unsigned long)beat_counts[i],
                   (unsigned long)res->beats_wall[i], (unsigned long)res->beats_busy[i]);
        printf("CALIB done\n");
    }
    snrt_cluster_hw_barrier();
    return 0;
}
