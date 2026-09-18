// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Isolate a hung SIMD task: run the suspect chain ONE TASK AT A TIME.
//
// A kernel that spins in `snax_simd_wait` tells you nothing about which of its tasks never
// retired -- an app reports its stage cycles at the end, and a hang never reaches the end. This
// runs the tasks separately, prints before and after each, and polls the finished-task counter
// with a BOUND instead of waiting forever, so a stuck task names itself and the run still exits.
// The last "begin" with no matching "end" is the culprit.
//
//   S1  reduce(SUMSQ)                 the fold path's ingredients, in the order it runs them
//   S2  elementwise(MUL), 2 operands
//   S3  BROADCAST reader, stride 0
//   S4  elementwise -||> Fp16ToInt8
//
// WHAT IT FOUND (2026-09-18), kept because it is the app's worked example. `snax-simd-*-fold`
// ran at rows=1 cols=64 and hung at 1x128. S1-S3 retired at both -- clearing the stride-0
// broadcast, which was the obvious suspect -- and S4 hung at BOTH, including the shape the app
// itself survived. That pointed at the quantiser's configuration rather than at any geometry:
// `snax_simd_enable_ext` writes SIMD_EXT_FP16TOINT8_CSR_NUM = 2 words from the caller's array
// and six apps passed a one-element one, so a stack word became `tailPeriod` and the writer
// waited for an unquantised tail beat the reader never sends. Shape-dependent because the stack
// word is. Fixed in those apps as `csr_q[2] = {scale, 0}`.
//
// Note the bisect reproduced it at 1x64 too, where the app did not: the same out-of-bounds read
// picks up a different word in a different frame. A harness that only reproduces the failing
// shape would have missed that the bug was never about the shape at all.
//
// This measures nothing. `snax-bingo-simd-sweep` is the cost sweep.

#include "data.h"
#include "snax-core-roles.h"
#include "snax-simd-lib.h"
#include "snrt.h"

#define BEAT_BYTES 64
#define OP_SUMSQ 2u
#define EW_MUL 0u
#define F32_ONE 0x3F800000u
#define FP16_HALF 0x3800u

// How long to let one task run before calling it hung. A whole 8x256 op measured ~20k cycles, so
// anything past this is not slow, it is stuck.
#define TASK_WATCHDOG 2000000u

// Poll the finished-task counter with a bound instead of `snax_simd_wait`, which spins forever.
// Returns 1 if the task retired, 0 if the watchdog fired.
static int wait_bounded(uint32_t id) {
    for (uint32_t i = 0; i < TASK_WATCHDOG; i++) {
        if (snax_read_simd_cfg_reg(SIMD_FINISHED_TASK_PTR) >= id) return 1;
    }
    return 0;
}

static int step(const char* name, uint32_t rows, uint32_t cols) {
    printf("[BISECT] begin %s rows=%u cols=%u\n", name, rows, cols);
    uint32_t id = snax_simd_start();
    int ok = wait_bounded(id);
    printf("[BISECT] %s %s\n", name, ok ? "end" : "HUNG");
    return ok;
}

int main() {
    uint32_t base = snrt_cluster_base_addrl();
    uint32_t rows = BSW_ROWS, d = BSW_D, beats = BSW_BEATS;
    uint32_t row_bytes = beats * BEAT_BYTES;
    uint32_t rows_bytes = rows * row_bytes;
    uint32_t scal_bytes = rows * BEAT_BYTES;

    uint8_t* x_in = (uint8_t*)base;
    uint8_t* second = x_in + rows_bytes;      // operand B for the interleaved elementwise
    uint8_t* out = second + rows_bytes;
    uint8_t* bc_out = out + rows_bytes;       // [rows, D] the broadcast writes
    uint8_t* scal = bc_out + rows_bytes;      // [rows] scalar beats
    uint8_t* q_out = scal + scal_bytes;
    uint32_t span = (uint32_t)((q_out + rows_bytes) - x_in);

    if (snax_is_simd_core()) {
        uint16_t* w = (uint16_t*)base;
        for (uint32_t i = 0; i < span / 2; i++) w[i] = FP16_HALF;
    }
    snrt_cluster_hw_barrier();

    if (!snax_is_simd_core()) {
        snrt_cluster_hw_barrier();
        return 0;
    }

    printf("[BISECT] rows=%u D=%u beats=%u\n", rows, d, beats);
    int all = 1;

    uint32_t red_str[2] = {BEAT_BYTES, row_bytes};
    uint32_t red_bnd[2] = {beats, rows};
    uint32_t w_rows_str[1] = {BEAT_BYTES};
    uint32_t w_rows_bnd[1] = {rows};
    uint32_t flat_str[1] = {BEAT_BYTES};
    uint32_t flat_bnd[1] = {rows * beats};

    // S1 reduce(SUMSQ) -- control
    uint32_t csr_ssq[2] = {beats, OP_SUMSQ};
    snax_simd_enable_ext(SIMD_EXT_STREAMREDUCE, csr_ssq);
    snax_simd_memcpy_nd_fast(x_in, scal, 8, 8, 2, red_str, red_bnd, 1, w_rows_str, w_rows_bnd,
                             0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    all &= step("S1_reduce_sumsq", rows, d);
    snax_simd_disable_ext(SIMD_EXT_STREAMREDUCE);

    // S2 elementwise(MUL), two interleaved operands -- control
    uint32_t ew_str[2] = {(uint32_t)(second - x_in), BEAT_BYTES};
    uint32_t ew_bnd[2] = {2, rows * beats};
    uint32_t csr_mul[2] = {2u, EW_MUL};
    snax_simd_enable_ext(SIMD_EXT_STREAMELEMENTWISE_1, csr_mul);
    snax_simd_memcpy_nd_fast(x_in, out, 8, 8, 2, ew_str, ew_bnd, 1, flat_str, flat_bnd,
                             0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    all &= step("S2_elementwise_mul", rows, d);
    snax_simd_disable_ext(SIMD_EXT_STREAMELEMENTWISE_1);

    // S3 BROADCAST: inner temporal dim stride 0 re-reads the row's scalar beat `beats` times.
    // This is rmsnorm-fold's B1, verbatim -- the shape it uses and the suspect.
    uint32_t bc_str[2] = {0, BEAT_BYTES};
    uint32_t bc_bnd[2] = {beats, rows};
    snax_simd_memcpy_nd_fast(scal, bc_out, 8, 8, 2, bc_str, bc_bnd, 1, flat_str, flat_bnd,
                             0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    all &= step("S3_broadcast_stride0", rows, d);

    // S4 elementwise -||> Fp16ToInt8, the chained quantise (halves the written beats)
    uint32_t q_bnd[1] = {(rows * beats) / 2};
    // TWO words: enable_ext writes SIMD_EXT_FP16TOINT8_CSR_NUM (2) from this array, and csr[1]
    // is tailPeriod. This is the step that found the bug -- see the note at the top.
    uint32_t csr_q[2] = {F32_ONE, 0u};
    snax_simd_enable_ext(SIMD_EXT_STREAMELEMENTWISE_1, csr_mul);
    snax_simd_enable_ext(SIMD_EXT_FP16TOINT8, csr_q);
    snax_simd_memcpy_nd_fast(x_in, q_out, 8, 8, 2, ew_str, ew_bnd, 1, flat_str, q_bnd,
                             0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    all &= step("S4_elementwise_quant", rows, d);
    snax_simd_disable_ext(SIMD_EXT_FP16TOINT8);
    snax_simd_disable_ext(SIMD_EXT_STREAMELEMENTWISE_1);

    // A dropped-as-degenerate task reports complete and costs nothing, so it would otherwise look
    // like a pass rather than a bad geometry.
    printf("[BISECT] bad_config=%d\n", snax_simd_bad_config() ? 1 : 0);
    printf("[BISECT] %s\n", all ? "ALL STEPS RETIRED" : "SOME STEP HUNG");
    snrt_cluster_hw_barrier();
    return all ? 0 : 1;
}
