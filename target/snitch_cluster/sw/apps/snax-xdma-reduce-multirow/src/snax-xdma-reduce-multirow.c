// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Standalone multi-row xDMA StreamReduce test. Drives S independent per-row reductions in ONE dispatch
// (the per-row form RMSNorm/Softmax need): a 2D reader {beats inner, rows outer} feeds StreamReduce,
// whose counter wraps every `beats` beats and re-inits the per-lane accumulators, emitting one splatted
// scalar beat per row; a 1D writer drains `rows` beats. No RTL change vs the single-row reduce -- only the
// AGU shape ({beats,rows} reader, {rows} writer) differs.
//
// It also exercises the mem_wide_narrow_mux DMAWriteDataCorrect assert: with the input fully initialized,
// the reduce writes defined (non-X) data, so the assert (gated on q.write) stays silent.
//
// LANE DATAFLOW -- StreamReduce (N beats in -> 1 beat out; a row's 32 lanes collapse to one scalar). This
// is NOT a 512b-in/512b-out element-wise op: the output beat carries ONE useful value, not 32.
//   in  beat k=0..N-1 : [ l0  l1  ... l31 ]   32 FP16 lanes/beat; N = beats per row (= D/32)
//        the 32 per-lane FP32 partials accumulate down the N beats; a horizontal reduce-tree then folds
//        them into ONE scalar s = max | sum | sum-of-squares over the whole D-element row.
//   out beat          : [ s   s   ...  s  ]   s SPLATTED across all 32 lanes (1 useful + 31 copies)
//   Read lane 0 only (stride 32 lanes = one 64-B beat). N input beats -> 1 output beat per row.
//
// EXAMPLE (params.hjson default: rows=4, D=64 -> beats=D/32=2; FP16, all L1-local)
//   in   x    [rows=4, D=64] fp16  = 4*128 = 512 B
//   out  one scalar PER ROW per op -> 4 splatted 64-B beats = 256 B/op (useful payload 4 fp16 = 8 B/op)
//
//   L1 memory map (byte offset from cluster base):
//     +0     x      512 B   [4,64] input
//     +512   ssq    256 B   [4] SUMSQ scalars (4 splatted 64-B beats)
//     +768   max    256 B   [4] MAX   scalars
//     +1024  add    256 B   [4] ADD   scalars
//     total  1280 B

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMREDUCE)
#error "Regenerate the XDMA CSR map with StreamReduce (CFG_OVERRIDE=cfg/snax_xdma_cluster.hjson)."
#endif

#define XDMA_BEAT_BYTES 64
#define FP16_PER_BEAT 32
#define OP_MAX 0u
#define OP_ADD 1u
#define OP_SUMSQ 2u

// FP16 bits -> monotonic ordering key (signed): adjacent FP16 values map to adjacent keys, so
// |key(a)-key(b)| is the FP16-ULP distance.
static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

// Minimal re-task launch: reuse the persisted 2D-reader/1D-writer shape, rewrite addresses + writer bound.
static uint32_t retask_and_run(void* src, void* dst, uint32_t dst_bound0) {
    if (snax_xdma_retask_1d(src, dst, dst_bound0) != 0) return 0xFFFFFFFFu;
    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    return snax_xdma_last_task_cycle();
}

// Check `rows` per-row scalars (one splatted beat each: lane 0 is at uint16 index r*FP16_PER_BEAT)
// against the golden. Returns the number of rows beyond tolerance.
static uint32_t check_rows(const char* name, uint16_t* out, const uint16_t* golden, uint32_t rows) {
    uint32_t worst = 0, bad = 0;
    for (uint32_t r = 0; r < rows; r++) {
        uint16_t got = out[r * FP16_PER_BEAT];
        uint32_t o = fp16_mono(got), g = fp16_mono(golden[r]);
        uint32_t ulp = (o > g) ? (o - g) : (g - o);
        if (ulp > worst) worst = ulp;
        if (ulp > 4) {
            if (bad < 6)
                printf("[ReduceMR] %s row %u: got %04x golden %04x (%u ulp)\n", name, r, got,
                       golden[r], ulp);
            bad++;
        }
    }
    printf("[ReduceMR] %s: worst FP16 ULP=%u over %u rows\n", name, worst, rows);
    return bad;
}

int main() {
    int err = 0;
    if (snrt_is_dm_core()) {
        uint32_t base = snrt_cluster_base_addrl();
        uint32_t rows = reduce_rows;
        uint32_t beats = reduce_beats;
        uint32_t row_bytes = beats * XDMA_BEAT_BYTES;  // bytes per input row
        uint32_t in_bytes = rows * row_bytes;          // all rows

        uint8_t* x_in = (uint8_t*)base;
        uint8_t* ssq_buf = x_in + in_bytes;                 // rows beats (one splatted scalar each)
        uint8_t* max_buf = ssq_buf + rows * XDMA_BEAT_BYTES;
        uint8_t* sum_buf = max_buf + rows * XDMA_BEAT_BYTES;

        printf("[ReduceMR] rows=%u D=%u beats=%u\n", rows, reduce_d, beats);

        // Fully initialize ALL rows of the input. This is the crux: a partially-written input would feed
        // X into the reduce, which would then write X (the failure symptom) and trip DMAWriteDataCorrect.
        snrt_dma_start_1d(x_in, reduce_input, in_bytes);
        snrt_dma_wait_all();

        // Program the 2D reader {beats inner, rows outer} -> 1D writer {rows} ONCE; each op is a retask.
        uint32_t str_beat[1] = {XDMA_BEAT_BYTES};
        uint32_t src_str_2d[2] = {XDMA_BEAT_BYTES, row_bytes};
        uint32_t src_bnd_2d[2] = {beats, rows};
        uint32_t bnd_rows[1] = {rows};

        int ok = (snax_xdma_memcpy_nd_fast(x_in, ssq_buf, 8, 8, 2, src_str_2d, src_bnd_2d, 1, str_beat,
                                           bnd_rows, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);

        // SUMSQ (RMSNorm), MAX + ADD (Softmax) -- one reduce op per dispatch, all multi-row.
        uint32_t csr_ssq[2] = {beats, OP_SUMSQ};
        ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_ssq) == 0);
        uint32_t c_ssq = retask_and_run(x_in, ssq_buf, rows);
        snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);

        uint32_t csr_max[2] = {beats, OP_MAX};
        ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_max) == 0);
        uint32_t c_max = retask_and_run(x_in, max_buf, rows);
        snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);

        uint32_t csr_add[2] = {beats, OP_ADD};
        ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_add) == 0);
        uint32_t c_add = retask_and_run(x_in, sum_buf, rows);
        snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);

        if (!ok || c_ssq == 0xFFFFFFFFu || c_max == 0xFFFFFFFFu || c_add == 0xFFFFFFFFu) {
            printf("[ReduceMR] xDMA task setup failed\n");
            return 1;
        }
        printf("[ReduceMR] cycles: sumsq=%u max=%u add=%u\n", c_ssq, c_max, c_add);

        err += (int)check_rows("SUMSQ", (uint16_t*)ssq_buf, reduce_ssq_golden, rows);
        err += (int)check_rows("MAX", (uint16_t*)max_buf, reduce_max_golden, rows);
        err += (int)check_rows("ADD", (uint16_t*)sum_buf, reduce_sum_golden, rows);

        if (err == 0)
            printf("[ReduceMR] PASS (all rows <=4 ULP)\n");
        else
            printf("[ReduceMR] FAIL (%d rows beyond 4 ULP)\n", err);
    }
    return err != 0;
}
