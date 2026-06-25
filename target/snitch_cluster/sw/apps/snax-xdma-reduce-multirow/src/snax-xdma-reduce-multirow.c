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
#define FP32_PER_BEAT 16
#define OP_MAX 0u
#define OP_ADD 1u
#define OP_SUMSQ 2u
#define REDUCE_OUT_FP32 (1u << 9)  // StreamReduce op CSR bit[9]: emit the scalar in FP32 (no FP16 narrow)

// FP16 bits -> monotonic ordering key (signed): adjacent FP16 values map to adjacent keys, so
// |key(a)-key(b)| is the FP16-ULP distance.
static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

// FP32 bits -> monotonic ordering key: flip the sign bit for positives, invert all bits for negatives.
// For two same-sign values |key(a)-key(b)| is their FP32-ULP distance (no FPU needed on the DM core).
static inline uint32_t fp32_mono(uint32_t b) { return (b & 0x80000000u) ? (~b) : (b | 0x80000000u); }

// Exact integer FP16 -> FP32 bit widen (normals/inf/nan; subnormals normalized). Lets the no-FPU DM core
// compare an fp16-out scalar against the true FP32 SUMSQ to show the fp16 path can't represent it.
static inline uint32_t fp16_to_fp32_bits(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    if (exp == 0x1Fu) return sign | 0x7F800000u | (man << 13);  // inf / nan
    if (exp == 0) {
        if (man == 0) return sign;  // +-0
        uint32_t e = 0;
        while ((man & 0x400u) == 0) {
            man <<= 1;
            e++;
        }
        man &= 0x3FFu;
        return sign | ((127u - 15u - e) << 23) | (man << 13);
    }
    return sign | ((exp - 15u + 127u) << 23) | (man << 13);
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

// Per-row FP32 scalars (one splatted beat each: lane 0 is at uint32 index r*FP32_PER_BEAT). Flags a row
// that is non-finite (inf/NaN exponent) or beyond an FP32-ULP tolerance of the golden. Returns bad rows.
static uint32_t check_rows_fp32(const char* name, const uint32_t* out, const uint32_t* golden,
                                uint32_t rows) {
    uint32_t worst = 0, bad = 0;
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t got = out[r * FP32_PER_BEAT];
        uint32_t exp = (got >> 23) & 0xFFu;
        uint32_t o = fp32_mono(got), g = fp32_mono(golden[r]);
        uint32_t ulp = (o > g) ? (o - g) : (g - o);
        if (ulp > worst) worst = ulp;
        if (exp == 0xFFu || ulp > 0x10000u) {  // non-finite, or > ~0.8% (tree-order vs sequential FP32)
            if (bad < 6)
                printf("[ReduceMR] %s row %u: got %08x golden %08x (%u fp32-ulp%s)\n", name, r, got,
                       golden[r], ulp, exp == 0xFFu ? ", NON-FINITE" : "");
            bad++;
        }
    }
    printf("[ReduceMR] %s: worst fp32 ULP=%u over %u rows\n", name, worst, rows);
    return bad;
}

// Show the default fp16-out path is unusable for a SUMSQ this large: widen each fp16-out scalar to fp32
// and report its FP32-ULP distance from the true SUMSQ. inf (0x7c00) or wrapped garbage both land
// millions of ULPs off; a row counts as "broken" when off by > ~25% of the golden. Returns broken rows.
static uint32_t count_fp16_broken(const char* name, const uint16_t* out, const uint32_t* golden_f32,
                                  uint32_t rows) {
    uint32_t broken = 0;
    for (uint32_t r = 0; r < rows; r++) {
        uint16_t got = out[r * FP16_PER_BEAT];
        uint32_t got32 = fp16_to_fp32_bits(got);
        uint32_t inf = ((got & 0x7FFFu) == 0x7C00u);
        uint32_t o = fp32_mono(got32), g = fp32_mono(golden_f32[r]);
        uint32_t ulp = (o > g) ? (o - g) : (g - o);
        int bad = inf || ulp > 0x01000000u;  // ~ >25% relative: unusable
        if (bad) broken++;
        printf("[ReduceMR] %s row %u: fp16-out=%04x vs true=%08x -> %s\n", name, r, got, golden_f32[r],
               inf ? "inf (UNUSABLE)" : (bad ? "garbage (UNUSABLE)" : "ok"));
    }
    printf("[ReduceMR] %s: %u/%u rows unrepresentable in fp16 (fp32out fixes this)\n", name, broken, rows);
    return broken;
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
        uint8_t* xbig_in = sum_buf + rows * XDMA_BEAT_BYTES;       // large-magnitude input (overflows FP16)
        uint8_t* ssq16_big_buf = xbig_in + in_bytes;              // FP16-out SUMSQ of xbig (-> inf)
        uint8_t* ssq32_big_buf = ssq16_big_buf + rows * XDMA_BEAT_BYTES;  // FP32-out SUMSQ of xbig (true)

        printf("[ReduceMR] rows=%u D=%u beats=%u\n", rows, reduce_d, beats);

        // Fully initialize ALL rows of both inputs. This is the crux: a partially-written input would feed
        // X into the reduce, which would then write X (the failure symptom) and trip DMAWriteDataCorrect.
        snrt_dma_start_1d(x_in, reduce_input, in_bytes);
        snrt_dma_start_1d(xbig_in, reduce_input_big, in_bytes);
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

        // Overflow A/B on the large-magnitude input (per-row SUMSQ ~1e9). Same SUMSQ op, two output modes:
        //   fp16-out (default): the scalar narrows to the FP16 grid -> +inf (the GAP-2 bug).
        //   fp32-out (bit[9]) : the scalar is splatted in FP32 -> the host gets the true SUMSQ.
        uint32_t csr_ssq16_big[2] = {beats, OP_SUMSQ};
        ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_ssq16_big) == 0);
        uint32_t c_big16 = retask_and_run(xbig_in, ssq16_big_buf, rows);
        snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);

        uint32_t csr_ssq32_big[2] = {beats, OP_SUMSQ | REDUCE_OUT_FP32};
        ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_ssq32_big) == 0);
        uint32_t c_big32 = retask_and_run(xbig_in, ssq32_big_buf, rows);
        snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);

        if (!ok || c_ssq == 0xFFFFFFFFu || c_max == 0xFFFFFFFFu || c_add == 0xFFFFFFFFu ||
            c_big16 == 0xFFFFFFFFu || c_big32 == 0xFFFFFFFFu) {
            printf("[ReduceMR] xDMA task setup failed\n");
            return 1;
        }
        printf("[ReduceMR] cycles: sumsq=%u max=%u add=%u big16=%u big32=%u\n", c_ssq, c_max, c_add,
               c_big16, c_big32);

        err += (int)check_rows("SUMSQ", (uint16_t*)ssq_buf, reduce_ssq_golden, rows);
        err += (int)check_rows("MAX", (uint16_t*)max_buf, reduce_max_golden, rows);
        err += (int)check_rows("ADD", (uint16_t*)sum_buf, reduce_sum_golden, rows);

        // The fp16-out scalar can't represent a SUMSQ this large (evidence of the bug); the fp32-out scalar
        // must be the true value (the fix). Only the fp32-out path is pass/fail; the fp16 breakage is logged
        // evidence (expected to be all rows -- it demonstrates the GAP-2 overflow the fp32out flag fixes).
        count_fp16_broken("SUMSQ-big-fp16", (uint16_t*)ssq16_big_buf, reduce_ssq_big_f32_golden, rows);
        err += (int)check_rows_fp32("SUMSQ-big-fp32", (uint32_t*)ssq32_big_buf, reduce_ssq_big_f32_golden,
                                    rows);

        if (err == 0)
            printf("[ReduceMR] PASS (all rows <=4 ULP; fp32out SUMSQ recovers the overflowing reduction)\n");
        else
            printf("[ReduceMR] FAIL (%d rows beyond tolerance)\n", err);
    }
    return err != 0;
}
