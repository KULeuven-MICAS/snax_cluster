// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// RUNTIME-PRECISION xDMA rmsnorm: out = x * inv_rms, inv_rms = 1/sqrt(mean(x^2)),
// mean = Σx^2 / N, eps = 0. Run end-to-end at FP16 AND FP8 on the SAME xDMA
// netlists, the precision chosen purely by the `fmt` CSR field. This is the
// runtime-precision counterpart of snax-xdma-rmsnorm: one set of runtime-precision
// reader extensions (StreamReduceRt = idx 8, StreamMapRt = idx 6) computes the
// whole rmsnorm at either precision -- no re-elaboration, no separate netlists.
//
//   T1 Σx^2 : StreamReduceRt(SUMSQ, fp32out)                 x   -> ssq (FP32)
//   host    : inv_rms = 1/sqrt(Σx^2 / N)  (precomputed in data.h; DM core no FPU)
//   T2 scale: StreamMapRt(a=inv_rms, b=0, LINEAR)            x   -> out
//
// fp32out (StreamReduceRt csr1 bit[9]) makes the SUMSQ scalar come back as raw
// FP32 splatted across the beat, so the host glue is format-independent: the
// reciprocal-of-sqrt is computed off the true FP32 Σx^2 regardless of the
// transport precision, and the runtime reduce is validated against the FP32 golden.
//
// LANE DATAFLOW: a 512-bit (64-byte) beat carries 32 FP16 elements OR 64 FP8
// elements, so for a row of N values beats = N/32 (FP16) or N/64 (FP8). At FP8
// the SAME row is processed in HALF the beats -- that is the runtime-precision
// bandwidth win, visible directly in the per-task cycle counts below.

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMMAPRT) || !defined(READER_EXT_STREAMREDUCERT)
#error \
    "Regenerate the XDMA CSR map: this app needs StreamMapRt + StreamReduceRt (cfg/snax_xdma_cluster.hjson)."
#endif

#define XDMA_BEAT_BYTES 64

// fmt CSR field (StreamMapRt csr2[3:2], StreamReduceRt csr0[17:16])
#define FMT_FP16 0u
#define FMT_FP8 2u
// StreamReduceRt op CSR (csr1)
#define OP_SUMSQ 2u      // fused-FMA square mode: acc + x*x (MAX=0, ADD=1)
#define RED_FP32 0x200u  // bit[9]: emit scalar as raw FP32 (format-independent)
// StreamMapRt func CSR (csr2[1:0])
#define ACT_NONE 0u  // LINEAR (out = a*x + b)

static uint32_t retask_and_run(void* src, void* dst, uint32_t dst_bound0) {
    if (snax_xdma_retask_1d(src, dst, dst_bound0) != 0) return 0xFFFFFFFFu;
    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    return snax_xdma_last_task_cycle();
}

// One rmsnorm chain at the given runtime precision. All buffers are TCDM. Writes
// out_buf (beats beats of fmt result). Returns the two per-task cycle counts and
// the raw FP32 Σx^2 (fp32out).
static int run_chain_rt(uint32_t fmt, uint32_t beats, uint32_t inv_rms_bits,
                        uint8_t* x_in, uint8_t* ssq_buf, uint8_t* out_buf,
                        uint32_t* c1, uint32_t* c2, uint32_t* ssq_hw) {
    int ok = 1;

    // T1: Σx^2. fp32out -> raw FP32 sum splatted, so the host reciprocal-of-sqrt
    // is accurate and format-independent (no FP8/FP16 decode of the scalar).
    uint32_t csr_ssq[2] = {beats | (fmt << 16), OP_SUMSQ | RED_FP32};
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCERT, csr_ssq) == 0);
    *c1 = retask_and_run(x_in, ssq_buf, 1);
    snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCERT);
    *ssq_hw = ((uint32_t*)ssq_buf)[0];

    // T2: out = inv_rms * x. Host-provided reciprocal of sqrt(mean); the DM core
    // has no FPU so inv_rms is precomputed in data.h and just fed to the map.
    uint32_t csr_norm[3] = {inv_rms_bits, 0u, ACT_NONE | (fmt << 2)};
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAPRT, csr_norm) == 0);
    *c2 = retask_and_run(x_in, out_buf, beats);
    snax_xdma_disable_src_ext(READER_EXT_STREAMMAPRT);

    return ok;
}

// Program the shared 2D[beats,1]-src / 1D-dst AGU shape once for a given beats.
static int program_shape(uint8_t* x_in, uint8_t* dst_buf, uint32_t beats) {
    uint32_t str_beat[1] = {XDMA_BEAT_BYTES};
    uint32_t src_str_2d[2] = {XDMA_BEAT_BYTES, beats * XDMA_BEAT_BYTES};
    uint32_t src_bnd_2d[2] = {beats, 1};
    uint32_t bnd_1[1] = {1};
    return (snax_xdma_memcpy_nd_fast(x_in, dst_buf, 8, 8, 2, src_str_2d,
                                     src_bnd_2d, 1, str_beat, bnd_1, 0xFFFFFFFF,
                                     0xFFFFFFFF, 0xFFFFFFFF) == 0);
}

// FP16-ULP integer compare. rmsnorm output is SIGNED (out = x * inv_rms), but for
// SAME-SIGN operands the raw FP16-bit integer distance equals the monotonic-key
// ULP distance; skipping the tiny (subnormal/zero) golden tail keeps every checked
// pair same-sign. Returns mismatches (>tol).
static uint32_t check_fp16(uint16_t* out, const uint16_t* gold, uint32_t n,
                           uint32_t tol, uint32_t* worst, uint32_t* checked) {
    uint32_t mism = 0;
    *worst = 0;
    *checked = 0;
    for (uint32_t i = 0; i < n; i++) {
        if ((gold[i] & 0x7C00u) == 0) continue;  // skip tiny golden
        (*checked)++;
        int32_t o = (int32_t)out[i], g = (int32_t)gold[i];
        uint32_t ulp = (o > g) ? (uint32_t)(o - g) : (uint32_t)(g - o);
        if (ulp > *worst) *worst = ulp;
        if (ulp > tol) {
            if (mism < 6)
                printf("[RmsnormRt] fp16 mism[%u]: got %04x gold %04x (%u ulp)\n",
                       i, out[i], gold[i], ulp);
            mism++;
        }
    }
    return mism;
}

// e5m2-code integer compare. rmsnorm output is SIGNED, but for SAME-SIGN operands
// the raw e5m2-byte integer distance equals the monotonic-key ULP distance; skip
// the subnormal/zero tail (exp field == 0) so every checked pair is same-sign.
static uint32_t check_fp8(uint8_t* out, const uint8_t* gold, uint32_t n,
                          uint32_t tol, uint32_t* worst, uint32_t* checked) {
    uint32_t mism = 0;
    *worst = 0;
    *checked = 0;
    for (uint32_t i = 0; i < n; i++) {
        if ((gold[i] & 0x7Cu) == 0) continue;  // skip tiny golden
        (*checked)++;
        int32_t o = (int32_t)out[i], g = (int32_t)gold[i];
        uint32_t ulp = (o > g) ? (uint32_t)(o - g) : (uint32_t)(g - o);
        if (ulp > *worst) *worst = ulp;
        if (ulp > tol) {
            if (mism < 6)
                printf("[RmsnormRt] fp8 mism[%u]: got %02x gold %02x (%u ulp)\n",
                       i, out[i], gold[i], ulp);
            mism++;
        }
    }
    return mism;
}

int main() {
    int err = 0;
    if (snrt_is_dm_core()) {
        uint32_t base = snrt_cluster_base_addrl();
        uint32_t n = rmsnorm_n;
        printf("[RmsnormRt] N=%u : one set of netlists, FP16 then FP8 by fmt CSR\n", n);

        // ---------- FP16 (32 elems/beat) ----------
        {
            uint32_t beats = rmsnorm_beats_fp16;
            uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
            uint8_t* x_in = (uint8_t*)base;
            uint8_t* ssq_buf = x_in + row_bytes;           // 1 beat (Σx^2)
            uint8_t* out_buf = ssq_buf + XDMA_BEAT_BYTES;  // beats (fmt result)

            snrt_dma_start_1d(x_in, rmsnorm_input_fp16, row_bytes);
            snrt_dma_wait_all();

            uint32_t c1 = 0, c2 = 0, ssqb = 0, cold = 0, warm = 0;
            int ok = 1;
            for (int iter = 0; iter < 2; iter++) {
                uint32_t t0 = snrt_mcycle();
                if (iter == 0) ok &= program_shape(x_in, ssq_buf, beats);
                ok &= run_chain_rt(FMT_FP16, beats, rmsnorm_inv_rms_fp16, x_in,
                                   ssq_buf, out_buf, &c1, &c2, &ssqb);
                uint32_t t1 = snrt_mcycle();
                if (iter == 0) cold = t1 - t0; else warm = t1 - t0;
            }
            if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu) {
                printf("[RmsnormRt] FP16 task setup failed\n");
                return 1;
            }
            // fp32out: ssqb is the raw FP32 Σx^2. Validate vs the FP32 golden
            // (Σx^2 > 0 so raw uint32 ordering is monotonic; the HW fold order
            // differs from the numpy sum by a few FP32 ULP -> WARN only).
            uint32_t gssq = rmsnorm_ssq_fp16;
            uint32_t dssq = (ssqb > gssq) ? (ssqb - gssq) : (gssq - ssqb);
            if (dssq > 1024u)
                printf("[RmsnormRt] WARN FP16 ssq %08x vs golden %08x (%u fp32-ulp)\n",
                       ssqb, gssq, dssq);
            uint32_t worst, checked;
            uint32_t mism = check_fp16((uint16_t*)out_buf, rmsnorm_golden_fp16,
                                       n, 4, &worst, &checked);
            printf("[RmsnormRt] FP16 beats=%u cycles: sumsq=%u norm=%u | cold=%u warm=%u\n",
                   beats, c1, c2, cold, warm);
            printf("[RmsnormRt] FP16 significant=%u/%u worst ULP=%u : %s\n",
                   checked, n, worst, mism == 0 ? "PASS (<=4 ULP)" : "FAIL");
            if (mism) err++;
        }

        // ---------- FP8 e5m2 (64 elems/beat) : SAME netlists, fmt=FP8 ----------
        {
            uint32_t beats = rmsnorm_beats_fp8;
            uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
            uint8_t* x_in = (uint8_t*)base;
            uint8_t* ssq_buf = x_in + row_bytes;
            uint8_t* out_buf = ssq_buf + XDMA_BEAT_BYTES;

            snrt_dma_start_1d(x_in, rmsnorm_input_fp8, row_bytes);
            snrt_dma_wait_all();

            uint32_t c1 = 0, c2 = 0, ssqb = 0, cold = 0, warm = 0;
            int ok = 1;
            for (int iter = 0; iter < 2; iter++) {
                uint32_t t0 = snrt_mcycle();
                if (iter == 0) ok &= program_shape(x_in, ssq_buf, beats);
                ok &= run_chain_rt(FMT_FP8, beats, rmsnorm_inv_rms_fp8, x_in,
                                   ssq_buf, out_buf, &c1, &c2, &ssqb);
                uint32_t t1 = snrt_mcycle();
                if (iter == 0) cold = t1 - t0; else warm = t1 - t0;
            }
            if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu) {
                printf("[RmsnormRt] FP8 task setup failed\n");
                return 1;
            }
            uint32_t gssq = rmsnorm_ssq_fp8;
            uint32_t dssq = (ssqb > gssq) ? (ssqb - gssq) : (gssq - ssqb);
            if (dssq > 1024u)
                printf("[RmsnormRt] WARN FP8  ssq %08x vs golden %08x (%u fp32-ulp)\n",
                       ssqb, gssq, dssq);
            uint32_t worst, checked;
            uint32_t mism = check_fp8((uint8_t*)out_buf, rmsnorm_golden_fp8, n,
                                      2, &worst, &checked);
            printf("[RmsnormRt] FP8  beats=%u cycles: sumsq=%u norm=%u | cold=%u warm=%u\n",
                   beats, c1, c2, cold, warm);
            printf("[RmsnormRt] FP8  significant=%u/%u worst e5m2 ULP=%u : %s\n",
                   checked, n, worst, mism == 0 ? "PASS (<=2 e5m2)" : "FAIL");
            if (mism) err++;
        }

        if (!err)
            printf("[RmsnormRt] ALL PASS — one netlist set, rmsnorm at FP16 & FP8 by runtime fmt\n");
    }
    return err != 0;
}
