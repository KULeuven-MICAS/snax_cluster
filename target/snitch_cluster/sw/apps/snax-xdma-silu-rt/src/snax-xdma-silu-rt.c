// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// RUNTIME-PRECISION xDMA SiLU: out = silu(x) = x * sigmoid(x), run end-to-end at
// FP16 AND FP8 on the SAME xDMA netlist, the precision chosen purely by the `fmt`
// CSR field. This is the runtime-precision counterpart of snax-xdma-silu: one set
// of runtime-precision reader extensions computes the kernel at either precision
// — no re-elaboration, no separate netlists.
//
// SiLU is UNARY, so it needs NO scalar glue (unlike softmax/rmsnorm): no reduce,
// no host reciprocal, no fp32out. A single StreamMapRt(SILU) pass does it:
//
//   T1  silu : StreamMapRt(a=1, b=0, func=SILU)   x -> out   (= x*sigmoid(x))
//
// SILU routes the affine result (a*x+b) through the merged FpActivation core (an
// odd-symmetry sigmoid LUT * x, sharing FP units with exp; see FpActivation).
// FP32-internal; the transport element width is the runtime `fmt`.
//
// LANE DATAFLOW: a 512-bit (64-byte) beat carries 32 FP16 elements OR 64 FP8
// elements, so for a row of N values beats = N/32 (FP16) or N/64 (FP8). At FP8
// the SAME row is processed in HALF the beats — that is the runtime-precision
// bandwidth win, visible directly in the per-task cycle count below.

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMMAPRT)
#error \
    "Regenerate the XDMA CSR map: this app needs StreamMapRt (cfg/snax_xdma_cluster.hjson)."
#endif

#define XDMA_BEAT_BYTES 64

// fmt CSR field (StreamMapRt csr2[3:2])
#define FMT_FP16 0u
#define FMT_FP8 2u
// StreamMapRt func CSR (csr2[1:0])
#define ACT_NONE 0u  // LINEAR (out = a*x + b)
#define ACT_EXP 1u
#define ACT_SILU 2u

static uint32_t retask_and_run(void* src, void* dst, uint32_t dst_bound0) {
    if (snax_xdma_retask_1d(src, dst, dst_bound0) != 0) return 0xFFFFFFFFu;
    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    return snax_xdma_last_task_cycle();
}

// One silu pass at the given runtime precision. All buffers are TCDM. Writes
// out_buf (beats beats of fmt result). Returns the single per-task cycle count.
static int run_chain_rt(uint32_t fmt, uint32_t beats, uint8_t* x_in,
                        uint8_t* out_buf, uint32_t* c1) {
    int ok = 1;

    // T1: out = silu(x). a=1.0, b=0, func=SILU. Single unary pass — no reduce, no
    // scalar glue (StreamElementwiseRt idx 7 / StreamReduceRt idx 8 stay disabled
    // = bypass).
    uint32_t csr_silu[3] = {0x3F800000u /*a=1.0f*/, 0u, ACT_SILU | (fmt << 2)};
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAPRT, csr_silu) == 0);
    *c1 = retask_and_run(x_in, out_buf, beats);
    snax_xdma_disable_src_ext(READER_EXT_STREAMMAPRT);

    return ok;
}

// Program the shared 2D[beats,1]-src / 1D-dst AGU shape once for a given beats.
static int program_shape(uint8_t* x_in, uint8_t* out_buf, uint32_t beats) {
    uint32_t str_beat[1] = {XDMA_BEAT_BYTES};
    uint32_t src_str_2d[2] = {XDMA_BEAT_BYTES, beats * XDMA_BEAT_BYTES};
    uint32_t src_bnd_2d[2] = {beats, 1};
    uint32_t bnd_1[1] = {1};
    return (snax_xdma_memcpy_nd_fast(x_in, out_buf, 8, 8, 2, src_str_2d,
                                     src_bnd_2d, 1, str_beat, bnd_1, 0xFFFFFFFF,
                                     0xFFFFFFFF, 0xFFFFFFFF) == 0);
}

// FP16-ULP integer compare. silu is SIGNED (small negative for x<0), but silu is
// monotonic in x and both HW and golden consume the same grid-snapped x, so they
// share sign — within one sign the FP16 bits are monotonic, so |o-g| is the ULP
// distance. Skip the negligible subnormal/zero tail (exp field == 0) the HW
// flushes near the zero crossing. Returns mismatches (>tol).
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
                printf("[SiluRt] fp16 mism[%u]: got %04x gold %04x (%u ulp)\n",
                       i, out[i], gold[i], ulp);
            mism++;
        }
    }
    return mism;
}

// e5m2-code integer compare. Same signed-monotonic argument as check_fp16: silu
// is monotonic and HW+golden share sign, so within one sign the e5m2 byte is
// monotonic and |o-g| is the ULP distance. Skip the subnormal/zero tail (exp
// field == 0).
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
                printf("[SiluRt] fp8 mism[%u]: got %02x gold %02x (%u ulp)\n",
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
        uint32_t n = silu_n;
        printf("[SiluRt] N=%u : one netlist, FP16 then FP8 by fmt CSR\n", n);

        // ---------- FP16 (32 elems/beat) ----------
        {
            uint32_t beats = silu_beats_fp16;
            uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
            uint8_t* x_in = (uint8_t*)base;
            uint8_t* out_buf = x_in + row_bytes;

            snrt_dma_start_1d(x_in, silu_input_fp16, row_bytes);
            snrt_dma_wait_all();

            uint32_t c1 = 0, cold = 0, warm = 0;
            int ok = 1;
            for (int iter = 0; iter < 2; iter++) {
                uint32_t t0 = snrt_mcycle();
                if (iter == 0) ok &= program_shape(x_in, out_buf, beats);
                ok &= run_chain_rt(FMT_FP16, beats, x_in, out_buf, &c1);
                uint32_t t1 = snrt_mcycle();
                if (iter == 0) cold = t1 - t0; else warm = t1 - t0;
            }
            if (!ok || c1 == 0xFFFFFFFFu) {
                printf("[SiluRt] FP16 task setup failed\n");
                return 1;
            }
            uint32_t worst, checked;
            uint32_t mism = check_fp16((uint16_t*)out_buf, silu_golden_fp16, n,
                                       4, &worst, &checked);
            printf("[SiluRt] FP16 beats=%u cycles: silu=%u | cold=%u warm=%u\n",
                   beats, c1, cold, warm);
            printf("[SiluRt] FP16 significant=%u/%u worst ULP=%u : %s\n",
                   checked, n, worst, mism == 0 ? "PASS (<=4 ULP)" : "FAIL");
            if (mism) err++;
        }

        // ---------- FP8 e5m2 (64 elems/beat) : SAME netlist, fmt=FP8 ----------
        {
            uint32_t beats = silu_beats_fp8;
            uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
            uint8_t* x_in = (uint8_t*)base;
            uint8_t* out_buf = x_in + row_bytes;

            snrt_dma_start_1d(x_in, silu_input_fp8, row_bytes);
            snrt_dma_wait_all();

            uint32_t c1 = 0, cold = 0, warm = 0;
            int ok = 1;
            for (int iter = 0; iter < 2; iter++) {
                uint32_t t0 = snrt_mcycle();
                if (iter == 0) ok &= program_shape(x_in, out_buf, beats);
                ok &= run_chain_rt(FMT_FP8, beats, x_in, out_buf, &c1);
                uint32_t t1 = snrt_mcycle();
                if (iter == 0) cold = t1 - t0; else warm = t1 - t0;
            }
            if (!ok || c1 == 0xFFFFFFFFu) {
                printf("[SiluRt] FP8 task setup failed\n");
                return 1;
            }
            uint32_t worst, checked;
            uint32_t mism = check_fp8((uint8_t*)out_buf, silu_golden_fp8, n, 2,
                                      &worst, &checked);
            printf("[SiluRt] FP8  beats=%u cycles: silu=%u | cold=%u warm=%u\n",
                   beats, c1, cold, warm);
            printf("[SiluRt] FP8  significant=%u/%u worst e5m2 ULP=%u : %s\n",
                   checked, n, worst, mism == 0 ? "PASS (<=2 e5m2)" : "FAIL");
            if (mism) err++;
        }

        if (!err)
            printf("[SiluRt] ALL PASS — one netlist, silu at FP16 & FP8 by runtime fmt\n");
    }
    return err != 0;
}
