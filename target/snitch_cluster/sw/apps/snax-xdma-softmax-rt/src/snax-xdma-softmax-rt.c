// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// RUNTIME-PRECISION xDMA softmax: out = exp(x - max(x)) / Σexp(x - max(x)),
// run end-to-end at FP16 AND FP8 on the SAME xDMA netlists, the precision chosen
// purely by the `fmt` CSR field. This is the vsim counterpart of the Chisel
// SoftmaxChainRtDemo: one set of runtime-precision reader extensions
// (StreamMapRt = idx 6, StreamReduceRt = idx 8) computes the whole softmax
// epilogue at either precision — no re-elaboration, no separate netlists.
//
//   T1 max  : StreamReduceRt(MAX, fp32out)                        x   -> max
//   T2 e+Σ  : StreamMapRt(EXP,b=-max) -||> StreamReduceRt(ADD,tap,fp32out)
//             x -> [exp row (N beats) , Σexp (1 trailing beat)]
//   host    : inv_sum = 1/Σexp  (precomputed in data.h; DM core has no FPU)
//   T3 norm : StreamMapRt(a=inv_sum, b=0, LINEAR)                 exp -> out
//
// fp32out (StreamReduceRt csr1 bit[9]) makes BOTH reduce scalars come back as
// raw FP32 splatted across the beat, so the scalar glue is format-independent:
// -max is a single integer sign-flip on the FP32 bits (no FP8/FP16 decode), and
// the FP32 Σexp reciprocal is accurate regardless of the transport precision.
//
// LANE DATAFLOW: a 512-bit (64-byte) beat carries 32 FP16 elements OR 64 FP8
// elements, so for a row of N values beats = N/32 (FP16) or N/64 (FP8). At FP8
// the SAME row is processed in HALF the beats — that is the runtime-precision
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
#define OP_MAX 0u
#define OP_ADD 1u
#define RED_TAP 0x100u   // bit[8]: pass row through + trailing scalar beat
#define RED_FP32 0x200u  // bit[9]: emit scalar as raw FP32 (format-independent)
// StreamMapRt func CSR (csr2[1:0])
#define ACT_NONE 0u  // LINEAR (out = a*x + b)
#define ACT_EXP 1u

static uint32_t retask_and_run(void* src, void* dst, uint32_t dst_bound0) {
    if (snax_xdma_retask_1d(src, dst, dst_bound0) != 0) return 0xFFFFFFFFu;
    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    return snax_xdma_last_task_cycle();
}

// One softmax chain at the given runtime precision. All buffers are TCDM. Writes
// out_buf (beats beats of fmt result). Returns the three per-task cycle counts.
static int run_chain_rt(uint32_t fmt, uint32_t beats, uint32_t inv_sum_bits,
                        uint8_t* x_in, uint8_t* max_buf, uint8_t* exp_buf,
                        uint8_t* out_buf, uint32_t* c1, uint32_t* c2,
                        uint32_t* c3, uint32_t* sum_hw_bits) {
    int ok = 1;

    // T1: row max. fp32out -> FP32 max splatted, so -max is a pure integer flip.
    uint32_t csr_max[2] = {beats | (fmt << 16), OP_MAX | RED_FP32};
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCERT, csr_max) == 0);
    *c1 = retask_and_run(x_in, max_buf, 1);
    snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCERT);
    uint32_t neg_max = ((uint32_t*)max_buf)[0] ^ 0x80000000u;

    // T2: exp(x-max) + Σexp fused. StreamMapRt(EXP) chains before StreamReduceRt
    // (ascending ext index; StreamElementwiseRt idx 7 stays disabled = bypass).
    // tap re-emits the exp row (N beats) then appends the FP32 Σ as beat N.
    uint32_t csr_exp[3] = {0x3F800000u /*a=1.0f*/, neg_max, ACT_EXP | (fmt << 2)};
    uint32_t csr_sumtap[2] = {beats | (fmt << 16), OP_ADD | RED_TAP | RED_FP32};
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAPRT, csr_exp) == 0);
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCERT, csr_sumtap) == 0);
    *c2 = retask_and_run(x_in, exp_buf, beats + 1);
    snax_xdma_disable_src_ext(READER_EXT_STREAMMAPRT);
    snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCERT);
    *sum_hw_bits = ((uint32_t*)(exp_buf + beats * XDMA_BEAT_BYTES))[0];

    // T3: out = inv_sum * exp (host-provided reciprocal of the FP32 Σexp).
    uint32_t csr_norm[3] = {inv_sum_bits, 0u, ACT_NONE | (fmt << 2)};
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAPRT, csr_norm) == 0);
    *c3 = retask_and_run(exp_buf, out_buf, beats);
    snax_xdma_disable_src_ext(READER_EXT_STREAMMAPRT);

    return ok;
}

// Program the shared 2D[beats,1]-src / 1D-dst AGU shape once for a given beats.
static int program_shape(uint8_t* x_in, uint8_t* max_buf, uint32_t beats) {
    uint32_t str_beat[1] = {XDMA_BEAT_BYTES};
    uint32_t src_str_2d[2] = {XDMA_BEAT_BYTES, beats * XDMA_BEAT_BYTES};
    uint32_t src_bnd_2d[2] = {beats, 1};
    uint32_t bnd_1[1] = {1};
    return (snax_xdma_memcpy_nd_fast(x_in, max_buf, 8, 8, 2, src_str_2d,
                                     src_bnd_2d, 1, str_beat, bnd_1, 0xFFFFFFFF,
                                     0xFFFFFFFF, 0xFFFFFFFF) == 0);
}

// FP16-ULP integer compare (softmax positive -> FP16 bits monotonic; skip the
// negligible subnormal/zero tail the HW flushes). Returns mismatches (>tol).
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
                printf("[SoftmaxRt] fp16 mism[%u]: got %04x gold %04x (%u ulp)\n",
                       i, out[i], gold[i], ulp);
            mism++;
        }
    }
    return mism;
}

// e5m2-code integer compare (positive softmax -> e5m2 byte monotonic; skip the
// subnormal/zero tail, exp field == 0).
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
                printf("[SoftmaxRt] fp8 mism[%u]: got %02x gold %02x (%u ulp)\n",
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
        uint32_t n = softmax_n;
        printf("[SoftmaxRt] N=%u : one set of netlists, FP16 then FP8 by fmt CSR\n", n);

        // ---------- FP16 (32 elems/beat) ----------
        {
            uint32_t beats = softmax_beats_fp16;
            uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
            uint8_t* x_in = (uint8_t*)base;
            uint8_t* max_buf = x_in + row_bytes;
            uint8_t* exp_buf = max_buf + XDMA_BEAT_BYTES;
            uint8_t* out_buf = exp_buf + (beats + 1) * XDMA_BEAT_BYTES;

            snrt_dma_start_1d(x_in, softmax_input_fp16, row_bytes);
            snrt_dma_wait_all();

            uint32_t c1 = 0, c2 = 0, c3 = 0, sumb = 0, cold = 0, warm = 0;
            int ok = 1;
            for (int iter = 0; iter < 2; iter++) {
                uint32_t t0 = snrt_mcycle();
                if (iter == 0) ok &= program_shape(x_in, max_buf, beats);
                ok &= run_chain_rt(FMT_FP16, beats, softmax_inv_sum_fp16, x_in,
                                   max_buf, exp_buf, out_buf, &c1, &c2, &c3,
                                   &sumb);
                uint32_t t1 = snrt_mcycle();
                if (iter == 0) cold = t1 - t0; else warm = t1 - t0;
            }
            if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu ||
                c3 == 0xFFFFFFFFu) {
                printf("[SoftmaxRt] FP16 task setup failed\n");
                return 1;
            }
            uint32_t worst, checked;
            uint32_t mism = check_fp16((uint16_t*)out_buf, softmax_golden_fp16,
                                       n, 4, &worst, &checked);
            printf("[SoftmaxRt] FP16 beats=%u cycles: max=%u exp+sum=%u norm=%u | cold=%u warm=%u\n",
                   beats, c1, c2, c3, cold, warm);
            printf("[SoftmaxRt] FP16 significant=%u/%u worst ULP=%u : %s\n",
                   checked, n, worst, mism == 0 ? "PASS (<=4 ULP)" : "FAIL");
            if (mism) err++;
        }

        // ---------- FP8 e5m2 (64 elems/beat) : SAME netlists, fmt=FP8 ----------
        {
            uint32_t beats = softmax_beats_fp8;
            uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
            uint8_t* x_in = (uint8_t*)base;
            uint8_t* max_buf = x_in + row_bytes;
            uint8_t* exp_buf = max_buf + XDMA_BEAT_BYTES;
            uint8_t* out_buf = exp_buf + (beats + 1) * XDMA_BEAT_BYTES;

            snrt_dma_start_1d(x_in, softmax_input_fp8, row_bytes);
            snrt_dma_wait_all();

            uint32_t c1 = 0, c2 = 0, c3 = 0, sumb = 0, cold = 0, warm = 0;
            int ok = 1;
            for (int iter = 0; iter < 2; iter++) {
                uint32_t t0 = snrt_mcycle();
                if (iter == 0) ok &= program_shape(x_in, max_buf, beats);
                ok &= run_chain_rt(FMT_FP8, beats, softmax_inv_sum_fp8, x_in,
                                   max_buf, exp_buf, out_buf, &c1, &c2, &c3,
                                   &sumb);
                uint32_t t1 = snrt_mcycle();
                if (iter == 0) cold = t1 - t0; else warm = t1 - t0;
            }
            if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu ||
                c3 == 0xFFFFFFFFu) {
                printf("[SoftmaxRt] FP8 task setup failed\n");
                return 1;
            }
            uint32_t worst, checked;
            uint32_t mism = check_fp8((uint8_t*)out_buf, softmax_golden_fp8, n,
                                      2, &worst, &checked);
            printf("[SoftmaxRt] FP8  beats=%u cycles: max=%u exp+sum=%u norm=%u | cold=%u warm=%u\n",
                   beats, c1, c2, c3, cold, warm);
            printf("[SoftmaxRt] FP8  significant=%u/%u worst e5m2 ULP=%u : %s\n",
                   checked, n, worst, mism == 0 ? "PASS (<=2 e5m2)" : "FAIL");
            if (mism) err++;
        }

        if (!err)
            printf("[SoftmaxRt] ALL PASS — one netlist set, softmax at FP16 & FP8 by runtime fmt\n");
    }
    return err != 0;
}
