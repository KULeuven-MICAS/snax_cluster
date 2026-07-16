// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// RUNTIME-PRECISION xDMA RoPE (Llama3 interleaved adjacent-pair / complex-
// rotation convention). Pair k = (x[2k], x[2k+1]) rotates by angle theta_k:
//   out[2k]   = x[2k]*cos_k - x[2k+1]*sin_k
//   out[2k+1] = x[2k]*sin_k + x[2k+1]*cos_k
// run end-to-end at FP16 AND FP8 on the SAME xDMA netlist, the precision chosen
// purely by the `fmt` CSR field. Same idea as the softmax-rt demo, but here the
// single runtime-precision reader extension is StreamElementwiseRt (idx 7): one
// netlist does the whole RoPE epilogue at either precision -- no re-elaboration,
// no separate netlists.
//
// ZERO new RTL and ZERO new passes vs the FP16 RoPE app: RoPE is 3 interleaved
// StreamElementwiseRt passes over per-pair-duplicated tables, plus one-time
// adjacent-element swap of x (rotate_half), offloaded to the iDMA:
//   swap  xswap = [x1,x0,x3,x2,...]              (iDMA, element size = fmt width)
//   P1    tmp1  = x     (.) cos_full   MUL       cos_full   = [c0,c0,c1,c1,...]
//   P2    tmp2  = xswap (.) sin_signed MUL       sin_signed = [-s0,+s0,-s1,+s1,]
//   P3    out   = tmp1  (+) tmp2       ADD
// The rotate_half sign lives in the precomputed sin_signed table (free); each
// pass reads its two operands as one INTERLEAVED src stream (AGU inner dim
// count=2 striding operand0 -> operand1 by one row), so buffer adjacency alone
// encodes which two buffers are the operand pair.
//
// LANE DATAFLOW: a 512-bit (64-byte) beat carries 32 FP16 elements OR 64 FP8
// elements, so for a row of N values beats = N/32 (FP16) or N/64 (FP8). At FP8
// the SAME row is processed in HALF the beats -- that is the runtime-precision
// bandwidth win, and the rotate_half staging tracks the precision too (2-byte
// halfword swap at FP16, 1-byte byte swap at FP8).

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMELEMENTWISERT)
#error \
    "Regenerate the XDMA CSR map: this app needs StreamElementwiseRt (cfg/snax_xdma_cluster.hjson)."
#endif

#define XDMA_BEAT_BYTES 64

// fmt CSR field (StreamElementwiseRt csr0[17:16])
#define FMT_FP16 0u
#define FMT_FP8 2u
// StreamElementwiseRt op CSR (csr1): fused-FMA op, 0=MUL (acc*x), 1=ADD (acc+x)
#define EW_MUL 0u
#define EW_ADD 1u

// FP16 bits -> monotonic ordering key (RoPE output is SIGNED): adjacent FP16
// values map to adjacent keys, so |key(a)-key(b)| is the FP16-ULP distance even
// across zero.
static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

// e5m2 bits -> monotonic ordering key (signed): same fold at 8-bit width.
static inline uint32_t fp8_mono(uint8_t b) {
    uint32_t mag = b & 0x7Fu;
    return (b & 0x80u) ? (0x80u - mag) : (0x80u + mag);
}

static uint32_t retask_and_run(void* src, void* dst, uint32_t dst_bound0) {
    if (snax_xdma_retask_1d(src, dst, dst_bound0) != 0) return 0xFFFFFFFFu;
    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    return snax_xdma_last_task_cycle();
}

// One RoPE chain at the given runtime precision. All buffers are TCDM; each
// pass's operand pair is the two buffers row_bytes apart (encoded by the shared
// interleaved [2,beats] src shape). Writes out_buf (beats beats of fmt result).
// Returns the three per-task cycle counts.
static int run_chain_rt(uint32_t fmt, uint32_t beats, uint8_t* x_in,
                        uint8_t* xswap, uint8_t* tmp1, uint8_t* tmp2,
                        uint8_t* out_buf, uint32_t* c1, uint32_t* c2,
                        uint32_t* c3) {
    int ok = 1;
    uint32_t csr_mul[2] = {2u /*operandCount*/ | (fmt << 16), EW_MUL};
    uint32_t csr_add[2] = {2u /*operandCount*/ | (fmt << 16), EW_ADD};

    // P1: tmp1 = x (.) cos_full. Interleaved pair = {x_in, cos_in=x_in+row}.
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISERT, csr_mul) ==
           0);
    *c1 = retask_and_run(x_in, tmp1, beats);
    snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISERT);

    // P2: tmp2 = xswap (.) sin_signed. Pair = {xswap, sin_in=xswap+row}.
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISERT, csr_mul) ==
           0);
    *c2 = retask_and_run(xswap, tmp2, beats);
    snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISERT);

    // P3: out = tmp1 (+) tmp2. Pair = {tmp1, tmp2=tmp1+row}.
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISERT, csr_add) ==
           0);
    *c3 = retask_and_run(tmp1, out_buf, beats);
    snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISERT);

    return ok;
}

// Program the shared interleaved 2D[2,beats]-src / 1D[beats]-dst AGU shape once
// for a given beats. The inner src dim (count=2, stride=row_bytes) walks
// operand0 -> operand1; retask_1d then only re-points the bases per pass.
static int program_shape(uint8_t* x_in, uint8_t* tmp1, uint32_t beats) {
    uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
    uint32_t src_str[2] = {row_bytes, XDMA_BEAT_BYTES};
    uint32_t src_bnd[2] = {2, beats};
    uint32_t dst_str[1] = {XDMA_BEAT_BYTES};
    uint32_t dst_bnd[1] = {beats};
    return (snax_xdma_memcpy_nd_fast(x_in, tmp1, 8, 8, 2, src_str, src_bnd, 1,
                                     dst_str, dst_bnd, 0xFFFFFFFF, 0xFFFFFFFF,
                                     0xFFFFFFFF) == 0);
}

// FP16-ULP integer compare over signed RoPE outputs (monotonic key handles the
// sign); skip the negligible subnormal/zero tail the HW flushes. Returns
// mismatches (>tol).
static uint32_t check_fp16(uint16_t* out, const uint16_t* gold, uint32_t n,
                           uint32_t tol, uint32_t* worst, uint32_t* checked) {
    uint32_t mism = 0;
    *worst = 0;
    *checked = 0;
    for (uint32_t i = 0; i < n; i++) {
        if ((gold[i] & 0x7C00u) == 0) continue;  // skip tiny golden
        (*checked)++;
        uint32_t o = fp16_mono(out[i]), g = fp16_mono(gold[i]);
        uint32_t ulp = (o > g) ? (o - g) : (g - o);
        if (ulp > *worst) *worst = ulp;
        if (ulp > tol) {
            if (mism < 6)
                printf("[RoPERt] fp16 mism[%u]: got %04x gold %04x (%u ulp)\n",
                       i, out[i], gold[i], ulp);
            mism++;
        }
    }
    return mism;
}

// e5m2-code integer compare over signed RoPE outputs (monotonic key); skip the
// subnormal/zero tail (exp field == 0).
static uint32_t check_fp8(uint8_t* out, const uint8_t* gold, uint32_t n,
                          uint32_t tol, uint32_t* worst, uint32_t* checked) {
    uint32_t mism = 0;
    *worst = 0;
    *checked = 0;
    for (uint32_t i = 0; i < n; i++) {
        if ((gold[i] & 0x7Cu) == 0) continue;  // skip tiny golden
        (*checked)++;
        uint32_t o = fp8_mono(out[i]), g = fp8_mono(gold[i]);
        uint32_t ulp = (o > g) ? (o - g) : (g - o);
        if (ulp > *worst) *worst = ulp;
        if (ulp > tol) {
            if (mism < 6)
                printf("[RoPERt] fp8 mism[%u]: got %02x gold %02x (%u ulp)\n",
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
        uint32_t n = rope_n;
        printf("[RoPERt] N=%u : one netlist, FP16 then FP8 by fmt CSR\n", n);

        // ---------- FP16 (32 elems/beat) ----------
        {
            uint32_t beats = rope_beats_fp16;
            uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
            // Layout: each pass's two operands are adjacent (row_bytes apart) so
            // the interleave stride = row_bytes.
            uint8_t* x_in = (uint8_t*)base;
            uint8_t* cos_in = x_in + row_bytes;    // cos_full (P1 operand 1)
            uint8_t* xswap = cos_in + row_bytes;   // rotate_half of x (P2 op 0)
            uint8_t* sin_in = xswap + row_bytes;   // sin_signed (P2 operand 1)
            uint8_t* tmp1 = sin_in + row_bytes;    // x (.) cos     (P3 operand 0)
            uint8_t* tmp2 = tmp1 + row_bytes;      // xswap (.) sin (P3 operand 1)
            uint8_t* out_buf = tmp2 + row_bytes;   // RoPE output (FP16)

            snrt_dma_start_1d(x_in, rope_x_fp16, row_bytes);
            snrt_dma_start_1d(cos_in, rope_cos_fp16, row_bytes);
            snrt_dma_start_1d(sin_in, rope_sin_fp16, row_bytes);
            snrt_dma_wait_all();

            // One-time rotate_half staging (FP16: 2-byte elems, 4-byte pairs):
            // xswap[2k]=x[2k+1], xswap[2k+1]=x[2k] as two strided halfword
            // copies (odd->even, even->odd). Measured once, reported separately.
            uint32_t sw = 0;
            {
                uint32_t ts0 = snrt_mcycle();
                uint32_t pairs = n / 2;
                snrt_dma_start_2d(xswap, x_in + 2, 2, 4, 4, pairs);
                snrt_dma_start_2d(xswap + 2, x_in, 2, 4, 4, pairs);
                snrt_dma_wait_all();
                sw = snrt_mcycle() - ts0;
            }

            uint32_t c1 = 0, c2 = 0, c3 = 0, cold = 0, warm = 0;
            int ok = 1;
            for (int iter = 0; iter < 2; iter++) {
                uint32_t t0 = snrt_mcycle();
                if (iter == 0) ok &= program_shape(x_in, tmp1, beats);
                ok &= run_chain_rt(FMT_FP16, beats, x_in, xswap, tmp1, tmp2,
                                   out_buf, &c1, &c2, &c3);
                uint32_t t1 = snrt_mcycle();
                if (iter == 0)
                    cold = t1 - t0;
                else
                    warm = t1 - t0;
            }
            if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu ||
                c3 == 0xFFFFFFFFu) {
                printf("[RoPERt] FP16 task setup failed\n");
                return 1;
            }
            uint32_t worst, checked;
            uint32_t mism = check_fp16((uint16_t*)out_buf, rope_golden_fp16, n,
                                       4, &worst, &checked);
            printf("[RoPERt] FP16 beats=%u cycles: swap=%u p1=%u p2=%u p3=%u | cold=%u warm=%u\n",
                   beats, sw, c1, c2, c3, cold, warm);
            printf("[RoPERt] FP16 significant=%u/%u worst ULP=%u : %s\n",
                   checked, n, worst, mism == 0 ? "PASS (<=4 ULP)" : "FAIL");
            if (mism) err++;
        }

        // ---------- FP8 e5m2 (64 elems/beat) : SAME netlist, fmt=FP8 ----------
        {
            uint32_t beats = rope_beats_fp8;
            uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
            uint8_t* x_in = (uint8_t*)base;
            uint8_t* cos_in = x_in + row_bytes;
            uint8_t* xswap = cos_in + row_bytes;
            uint8_t* sin_in = xswap + row_bytes;
            uint8_t* tmp1 = sin_in + row_bytes;
            uint8_t* tmp2 = tmp1 + row_bytes;
            uint8_t* out_buf = tmp2 + row_bytes;

            snrt_dma_start_1d(x_in, rope_x_fp8, row_bytes);
            snrt_dma_start_1d(cos_in, rope_cos_fp8, row_bytes);
            snrt_dma_start_1d(sin_in, rope_sin_fp8, row_bytes);
            snrt_dma_wait_all();

            // One-time rotate_half staging (FP8: 1-byte elems, 2-byte pairs):
            // xswap[2k]=x[2k+1], xswap[2k+1]=x[2k] as two strided byte copies.
            uint32_t sw = 0;
            {
                uint32_t ts0 = snrt_mcycle();
                uint32_t pairs = n / 2;
                snrt_dma_start_2d(xswap, x_in + 1, 1, 2, 2, pairs);
                snrt_dma_start_2d(xswap + 1, x_in, 1, 2, 2, pairs);
                snrt_dma_wait_all();
                sw = snrt_mcycle() - ts0;
            }

            uint32_t c1 = 0, c2 = 0, c3 = 0, cold = 0, warm = 0;
            int ok = 1;
            for (int iter = 0; iter < 2; iter++) {
                uint32_t t0 = snrt_mcycle();
                if (iter == 0) ok &= program_shape(x_in, tmp1, beats);
                ok &= run_chain_rt(FMT_FP8, beats, x_in, xswap, tmp1, tmp2,
                                   out_buf, &c1, &c2, &c3);
                uint32_t t1 = snrt_mcycle();
                if (iter == 0)
                    cold = t1 - t0;
                else
                    warm = t1 - t0;
            }
            if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu ||
                c3 == 0xFFFFFFFFu) {
                printf("[RoPERt] FP8 task setup failed\n");
                return 1;
            }
            uint32_t worst, checked;
            uint32_t mism = check_fp8((uint8_t*)out_buf, rope_golden_fp8, n, 3,
                                      &worst, &checked);
            printf("[RoPERt] FP8  beats=%u cycles: swap=%u p1=%u p2=%u p3=%u | cold=%u warm=%u\n",
                   beats, sw, c1, c2, c3, cold, warm);
            printf("[RoPERt] FP8  significant=%u/%u worst e5m2 ULP=%u : %s\n",
                   checked, n, worst, mism == 0 ? "PASS (<=3 e5m2)" : "FAIL");
            if (mism) err++;
        }

        if (!err)
            printf("[RoPERt] ALL PASS — one netlist, RoPE at FP16 & FP8 by runtime fmt\n");
    }
    return err != 0;
}
