// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// RUNTIME-PRECISION xDMA SwiGLU: out = silu(gate) (.) up, run end-to-end at FP16
// AND FP8 on the SAME xDMA netlists, the precision chosen purely by the `fmt`
// CSR field. This is the runtime-precision counterpart of snax-xdma-swiglu:
// one set of reader extensions (StreamMapRt = idx 6 for silu, StreamElementwiseRt
// = idx 7 for the two-tensor multiply) computes the whole SwiGLU at either
// precision -- no re-elaboration, no separate netlists, no scalar glue.
//
//   T1 silu : StreamMapRt(a=1, b=0, SILU)                 gate -> sg (xDMA)
//   T2 (.)up: StreamElementwiseRt(MUL, operandCount=2)  {sg, up} -> out (xDMA)
//
// T2 reads sg and up as one INTERLEAVED stream (AGU inner dim count=2 striding
// sg_buf -> up_in), exactly like the FP16 baseline swiglu. T1 and T2 use
// DIFFERENT src AGU shapes, so each pass programs its own memcpy_nd_fast (the
// single-shape retask trick does not apply here).
//
// LANE DATAFLOW: a 512-bit (64-byte) beat carries 32 FP16 elements OR 64 FP8
// elements, so for a row of N values beats = N/32 (FP16) or N/64 (FP8). At FP8
// the SAME row is processed in HALF the beats -- the runtime-precision bandwidth
// win, visible directly in the per-task cycle counts.
//   T1 StreamMapRt(SILU)          1 beat  -> 1 beat  (clean 512b->512b)
//   T2 StreamElementwiseRt(MUL)   2 beats -> 1 beat  (2:1, interleaved {sg,up})
//     sg [ s0 .. sK ]  up [ u0 .. uK ]  ->  out [ s0*u0 .. sK*uK ]

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMMAPRT) || !defined(READER_EXT_STREAMELEMENTWISERT)
#error \
    "Regenerate the XDMA CSR map: this app needs StreamMapRt + StreamElementwiseRt (cfg/snax_xdma_cluster.hjson)."
#endif

#define XDMA_BEAT_BYTES 64

// fmt CSR field (StreamMapRt csr2[3:2], StreamElementwiseRt csr0[17:16])
#define FMT_FP16 0u
#define FMT_FP8 2u
// StreamMapRt func CSR (csr2[1:0]): 0=LINEAR, 1=EXP, 2=SILU
#define ACT_SILU 2u
// StreamElementwiseRt op CSR (csr1): 0=MUL (acc*x), 1=ADD (acc+x)
#define EW_MUL 0u

// FP16 bits -> monotonic sign-magnitude key: adjacent FP16 values map to
// adjacent keys, so |key(a)-key(b)| is the FP16-ULP distance even across zero
// (SwiGLU outputs are SIGNED). DM core has no FPU -> integer compare.
static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

// e5m2 byte -> monotonic sign-magnitude key (same trick, signed FP8 outputs).
static inline uint32_t fp8_mono(uint8_t b) {
    uint32_t mag = b & 0x7Fu;
    return (b & 0x80u) ? (0x80u - mag) : (0x80u + mag);
}

// One SwiGLU chain at the given runtime precision. All buffers are TCDM. Writes
// out_buf (beats beats of fmt result). Returns the two per-task cycle counts.
static int run_chain_rt(uint32_t fmt, uint32_t beats, uint8_t* gate_in,
                        uint8_t* sg_buf, uint8_t* up_in, uint8_t* out_buf,
                        uint32_t* c1, uint32_t* c2) {
    int ok = 1;
    uint32_t row_bytes = beats * XDMA_BEAT_BYTES;

    uint32_t dst_str[1] = {XDMA_BEAT_BYTES};
    uint32_t dst_bnd[1] = {beats};
    // T1 src: 2D [beats,1] reading gate_in beat-by-beat.
    uint32_t t1_src_str[2] = {XDMA_BEAT_BYTES, beats * XDMA_BEAT_BYTES};
    uint32_t t1_src_bnd[2] = {beats, 1};
    // T2 src: interleaved 2D [2,beats] -> {sg[beat], up[beat]} pairs (inner dim
    // strides sg_buf -> up_in, so up_in - sg_buf = row_bytes).
    uint32_t t2_src_str[2] = {row_bytes, XDMA_BEAT_BYTES};
    uint32_t t2_src_bnd[2] = {2, beats};

    // T1: sg = silu(gate). StreamMapRt(SILU) at runtime precision fmt.
    uint32_t csr_silu[3] = {0x3F800000u /*a=1.0f*/, 0u /*b=0*/,
                            ACT_SILU | (fmt << 2)};
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAPRT, csr_silu) == 0);
    ok &= (snax_xdma_memcpy_nd_fast(gate_in, sg_buf, 8, 8, 2, t1_src_str,
                                    t1_src_bnd, 1, dst_str, dst_bnd, 0xFFFFFFFF,
                                    0xFFFFFFFF, 0xFFFFFFFF) == 0);
    {
        int task_id = snax_xdma_start();
        snax_xdma_local_wait(task_id);
        *c1 = snax_xdma_last_task_cycle();
    }
    snax_xdma_disable_src_ext(READER_EXT_STREAMMAPRT);

    // T2: out = sg (.) up. StreamElementwiseRt(MUL), operandCount=2 over the
    // interleaved {sg,up} stream (StreamMapRt idx 6 stays disabled = bypass).
    uint32_t csr_mul[2] = {2u /*operandCount*/ | (fmt << 16), EW_MUL};
    ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISERT, csr_mul) ==
           0);
    ok &= (snax_xdma_memcpy_nd_fast(sg_buf, out_buf, 8, 8, 2, t2_src_str,
                                    t2_src_bnd, 1, dst_str, dst_bnd, 0xFFFFFFFF,
                                    0xFFFFFFFF, 0xFFFFFFFF) == 0);
    {
        int task_id = snax_xdma_start();
        snax_xdma_local_wait(task_id);
        *c2 = snax_xdma_last_task_cycle();
    }
    snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISERT);

    return ok;
}

// FP16-ULP compare on the signed SwiGLU output (mono key across zero); skip the
// negligible subnormal/zero golden the HW flushes. Returns mismatches (>tol).
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
                printf("[SwiGLURt] fp16 mism[%u]: got %04x gold %04x (%u ulp)\n",
                       i, out[i], gold[i], ulp);
            mism++;
        }
    }
    return mism;
}

// e5m2-ULP compare on the signed SwiGLU output (mono key across zero); skip the
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
                printf("[SwiGLURt] fp8 mism[%u]: got %02x gold %02x (%u ulp)\n",
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
        uint32_t n = swiglu_n;
        printf("[SwiGLURt] N=%u : one set of netlists, FP16 then FP8 by fmt CSR\n", n);

        // ---------- FP16 (32 elems/beat) ----------
        {
            uint32_t beats = swiglu_beats_fp16;
            uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
            // sg_buf and up_in are adjacent (row_bytes apart) so T2's interleave
            // stride = row_bytes.
            uint8_t* gate_in = (uint8_t*)base;
            uint8_t* sg_buf = gate_in + row_bytes;   // silu(gate): T1 out, T2 op0
            uint8_t* up_in = sg_buf + row_bytes;     // up: T2 op1 (up_in-sg_buf=row_bytes)
            uint8_t* out_buf = up_in + row_bytes;    // swiglu output

            snrt_dma_start_1d(gate_in, swiglu_gate_fp16, row_bytes);
            snrt_dma_start_1d(up_in, swiglu_up_fp16, row_bytes);
            snrt_dma_wait_all();

            uint32_t c1 = 0, c2 = 0, cold = 0, warm = 0;
            int ok = 1;
            for (int iter = 0; iter < 2; iter++) {
                uint32_t t0 = snrt_mcycle();
                ok &= run_chain_rt(FMT_FP16, beats, gate_in, sg_buf, up_in,
                                   out_buf, &c1, &c2);
                uint32_t t1 = snrt_mcycle();
                if (iter == 0) cold = t1 - t0; else warm = t1 - t0;
            }
            if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu) {
                printf("[SwiGLURt] FP16 task setup failed\n");
                return 1;
            }
            uint32_t worst, checked;
            uint32_t mism = check_fp16((uint16_t*)out_buf, swiglu_golden_fp16, n,
                                       4, &worst, &checked);
            printf("[SwiGLURt] FP16 beats=%u cycles: silu=%u mul=%u | cold=%u warm=%u\n",
                   beats, c1, c2, cold, warm);
            printf("[SwiGLURt] FP16 significant=%u/%u worst ULP=%u : %s\n",
                   checked, n, worst, mism == 0 ? "PASS (<=4 ULP)" : "FAIL");
            if (mism) err++;
        }

        // ---------- FP8 e5m2 (64 elems/beat) : SAME netlists, fmt=FP8 ----------
        {
            uint32_t beats = swiglu_beats_fp8;
            uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
            uint8_t* gate_in = (uint8_t*)base;
            uint8_t* sg_buf = gate_in + row_bytes;
            uint8_t* up_in = sg_buf + row_bytes;
            uint8_t* out_buf = up_in + row_bytes;

            snrt_dma_start_1d(gate_in, swiglu_gate_fp8, row_bytes);
            snrt_dma_start_1d(up_in, swiglu_up_fp8, row_bytes);
            snrt_dma_wait_all();

            uint32_t c1 = 0, c2 = 0, cold = 0, warm = 0;
            int ok = 1;
            for (int iter = 0; iter < 2; iter++) {
                uint32_t t0 = snrt_mcycle();
                ok &= run_chain_rt(FMT_FP8, beats, gate_in, sg_buf, up_in,
                                   out_buf, &c1, &c2);
                uint32_t t1 = snrt_mcycle();
                if (iter == 0) cold = t1 - t0; else warm = t1 - t0;
            }
            if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu) {
                printf("[SwiGLURt] FP8 task setup failed\n");
                return 1;
            }
            uint32_t worst, checked;
            uint32_t mism = check_fp8((uint8_t*)out_buf, swiglu_golden_fp8, n, 2,
                                      &worst, &checked);
            printf("[SwiGLURt] FP8  beats=%u cycles: silu=%u mul=%u | cold=%u warm=%u\n",
                   beats, c1, c2, cold, warm);
            printf("[SwiGLURt] FP8  significant=%u/%u worst e5m2 ULP=%u : %s\n",
                   checked, n, worst, mism == 0 ? "PASS (<=2 e5m2)" : "FAIL");
            if (mism) err++;
        }

        if (!err)
            printf("[SwiGLURt] ALL PASS — one netlist set, SwiGLU at FP16 & FP8 by runtime fmt\n");
    }
    return err != 0;
}
