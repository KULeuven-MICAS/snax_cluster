// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 SwiGLU: out = silu(gate) (.) up (gate, up are two prior GEMMs). FULLY ON THE xDMA -- both the
// unary silu(gate) and the two-tensor multiply by `up` are offloaded; the host keeps nothing for swiglu.
//
//   T1  silu(gate) : StreamMap(a=1, b=0, func=SILU)         gate -> sg        (xDMA)
//   T2  (.)up      : StreamElementwise(MUL, operandCount=2) {sg, up} -> out   (xDMA)
//
// T2 reads sg and up as one INTERLEAVED stream (AGU inner dim count=2 striding sg_buf -> up_in), like the
// elementwise-add app. T1/T2 use different AGU shapes, so each does its own memcpy_nd_fast. FP16 transport,
// FP32-internal; out matches the FP64 golden to <=2 FP16 ULP.
//
// Performance (FP16, vsim, L1<->L1). host(full) = a single CVA6+Ara core, full FP32 silu_mul (op-LUT).
//
//   N      beats   silu  mul    xdma_total   warm    cold    host(full)   warm speedup
//   ----   -----   ----  -----  ----------   -----   -----   ----------   ------------
//   64     2       22    32     54           2,107   2,610   2,308        1.1x
//   256    8       58    98     156          2,207   2,714   8,662        3.9x
//   1024   32      202   362    564          2,590   3,089   34,493       13.3x
//   4096   128     778   1,418  2,196        4,222   4,721   137,799      32.6x
//
// The mul datapath (32..1,418 cc) replaces the host's O(n) (.)up, so the speedup climbs with n instead of
// collapsing. warm ~= 2,050 (two AGU setups, no retask) + xdma_total; the 2-setup overhead amortizes in
// batched inference (one silu over all [S,F], then one mul).

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMMAP) || !defined(READER_EXT_STREAMELEMENTWISE)
#error "Regenerate the XDMA CSR map with StreamMap (func=SILU) and StreamElementwise (op=MUL)."
#endif

#define XDMA_BEAT_BYTES 64
#define ACT_SILU 2u  // StreamMap func CSR bits[1:0]: 0=LINEAR, 1=EXP, 2=SILU
#define EW_MUL 0u     // StreamElementwise op CSR: 0=MUL, 1=ADD

// FP16 bits -> monotonic ordering key (handles signed outputs): adjacent FP16 values map to adjacent
// keys, so |key(a)-key(b)| is the FP16-ULP distance even across zero.
static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

int main() {
    int err = 0;
    if (snrt_is_dm_core()) {
        uint32_t base = snrt_cluster_base_addrl();
        uint32_t beats = swiglu_beats;
        uint32_t row_bytes = beats * XDMA_BEAT_BYTES;

        // Layout: sg_buf and up_in are adjacent (row_bytes apart) so T2's interleave stride = row_bytes.
        uint8_t* gate_in = (uint8_t*)base;
        uint8_t* sg_buf  = gate_in + row_bytes;        // silu(gate) (T1 output, T2 operand 0)
        uint8_t* up_in   = sg_buf + row_bytes;         // up (T2 operand 1); up_in - sg_buf = row_bytes
        uint8_t* out_buf = up_in + row_bytes;          // swiglu output

        printf("[SwiGLU] N=%u beats=%u\n", swiglu_n, beats);

        snrt_dma_start_1d(gate_in, swiglu_gate, row_bytes);
        snrt_dma_start_1d(up_in, swiglu_up, row_bytes);
        snrt_dma_wait_all();

        uint32_t dst_str[1]    = {XDMA_BEAT_BYTES};
        uint32_t dst_bnd[1]    = {beats};
        // T1 src: 2D [beats,1] reading gate_in beat-by-beat.
        uint32_t t1_src_str[2] = {XDMA_BEAT_BYTES, beats * XDMA_BEAT_BYTES};
        uint32_t t1_src_bnd[2] = {beats, 1};
        // T2 src: interleaved 2D [2,beats] -> {sg[beat], up[beat]} pairs (inner dim strides sg_buf->up_in).
        uint32_t t2_src_str[2] = {row_bytes, XDMA_BEAT_BYTES};
        uint32_t t2_src_bnd[2] = {2, beats};

        uint32_t c1 = 0, c2 = 0, lat_cold = 0, lat_warm = 0;
        int ok = 1;
        // Run twice: iter 0 = cold (first call, icache cold); iter 1 = warm / steady-state.
        for (int iter = 0; iter < 2; iter++) {
            uint32_t t0 = snrt_mcycle();

            // T1: sg = silu(gate).
            uint32_t csr_silu[3] = {0x3F800000u /*1.0f*/, 0u, ACT_SILU};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_silu) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(gate_in, sg_buf, 8, 8, 2, t1_src_str, t1_src_bnd,
                                            1, dst_str, dst_bnd,
                                            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            {
                int task_id = snax_xdma_start();
                snax_xdma_local_wait(task_id);
                c1 = snax_xdma_last_task_cycle();
            }
            snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);

            // T2: out = sg (.) up. StreamElementwise(MUL), operandCount=2 over the interleaved {sg,up} stream.
            uint32_t csr_mul[2] = {2u /*operandCount*/, EW_MUL};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_mul) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(sg_buf, out_buf, 8, 8, 2, t2_src_str, t2_src_bnd,
                                            1, dst_str, dst_bnd,
                                            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            {
                int task_id = snax_xdma_start();
                snax_xdma_local_wait(task_id);
                c2 = snax_xdma_last_task_cycle();
            }
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);

            uint32_t t1 = snrt_mcycle();
            if (iter == 0) lat_cold = t1 - t0; else lat_warm = t1 - t0;
        }

        if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu) {
            printf("[SwiGLU] xDMA task setup failed\n");
            return 1;
        }
        printf("[SwiGLU] cycles: silu=%u mul=%u xdma_total=%u\n", c1, c2, c1 + c2);
        printf("[SwiGLU] full latency: cold=%u warm=%u cycles\n", lat_cold, lat_warm);

        // Verify the full swiglu out == silu(gate)*up. Integer FP16-ULP check on significant (signed)
        // outputs; skip the negligible subnormal/zero golden (FTZ tail). DM core has no FPU -> integer compare.
        uint16_t* out_h = (uint16_t*)out_buf;
        uint32_t mism = 0, worst = 0, checked = 0;
        for (uint32_t i = 0; i < swiglu_n; i++) {
            if ((swiglu_golden[i] & 0x7C00u) == 0) continue;  // skip subnormal/zero golden
            checked++;
            uint32_t o = fp16_mono(out_h[i]), g = fp16_mono(swiglu_golden[i]);
            uint32_t ulp = (o > g) ? (o - g) : (g - o);
            if (ulp > worst) worst = ulp;
            if (ulp > 4) {
                if (mism < 6)
                    printf("[SwiGLU] mismatch[%u]: got %04x golden %04x (%u ulp)\n",
                           i, out_h[i], swiglu_golden[i], ulp);
                mism++;
            }
        }
        printf("[SwiGLU] significant=%u/%u worst FP16 ULP=%u\n", checked, swiglu_n, worst);
        if (mism == 0) {
            printf("[SwiGLU] PASS (%u significant elements, <=4 ULP)\n", checked);
        } else {
            printf("[SwiGLU] FAIL (%u/%u significant beyond 4 ULP)\n", mism, checked);
            err++;
        }
    }
    return err != 0;
}
