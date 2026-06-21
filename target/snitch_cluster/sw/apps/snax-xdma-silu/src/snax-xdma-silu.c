// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 SiLU: out = silu(x) = x * sigmoid(x). ONE xDMA task, L1<->L1. The whole op is offloaded to
// the xDMA reader SIMD extension; the DM core (rv32ima, no FPU) only programs the descriptor. Because
// silu is unary, it needs no scalar glue at all (unlike rmsnorm/softmax which form a reduction scalar
// between passes) -- a single StreamMap pass with the SILU activation:
//
//   T1  silu : StreamMap(a=1, b=0, func=SILU)   x -> out   (out = silu(1*x + 0) = x*sigmoid(x))
//
// SILU routes the per-lane affine result through FpSilu (a 512-entry sigmoid LUT * x); see FpSilu.scala.
// FP16 transport, FP32-internal. Significant outputs match the FP64 golden to <= a few FP16 ULP.
//
// Performance (FP16, measured under vsim, L1<->L1). warm = steady-state (AGU shape persists); cold = first
// call (one-time shape setup + icache warm). SiLU is unary, so it is FULLY offloaded -- there is no host
// step. host = a single CVA6+Ara core running the full FP32 silu (op-LUT baseline). Significant outputs
// match the FP64 golden to <=1 FP16 ULP at every N.
//
//   N      beats   warm    cold    host       warm speedup
//   ----   -----   -----   -----   -------    ------------
//   64     2       313     1,554   2,258      7.2x
//   256    8       341     1,590   8,264      24.2x
//   1024   32      483     1,733   32,718     67.7x
//   4096   128     1,059   2,309   130,686    123.4x

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMMAP)
#error "Regenerate the XDMA CSR map with StreamMap (and func=SILU in the cfg)."
#endif

#define XDMA_BEAT_BYTES 64
#define ACT_SILU 2u  // StreamMap func CSR bits[1:0]: 0=LINEAR, 1=EXP, 2=SILU

// FP16 bits -> monotonic ordering key (handles signed outputs): adjacent FP16 values map to adjacent
// keys, so |key(a)-key(b)| is the FP16-ULP distance even across zero.
static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

// Minimal re-task launch: reuse the persisted temporal shape, rewrite only addresses + dst bound.
static uint32_t retask_and_run(void* src, void* dst, uint32_t dst_bound0) {
    if (snax_xdma_retask_1d(src, dst, dst_bound0) != 0) return 0xFFFFFFFFu;
    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    return snax_xdma_last_task_cycle();
}

int main() {
    int err = 0;
    if (snrt_is_dm_core()) {
        uint32_t base = snrt_cluster_base_addrl();
        uint32_t beats = silu_beats;
        uint32_t row_bytes = beats * XDMA_BEAT_BYTES;

        uint8_t* x_in    = (uint8_t*)base;
        uint8_t* out_buf = x_in + row_bytes;  // beats

        printf("[Silu] N=%u beats=%u\n", silu_n, beats);

        snrt_dma_start_1d(x_in, silu_input, row_bytes);
        snrt_dma_wait_all();

        // The AGU shape (2D [beats,1] src reading x_in, 1D dst of `beats`) is identical across rows, so it
        // is programmed ONCE (full _fast); the task then needs only a 5-write retask (dst addr + writer
        // beat count; the src addr x_in is common). xDMA regs reset to 0, so unused multicast dsts need no
        // zeroing.
        uint32_t str_beat[1]   = {XDMA_BEAT_BYTES};
        uint32_t src_str_2d[2] = {XDMA_BEAT_BYTES, beats * XDMA_BEAT_BYTES};
        uint32_t src_bnd_2d[2] = {beats, 1};
        uint32_t bnd_1[1]      = {1};

        uint32_t c1 = 0, lat_cold = 0, lat_warm = 0;
        int ok = 1;
        // Run twice: iter 0 = cold (one-time AGU shape setup + icache warming); iter 1 = warm / steady-state.
        for (int iter = 0; iter < 2; iter++) {
            uint32_t t0 = snrt_mcycle();
            if (iter == 0)
                ok &= (snax_xdma_memcpy_nd_fast(x_in, out_buf, 8, 8, 2, src_str_2d,
                                                src_bnd_2d, 1, str_beat, bnd_1,
                                                0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);

            // T1: out = silu(x). a=1.0, b=0, func=SILU.
            uint32_t csr_silu[3] = {0x3F800000u /*1.0f*/, 0u, ACT_SILU};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_silu) == 0);
            c1 = retask_and_run(x_in, out_buf, beats);
            snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);

            uint32_t t1 = snrt_mcycle();
            if (iter == 0) lat_cold = t1 - t0; else lat_warm = t1 - t0;
        }

        if (!ok || c1 == 0xFFFFFFFFu) {
            printf("[Silu] xDMA task setup failed\n");
            return 1;
        }
        printf("[Silu] cycles: silu=%u\n", c1);
        printf("[Silu] full latency: cold=%u warm=%u cycles\n", lat_cold, lat_warm);

        // Integer FP16-ULP check on the significant (signed) outputs; skip the negligible subnormal/zero
        // golden (FTZ tail, e.g. silu near 0). The DM core has no FPU, so the compare stays integer.
        uint16_t* out_h = (uint16_t*)out_buf;
        uint32_t mism = 0, worst = 0, checked = 0;
        for (uint32_t i = 0; i < silu_n; i++) {
            if ((silu_golden[i] & 0x7C00u) == 0) continue;  // skip subnormal/zero golden
            checked++;
            uint32_t o = fp16_mono(out_h[i]), g = fp16_mono(silu_golden[i]);
            uint32_t ulp = (o > g) ? (o - g) : (g - o);
            if (ulp > worst) worst = ulp;
            if (ulp > 4) {
                if (mism < 6)
                    printf("[Silu] mismatch[%u]: got %04x golden %04x (%u ulp)\n",
                           i, out_h[i], silu_golden[i], ulp);
                mism++;
            }
        }
        printf("[Silu] significant=%u/%u worst FP16 ULP=%u\n", checked, silu_n, worst);
        if (mism == 0) {
            printf("[Silu] PASS (%u significant elements, <=4 ULP)\n", checked);
        } else {
            printf("[Silu] FAIL (%u/%u significant beyond 4 ULP)\n", mism, checked);
            err++;
        }
    }
    return err != 0;
}
