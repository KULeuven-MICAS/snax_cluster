// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 SwiGLU: out = silu(gate) (.) up, the SwiGLU MLP activation (gate = x*Wgate, up = x*Wup are
// two prior GEMMs). The *unary* half -- silu(gate) -- is offloaded to the xDMA reader SIMD extension as a
// single StreamMap pass; the *binary* half -- the element-wise multiply by `up` -- is left to the host
// (CVA6, FP-capable), because no current xDMA reader block multiplies two distinct tensor streams. That
// split is deliberate: from the measured host op-LUTs the extra (.)up is only ~5% of the SwiGLU kernel
// (silu 130,686 vs silu_mul 137,799 at n=4096), so FpSilu alone retires ~95% of the host cost without a
// new two-operand datapath block.
//
//   T1  silu(gate) : StreamMap(a=1, b=0, func=SILU)   gate -> sg     (xDMA, L1<->L1)
//   host           : out[i] = sg[i] * up[i]                          (host FP multiply; NOT done here --
//                                                                     the DM core is rv32ima, no FPU)
//
// This cluster app exercises and verifies the offloaded part (sg == silu(gate)). `up` and the full
// swiglu_golden are emitted in data.h for the host's downstream multiply / end-to-end reference.
// FP16 transport, FP32-internal; significant outputs match the FP64 golden to <=1 FP16 ULP at every N.
//
// Performance (FP16, measured under vsim, L1<->L1). "xDMA silu(gate)" warm = the offloaded SILU pass;
// "+host (.)up" adds the host element-wise multiply by up, estimated from the op-LUT delta
// silu_mul_host - silu_host (~1.7 cc/elem). host(full) = a single CVA6+Ara core running the full FP32
// silu_mul (op-LUT baseline). NOTE the (.)up is O(n) on the host, so once silu(gate) is offloaded it
// becomes the bottleneck and caps the speedup -- this is exactly what a future StreamBinary(MUL) would
// remove (it would take (.)up onto the xDMA too).
//
//   N      beats   xDMA silu(gate)   +host (.)up est   host(full)   speedup(+host)
//   ----   -----   --------------    ---------------   ----------   --------------
//   64     2       327               ~377              2,308        6.1x
//   256    8       363               ~761              8,662        11.4x
//   1024   32      483               ~2,258            34,493       15.3x
//   4096   128     1,059             ~8,172            137,799      16.9x

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
        uint32_t beats = swiglu_beats;
        uint32_t row_bytes = beats * XDMA_BEAT_BYTES;

        uint8_t* gate_in = (uint8_t*)base;
        uint8_t* up_in   = gate_in + row_bytes;   // staged for the host's (.)up multiply
        uint8_t* sg_buf  = up_in + row_bytes;     // silu(gate) (xDMA output)

        printf("[SwiGLU] N=%u beats=%u\n", swiglu_n, beats);

        snrt_dma_start_1d(gate_in, swiglu_gate, row_bytes);
        snrt_dma_start_1d(up_in, swiglu_up, row_bytes);
        snrt_dma_wait_all();

        // The AGU shape (2D [beats,1] src reading gate_in, 1D dst of `beats`) is identical across rows, so
        // it is programmed ONCE (full _fast); the task then needs only a 5-write retask.
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
                ok &= (snax_xdma_memcpy_nd_fast(gate_in, sg_buf, 8, 8, 2, src_str_2d,
                                                src_bnd_2d, 1, str_beat, bnd_1,
                                                0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);

            // T1: sg = silu(gate). a=1.0, b=0, func=SILU.
            uint32_t csr_silu[3] = {0x3F800000u /*1.0f*/, 0u, ACT_SILU};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_silu) == 0);
            c1 = retask_and_run(gate_in, sg_buf, beats);
            snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);

            uint32_t t1 = snrt_mcycle();
            if (iter == 0) lat_cold = t1 - t0; else lat_warm = t1 - t0;
        }

        if (!ok || c1 == 0xFFFFFFFFu) {
            printf("[SwiGLU] xDMA task setup failed\n");
            return 1;
        }
        printf("[SwiGLU] cycles: silu_gate=%u  (host adds the (.)up multiply, ~5%% of the kernel)\n", c1);
        printf("[SwiGLU] full latency: cold=%u warm=%u cycles\n", lat_cold, lat_warm);

        // Verify the offloaded part: sg == silu(gate). Integer FP16-ULP check on significant (signed)
        // outputs; skip the negligible subnormal/zero golden (FTZ tail). DM core has no FPU -> integer compare.
        uint16_t* sg_h = (uint16_t*)sg_buf;
        uint32_t mism = 0, worst = 0, checked = 0;
        for (uint32_t i = 0; i < swiglu_n; i++) {
            if ((swiglu_silu_gate_golden[i] & 0x7C00u) == 0) continue;  // skip subnormal/zero golden
            checked++;
            uint32_t o = fp16_mono(sg_h[i]), g = fp16_mono(swiglu_silu_gate_golden[i]);
            uint32_t ulp = (o > g) ? (o - g) : (g - o);
            if (ulp > worst) worst = ulp;
            if (ulp > 4) {
                if (mism < 6)
                    printf("[SwiGLU] silu(gate) mismatch[%u]: got %04x golden %04x (%u ulp)\n",
                           i, sg_h[i], swiglu_silu_gate_golden[i], ulp);
                mism++;
            }
        }
        printf("[SwiGLU] silu(gate) significant=%u/%u worst FP16 ULP=%u\n", checked, swiglu_n, worst);
        if (mism == 0) {
            printf("[SwiGLU] PASS silu(gate) (%u significant elements, <=4 ULP); (.)up is the host's step\n",
                   checked);
        } else {
            printf("[SwiGLU] FAIL silu(gate) (%u/%u significant beyond 4 ULP)\n", mism, checked);
            err++;
        }
    }
    return err != 0;
}
