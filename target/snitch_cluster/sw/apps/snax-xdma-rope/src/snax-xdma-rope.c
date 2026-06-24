// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 RoPE (Llama3, interleaved adjacent-pair / complex-rotation convention). Pair k = (x[2k],
// x[2k+1]) rotates by angle theta_k:
//   out[2k]   = x[2k]*cos_k - x[2k+1]*sin_k
//   out[2k+1] = x[2k]*sin_k + x[2k+1]*cos_k
//
// ZERO new RTL: reuse the existing StreamElementwise extension (MUL/ADD). RoPE is 3 passes over
// per-pair-duplicated tables, plus one adjacent halfword swap of x:
//   swap  xswap = [x1,x0,x3,x2,...]                       (host, integer: 16-bit rotate per word)
//   P1    tmp1  = x     (.) cos_full   StreamElementwise MUL   cos_full   = [c0,c0,c1,c1,...]
//   P2    tmp2  = xswap (.) sin_signed StreamElementwise MUL   sin_signed = [-s0,+s0,-s1,+s1,...]
//   P3    out   = tmp1  (+) tmp2       StreamElementwise ADD
// The rotate_half sign lives in the precomputed sin_signed table (free); the only runtime data move is
// the adjacent swap, cheap because pairs are adjacent. FP16 transport, FP32-internal; out matches the
// FP64 golden to <=4 FP16 ULP. Each pass reads its two operands as one INTERLEAVED stream (AGU inner dim
// count=2 striding operand0 -> operand1), exactly like swiglu's mul pass.
//
// LANE DATAFLOW (FP16 transport = 32 lanes/512-b beat; all 3 passes are StreamElementwise, 2 beats -> 1
// beat, op per lane). RoPE works on adjacent PAIRS (lanes 2k, 2k+1 share angle theta_k), encoded purely
// in how the operand tables are laid out across the lanes -- no special HW:
//   lane:       0     1     2     3   ...          (pair k uses lanes 2k, 2k+1)
//   x        [  x0    x1    x2    x3  ... ]
//   xswap    [  x1    x0    x3    x2  ... ]         adjacent halfword swap (iDMA, one-time)
//   cos_full [  c0    c0    c1    c1  ... ]         each cos_k duplicated across the pair
//   sin_sgn  [ -s0   +s0   -s1   +s1  ... ]         rotate_half sign baked into the table
//   P1 MUL:  tmp1 = x     (.) cos_full -> [ x0*c0    x1*c0    x2*c1  ... ]
//   P2 MUL:  tmp2 = xswap (.) sin_sgn  -> [ -x1*s0   x0*s0   -x3*s1  ... ]
//   P3 ADD:  out  = tmp1  (+) tmp2     -> [ x0*c0-x1*s0   x1*c0+x0*s0  ... ]   (the rotation)
//   Each pass is a clean 32-lane-in/32-lane-out element-wise op (the 2:1 is the interleaved-operand read).
//
// Performance (FP16, vsim, L1<->L1), measured on the area/timing-optimized RTL: pipelined StreamElementwise
// + time-mux computeLanes=2. swap* = one-time rotate_half staging (iDMA, ~2 cyc/elem); p1/p2/p3 = the
// StreamElementwise passes; xdma_total = p1+p2+p3 (pure datapath); warm/cold = the 3-pass offload only
// (swap excluded, comparable to swiglu's warm/cold); per-call = swap+warm.
//
//   N      beats   swap*   p1     p2     p3     xdma_total   warm     cold     per-call(swap+warm)
//   ----   -----   -----   ----   ----   ----   ----------   ------   ------   -------------------
//   64     2       166     89     89     89     267          1,864    2,032    2,030
//   256    8       551     323    323    323    969          2,568    2,736    3,119
//   1024   32      2,090   1,259  1,259  1,259  3,777        5,326    5,532    7,416
//   4096   128     8,235   5,003  5,003  5,003  15,009       16,558   16,764   24,793
//
// worst FP16 ULP vs the 3-pass golden = 0 (N<=1024), 1 (N=4096). The datapath is ~40 cyc/beat/pass (the
// computeLanes=2 time-mux); the ~1.6k fixed warm cost is the 3x CSR orchestration (3 memcpy_nd setups), not
// the math (skill: orchestration is the bottleneck). At large N the swap (the rotate_half tax of the
// zero-HW path) and the time-muxed datapath dominate. Reference:
// swiglu's measured host(full) silu+mul is ~33.6*N cyc (2,308..137,799 for N=64..4096); a host
// RoPE has no transcendental so it is cheaper, hence the offload pays off only at larger N. A fused
// single-pass StreamRoPE (intra-beat adjacent-pair) would remove the swap and 2 of the 3 passes.

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMELEMENTWISE)
#error "Regenerate the XDMA CSR map with StreamElementwise (op=MUL/ADD)."
#endif

#define XDMA_BEAT_BYTES 64
#define EW_MUL 0u  // StreamElementwise op CSR: 0=MUL, 1=ADD
#define EW_ADD 1u

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
        uint32_t beats = rope_beats;
        uint32_t row_bytes = beats * XDMA_BEAT_BYTES;

        // Layout: each pass's two operands are adjacent (row_bytes apart) so the interleave stride =
        // row_bytes. P1 {x,cos}->tmp1, P2 {xswap,sin}->tmp2, P3 {tmp1,tmp2}->out.
        uint8_t* x_in   = (uint8_t*)base;
        uint8_t* cos_in = x_in   + row_bytes;   // cos_full (P1 operand 1)
        uint8_t* xswap  = cos_in + row_bytes;   // adjacent swap of x (P2 operand 0)
        uint8_t* sin_in = xswap  + row_bytes;   // sin_signed (P2 operand 1)
        uint8_t* tmp1   = sin_in + row_bytes;   // x (.) cos       (P3 operand 0)
        uint8_t* tmp2   = tmp1   + row_bytes;   // xswap (.) sin   (P3 operand 1)
        uint8_t* out_buf = tmp2  + row_bytes;   // RoPE output (FP16)

        printf("[RoPE] N=%u beats=%u\n", rope_n, beats);

        snrt_dma_start_1d(x_in, rope_x, row_bytes);
        snrt_dma_start_1d(cos_in, rope_cos, row_bytes);
        snrt_dma_start_1d(sin_in, rope_sin, row_bytes);
        snrt_dma_wait_all();

        // One-time input staging (like swiglu's gate/up load): adjacent halfword swap
        // xswap[2k]=x[2k+1], xswap[2k+1]=x[2k], offloaded to the iDMA as two strided 2-byte copies
        // (odd->even, even->odd halfwords). The DM core stays free; measured once, reported separately.
        uint32_t sw = 0;
        {
            uint32_t ts0 = snrt_mcycle();
            uint32_t pairs = beats * 16;  // N/2
            snrt_dma_start_2d(xswap, x_in + 2, 2, 4, 4, pairs);   // xswap[2k]   = x[2k+1]
            snrt_dma_start_2d(xswap + 2, x_in, 2, 4, 4, pairs);   // xswap[2k+1] = x[2k]
            snrt_dma_wait_all();
            sw = snrt_mcycle() - ts0;
        }

        // All passes share this interleaved [2,beats] src shape and 1D dst shape; only the bases change.
        uint32_t src_str[2] = {row_bytes, XDMA_BEAT_BYTES};  // inner = operand jump, outer = beat step
        uint32_t src_bnd[2] = {2, beats};                    // 2 operands, beats beats
        uint32_t dst_str[1] = {XDMA_BEAT_BYTES};
        uint32_t dst_bnd[1] = {beats};

        uint32_t c1 = 0, c2 = 0, c3 = 0, lat_cold = 0, lat_warm = 0;
        int ok = 1;
        // Run twice: iter 0 = cold (first call, icache cold); iter 1 = warm / steady-state. Measures the
        // 3-pass xDMA offload only (swap is one-time staging above), comparable to swiglu's warm/cold.
        for (int iter = 0; iter < 2; iter++) {
            uint32_t t0 = snrt_mcycle();

            uint32_t csr_mul[2] = {2u /*operandCount*/, EW_MUL};
            uint32_t csr_add[2] = {2u /*operandCount*/, EW_ADD};

            // P1: tmp1 = x (.) cos.
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_mul) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(x_in, tmp1, 8, 8, 2, src_str, src_bnd,
                                            1, dst_str, dst_bnd,
                                            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            {
                int task_id = snax_xdma_start();
                snax_xdma_local_wait(task_id);
                c1 = snax_xdma_last_task_cycle();
            }
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);

            // P2: tmp2 = xswap (.) sin_signed.
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_mul) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(xswap, tmp2, 8, 8, 2, src_str, src_bnd,
                                            1, dst_str, dst_bnd,
                                            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            {
                int task_id = snax_xdma_start();
                snax_xdma_local_wait(task_id);
                c2 = snax_xdma_last_task_cycle();
            }
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);

            // P3: out = tmp1 (+) tmp2.
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_add) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(tmp1, out_buf, 8, 8, 2, src_str, src_bnd,
                                            1, dst_str, dst_bnd,
                                            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            {
                int task_id = snax_xdma_start();
                snax_xdma_local_wait(task_id);
                c3 = snax_xdma_last_task_cycle();
            }
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);

            uint32_t t1 = snrt_mcycle();
            if (iter == 0) lat_cold = t1 - t0; else lat_warm = t1 - t0;
        }

        if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu || c3 == 0xFFFFFFFFu) {
            printf("[RoPE] xDMA task setup failed\n");
            return 1;
        }
        printf("[RoPE] cycles: swap=%u p1=%u p2=%u p3=%u xdma_total=%u\n",
               sw, c1, c2, c3, c1 + c2 + c3);
        printf("[RoPE] full latency: cold=%u warm=%u cycles\n", lat_cold, lat_warm);

        // Verify out == interleaved RoPE(x). Integer FP16-ULP check on significant (signed) outputs;
        // skip the negligible subnormal/zero golden (FTZ tail). DM core has no FPU -> integer compare.
        uint16_t* out_h = (uint16_t*)out_buf;
        uint32_t mism = 0, worst = 0, checked = 0;
        for (uint32_t i = 0; i < rope_n; i++) {
            if ((rope_golden[i] & 0x7C00u) == 0) continue;  // skip subnormal/zero golden
            checked++;
            uint32_t o = fp16_mono(out_h[i]), g = fp16_mono(rope_golden[i]);
            uint32_t ulp = (o > g) ? (o - g) : (g - o);
            if (ulp > worst) worst = ulp;
            if (ulp > 4) {
                if (mism < 6)
                    printf("[RoPE] mismatch[%u]: got %04x golden %04x (%u ulp)\n",
                           i, out_h[i], rope_golden[i], ulp);
                mism++;
            }
        }
        printf("[RoPE] significant=%u/%u worst FP16 ULP=%u\n", checked, rope_n, worst);
        if (mism == 0) {
            printf("[RoPE] PASS (%u significant elements, <=4 ULP)\n", checked);
        } else {
            printf("[RoPE] FAIL (%u/%u significant beyond 4 ULP)\n", mism, checked);
            err++;
        }
    }
    return err != 0;
}
