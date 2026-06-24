// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 multi-row RMSNorm: per row, out[r,:] = x[r,:] * inv_rms[r], inv_rms[r]=1/sqrt(mean(x[r,:]^2)).
// This is the PER-ROW form the llama3 layer needs -- S independent rows scaled in one dispatch. The
// single-row app applies one StreamMap scalar to the whole row, which can't vary per row; the multi-row
// pipeline is instead:
//
//   T1  Sx^2   : StreamReduce(SUMSQ, rows)            x[rows,D] -> ssq[rows]        [xDMA, multi-row]
//   bcast      : inv_rms[r] -> inv_bcast[r,:]                                       [DM core, integer]
//   T2  scale  : StreamElementwise(MUL)              x (.) inv_bcast -> out[rows,D] [xDMA, element-wise]
//
// LANE DATAFLOW (FP16 transport = 32 lanes per 512-b beat):
//   T1 StreamReduce(SUMSQ, rows)  per row N beats -> 1 beat: the row's 32 lanes COLLAPSE to a scalar
//      ssq[r]=sum(x[r,:]^2) SPLATTED across all 32 lanes (read lane 0). rows*N beats in -> rows beats out.
//   bcast (DM core): inv_rms[r] written to ALL D lanes of inv_bcast[r,:] ([rows,D] buffer; integer copy).
//   T2 StreamElementwise(MUL)     2 beats -> 1 beat: out[i] = x[i] * inv_bcast[i], full 32 lanes. The
//      reader interleaves {x_beat, inv_bcast_beat} over rows*beats beats (reads 2x what the writer emits).
//
// The DM core (rv32ima, no FPU) can't do the rsqrt; a host core does it. Here inv_rms[rows] is precomputed
// in data.h (derived from the FP16-narrowed per-row Sx^2 so it matches the runtime reduce) and the DM core
// only replicates it across the row. Checks out vs rmsmr_out_golden.
//
// EXAMPLE (params.hjson default: rows=4, D=64 -> beats=D/32=2; FP16, all L1-local)
//   in   x    [rows=4, D=64] fp16  = 4*128 = 512 B
//   out  out  [rows=4, D=64] fp16  = 4*128 = 512 B   (per-row out[r,:] = x[r,:] * inv_rms[r])
//   intermediates: ssq[4] (one splatted 64-B beat/row), inv_bcast[4,64] (inv_rms[r] replicated).
//
//   L1 memory map (byte offset from cluster base):
//     +0     x          512 B   [4,64] input
//     +512   ssq        256 B   [4] SUMSQ scalars (splatted beats; T1 writer = 4 beats)
//     +768   inv_bcast  512 B   [4,64] inv_rms[r] broadcast (DM-core fill; T2 operand 1)
//     +1280  out        512 B   [4,64] result (T2 writer)
//     total  1792 B

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMREDUCE) || !defined(READER_EXT_STREAMELEMENTWISE)
#error "Regenerate the XDMA CSR map with StreamReduce and StreamElementwise."
#endif

#define XDMA_BEAT_BYTES 64
#define FP16_PER_BEAT 32
#define OP_SUMSQ 2u
#define EW_MUL 0u  // StreamElementwise op CSR: 0=MUL, 1=ADD

static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

static uint32_t run_task(void) {
    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    return snax_xdma_last_task_cycle();
}

int main() {
    int err = 0;
    if (snrt_is_dm_core()) {
        uint32_t base = snrt_cluster_base_addrl();
        uint32_t rows = rmsmr_rows;
        uint32_t d = rmsmr_d;
        uint32_t beats = rmsmr_beats;
        uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
        uint32_t rows_bytes = rows * row_bytes;  // whole [rows,D] FP16 buffer

        uint8_t* x_in = (uint8_t*)base;
        uint8_t* ssq_buf = x_in + rows_bytes;                    // rows splatted scalar beats
        uint8_t* inv_bcast = ssq_buf + rows * XDMA_BEAT_BYTES;   // [rows,D] broadcast of inv_rms[r]
        uint8_t* out_buf = inv_bcast + rows_bytes;               // [rows,D] result

        printf("[RmsMR] rows=%u D=%u beats=%u\n", rows, d, beats);

        snrt_dma_start_1d(x_in, rmsmr_input, rows_bytes);
        snrt_dma_wait_all();

        // T1: per-row Sx^2 -> ssq[rows]. 2D reader {beats inner, rows outer} -> 1D writer {rows}.
        uint32_t r_str[2] = {XDMA_BEAT_BYTES, row_bytes};
        uint32_t r_bnd[2] = {beats, rows};
        uint32_t w_str1[1] = {XDMA_BEAT_BYTES};
        uint32_t w_bnd_rows[1] = {rows};
        uint32_t csr_ssq[2] = {beats, OP_SUMSQ};
        int ok = (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_ssq) == 0);
        ok &= (snax_xdma_memcpy_nd_fast(x_in, ssq_buf, 8, 8, 2, r_str, r_bnd, 1, w_str1, w_bnd_rows,
                                        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
        uint32_t c1 = run_task();
        snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);

        // bcast: inv_rms[r] (precomputed FP16) -> inv_bcast[r, 0..D). Pure integer replication (no FPU).
        uint16_t* bc = (uint16_t*)inv_bcast;
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t c = 0; c < d; c++) bc[r * d + c] = rmsmr_inv_rms[r];

        // T2: out = x (.) inv_bcast. StreamElementwise(MUL), operandCount=2 over the {x, inv_bcast}
        // interleave: inner dim picks the two operands (stride = inv_bcast - x_in), outer sweeps the
        // rows*beats beats. Element-wise, so multi-row is just the larger outer bound.
        uint32_t ew_str[2] = {(uint32_t)(inv_bcast - x_in), XDMA_BEAT_BYTES};
        uint32_t ew_bnd[2] = {2, rows * beats};
        uint32_t w_str_all[1] = {XDMA_BEAT_BYTES};
        uint32_t w_bnd_all[1] = {rows * beats};
        uint32_t csr_mul[2] = {2u /*operandCount*/, EW_MUL};
        ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_mul) == 0);
        ok &= (snax_xdma_memcpy_nd_fast(x_in, out_buf, 8, 8, 2, ew_str, ew_bnd, 1, w_str_all, w_bnd_all,
                                        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
        uint32_t c2 = run_task();
        snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);

        if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu) {
            printf("[RmsMR] xDMA task setup failed\n");
            return 1;
        }
        printf("[RmsMR] cycles: sumsq=%u scale=%u\n", c1, c2);

        // Per-row integer FP16-ULP check on the significant outputs (skip subnormal/zero golden tail).
        uint16_t* out_h = (uint16_t*)out_buf;
        uint32_t mism = 0, worst = 0, checked = 0;
        for (uint32_t i = 0; i < rows * d; i++) {
            if ((rmsmr_out_golden[i] & 0x7C00u) == 0) continue;
            checked++;
            uint32_t o = fp16_mono(out_h[i]), g = fp16_mono(rmsmr_out_golden[i]);
            uint32_t ulp = (o > g) ? (o - g) : (g - o);
            if (ulp > worst) worst = ulp;
            if (ulp > 4) {
                if (mism < 6)
                    printf("[RmsMR] mismatch[%u] (row %u): got %04x golden %04x (%u ulp)\n", i, i / d,
                           out_h[i], rmsmr_out_golden[i], ulp);
                mism++;
            }
        }
        printf("[RmsMR] significant=%u/%u worst FP16 ULP=%u\n", checked, rows * d, worst);
        if (mism == 0)
            printf("[RmsMR] PASS (%u significant elements over %u rows, <=4 ULP)\n", checked, rows);
        else {
            printf("[RmsMR] FAIL (%u/%u significant beyond 4 ULP)\n", mism, checked);
            err++;
        }
    }
    return err != 0;
}
