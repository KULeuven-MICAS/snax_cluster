// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 multi-row softmax: per row, out[r,:] = exp(x[r,:]-max[r]) / Sexp[r]. The PER-ROW form the
// llama3 attention needs -- S independent rows softmaxed in one dispatch. Each row has its own max and
// 1/Sexp, which StreamMap's single scalar can't express; the multi-row pipeline uses a host scalar-broadcast +
// StreamElementwise:
//
//   T1 max    : StreamReduce(MAX, rows)               x       -> max[rows]          [xDMA, multi-row]
//   bcast     : -max[r] -> negmax_bcast[r,:]   (FP16 sign flip, integer)            [DM core]
//   T2 sub    : StreamElementwise(ADD)               x (.) negmax -> xs[rows,D]      [xDMA, element-wise]
//   T3 exp    : StreamMap(EXP, a=1, b=0)             xs      -> expb[rows,D]         [xDMA, element-wise]
//   T4 sum    : StreamReduce(ADD, rows)              expb    -> sum[rows]            [xDMA, multi-row]
//   bcast     : 1/sum[r] -> recip_bcast[r,:]   (recip precomputed, integer copy)    [DM core]
//   T5 norm   : StreamElementwise(MUL)              expb (.) recip -> out[rows,D]    [xDMA, element-wise]
//
// LANE DATAFLOW (FP16 transport = 32 lanes per 512-b beat). Only the two reduces change the beat shape;
// the sew/smap passes are clean element-wise over the flat rows*beats stream:
//   T1 StreamReduce(MAX)  per row N beats -> 1 beat: 32 lanes collapse to max[r], SPLATTED (read lane 0).
//   bcast: -max[r] -> negmax_bcast[r,:]  (FP16 sign flip ^0x8000, replicated across the D lanes).
//   T2 StreamElementwise(ADD)  2 beats -> 1 beat: xs[i]   = x[i] + (-max[i])        (full 32 lanes)
//   T3 StreamMap(EXP)          1 beat  -> 1 beat: expb[i] = exp(xs[i])              (full 32 lanes)
//   T4 StreamReduce(ADD)  per row N beats -> 1 beat: 32 lanes collapse to Sexp[r], SPLATTED.
//   bcast: 1/sum[r] -> recip_bcast[r,:]  (replicated across the D lanes).
//   T5 StreamElementwise(MUL)  2 beats -> 1 beat: out[i]  = expb[i] * recip[i]      (full 32 lanes)
//   (each sew reads its two operands interleaved -> 2x the beats it writes; the reduces are rows*N -> rows.)
//
// The DM core (rv32ima, no FPU) does -max as an integer sign flip; the reciprocal is precomputed in
// data.h (a host core does it). Checks out vs smr_out_golden, which mirrors the FP16 chain.
//
// EXAMPLE (params.hjson default: rows=4, D=64 -> beats=D/32=2; FP16, all L1-local)
//   in   x    [rows=4, D=64] fp16  = 4*128 = 512 B
//   out  out  [rows=4, D=64] fp16  = 4*128 = 512 B   (per-row softmax)
//   The 5 xDMA passes thread 6 intermediate [rows,D]/[rows] buffers (max, negmax, xs, expb, sum, recip).
//
//   L1 memory map (byte offset from cluster base):
//     +0     x          512 B   [4,64] input
//     +512   max        256 B   [4] MAX scalars (splatted beats; T1)
//     +768   negmax_bc  512 B   [4,64] -max[r] broadcast (DM core; T2 operand 1)
//     +1280  xs         512 B   [4,64] x-max          (T2 out / T3 in)
//     +1792  expb       512 B   [4,64] exp(x-max)     (T3 out / T4,T5 in)
//     +2304  sum        256 B   [4] Sexp scalars      (T4)
//     +2560  recip_bc   512 B   [4,64] 1/Sexp[r] bcast (DM core; T5 operand 1)
//     +3072  out        512 B   [4,64] softmax result (T5)
//     total  3584 B

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMREDUCE) || !defined(READER_EXT_STREAMMAP) || \
    !defined(READER_EXT_STREAMELEMENTWISE)
#error "Regenerate the XDMA CSR map with StreamReduce, StreamMap and StreamElementwise."
#endif

#define XDMA_BEAT_BYTES 64
#define FP16_PER_BEAT 32
#define OP_MAX 0u
#define OP_ADD 1u
#define EW_MUL 0u  // StreamElementwise op CSR: 0=MUL, 1=ADD
#define EW_ADD 1u
#define ACT_EXP 1u  // StreamMap func CSR bits[1:0]=1: EXP (out = exp(a*x + b))

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
        uint32_t rows = smr_rows;
        uint32_t d = smr_d;
        uint32_t beats = smr_beats;
        uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
        uint32_t rows_bytes = rows * row_bytes;   // whole [rows,D] FP16 buffer
        uint32_t scal_bytes = rows * XDMA_BEAT_BYTES;  // rows splatted scalar beats

        uint8_t* x_in = (uint8_t*)base;
        uint8_t* max_buf = x_in + rows_bytes;            // reduce(MAX) out
        uint8_t* negmax_bc = max_buf + scal_bytes;       // [rows,D] -max broadcast
        uint8_t* xs_buf = negmax_bc + rows_bytes;        // sew(ADD) out = x - max
        uint8_t* expb_buf = xs_buf + rows_bytes;         // smap(EXP) out
        uint8_t* sum_buf = expb_buf + rows_bytes;        // reduce(ADD) out
        uint8_t* recip_bc = sum_buf + scal_bytes;        // [rows,D] 1/sum broadcast
        uint8_t* out_buf = recip_bc + rows_bytes;        // sew(MUL) out = softmax

        printf("[SmMR] rows=%u D=%u beats=%u\n", rows, d, beats);

        snrt_dma_start_1d(x_in, smr_input, rows_bytes);
        snrt_dma_wait_all();

        // Common AGU shapes.
        uint32_t red_str[2] = {XDMA_BEAT_BYTES, row_bytes};     // 2D reader {beats, rows} for the reduces
        uint32_t red_bnd[2] = {beats, rows};
        uint32_t w_rows_str[1] = {XDMA_BEAT_BYTES};
        uint32_t w_rows_bnd[1] = {rows};
        uint32_t flat_str[1] = {XDMA_BEAT_BYTES};               // 1D reader/writer over rows*beats beats
        uint32_t flat_bnd[1] = {rows * beats};

        // T1: per-row max.
        uint32_t csr_max[2] = {beats, OP_MAX};
        int ok = (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_max) == 0);
        ok &= (snax_xdma_memcpy_nd_fast(x_in, max_buf, 8, 8, 2, red_str, red_bnd, 1, w_rows_str,
                                        w_rows_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
        uint32_t c1 = run_task();
        snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);

        // bcast -max[r] (FP16 sign flip of the runtime max) over negmax_bc[r, 0..D).
        uint16_t* mx = (uint16_t*)max_buf;
        uint16_t* nbc = (uint16_t*)negmax_bc;
        for (uint32_t r = 0; r < rows; r++) {
            uint16_t neg = (uint16_t)(mx[r * FP16_PER_BEAT] ^ 0x8000u);  // low lane of splatted beat r
            for (uint32_t c = 0; c < d; c++) nbc[r * d + c] = neg;
        }

        // T2: xs = x + (-max). StreamElementwise(ADD), {x, negmax} interleave over rows*beats beats.
        uint32_t ew_add_str[2] = {(uint32_t)(negmax_bc - x_in), XDMA_BEAT_BYTES};
        uint32_t ew_bnd[2] = {2, rows * beats};
        uint32_t csr_add[2] = {2u, EW_ADD};
        ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_add) == 0);
        ok &= (snax_xdma_memcpy_nd_fast(x_in, xs_buf, 8, 8, 2, ew_add_str, ew_bnd, 1, flat_str, flat_bnd,
                                        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
        uint32_t c2 = run_task();
        snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);

        // T3: expb = exp(xs). StreamMap(EXP, a=1.0, b=0), element-wise over rows*beats beats.
        uint32_t csr_exp[3] = {0x3F800000u /*1.0f*/, 0u, ACT_EXP};
        ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_exp) == 0);
        ok &= (snax_xdma_memcpy_nd_fast(xs_buf, expb_buf, 8, 8, 1, flat_str, flat_bnd, 1, flat_str,
                                        flat_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
        uint32_t c3 = run_task();
        snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);

        // T4: per-row Sexp.
        uint32_t csr_sum[2] = {beats, OP_ADD};
        ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_sum) == 0);
        ok &= (snax_xdma_memcpy_nd_fast(expb_buf, sum_buf, 8, 8, 2, red_str, red_bnd, 1, w_rows_str,
                                        w_rows_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
        uint32_t c4 = run_task();
        snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);

        // bcast 1/sum[r] (precomputed FP16 reciprocal) over recip_bc[r, 0..D).
        uint16_t* rbc = (uint16_t*)recip_bc;
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t c = 0; c < d; c++) rbc[r * d + c] = smr_inv_sum[r];

        // T5: out = expb * recip. StreamElementwise(MUL), {expb, recip} interleave.
        uint32_t ew_mul_str[2] = {(uint32_t)(recip_bc - expb_buf), XDMA_BEAT_BYTES};
        uint32_t csr_mul[2] = {2u, EW_MUL};
        ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_mul) == 0);
        ok &= (snax_xdma_memcpy_nd_fast(expb_buf, out_buf, 8, 8, 2, ew_mul_str, ew_bnd, 1, flat_str,
                                        flat_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
        uint32_t c5 = run_task();
        snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);

        if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu || c3 == 0xFFFFFFFFu || c4 == 0xFFFFFFFFu ||
            c5 == 0xFFFFFFFFu) {
            printf("[SmMR] xDMA task setup failed\n");
            return 1;
        }
        printf("[SmMR] cycles: max=%u sub=%u exp=%u sum=%u norm=%u\n", c1, c2, c3, c4, c5);

        // Per-row FP16-ULP check on the significant outputs. Tolerance 8: the multi-row chain narrows to
        // FP16 once more than the single-row fused path (xs and recip both narrowed), and the HW exp is a
        // LUT vs the np.exp golden -- a broken row/AGU would be off by hundreds of ULP, not <=8.
        uint16_t* out_h = (uint16_t*)out_buf;
        uint32_t mism = 0, worst = 0, checked = 0;
        for (uint32_t i = 0; i < rows * d; i++) {
            if ((smr_out_golden[i] & 0x7C00u) == 0) continue;  // skip negligible subnormal/zero golden
            checked++;
            uint32_t o = fp16_mono(out_h[i]), g = fp16_mono(smr_out_golden[i]);
            uint32_t ulp = (o > g) ? (o - g) : (g - o);
            if (ulp > worst) worst = ulp;
            if (ulp > 8) {
                if (mism < 6)
                    printf("[SmMR] mismatch[%u] (row %u): got %04x golden %04x (%u ulp)\n", i, i / d,
                           out_h[i], smr_out_golden[i], ulp);
                mism++;
            }
        }
        printf("[SmMR] significant=%u/%u worst FP16 ULP=%u\n", checked, rows * d, worst);
        if (mism == 0)
            printf("[SmMR] PASS (%u significant elements over %u rows, <=8 ULP)\n", checked, rows);
        else {
            printf("[SmMR] FAIL (%u/%u significant beyond 8 ULP)\n", mism, checked);
            err++;
        }
    }
    return err != 0;
}
