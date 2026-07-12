// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 multi-row RMSNorm, A/B: per row, out[r,:] = x[r,:] * inv_rms[r],
// inv_rms[r] = 1/sqrt(mean(x[r,:]^2)). Same kernel as snax-xdma-rmsnorm-multirow
// (same tensor, same golden); this app runs the DEPLOYED path and a SOFTWARE-ONLY
// optimized path back to back on the same data, and reports both.
//
// The per-row scalar cannot be held constant across a row by the reader AGU (a single
// affine address stream cannot keep one operand's address fixed while the other walks
// the row), so inv_rms[r] must be materialized as a full [rows,D] operand tensor. That
// is unavoidable without an RTL change. What IS avoidable is materializing it with the
// DM core, and re-reading the result for a separate quantize pass:
//
//   A (deployed)   T1 StreamReduce(SUMSQ)                     x -> ssq[rows]
//                  A1 DM-core broadcast loop                  inv_rms[r] -> [rows,D]
//                     (rows*D half-word stores -- O(rows*D), scales with D)
//                  A2 StreamElementwise(MUL)                  x (.) inv_bc -> out
//                  A3 Fp16ToInt8 (separate pass)              out -> int8   [re-reads out]
//
//   B (this app)   T1 StreamReduce(SUMSQ)                     x -> ssq[rows]
//                  B0 DM-core splat, ONE beat per row         inv_rms[r] -> [rows] beats
//                     (rows*16 word stores -- O(rows), independent of D)
//                  B1 xDMA BROADCAST pass, reader stride 0    [rows] beats -> [rows,D]
//                     (the inner temporal dim re-reads the row's scalar beat `beats`
//                      times: the AGU counter's trip count comes from its own index
//                      counter, so step=0 holds the address while the loop still runs)
//                  B2 StreamElementwise(MUL) -||> Fp16ToInt8  x (.) inv_bc -> int8
//                     (chained in ONE task: the quantize rides the scale pass, no re-read)
//
// Both paths are exercised here and cross-checked bit-exactly against each other:
//   inv_bc_b == inv_bc_a  -> the stride-0 broadcast really does replicate the beat
//   out_b    == out_a     -> same EW operands => identical FP16 result
//   q_fused  == q_sep     -> chaining the quant onto the EW pass is bit-identical
// and out_a is checked against the FP64-derived golden (<=4 FP16 ULP), as in the
// deployed app.
//
// inv_rms[] is DMA'd to L1 up front and BOTH paths read it from there: in data.h it lives
// in L3, and a per-row DRAM load would tax both core loops (~146 cc/row) for reasons that
// have nothing to do with the kernel. Zeroing the lane-0 beat buffer is likewise a
// one-time scratch init (lanes 1..31 stay zero across calls), so it is not timed.
//
// MEASURED (vsim, warm, L1<->L1; A's xDMA task cycles match snax-xdma-rmsnorm-multirow
// exactly at 4x64: sumsq 197, scale 144). The win scales with D: A's broadcast loop is
// rows*D core stores (~4 cc each), B's splat is rows*16 -- independent of D.
//
//   rows x D    A fp16   B fp16   x     A int8   B int8   x     A1 loop   B0 splat
//   --------    ------   ------   ---   ------   ------   ---   -------   --------
//   4 x 64       2,354    1,735   1.4    2,905    1,873   1.6     1,075        102
//   16 x 256    21,062    5,360   3.9   22,573    5,500   4.1    16,567        390
//   8 x 1024    40,206    8,144   4.9   42,741    8,284   5.2    32,863        198
//
// EXAMPLE (params.hjson default: rows=16, D=256 -> beats=D/32=8; FP16, all L1-local)
//   L1 memory map (byte offset from cluster base, rows_bytes = rows*beats*64 = 8 KiB):
//     +0       x         8192 B  [16,256] input
//     +8192    ssq       1024 B  [16] SUMSQ scalars (splatted beats; T1)
//     +9216    inv_l1      64 B  [16] inv_rms, L1 copy (read by both core loops)
//     +9280    inv_beat  1024 B  [16] inv_rms[r] splatted, one beat per row (B0)
//     +10304   inv_bc_a  8192 B  [16,256] broadcast, DM-core loop      (A1; A2 operand 1)
//     +18496   inv_bc_b  8192 B  [16,256] broadcast, xDMA stride-0     (B1; B2 operand 1)
//     +26688   out_a     8192 B  [16,256] FP16 result, path A
//     +34880   out_b     8192 B  [16,256] FP16 result, path B
//     +43072   q_sep     4096 B  [16,256] int8, separate quant pass    (A3)
//     +47168   q_fused   4096 B  [16,256] int8, quant chained onto EW  (B2)
//     total    51264 B (~50 KiB)

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMREDUCE) || !defined(READER_EXT_STREAMELEMENTWISE) || \
    !defined(READER_EXT_FP16TOINT8)
#error \
    "Regenerate the XDMA CSR map with StreamReduce, StreamElementwise and Fp16ToInt8."
#endif

#define XDMA_BEAT_BYTES 64
#define FP16_PER_BEAT 32
#define OP_SUMSQ 2u  // StreamReduce fused-FMA op, square mode (acc + x*x)
#define EW_MUL 0u    // StreamElementwise fused-FMA op CSR: 0=MUL (acc*x), 1=ADD

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
        uint32_t rows = rmsf_rows;
        uint32_t d = rmsf_d;
        uint32_t beats = rmsf_beats;
        uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
        uint32_t rows_bytes = rows * row_bytes;   // whole [rows,D] FP16 buffer
        uint32_t scal_bytes = rows * XDMA_BEAT_BYTES;
        uint32_t all_beats = rows * beats;        // beats in a whole [rows,D] tensor

        uint8_t* x_in = (uint8_t*)base;
        uint8_t* ssq_buf = x_in + rows_bytes;
        uint8_t* inv_l1 = ssq_buf + scal_bytes;      // [rows] inv_rms, L1 copy
        uint8_t* inv_lane0 = inv_l1 + XDMA_BEAT_BYTES;  // [rows] beats, scalar in lane 0
        uint8_t* inv_bc_a = inv_lane0 + scal_bytes;  // [rows,D] bcast, DM-core loop
        uint8_t* inv_bc_b = inv_bc_a + rows_bytes;   // [rows,D] bcast, xDMA stride-0
        uint8_t* out_a = inv_bc_b + rows_bytes;      // FP16 result, path A
        uint8_t* out_b = out_a + rows_bytes;         // FP16 result, path B
        uint8_t* q_sep = out_b + rows_bytes;         // int8, separate quant pass
        uint8_t* q_fused = q_sep + rows_bytes / 2;   // int8, quant chained onto the EW

        printf("[RmsFold] rows=%u D=%u beats=%u\n", rows, d, beats);

        // Input + the per-row inv_rms scalars into L1 (both core loops read inv_rms from
        // L1, so neither pays a per-row DRAM load for a data.h array).
        snrt_dma_start_1d(x_in, rmsf_input, rows_bytes);
        snrt_dma_start_1d(inv_l1, rmsf_inv_rms, rows * sizeof(uint16_t));
        snrt_dma_wait_all();

        // One-time scratch init: lanes 1..31 of each lane-0 beat stay zero across calls,
        // so B0 only ever rewrites lane 0. Not part of the per-call cost.
        for (uint32_t i = 0; i < scal_bytes / 4; i++) ((uint32_t*)inv_lane0)[i] = 0;

        // ---- AGU shapes ------------------------------------------------------------
        // reduce: 2D reader {beats inner, rows outer} -> 1D writer {rows}
        uint32_t red_str[2] = {XDMA_BEAT_BYTES, row_bytes};
        uint32_t red_bnd[2] = {beats, rows};
        uint32_t w_rows_str[1] = {XDMA_BEAT_BYTES};
        uint32_t w_rows_bnd[1] = {rows};
        // flat: 1D over the whole [rows,D] tensor
        uint32_t flat_str[1] = {XDMA_BEAT_BYTES};
        uint32_t flat_bnd[1] = {all_beats};
        // int8 writer: Fp16ToInt8 packs 2 FP16 beats -> 1 INT8 beat
        uint32_t q_bnd[1] = {all_beats / 2};
        // BROADCAST reader: inner dim stride 0 -> the row's scalar beat is re-read
        // `beats` times; outer dim steps one scalar beat per row.
        uint32_t bc_str[2] = {0, XDMA_BEAT_BYTES};
        uint32_t bc_bnd[2] = {beats, rows};
        // elementwise interleave {x_beat, bcast_beat}: inner dim picks the operand
        // (stride = the two regions' distance), outer sweeps the tensor's beats.
        uint32_t ew_a_str[2] = {(uint32_t)(inv_bc_a - x_in), XDMA_BEAT_BYTES};
        uint32_t ew_b_str[2] = {(uint32_t)(inv_bc_b - x_in), XDMA_BEAT_BYTES};
        uint32_t ew_bnd[2] = {2, all_beats};

        uint32_t csr_ssq[2] = {beats, OP_SUMSQ};
        uint32_t csr_mul[2] = {2u /*operandCount*/, EW_MUL};
        uint32_t csr_q[1] = {rmsf_inv_scale};

        uint16_t* bc_a = (uint16_t*)inv_bc_a;
        uint16_t* inv = (uint16_t*)inv_l1;

        // Per-stage wall-clock (CSR config + start + wait), and the xDMA datapath-only
        // cycles. Two iterations: 0 = cold (one-time AGU shape setup + icache), 1 = warm.
        uint32_t t_red = 0, t_loop = 0, t_ewa = 0, t_qsep = 0;
        uint32_t t_splat = 0, t_bcast = 0, t_ewq = 0, t_ewb = 0;
        uint32_t d_red = 0, d_ewa = 0, d_qsep = 0, d_bcast = 0, d_ewq = 0;
        int ok = 1;

        for (int it = 0; it < 2; it++) {
            uint32_t t0, t1;

            // ================= T1 (shared): per-row Sx^2 =================
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_ssq) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(x_in, ssq_buf, 8, 8, 2, red_str, red_bnd,
                                            1, w_rows_str, w_rows_bnd, 0xFFFFFFFF,
                                            0xFFFFFFFF, 0xFFFFFFFF) == 0);
            d_red = run_task();
            snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);
            t1 = snrt_mcycle();
            t_red = t1 - t0;

            // ================= PATH A: deployed =================
            // A1: DM-core broadcast loop -- rows*D half-word stores.
            t0 = snrt_mcycle();
            for (uint32_t r = 0; r < rows; r++)
                for (uint32_t c = 0; c < d; c++) bc_a[r * d + c] = inv[r];
            t1 = snrt_mcycle();
            t_loop = t1 - t0;

            // A2: out_a = x (.) inv_bc_a
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_mul) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(x_in, out_a, 8, 8, 2, ew_a_str, ew_bnd, 1,
                                            flat_str, flat_bnd, 0xFFFFFFFF, 0xFFFFFFFF,
                                            0xFFFFFFFF) == 0);
            d_ewa = run_task();
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);
            t1 = snrt_mcycle();
            t_ewa = t1 - t0;

            // A3: separate quantize pass -- re-reads the whole FP16 result.
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_FP16TOINT8, csr_q) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(out_a, q_sep, 8, 8, 1, flat_str, flat_bnd,
                                            1, flat_str, q_bnd, 0xFFFFFFFF, 0xFFFFFFFF,
                                            0xFFFFFFFF) == 0);
            d_qsep = run_task();
            snax_xdma_disable_src_ext(READER_EXT_FP16TOINT8);
            t1 = snrt_mcycle();
            t_qsep = t1 - t0;

            // ================= PATH B: software-only fold =================
            // B0: splat inv_rms[r] across the 32 lanes of ONE beat per row, as 16 word
            // stores (2 FP16 lanes each). rows*16 stores -- O(rows), independent of D
            // (the A1 loop is rows*D). Doing this splat in HW instead (a stride-0 read
            // through StreamReduce(MAX, operandCount=1)) is NOT possible on this RTL: a
            // 1-beat row retires every cycle and trips StreamReduce's "row completed
            // while the fold was still busy" assert -- the pipelined horizontal fold
            // needs rows longer than its latency.
            t0 = snrt_mcycle();
            for (uint32_t r = 0; r < rows; r++) {
                uint32_t w = ((uint32_t)inv[r] << 16) | (uint32_t)inv[r];
                uint32_t* beat = (uint32_t*)(inv_lane0 + r * XDMA_BEAT_BYTES);
                for (uint32_t l = 0; l < FP16_PER_BEAT / 2; l++) beat[l] = w;
            }
            t1 = snrt_mcycle();
            t_splat = t1 - t0;

            // B1: xDMA broadcast -- reader inner stride 0 re-reads the row's scalar beat
            // `beats` times. No extension: a plain strided copy.
            t0 = snrt_mcycle();
            ok &= (snax_xdma_memcpy_nd_fast(inv_lane0, inv_bc_b, 8, 8, 2, bc_str, bc_bnd,
                                            1, flat_str, flat_bnd, 0xFFFFFFFF,
                                            0xFFFFFFFF, 0xFFFFFFFF) == 0);
            d_bcast = run_task();
            t1 = snrt_mcycle();
            t_bcast = t1 - t0;

            // B2: out_b = x (.) inv_bc_b, with Fp16ToInt8 CHAINED -- one task emits both
            // the FP16 scale result (through the chain) and the int8 quant. The writer
            // drains the int8 stream (all_beats/2); out_b is written by the unfused
            // twin below only to prove the FP16 datapath is unchanged.
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_mul) == 0);
            ok &= (snax_xdma_enable_src_ext(READER_EXT_FP16TOINT8, csr_q) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(x_in, q_fused, 8, 8, 2, ew_b_str, ew_bnd, 1,
                                            flat_str, q_bnd, 0xFFFFFFFF, 0xFFFFFFFF,
                                            0xFFFFFFFF) == 0);
            d_ewq = run_task();
            snax_xdma_disable_src_ext(READER_EXT_FP16TOINT8);
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);
            t1 = snrt_mcycle();
            t_ewq = t1 - t0;

            // B2': the same EW MUL WITHOUT the chained quant -- path B's FP16-out
            // variant, and the buffer that lets out_b be compared to out_a bit-exactly.
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE, csr_mul) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(x_in, out_b, 8, 8, 2, ew_b_str, ew_bnd, 1,
                                            flat_str, flat_bnd, 0xFFFFFFFF, 0xFFFFFFFF,
                                            0xFFFFFFFF) == 0);
            (void)run_task();
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);
            t1 = snrt_mcycle();
            t_ewb = t1 - t0;
        }

        if (!ok) {
            printf("[RmsFold] xDMA task setup FAILED\n");
            return 1;
        }

        // ---- cycles ----------------------------------------------------------------
        uint32_t base_fp16 = t_red + t_loop + t_ewa;
        uint32_t opt_fp16 = t_red + t_splat + t_bcast + t_ewb;
        uint32_t base_int8 = base_fp16 + t_qsep;
        uint32_t opt_int8 = t_red + t_splat + t_bcast + t_ewq;

        printf("[RmsFold] --- stage cycles (warm; wall = CSR cfg + start + wait) ---\n");
        printf("[RmsFold] T1 reduce(SUMSQ)      wall=%u  datapath=%u\n", t_red, d_red);
        printf("[RmsFold] A1 DM-core bcast loop wall=%u  (rows*D=%u stores)\n", t_loop,
               rows * d);
        printf("[RmsFold] A2 EW(MUL)            wall=%u  datapath=%u\n", t_ewa, d_ewa);
        printf("[RmsFold] A3 quant (separate)   wall=%u  datapath=%u\n", t_qsep, d_qsep);
        printf("[RmsFold] B0 DM-core splat      wall=%u  (rows*16=%u stores)\n", t_splat,
               rows * 16);
        printf("[RmsFold] B1 xDMA bcast (str0)  wall=%u  datapath=%u\n", t_bcast,
               d_bcast);
        printf("[RmsFold] B2 EW(MUL)-||>quant   wall=%u  datapath=%u\n", t_ewq, d_ewq);
        printf("[RmsFold] B2' EW(MUL) fp16 out  wall=%u\n", t_ewb);
        printf("[RmsFold] === fp16 out: A=%u  B=%u cycles ===\n", base_fp16, opt_fp16);
        printf("[RmsFold] === int8 out: A=%u  B=%u cycles ===\n", base_int8, opt_int8);

        // ---- correctness -----------------------------------------------------------
        // 1) path A vs the FP64-derived golden (the deployed app's check).
        uint16_t* oa = (uint16_t*)out_a;
        uint32_t mism = 0, worst = 0, checked = 0;
        for (uint32_t i = 0; i < rows * d; i++) {
            if ((rmsf_out_golden[i] & 0x7C00u) == 0) continue;
            checked++;
            uint32_t o = fp16_mono(oa[i]), g = fp16_mono(rmsf_out_golden[i]);
            uint32_t ulp = (o > g) ? (o - g) : (g - o);
            if (ulp > worst) worst = ulp;
            if (ulp > 4) {
                if (mism < 6)
                    printf("[RmsFold] mismatch[%u] (row %u): got %04x golden %04x\n", i,
                           i / d, oa[i], rmsf_out_golden[i]);
                mism++;
            }
        }
        printf("[RmsFold] A vs golden: significant=%u/%u worst FP16 ULP=%u\n", checked,
               rows * d, worst);
        if (mism) {
            printf("[RmsFold] FAIL: %u/%u beyond 4 ULP\n", mism, checked);
            err++;
        }

        // 2) the stride-0 broadcast must reproduce the DM-core fill BIT-EXACTLY.
        uint32_t bad = 0;
        uint16_t* bc_b = (uint16_t*)inv_bc_b;
        for (uint32_t i = 0; i < rows * d; i++)
            if (bc_a[i] != bc_b[i]) bad++;
        printf("[RmsFold] stride-0 bcast vs DM-core fill: %s (%u/%u differ)\n",
               bad ? "FAIL" : "bit-exact", bad, rows * d);
        if (bad) err++;

        // 3) same EW operands => the FP16 result must be bit-identical.
        uint16_t* ob = (uint16_t*)out_b;
        bad = 0;
        for (uint32_t i = 0; i < rows * d; i++)
            if (oa[i] != ob[i]) bad++;
        printf("[RmsFold] out_b vs out_a: %s (%u/%u differ)\n", bad ? "FAIL" : "bit-exact",
               bad, rows * d);
        if (bad) err++;

        // 4) chaining the quant onto the EW pass must be bit-identical to a separate pass.
        int8_t* qs = (int8_t*)q_sep;
        int8_t* qf = (int8_t*)q_fused;
        bad = 0;
        for (uint32_t i = 0; i < rows * d; i++)
            if (qs[i] != qf[i]) bad++;
        printf("[RmsFold] q_fused vs q_sep: %s (%u/%u differ)\n",
               bad ? "FAIL" : "bit-exact", bad, rows * d);
        if (bad) err++;

        printf(err ? "[RmsFold] FAIL\n" : "[RmsFold] PASS\n");
    }
    return err != 0;
}
