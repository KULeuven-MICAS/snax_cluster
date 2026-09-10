// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// SIMD FP16 multi-row softmax, A/B: per row,
// out[r,:] = exp(x[r,:] - max[r]) / Sexp[r]. Same kernel, tensor and golden as
// snax-simd-softmax-multirow; this app runs the DEPLOYED path and a
// SOFTWARE-ONLY optimized path back to back on the same data, and reports both.
//
// The per-row scalars (max[r], 1/Sexp[r]) cannot be held constant across a row
// by the reader AGU -- a single affine address stream cannot keep one operand's
// address fixed while the other walks the row -- so they must be materialized
// as full [rows,D] operand tensors, which no software can avoid on this RTL.
// What IS avoidable is materializing them with the DM core, and paying a
// separate pass for the row-sum and for the quantize:
//
//   A (deployed)
//     T1  StreamReduce(MAX)          x -> max[rows]
//     A1  DM-core broadcast loop     -max[r] -> [rows,D]   (rows*D stores)
//     A2  StreamElementwise(ADD)     x (.) -max -> xs
//     A3  StreamMap(EXP)             xs -> expb
//     A4  StreamReduce(ADD)          expb -> sum[rows]  (re-reads expb)
//     A5  DM-core broadcast loop     1/sum[r] -> [rows,D] (rows*D stores)
//     A6  StreamElementwise(MUL)     expb (.) recip -> out
//     A7  Fp16ToInt8, separate pass  out -> int8        (re-reads out)
//
//   B (this app)
//     T1  StreamReduce(MAX)          x -> max[rows]
//     B0  DM-core sign-flip of the SPLATTED max beat, one beat per row
//         (rows*16 word stores: O(rows), independent of D)
//     B1  xDMA BROADCAST pass, stride 0    [rows] beats -> [rows,D]
//         The inner temporal dim re-reads the row's scalar beat `beats` times:
//         the AGU counter takes its trip count from its own index counter, so a
//         step of 0 holds the address while the loop still runs.
//     B2  StreamElementwise(ADD)     x (.) -max -> xs
//     B3  StreamMap(EXP) -||> StreamReduce(ADD, TAP)   xs -> expb + Sexp
//         Chained in ONE task: TAP passes the exp row through AND appends the
//         row-sum as a trailing beat, so A3+A4 collapse and expb is never
//         re-read. The output is beats+1 beats per row: a PADDED row layout.
//     B4  DM-core splat of 1/Sexp[r]  (rows*16 word stores)
//     B5  xDMA BROADCAST pass, stride 0, written at the PADDED row stride so it
//         lines up with expb's padded rows (the EW interleave needs both
//         operands to share a row stride).
//     B6  StreamElementwise(MUL) -||> Fp16ToInt8   expb (.) recip -> int8
//         Chained in ONE task: the quantize rides the norm pass, no re-read.
//         Its 3D reader {operand, beat, row} never addresses the trailing sum
//         beat.
//
// The subtract cannot fuse with the exp: the reader extensions are chained in
// the hjson's list order (... StreamMap, StreamReduce, StreamElementwise,
// Fp16ToInt8), so an elementwise op can never precede a map in one task. B
// therefore keeps A2 as its own pass.
//
// Cross-checks, all bit-exact, all in this app:
//   out_b   == out_a     the whole B chain reproduces the deployed FP16 result
//   tap sum == A4's sum  the multi-row TAP scalar equals a standalone reduce
//   q_fused == q_sep     chaining the quant onto the norm pass is bit-identical
// and out_a is checked against the FP64-derived golden (<=4 FP16 ULP), as the
// deployed app does.
//
// inv_sum[] is DMA'd to L1 up front and both core loops read it from there: in
// data.h it lives in L3, and a per-row DRAM load would tax both paths for
// reasons that have nothing to do with the kernel.
//
// MEASURED (vsim, warm, L1<->L1). A's xDMA task cycles match
// snax-simd-softmax-multirow exactly at 4x64 (max 197, sub 144, exp 86, sum
// 197, norm 144), so A is the deployed kernel, not a strawman. The win scales
// with D: A's two broadcast loops are 2*rows*D core stores (~4 cc each), B's
// two splats are rows*16 each, independent of D.
//
//   rows x D   A fp16  B fp16    x   A int8  B int8    x   A loops  B splats
//   --------   ------  ------   ---  ------  ------   ---  -------  --------
//    4 x   64   5,284   3,700   1.4   5,841   3,801   1.5   ~1,100      ~200
//    8 x  256  23,080   6,954   3.3  24,085   7,057   3.4   32,900      ~400
//   16 x  256  43,696  11,066   3.9  45,213  11,169   4.0   33,186      ~800
//    4 x 1024  42,724   9,698   4.4  44,241   9,801   4.5   32,886      ~210
//
// The TAP fusion also shrinks the datapath itself (B3 = 242 cc vs A3+A4 =
// 86+197 = 283 cc at 4x64) and removes a whole task setup (~600 cc of CSR
// orchestration).
//
// EXAMPLE (params.hjson default: rows=16, D=256 -> beats=8; FP16, L1-local):
// row_bytes = beats*64 = 512, rows_bytes = 8 KiB, padded row = (beats+1)*64.

#include "data.h"
#include "snax-simd-compat.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMREDUCE) || !defined(READER_EXT_STREAMMAP) || \
    !defined(READER_EXT_STREAMELEMENTWISE) || !defined(READER_EXT_FP16TOINT8)
#error \
    "Regenerate the XDMA CSR map with StreamReduce, StreamMap, StreamElementwise and Fp16ToInt8."
#endif

#define XDMA_BEAT_BYTES 64
#define FP16_PER_BEAT 32
#define OP_MAX 0u  // StreamReduce op CSR: MAX=0 (compare)
#define OP_ADD 1u  // StreamReduce op CSR: ADD=1 (fused FMA)
#define RED_TAP \
    0x100u  // StreamReduce op CSR bit[8]: pass the row through + trailing
            // scalar
#define EW_ADD 1u  // StreamElementwise fused-FMA op CSR: 0=MUL, 1=ADD
#define EW_MUL 0u
#define ACT_EXP 1u  // StreamMap func CSR bits[1:0]=1: EXP (out = exp(a*x + b))
#define FP32_ONE 0x3F800000u

static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

static uint32_t run_task(void) {
    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    return snax_xdma_last_task_cycle();
}

// Splat one FP16 value across the 32 lanes of a beat, as 16 word stores.
static inline void splat_beat(uint8_t* beat, uint16_t h) {
    uint32_t w = ((uint32_t)h << 16) | (uint32_t)h;
    uint32_t* p = (uint32_t*)beat;
    for (uint32_t l = 0; l < FP16_PER_BEAT / 2; l++) p[l] = w;
}

int main() {
    int err = 0;
    if (snax_is_simd_core()) {
        uint32_t base = snrt_cluster_base_addrl();
        uint32_t rows = smf_rows;
        uint32_t d = smf_d;
        uint32_t beats = smf_beats;
        uint32_t row_bytes = beats * XDMA_BEAT_BYTES;
        uint32_t rows_bytes = rows * row_bytes;
        uint32_t scal_bytes = rows * XDMA_BEAT_BYTES;
        uint32_t all_beats = rows * beats;
        uint32_t pad_row_bytes =
            (beats + 1) * XDMA_BEAT_BYTES;  // + the trailing sum beat
        uint32_t pad_bytes = rows * pad_row_bytes;

        uint8_t* x_in = (uint8_t*)base;
        uint8_t* max_buf = x_in + rows_bytes;     // T1 out: splatted max beats
        uint8_t* sum_buf = max_buf + scal_bytes;  // A4 out: splatted sum beats
        uint8_t* inv_l1 = sum_buf + scal_bytes;   // [rows] 1/Sexp, L1 copy
        uint8_t* neg_beat =
            inv_l1 + XDMA_BEAT_BYTES;  // B0: -max[r], one beat per row
        uint8_t* rcp_beat =
            neg_beat + scal_bytes;  // B4: 1/Sexp[r], one beat per row
        // ORDER MATTERS. The elementwise interleave addresses operand 1 as
        // operand0_base + delta, and the AGU adds `delta` as an UNSIGNED stride
        // -- a broadcast buffer placed BELOW its data operand makes the delta
        // wrap and the reader walks off outside the TCDM (it returns X, the
        // task still "completes", and X lands in the result). So every
        // broadcast buffer must sit ABOVE every data operand it is interleaved
        // with: bc_a above x_in AND above expb_a; bc_b above x_in AND above
        // expb_pad.
        uint8_t* xs_buf = rcp_beat + scal_bytes;  // x - max (A2 / B2 out)
        uint8_t* expb_a = xs_buf + rows_bytes;    // A3 out
        uint8_t* bc_a = expb_a + rows_bytes;  // A: -max bcast, then 1/sum bcast
        uint8_t* out_a = bc_a + rows_bytes;   // A6 out
        uint8_t* q_sep = out_a + rows_bytes;  // A7 out (int8)
        uint8_t* expb_pad =
            q_sep + rows_bytes / 2;            // B3 out: [rows, beats+1] beats
        uint8_t* bc_b = expb_pad + pad_bytes;  // B: -max bcast (flat), then
                                               //    1/sum bcast (padded stride)
        uint8_t* out_b = bc_b + pad_bytes;     // B6 out (fp16)
        uint8_t* q_fused = out_b + rows_bytes;  // B6 out (int8, chained quant)

        printf("[SmFold] rows=%u D=%u beats=%u\n", rows, d, beats);

        snax_stage_1d(x_in, smf_input, rows_bytes);
        snax_stage_1d(inv_l1, smf_inv_sum, rows * sizeof(uint16_t));

        // ---- AGU shapes ----
        uint32_t red_str[2] = {XDMA_BEAT_BYTES,
                               row_bytes};  // reduce reader {beats, rows}
        uint32_t red_bnd[2] = {beats, rows};
        uint32_t w_rows_str[1] = {XDMA_BEAT_BYTES};
        uint32_t w_rows_bnd[1] = {rows};
        uint32_t flat_str[1] = {
            XDMA_BEAT_BYTES};  // 1D over the whole [rows,D] tensor
        uint32_t flat_bnd[1] = {all_beats};
        uint32_t q_bnd[1] = {all_beats /
                             2};  // Fp16ToInt8: 2 FP16 beats -> 1 INT8
        uint32_t bc_str[2] = {0, XDMA_BEAT_BYTES};  // BROADCAST: inner stride 0
        uint32_t bc_bnd[2] = {beats, rows};
        uint32_t ew_bnd[2] = {2, all_beats};  // {operand, beat} interleave
        uint32_t ew_a_add_str[2] = {(uint32_t)(bc_a - x_in), XDMA_BEAT_BYTES};
        uint32_t ew_a_mul_str[2] = {(uint32_t)(bc_a - expb_a), XDMA_BEAT_BYTES};
        uint32_t ew_b_add_str[2] = {(uint32_t)(bc_b - x_in), XDMA_BEAT_BYTES};
        // B3 writer: beats passthrough beats + 1 trailing sum beat, per row.
        uint32_t tap_w_str[2] = {XDMA_BEAT_BYTES, pad_row_bytes};
        uint32_t tap_w_bnd[2] = {beats + 1, rows};
        // B5 writer: the bcast lands at the PADDED row stride (only the beats
        // data slots), so expb_pad and bc_b share a row stride and the B6
        // interleave delta is constant.
        uint32_t padw_str[2] = {XDMA_BEAT_BYTES, pad_row_bytes};
        uint32_t padw_bnd[2] = {beats, rows};
        // B6 reader: 3D {operand, beat, row} over the padded rows -- the
        // trailing sum beat of each row is simply never addressed.
        uint32_t ew_b_mul_str[3] = {(uint32_t)(bc_b - expb_pad),
                                    XDMA_BEAT_BYTES, pad_row_bytes};
        uint32_t ew_b_mul_bnd[3] = {2, beats, rows};

        // Guard the unsigned-delta rule above: an interleave whose operand-1
        // region sits below its operand-0 region silently reads outside the
        // TCDM.
        if (bc_a <= x_in || bc_a <= expb_a || bc_b <= x_in ||
            bc_b <= expb_pad) {
            printf("[SmFold] FAIL: interleave delta would be negative\n");
            return 1;
        }

        uint32_t csr_max[2] = {beats, OP_MAX};
        uint32_t csr_add[2] = {2u, EW_ADD};
        uint32_t csr_exp[3] = {FP32_ONE /*a=1.0*/, 0u /*b=0*/, ACT_EXP};
        uint32_t csr_sum[2] = {beats, OP_ADD};
        uint32_t csr_sum_tap[2] = {beats, OP_ADD | RED_TAP};
        uint32_t csr_mul[2] = {2u, EW_MUL};
        uint32_t csr_q[1] = {smf_inv_scale};

        uint16_t* mx = (uint16_t*)max_buf;
        uint16_t* inv = (uint16_t*)inv_l1;
        uint16_t* bca = (uint16_t*)bc_a;

        uint32_t t_max = 0, t_negloop = 0, t_sub = 0, t_exp = 0, t_sum = 0;
        uint32_t t_rcploop = 0, t_norm = 0, t_qsep = 0;
        uint32_t t_neg0 = 0, t_bc1 = 0, t_sub_b = 0, t_expsum = 0, t_rcp0 = 0;
        uint32_t t_bc2 = 0, t_norm_b = 0, t_normq = 0;
        uint32_t d_max = 0, d_sub = 0, d_exp = 0, d_sum = 0, d_norm = 0,
                 d_qsep = 0;
        uint32_t d_expsum = 0, d_bc1 = 0, d_normq = 0;
        int ok = 1;

        for (int it = 0; it < 2; it++) {
            uint32_t t0, t1;

            // ================= T1 (shared): per-row max =================
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_max) ==
                   0);
            ok &= (snax_xdma_memcpy_nd_fast(
                       x_in, max_buf, 8, 8, 2, red_str, red_bnd, 1, w_rows_str,
                       w_rows_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            d_max = run_task();
            snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);
            t1 = snrt_mcycle();
            t_max = t1 - t0;

            // ================= PATH A: deployed =================
            // A1: DM-core broadcast of -max[r] (FP16 sign flip) -- rows*D
            // stores.
            t0 = snrt_mcycle();
            for (uint32_t r = 0; r < rows; r++) {
                uint16_t neg = (uint16_t)(mx[r * FP16_PER_BEAT] ^ 0x8000u);
                for (uint32_t c = 0; c < d; c++) bca[r * d + c] = neg;
            }
            t1 = snrt_mcycle();
            t_negloop = t1 - t0;

            // A2: xs = x + (-max)
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE,
                                            csr_add) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(
                       x_in, xs_buf, 8, 8, 2, ew_a_add_str, ew_bnd, 1, flat_str,
                       flat_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            d_sub = run_task();
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);
            t1 = snrt_mcycle();
            t_sub = t1 - t0;

            // A3: expb = exp(xs)
            t0 = snrt_mcycle();
            ok &=
                (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_exp) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(
                       xs_buf, expb_a, 8, 8, 1, flat_str, flat_bnd, 1, flat_str,
                       flat_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            d_exp = run_task();
            snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);
            t1 = snrt_mcycle();
            t_exp = t1 - t0;

            // A4: per-row Sexp -- a second full read of expb.
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_sum) ==
                   0);
            ok &=
                (snax_xdma_memcpy_nd_fast(
                     expb_a, sum_buf, 8, 8, 2, red_str, red_bnd, 1, w_rows_str,
                     w_rows_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            d_sum = run_task();
            snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);
            t1 = snrt_mcycle();
            t_sum = t1 - t0;

            // A5: DM-core broadcast of 1/Sexp[r] -- rows*D stores.
            t0 = snrt_mcycle();
            for (uint32_t r = 0; r < rows; r++)
                for (uint32_t c = 0; c < d; c++) bca[r * d + c] = inv[r];
            t1 = snrt_mcycle();
            t_rcploop = t1 - t0;

            // A6: out = expb * recip
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE,
                                            csr_mul) == 0);
            ok &=
                (snax_xdma_memcpy_nd_fast(
                     expb_a, out_a, 8, 8, 2, ew_a_mul_str, ew_bnd, 1, flat_str,
                     flat_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            d_norm = run_task();
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);
            t1 = snrt_mcycle();
            t_norm = t1 - t0;

            // A7: separate quantize pass -- re-reads the whole FP16 result.
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_FP16TOINT8, csr_q) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(
                       out_a, q_sep, 8, 8, 1, flat_str, flat_bnd, 1, flat_str,
                       q_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            d_qsep = run_task();
            snax_xdma_disable_src_ext(READER_EXT_FP16TOINT8);
            t1 = snrt_mcycle();
            t_qsep = t1 - t0;

            // ================= PATH B: software-only fold =================
            // B0: sign-flip the already-SPLATTED max beat into -max, one beat
            // per row.
            t0 = snrt_mcycle();
            for (uint32_t r = 0; r < rows; r++)
                splat_beat(neg_beat + r * XDMA_BEAT_BYTES,
                           (uint16_t)(mx[r * FP16_PER_BEAT] ^ 0x8000u));
            t1 = snrt_mcycle();
            t_neg0 = t1 - t0;

            // B1: xDMA broadcast of -max (stride-0 reader), flat [rows,D].
            t0 = snrt_mcycle();
            ok &= (snax_xdma_memcpy_nd_fast(
                       neg_beat, bc_b, 8, 8, 2, bc_str, bc_bnd, 1, flat_str,
                       flat_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            d_bc1 = run_task();
            t1 = snrt_mcycle();
            t_bc1 = t1 - t0;

            // B2: xs = x + (-max)
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE,
                                            csr_add) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(
                       x_in, xs_buf, 8, 8, 2, ew_b_add_str, ew_bnd, 1, flat_str,
                       flat_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            (void)run_task();
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);
            t1 = snrt_mcycle();
            t_sub_b = t1 - t0;

            // B3: expb + Sexp in ONE task -- StreamMap(EXP) chained into
            // StreamReduce(ADD, TAP). Writes beats+1 beats per row (padded
            // rows).
            t0 = snrt_mcycle();
            ok &=
                (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_exp) == 0);
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE,
                                            csr_sum_tap) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(xs_buf, expb_pad, 8, 8, 1, flat_str,
                                            flat_bnd, 2, tap_w_str, tap_w_bnd,
                                            0xFFFFFFFF, 0xFFFFFFFF,
                                            0xFFFFFFFF) == 0);
            d_expsum = run_task();
            snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);
            snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);
            t1 = snrt_mcycle();
            t_expsum = t1 - t0;

            // B4: splat 1/Sexp[r], one beat per row.
            t0 = snrt_mcycle();
            for (uint32_t r = 0; r < rows; r++)
                splat_beat(rcp_beat + r * XDMA_BEAT_BYTES, inv[r]);
            t1 = snrt_mcycle();
            t_rcp0 = t1 - t0;

            // B5: xDMA broadcast of 1/Sexp at the PADDED row stride.
            t0 = snrt_mcycle();
            ok &= (snax_xdma_memcpy_nd_fast(
                       rcp_beat, bc_b, 8, 8, 2, bc_str, bc_bnd, 2, padw_str,
                       padw_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            (void)run_task();
            t1 = snrt_mcycle();
            t_bc2 = t1 - t0;

            // B6: out = expb * recip, with Fp16ToInt8 CHAINED (int8 out, no
            // re-read).
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE,
                                            csr_mul) == 0);
            ok &= (snax_xdma_enable_src_ext(READER_EXT_FP16TOINT8, csr_q) == 0);
            ok &=
                (snax_xdma_memcpy_nd_fast(
                     expb_pad, q_fused, 8, 8, 3, ew_b_mul_str, ew_b_mul_bnd, 1,
                     flat_str, q_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            d_normq = run_task();
            snax_xdma_disable_src_ext(READER_EXT_FP16TOINT8);
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);
            t1 = snrt_mcycle();
            t_normq = t1 - t0;

            // B6': the same norm WITHOUT the chained quant -- path B's FP16-out
            // variant, and the buffer that lets out_b be compared to out_a
            // bit-exactly.
            t0 = snrt_mcycle();
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMELEMENTWISE,
                                            csr_mul) == 0);
            ok &= (snax_xdma_memcpy_nd_fast(expb_pad, out_b, 8, 8, 3,
                                            ew_b_mul_str, ew_b_mul_bnd, 1,
                                            flat_str, flat_bnd, 0xFFFFFFFF,
                                            0xFFFFFFFF, 0xFFFFFFFF) == 0);
            (void)run_task();
            snax_xdma_disable_src_ext(READER_EXT_STREAMELEMENTWISE);
            t1 = snrt_mcycle();
            t_norm_b = t1 - t0;
        }

        if (!ok) {
            printf("[SmFold] xDMA task setup FAILED\n");
            return 1;
        }

        uint32_t a_fp16 =
            t_max + t_negloop + t_sub + t_exp + t_sum + t_rcploop + t_norm;
        uint32_t b_fp16 = t_max + t_neg0 + t_bc1 + t_sub_b + t_expsum + t_rcp0 +
                          t_bc2 + t_norm_b;
        uint32_t a_int8 = a_fp16 + t_qsep;
        uint32_t b_int8 = b_fp16 - t_norm_b + t_normq;

        printf(
            "[SmFold] --- stage cycles (warm; wall = CSR cfg + start + wait) "
            "---\n");
        printf("[SmFold] T1 reduce(MAX)         wall=%u  datapath=%u\n", t_max,
               d_max);
        printf("[SmFold] A1 DM-core bcast -max  wall=%u  (rows*D=%u stores)\n",
               t_negloop, rows * d);
        printf("[SmFold] A2 EW(ADD)             wall=%u  datapath=%u\n", t_sub,
               d_sub);
        printf("[SmFold] A3 StreamMap(EXP)      wall=%u  datapath=%u\n", t_exp,
               d_exp);
        printf("[SmFold] A4 reduce(ADD)         wall=%u  datapath=%u\n", t_sum,
               d_sum);
        printf("[SmFold] A5 DM-core bcast 1/sum wall=%u  (rows*D=%u stores)\n",
               t_rcploop, rows * d);
        printf("[SmFold] A6 EW(MUL)             wall=%u  datapath=%u\n", t_norm,
               d_norm);
        printf("[SmFold] A7 quant (separate)    wall=%u  datapath=%u\n", t_qsep,
               d_qsep);
        printf("[SmFold] B0 DM-core splat -max  wall=%u  (rows*16=%u stores)\n",
               t_neg0, rows * 16);
        printf("[SmFold] B1 xDMA bcast (str0)   wall=%u  datapath=%u\n", t_bc1,
               d_bc1);
        printf("[SmFold] B2 EW(ADD)             wall=%u\n", t_sub_b);
        printf(
            "[SmFold] B3 EXP-||>reduce(TAP)  wall=%u  datapath=%u  (fuses "
            "A3+A4)\n",
            t_expsum, d_expsum);
        printf("[SmFold] B4 DM-core splat 1/sum wall=%u  (rows*16=%u stores)\n",
               t_rcp0, rows * 16);
        printf("[SmFold] B5 xDMA bcast (padded) wall=%u\n", t_bc2);
        printf("[SmFold] B6 EW(MUL)-||>quant    wall=%u  datapath=%u\n",
               t_normq, d_normq);
        printf("[SmFold] B6' EW(MUL) fp16 out   wall=%u\n", t_norm_b);
        printf("[SmFold] === fp16 out: A=%u  B=%u cycles ===\n", a_fp16,
               b_fp16);
        printf("[SmFold] === int8 out: A=%u  B=%u cycles ===\n", a_int8,
               b_int8);

        // ---- correctness ----
        uint16_t* oa = (uint16_t*)out_a;
        uint32_t mism = 0, worst = 0, checked = 0;
        for (uint32_t i = 0; i < rows * d; i++) {
            if ((smf_out_golden[i] & 0x7C00u) == 0) continue;
            checked++;
            uint32_t o = fp16_mono(oa[i]), g = fp16_mono(smf_out_golden[i]);
            uint32_t ulp = (o > g) ? (o - g) : (g - o);
            if (ulp > worst) worst = ulp;
            if (ulp > 4) {
                if (mism < 6)
                    printf(
                        "[SmFold] mismatch[%u] (row %u): got %04x golden "
                        "%04x\n",
                        i, i / d, oa[i], smf_out_golden[i]);
                mism++;
            }
        }
        printf("[SmFold] A vs golden: significant=%u/%u worst FP16 ULP=%u\n",
               checked, rows * d, worst);
        if (mism) {
            printf("[SmFold] FAIL: %u/%u beyond 4 ULP\n", mism, checked);
            err++;
        }

        // The TAP scalar (trailing beat of each padded row) must equal the
        // standalone reduce's per-row sum.
        uint32_t bad = 0;
        uint16_t* sm = (uint16_t*)sum_buf;
        for (uint32_t r = 0; r < rows; r++) {
            uint16_t tap = *(uint16_t*)(expb_pad + r * pad_row_bytes +
                                        beats * XDMA_BEAT_BYTES);
            if (tap != sm[r * FP16_PER_BEAT]) bad++;
        }
        printf("[SmFold] TAP sum vs reduce(ADD): %s (%u/%u rows differ)\n",
               bad ? "FAIL" : "bit-exact", bad, rows);
        if (bad) err++;

        // The whole B chain must reproduce A's FP16 result bit-exactly.
        uint16_t* ob = (uint16_t*)out_b;
        bad = 0;
        for (uint32_t i = 0; i < rows * d; i++)
            if (oa[i] != ob[i]) bad++;
        printf("[SmFold] out_b vs out_a: %s (%u/%u differ)\n",
               bad ? "FAIL" : "bit-exact", bad, rows * d);
        if (bad) err++;

        // Chaining the quant onto the norm pass must equal a separate quant
        // pass.
        int8_t* qs = (int8_t*)q_sep;
        int8_t* qf = (int8_t*)q_fused;
        bad = 0;
        for (uint32_t i = 0; i < rows * d; i++)
            if (qs[i] != qf[i]) bad++;
        printf("[SmFold] q_fused vs q_sep: %s (%u/%u differ)\n",
               bad ? "FAIL" : "bit-exact", bad, rows * d);
        if (bad) err++;

        printf(err ? "[SmFold] FAIL\n" : "[SmFold] PASS\n");
    }
    return err != 0;
}
