// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// RoPE on the four-engine cluster, and why it is one task rather than three.
//
// WHAT ROPE IS. The positional encoding every Llama-family attention layer applies to Q
// and K before the scores are computed. The head vector is read as a list of 2-D points,
// and point k is ROTATED by an angle that depends on both its index and the token's
// position:
//
//     out[2k]   = x[2k]*cos_k - x[2k+1]*sin_k
//     out[2k+1] = x[2k]*sin_k + x[2k+1]*cos_k
//
// (the interleaved adjacent-pair convention: pair k is the two ADJACENT elements
// x[2k], x[2k+1], not element k and element k+d/2).
//
// It is the cheapest of the SIMD kernels and the one that fits the machine worst, and
// those two facts have the same cause. There is no reduction and no per-row scalar, so
// none of the fold-and-broadcast machinery that dominates RMSNorm and softmax applies
// here at all. What RoPE has instead is a coupling between ADJACENT LANES -- and that is
// the one direction this datapath cannot address.
//
// ======================================================================================
// THE DATAFLOW -- WHAT THE BLOCK ACTUALLY SEES
// ======================================================================================
//
// The SIMD block is a stream machine. One reader sweeps an affine iteration space, the
// armed operators transform the stream in a FIXED order, one writer drains it:
//
//     read -> EW0 -> Map -> Reduce -> EW1 -> Fp16ToInt8 -> write
//
// The unit of that stream is a BEAT: 512 bits = 64 B = 32 FP16 LANES. Every operator is
// PER LANE: lane j of the output is a function of lane j of the inputs and of nothing
// else. Data crosses lanes in exactly one place in the whole block -- StreamReduce's
// cross-lane fold -- and that is a fold to a scalar, not a permutation.
//
// So the reader is where any rearrangement has to happen, and the reader is affine:
//
//     addr = base + o*operand_stride + b*64        (+ further nested axes)
//
// One stride per axis, whole beats at a time, plus a single `lane_stride` inside the beat.
// A pair swap -- lane 2k takes lane 2k+1's value and vice versa -- is not any stride: it
// is a permutation, and it alternates. NOTHING in the reader expresses it. That is the
// whole reason this kernel has a staging step, and it is why the arithmetic below is
// arranged the way it is.
//
// What the reader IS good at is presenting several operands per output beat. The operand
// axis walks whole buffers -- o = 0, 1, 2, 3 at a fixed spacing -- and the two elementwise
// slots, one on each side of the Map, consume them in pairs. RoPE is two products and
// their sum. That is a combine, then a combine. It fits exactly.
//
// ======================================================================================
// A WORKED EXAMPLE -- 4 pairs, 8 lanes to a beat
// ======================================================================================
//
// Real shapes are 32 lanes and N/32 beats; shrink to 8 lanes and the whole thing is one
// beat. The angles are the four right angles, so every number below is exact:
//
//     pair k      0        1        2        3
//     theta       0      90 deg   180 deg  270 deg
//     cos_k       1        0        -1        0
//     sin_k       0        1         0       -1
//
//     x    [  1    2  |  3    4  |  5    6  |  7    8  ]      lane index ->
//            pair 0      pair 1     pair 2     pair 3
//
// Applying the definition once per pair, by hand:
//
//     pair 0   ( 1, 2)  rotated by 0     ->  ( 1,  2)     unchanged
//     pair 1   ( 3, 4)  rotated by 90    ->  (-4,  3)     (a,b) -> (-b, a)
//     pair 2   ( 5, 6)  rotated by 180   ->  (-5, -6)     (a,b) -> (-a,-b)
//     pair 3   ( 7, 8)  rotated by 270   ->  ( 8, -7)     (a,b) -> ( b,-a)
//
//     out  [  1    2  | -4    3  | -5   -6  |  8   -7  ]
//
// ---- HOW THE MACHINE GETS THERE -----------------------------------------------------
//
// Write the pair of formulas as one expression that is the SAME in both lanes of a pair,
// and the kernel falls out:
//
//     out[i] = x[i] * cos_of_my_pair  +  x[partner of i] * (signed sin of my pair)
//
// The only thing that differs between the two lanes of a pair is the SIGN on the sine
// term -- minus on the even lane, plus on the odd one -- and a sign is just a value, so
// it is folded into a table that is built once, offline, for free. Three operand tables,
// each the same shape as x:
//
//     lane:        0     1  |  2     3  |  4     5  |  6     7
//     x        [   1     2  |  3     4  |  5     6  |  7     8  ]
//     cos_full [   1     1  |  0     0  | -1    -1  |  0     0  ]   c_k, duplicated
//     xswap    [   2     1  |  4     3  |  6     5  |  8     7  ]   the partner lane
//     sin_sgn  [   0     0  | -1    +1  |  0     0  | +1    -1  ]   -s_k, +s_k
//
// and then the rotation is two multiplies and an add, per lane, with nothing crossing
// lanes:
//
//     x        * cos_full  =  [  1     2  |  0     0  | -5    -6  |  0     0  ]
//     xswap    * sin_sgn   =  [  0     0  | -4    +3  |  0     0  | +8    -7  ]
//     sum                  =  [  1     2  | -4     3  | -5    -6  |  8    -7  ]
//
// which is the hand-computed answer above, lane for lane.
//
// Note what each table cost. cos_full and sin_sgn are built offline and carry the pairing
// and the sign for nothing. xswap is the one that cannot be: it is x itself, permuted,
// and the permutation is the adjacent swap the reader has no stride for.
//
// ======================================================================================
// THE TWO PATHS THIS APP RUNS
// ======================================================================================
//
//   THREE PASSES -- the deployed form, one elementwise op per task.
//     0  iDMA         xswap = [x1,x0,x3,x2,...]    two strided 2-byte copies, on the hart
//                                                  that owns the iDMA
//     1  ew(MUL)      (x, cos_full)   -> tmp1
//     2  ew(MUL)      (xswap, sin_sgn) -> tmp2
//     3  ew(ADD)      (tmp1, tmp2)    -> out
//
//     Each pass reads its two operands interleaved, so it moves 2 beats to write 1: six
//     operand beats per output beat over the three passes, plus two whole intermediate
//     tiles written to TCDM and read straight back.
//
//   ONE TASK -- the same arithmetic, using both elementwise slots.
//     0  iDMA         xswap, unchanged -- see below
//     1  read   [ x_b , cos_b , xswap_b , sin_b ]    the operand axis, FOUR deep
//        EW0    MUL, operandCount 2  ->  [ x*cos , xswap*sin ]    pairs consecutive beats
//        EW1    ADD, operandCount 2  ->  [ x*cos + xswap*sin ]    pairs those
//        write  one beat
//
//     Four operand beats per output beat instead of six, and the two intermediates never
//     exist. The only requirement is that the four operands be EQUALLY SPACED, because
//     the reader adds one stride per axis -- and the buffers were already laid out that
//     way, x / cos / xswap / sin one row apart in the order the pairing needs, so the
//     fusion costs no restaging at all.
//
//     The pre-map elementwise slot was added for attention's quantise(exp(S - m)), which
//     needs a combine BEFORE the map. RoPE needs one on EACH side, and is the second
//     kernel to find a use for it. Nothing here is RoPE-specific: any a*b + c*d over four
//     equally spaced operands is one task.
//
// MEASURED, warm, Verilator, snax_split_cluster, N=256:
//
//     3 passes   datapath 207 cc  (69 + 69 + 69)      wall 1,184 cc
//     1 task     datapath 131 cc                      wall   304 cc
//
// Two numbers because they measure two different things. The DATAPATH figure is the
// fusion alone -- 36% less, from 6 operand beats per output beat down to 4 and from two
// intermediate tiles down to none. The WALL figure also folds in the orchestration: the
// fused path programs its descriptor with snax_simd_program_fast and constant-address
// operator writes, where the 3-pass path pays snax_simd_memcpy_nd_fast plus three
// enable/disable round trips. So 74% is the two effects together, not the fusion alone.
//
// The fused result is BIT-EXACT against the 3-pass result, and it has to be: fusing
// widens nothing. EW0's product is narrowed to FP16 on its way to EW1 exactly as tmp1 and
// tmp2 were narrowed on their way to TCDM -- same operands, same units, same two
// roundings. Anything but bit-exact would be a wiring bug, so that is what the app checks.
//
// ======================================================================================
// WHAT IS NOT REMOVED, AND WHY -- THE SWAP
// ======================================================================================
//
// xswap is still staged by the iDMA (532 cc at N=256, on hart 3, off the SIMD block's
// critical path). It survives the fusion because it is a permutation of ADJACENT LANES,
// and the reader has no lane permutation: the operand and beat axes move whole beats, and
// `lane_stride` is one stride, not a shuffle.
//
// WOULD TRANSPOSING HELP, as it does for RMSNorm and softmax? No -- and the reason is
// worth stating, because the answer for those two was yes. There the problem was that a
// REDUCTION ran across lanes; transposing turned it into a reduction along beats, which
// the accumulators do for free. RoPE has no reduction. Its coupling is between the two
// halves of a pair, and transposing moves that coupling from adjacent LANES to adjacent
// BEATS -- which the operand axis can address. But the two outputs of a pair need the cos
// and sin tables EXCHANGED between them, and that exchange rides the same axis as the
// operand selection, so it needs a second stride on a shared axis. One materialisation is
// traded for another. The swap stays either way, without RTL.
//
// A fused StreamRoPE that rotated adjacent lanes inside the beat would remove it. That is
// the only part of this kernel still asking for hardware, and it is now the ONLY part:
// before this app, two of the three passes were asking too.

#include "data.h"
#include "snax-core-roles.h"
#include "snax-simd-lib.h"
#include "snrt.h"

// The extensions this kernel needs, and WHAT IT NEEDS THEM TO DO. An op/func CSR is a
// runtime select over the set the cfg elaborated, and selecting outside that set does not
// fault -- it returns another op's answer. So the gate names capabilities, not extensions;
// each _HAS_ macro implies its extension exists. See the note in snax-simd-lib.h.
//
// BOTH elementwise slots, and for different reasons: the 3-pass path runs everything on
// the post-map instance, while the fused pass puts the two products on the PRE-map one
// and their sum on the post-map one. A cluster with only one of them can still run the
// deployed form, but this app runs both, so it needs both.
#if !defined(SIMD_EXT_STREAMELEMENTWISE_0_HAS_MUL) || \
    !defined(SIMD_EXT_STREAMELEMENTWISE_1_HAS_MUL) || \
    !defined(SIMD_EXT_STREAMELEMENTWISE_1_HAS_ADD)
#error \
    "This cluster cannot run RoPE fused: it needs a PRE-map StreamElementwise MUL and a post-map one with MUL and ADD."
#endif

// Op selectors and SIMD_BEAT_BYTES come from snax-simd-lib.h, not from local #defines. A
// local copy of the op encoding compiles against any cluster, including one whose SIMD
// block never built that op -- which is exactly the silent-wrong-answer the capability
// macros above exist to stop.

// FP16 bits -> monotonic ordering key (handles signed outputs): adjacent FP16
// values map to adjacent keys, so |key(a)-key(b)| is the FP16-ULP distance even
// across zero.
static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

// ============================================================ the fused pass
//
// One task for the whole rotation. The reader sweeps FOUR operands per beat and the two
// elementwise slots consume them in pairs -- EW0 in front of the Map, EW1 behind it:
//
//   read   [ x_b , cos_b , xswap_b , sin_b ]      operand dim, 4 deep
//   EW0    MUL, operandCount 2  ->  [ x*cos , xswap*sin ]
//   EW1    ADD, operandCount 2  ->  [ x*cos + xswap*sin ]     <- the rotation
//   write  one beat
//
// The four operands must be EQUALLY SPACED, because the reader adds a single stride per
// axis: addr = base + o*operand_stride + b*BEAT. The buffers below are already laid out
// that way, one row_bytes apart in the order the pairing needs.
//
// Nothing new in the hardware -- the pre-map elementwise slot was added for attention's
// quantise(exp(S - m)), and RoPE is the second kernel that turns out to need a combine on
// each side of the Map.
// How long to let one task run before calling it hung. The whole 3-pass kernel measures
// ~2k cycles at N=256, so anything past this is not slow, it is stuck -- and a bounded wait
// is what lets the run say which pass died instead of producing no output at all.
#define ROPE_TASK_BUDGET 50000u

static uint32_t rope_fused(void *ops_base, void *dst, uint32_t beats,
                           uint32_t operand_stride) {
    snax_simd_shape_t in, out;
    snax_simd_shape_2d(&in, ops_base, 4u, operand_stride, beats, SIMD_BEAT_BYTES);
    snax_simd_shape_flat(&out, dst, beats);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                                     (1u << SIMD_EXT_STREAMELEMENTWISE_1));
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 0, 2u);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_0_CSR + 1, SIMD_EW_MUL);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_1_CSR + 0, 2u);
    snax_write_simd_cfg_reg(SIMD_EXT_STREAMELEMENTWISE_1_CSR + 1, SIMD_EW_ADD);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    if (snax_simd_wait_all_timeout(ROPE_TASK_BUDGET)) return 0xFFFFFFFFu;
    // DISARM. This routine writes the enable mask ABSOLUTELY, while the 3-pass path below
    // uses snax_simd_enable_ext/disable_ext, which read-modify-WRITE it. Mixing the two and
    // leaving EW0 set turns the next 2-operand task into an unintended two-stage chain:
    // EW0 folds its 2 beats into 1, EW1 then waits for a partner beat that no longer
    // exists, and the writer waits forever. That is not a hypothetical -- it is what the
    // watchdog above caught, reported against the innocent pass that ran afterwards.
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0u);
    return snax_simd_last_task_cycle();
}

int main() {
    int err = 0;
    // TCDM layout, derived on EVERY hart and not on the engine core
    // alone: the staging core below has to land the data where the
    // engine core will read it, and snrt_cluster_base_addrl() is the
    // same on every hart, so both derive it rather than communicate.
    uint32_t base = snrt_cluster_base_addrl();
    uint32_t beats = rope_beats;
    uint32_t row_bytes = beats * SIMD_BEAT_BYTES;

    // Layout: each pass's two operands are adjacent (row_bytes apart) so
    // the interleave stride = row_bytes. P1 {x,cos}->tmp1, P2
    // {xswap,sin}->tmp2, P3 {tmp1,tmp2}->out.
    uint8_t* x_in = (uint8_t*)base;
    uint8_t* cos_in = x_in + row_bytes;  // cos_full (P1 operand 1)
    uint8_t* xswap =
        cos_in + row_bytes;  // adjacent swap of x (P2 operand 0)
    uint8_t* sin_in = xswap + row_bytes;  // sin_signed (P2 operand 1)
    uint8_t* tmp1 = sin_in + row_bytes;   // x (.) cos       (P3 operand 0)
    uint8_t* tmp2 = tmp1 + row_bytes;     // xswap (.) sin   (P3 operand 1)
    uint8_t* out_buf = tmp2 + row_bytes;  // RoPE output, 3-pass path (FP16)
    uint8_t* out_fus = out_buf + row_bytes;  // RoPE output, ONE task

    // Stage L3 -> TCDM on the core that OWNS the iDMA. On a split cluster
    // that is a different hart from the engine block, which carries no DMA
    // ISA at all -- a dm* instruction there traps. Where the two roles share
    // one hart this reads exactly the same.
    if (snax_is_idma_core()) {
        snrt_dma_start_1d(x_in, rope_x, row_bytes);
        snrt_dma_start_1d(cos_in, rope_cos, row_bytes);
        snrt_dma_start_1d(sin_in, rope_sin, row_bytes);
        snrt_dma_wait_all();

        // The adjacent-halfword swap xswap[2k]=x[2k+1], xswap[2k+1]=x[2k], as
        // two strided 2-byte copies (odd->even, even->odd). This is iDMA work,
        // not SIMD work, so it belongs on this hart with the rest of the input
        // prep -- and it is timed and reported HERE, because a cycle count read
        // on the engine core would not be measuring the core that did it.
        uint32_t ts0 = snrt_mcycle();
        uint32_t pairs = beats * 16;  // N/2
        snrt_dma_start_2d(xswap, x_in + 2, 2, 4, 4, pairs);
        snrt_dma_start_2d(xswap + 2, x_in, 2, 4, 4, pairs);
        snrt_dma_wait_all();
        printf("[RoPE] cycles: swap=%u\n", snrt_mcycle() - ts0);
    }
    // Unconditional: the hardware barrier counts every core in the cluster,
    // so a hart that skipped it would hang the ones that did not.
    snrt_cluster_hw_barrier();

    if (snax_is_simd_core()) {
        printf("[RoPE] N=%u beats=%u\n", rope_n, beats);

        // All passes share this interleaved [2,beats] src shape and 1D dst
        // shape; only the bases change.
        uint32_t src_str[2] = {
            row_bytes,
            SIMD_BEAT_BYTES};  // inner = operand jump, outer = beat step
        uint32_t src_bnd[2] = {2, beats};  // 2 operands, beats beats
        uint32_t dst_str[1] = {SIMD_BEAT_BYTES};
        uint32_t dst_bnd[1] = {beats};

        uint32_t c1 = 0, c2 = 0, c3 = 0, lat_cold = 0, lat_warm = 0;
        uint32_t cf = 0, lat_fused = 0;
        int ok = 1;
        // Run twice: iter 0 = cold (first call, icache cold); iter 1 = warm /
        // steady-state. Measures the 3-pass xDMA offload only (swap is one-time
        // staging above), comparable to swiglu's warm/cold.
        for (int iter = 0; iter < 2; iter++) {
            uint32_t t0 = snrt_mcycle();

            uint32_t csr_mul[2] = {2u /*operandCount*/, SIMD_EW_MUL};
            uint32_t csr_add[2] = {2u /*operandCount*/, SIMD_EW_ADD};

            // P1: tmp1 = x (.) cos.
            ok &= (snax_simd_enable_ext(SIMD_EXT_STREAMELEMENTWISE_1,
                                            csr_mul) == 0);
            ok &= (snax_simd_memcpy_nd_fast(
                       x_in, tmp1, 8, 8, 2, src_str, src_bnd, 1, dst_str,
                       dst_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            {
                uint32_t task_id = snax_simd_start();
                if (snax_simd_wait_all_timeout(ROPE_TASK_BUDGET)) {
                    printf("[RoPE] HUNG: P1 {x,cos} did not retire in %u cycles\n",
                           ROPE_TASK_BUDGET);
                    return 1;
                }
                (void)task_id;
                c1 = snax_simd_last_task_cycle();
            }
            snax_simd_disable_ext(SIMD_EXT_STREAMELEMENTWISE_1);

            // P2: tmp2 = xswap (.) sin_signed.
            ok &= (snax_simd_enable_ext(SIMD_EXT_STREAMELEMENTWISE_1,
                                            csr_mul) == 0);
            ok &= (snax_simd_memcpy_nd_fast(
                       xswap, tmp2, 8, 8, 2, src_str, src_bnd, 1, dst_str,
                       dst_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            {
                uint32_t task_id = snax_simd_start();
                if (snax_simd_wait_all_timeout(ROPE_TASK_BUDGET)) {
                    printf("[RoPE] HUNG: P2 {xswap,sin} did not retire in %u cycles\n",
                           ROPE_TASK_BUDGET);
                    return 1;
                }
                (void)task_id;
                c2 = snax_simd_last_task_cycle();
            }
            snax_simd_disable_ext(SIMD_EXT_STREAMELEMENTWISE_1);

            // P3: out = tmp1 (+) tmp2.
            ok &= (snax_simd_enable_ext(SIMD_EXT_STREAMELEMENTWISE_1,
                                            csr_add) == 0);
            ok &= (snax_simd_memcpy_nd_fast(
                       tmp1, out_buf, 8, 8, 2, src_str, src_bnd, 1, dst_str,
                       dst_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
            {
                uint32_t task_id = snax_simd_start();
                if (snax_simd_wait_all_timeout(ROPE_TASK_BUDGET)) {
                    printf("[RoPE] HUNG: P3 {tmp1,tmp2} did not retire in %u cycles\n",
                           ROPE_TASK_BUDGET);
                    return 1;
                }
                (void)task_id;
                c3 = snax_simd_last_task_cycle();
            }
            snax_simd_disable_ext(SIMD_EXT_STREAMELEMENTWISE_1);

            uint32_t t1 = snrt_mcycle();
            if (iter == 0)
                lat_cold = t1 - t0;
            else
                lat_warm = t1 - t0;

            // The same rotation as ONE task. x_in, cos_in, xswap and sin_in are already
            // row_bytes apart and in the order the pairing needs, so the fused reader is
            // the 3-pass reader with its operand axis 4 deep instead of 2.
            uint32_t f0 = snrt_mcycle();
            cf = rope_fused(x_in, out_fus, beats, row_bytes);
            uint32_t f1 = snrt_mcycle();
            lat_fused = f1 - f0;
            if (cf == 0xFFFFFFFFu) {
                printf("[RoPE] HUNG: EW0(MUL,2) -||> EW1(ADD,2) did not retire in %u "
                       "cycles\n",
                       ROPE_TASK_BUDGET);
                return 1;
            }
        }

        if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu ||
            c3 == 0xFFFFFFFFu) {
            printf("[RoPE] xDMA task setup failed\n");
            return 1;
        }
        printf("[RoPE] cycles: p1=%u p2=%u p3=%u xdma_total=%u\n",
               c1, c2, c3, c1 + c2 + c3);
        printf("[RoPE] full latency: cold=%u warm=%u cycles\n", lat_cold,
               lat_warm);
        printf("[RoPE] FUSED (one task): datapath=%u  wall=%u\n", cf, lat_fused);
        printf("[RoPE] === datapath %u -> %u (%u%% less) | wall %u -> %u (%u%% less) ===\n",
               c1 + c2 + c3, cf,
               (c1 + c2 + c3) ? (100u * ((c1 + c2 + c3) - cf)) / (c1 + c2 + c3) : 0u,
               lat_warm, lat_fused,
               lat_warm ? (100u * (lat_warm - lat_fused)) / lat_warm : 0u);

        // Verify out == interleaved RoPE(x). Integer FP16-ULP check on
        // significant (signed) outputs; skip the negligible subnormal/zero
        // golden (FTZ tail). DM core has no FPU -> integer compare.
        uint16_t* out_h = (uint16_t*)out_buf;
        uint32_t mism = 0, worst = 0, checked = 0;
        for (uint32_t i = 0; i < rope_n; i++) {
            if ((rope_golden[i] & 0x7C00u) == 0)
                continue;  // skip subnormal/zero golden
            checked++;
            uint32_t o = fp16_mono(out_h[i]), g = fp16_mono(rope_golden[i]);
            uint32_t ulp = (o > g) ? (o - g) : (g - o);
            if (ulp > worst) worst = ulp;
            if (ulp > 4) {
                if (mism < 6)
                    printf(
                        "[RoPE] mismatch[%u]: got %04x golden %04x (%u ulp)\n",
                        i, out_h[i], rope_golden[i], ulp);
                mism++;
            }
        }
        printf("[RoPE] significant=%u/%u worst FP16 ULP=%u\n", checked, rope_n,
               worst);
        if (mism == 0) {
            printf("[RoPE] PASS (%u significant elements, <=4 ULP)\n", checked);
        } else {
            printf("[RoPE] FAIL (%u/%u significant beyond 4 ULP)\n", mism,
                   checked);
            err++;
        }

        // The fused task must reproduce the 3-pass result BIT FOR BIT, and it is worth
        // saying why that is the right bar rather than an ULP tolerance. Fusing does not
        // widen anything: EW0's product is narrowed to FP16 on its way to EW1, exactly as
        // tmp1/tmp2 were narrowed on their way to TCDM. Same operands, same two roundings,
        // same order -- so any difference at all is a wiring bug, not arithmetic.
        uint16_t* fus_h = (uint16_t*)out_fus;
        uint32_t fbad = 0;
        for (uint32_t i = 0; i < rope_n; i++)
            if (fus_h[i] != out_h[i]) {
                if (fbad < 4)
                    printf("[RoPE] fused[%u]: got %04x 3-pass %04x\n", i, fus_h[i],
                           out_h[i]);
                fbad++;
            }
        printf("[RoPE] fused vs 3-pass: %s (%u/%u differ)\n",
               fbad ? "FAIL" : "bit-exact", fbad, rope_n);
        if (fbad) err++;
    }
    return err != 0;
}
