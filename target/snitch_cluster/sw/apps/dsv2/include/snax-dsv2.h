// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Kernels shared by the snax-dsv2-* apps (DeepSeek-V2-Lite layer 1 on snax_split_cluster).
//
// Everything here is always_inline with compile-time-constant CSR addresses: out of line, a
// csrw_ss to a computed address is a jump-table load from DRAM per write.
//
// ======================================================================================
// THE GROUPED GEMV
// ======================================================================================
//
// One GEMM task runs `groups` GEMVs of depth K = 4 kt and width 16 nb:
//
//     y_g (1 x 16 nb, FP16) = RNE(x_g . W_g * 2^-k)        g = 0 .. groups-1
//
//   A  x_g in row 0 of a 16-row operand (16 x 4 blocks), group g at a + g * a_step. a_step = 0
//      when the groups share one x -- the chunks of one weight matrix -- and 16 K when each has
//      its own -- the absorbed per-head GEMVs. Only row 0 is read (the A reader's channel mask):
//      the quantiser writes row 0 only (dsv2_quant_a); the datagens fill all 16 rows
//      (layout.gemv_a_rep).
//   B  W_g in B-layout (4 x 16 blocks), group g at b + g * K * 16 nb: the groups' blocks are
//      consecutive.
//   D  y_g at d + g * slot, slot = 32 nb bytes: FP16, contiguous.
//
// THE ARRAY SHAPE (DSV2_GEMV, sw/apps/dsv2/dsv2.mk). VersaCore runs one of two unrollings per
// task on this cluster, both with Ku = 4, so both read the layouts above:
//
//   1  (1, 4, 32)   one row, two column blocks a pass: 128 weight bytes, B on 16 channels
//   0  (16, 4, 16)  the 16-row GEMM shape, row 0 kept (below)
//
// A GEMV does one multiply-add per weight byte, so the shape's weight bytes per pass are its
// rate. Shape 1 takes the column blocks in pairs. A GEMV of an odd nb runs it at half width, one
// block a pass: B's channels 8..15 masked, so the upper 16 columns compute zero, and the D mask
// writing the lower 16 values alone. Its output block is one D beat, which the converter's
// single-beat mode turns into FP16 in place.
//
// SHAPE 0 computes 16 rows; row 0 of the output is y_g. The D port writes the 16 x 16 nb tile
// PACKED with a row pitch of one slot, so row r of group g lands in slot g + r: row 0 is y_g,
// and rows 1..15 land in the slots of LATER groups, which write their own row 0 there
// afterwards (groups run in order within a task, tasks retire in order, and the next group's
// first write is kt array passes later). When the last group retires, the slots hold y_0,
// y_1, ... contiguously, followed by 15 slots of spill. The spill is why an output buffer needs
// DSV2_SPILL(nb) bytes past its end.
//
// C is read with every channel masked: a disabled channel presents zero and issues no TCDM
// request -- the fresh accumulator every GEMV wants. take_in_new_c must stay 1, or the C
// FIFO never drains.
//
// TWO TOKENS (dsv2_gemv_arm_ntok, then dsv2_gemv_two_tokens; one group per task) run shape 0:
// A holds token 0 in row 0 and token 1 in row 4 (dsv2_quant_a2), so the array computes both
// from one pass over the weights. The D port emits a block as four beats of four rows; the walk
// writes beat 0 (row 0 first) to four slots, beat 1 (row 4 first) to four slots r bytes
// further, and beats 2 and 3 one slot past beats 0 and 1, into slots the later groups
// overwrite. Each token's outputs are then contiguous with 4 slots of spill, token 1's r bytes
// after token 0's: r must cover a whole output plus DSV2_SPILL2(nb).
//
// WEIGHT WIDTH (DSV2_WBITS, sw/apps/dsv2/dsv2.mk: 8 or 4). An INT4 GEMV (the arms' w4) runs the
// same passes: B's converter (cfg: HasIntlowToInthighConverter) sign-extends the nibbles of the
// beat's low half into the INT8 beat the array reads. What changes is where B's bytes are:
//
//   W_g in the PAIRED layout, nibble-packed (util/layout.py to_b_pairs, pack_int4): per pair of
//   column blocks, per k, block n's 32 bytes then block n + 1's. Group g at b + g * K * 8 nb.
//   (1, 4, 32)   one pair a pass: 64 bytes on channels 0..7. nb must be even.
//   (16, 4, 16)  one block a pass: 32 bytes on channels 0..3, walked k, half of the pair, pair;
//                the three loops leave none for groups, so one group per task (two tokens).
//
// Every arm here writes the converter's enable, so an INT8 task never inherits an INT4 one's.

#pragma once

#include "snax-simd-lib.h"
#include "snax-tcdm-priority.h"
#include "snax-versacore-to-lib.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !VERSACORE_HAS_D_SHIFT
#error "the GEMM's D-port Int32ToFp16Converter has no power-of-two shift (build it with shift: 1)"
#endif

#define DSV2_MR 16  // VersaCore (Mu, Ku, Nu) = (16, 4, 16), array shape 0
#define DSV2_KU 4
#define DSV2_NU 16
#define DSV2_BEAT 64u
#define DSV2_SLOT(nb) (32u * (nb))                       // one group's outputs, FP16
#define DSV2_SPILL(nb) ((DSV2_MR - 1u) * DSV2_SLOT(nb))  // past the last group's slot
#define DSV2_SPILL2(nb) (4u * DSV2_SLOT(nb))            // past each token's last slot, two tokens
#define DSV2_SPIN_LIMIT 4000000u

// What an app checks (sw/apps/dsv2/dsv2.mk): 0, the final results only, after the kernel; 1, every
// intermediate stage as well.
#ifndef DSV2_STAGE_CHECKS
#define DSV2_STAGE_CHECKS 0
#endif

// How much of a tensor a check compares (sw/apps/dsv2/dsv2.mk): 0, every term; N, its first N
// terms only, for regressions. A check runs on one hart with one load in flight, so comparing a
// large tensor whole can take more simulated cycles than the kernel that made it.
#ifndef DSV2_CHECK_TERMS
#define DSV2_CHECK_TERMS 0
#endif

// The array shape a one-token GEMV runs (see the header): 1 or 0.
#ifndef DSV2_GEMV
#define DSV2_GEMV 1
#endif
#if DSV2_GEMV != 0 && DSV2_GEMV != 1
#error "DSV2_GEMV is the one-token GEMV's array shape: 1, (1, 4, 32), or 0, (16, 4, 16)"
#endif
#define DSV2_SHAPE_GEMM 0u  // (16, 4, 16)
#define DSV2_SHAPE_GEMV 1u  // (1, 4, 32)

// The weights' width (see the header): 8, or 4 through B's converter.
#ifndef DSV2_WBITS
#define DSV2_WBITS 8
#endif
#if DSV2_WBITS != 8 && DSV2_WBITS != 4
#error "DSV2_WBITS is the weights' width: 8 or 4"
#endif
#define DSV2_W4 (DSV2_WBITS == 4)
#if defined(DSV2_DATA_WBITS) && DSV2_DATA_WBITS != DSV2_WBITS
#error "data.h holds weights of another width than DSV2_WBITS: regenerate it (dsv2.mk)"
#endif
#if defined(READER_EXTENSION_1_CSR_BASE) && READER_EXTENSION_1_CSR_NUM != 1
#error "B's extension host must hold the INT4 converter alone: one enable register"
#endif
#if DSV2_W4 && !defined(READER_EXTENSION_1_CSR_BASE)
#error "INT4 weights need B's HasIntlowToInthighConverter (cfg/snax_split_cluster.hjson)"
#endif
#if DSV2_W4 && !DSV2_GEMV
#error "INT4 weights run the one-token GEMV in shape 1: shape 0 walks one group per task"
#endif
// Bytes of n weights stored at width w4 ? 4 : 8.
#define DSV2_WB(n, w4) ((w4) ? (n) / 2u : (n))

// B's INT4 converter on or off, for the tasks armed next.
__attribute__((always_inline)) static inline void dsv2_b_int4(uint32_t on) {
#ifdef READER_EXTENSION_1_CSR_BASE
    csrw_ss(READER_EXTENSION_1_CSR_BASE, on);
#else
    (void)on;
#endif
}

// The shape a one-token GEMV runs.
static inline uint32_t dsv2_gemv_shape(void) {
    return DSV2_GEMV ? DSV2_SHAPE_GEMV : DSV2_SHAPE_GEMM;
}

// Column blocks a one-token GEMV of nb blocks takes a pass: two when shape 1 can pair them.
static inline uint32_t dsv2_gemv_blocks_per_pass(uint32_t nb) {
    return DSV2_GEMV && !(nb & 1u) ? 2u : 1u;
}

// The TCDM arbitration policy for a whole run (snax-tcdm-priority.h, SNAX_TCDM_POLICY_*): 0, the
// hardware's own (every requester's urgency and the starvation guard), unless an app is tuned
// otherwise. DSV2_TCDM_GUARD: the guard's wait in cycles, 0 for the hardware's reset value.
#ifndef DSV2_TCDM_PRIO
#define DSV2_TCDM_PRIO 0
#endif
#ifndef DSV2_TCDM_GUARD
#define DSV2_TCDM_GUARD 0
#endif

static inline void dsv2_tcdm_prio_install(uint32_t policy) {
    snax_tcdm_policy_install(policy, DSV2_TCDM_GUARD);
}

// Per-kernel policies, installed by the GEMM hart as each of its phases starts; a class not
// named runs DSV2_TCDM_PRIO. The register is cluster-wide, so a phase's policy also covers the
// loads and SIMD tasks that overlap it.
#ifndef DSV2_TCDM_PRIO_STREAM  // GEMVs over streamed weights: W_DKV, W_Q, W_O
#define DSV2_TCDM_PRIO_STREAM DSV2_TCDM_PRIO
#endif
#ifndef DSV2_TCDM_PRIO_HEAD  // per-head GEMVs out of L1: W_UK, W_UV
#define DSV2_TCDM_PRIO_HEAD DSV2_TCDM_PRIO
#endif
#ifndef DSV2_TCDM_PRIO_ATTN  // attention: QK, the softmax, PV
#define DSV2_TCDM_PRIO_ATTN DSV2_TCDM_PRIO
#endif
#define DSV2_TCDM_PRIO_PER_KERNEL                                                     \
    (DSV2_TCDM_PRIO_STREAM != DSV2_TCDM_PRIO || DSV2_TCDM_PRIO_HEAD != DSV2_TCDM_PRIO || \
     DSV2_TCDM_PRIO_ATTN != DSV2_TCDM_PRIO)

// Arm the 16-row grouped GEMV (shape 0) for a run of tasks. k: the D port's RNE(acc * 2^-k).
// amask: the A channels holding the tokens' rows, two rows a channel -- 0x01 for row 0, 0x05 for
// rows 0 and 4. The other rows read zero. w4: INT4 weights (groups must be 1, nb even).
__attribute__((always_inline)) static inline void dsv2_gemm16_arm(uint32_t kt, uint32_t nb,
                                                                  uint32_t groups,
                                                                  uint32_t a_step, uint32_t k,
                                                                  uint32_t amask, uint32_t w4) {
    const uint32_t blk = DSV2_MR * DSV2_KU;  // one A or B block = 64 B
    const uint32_t slot = DSV2_SLOT(nb);
    // A -- k inner, n broadcast (the same x for every block of a group), then the group.
    csrw_ss(ENABLED_CHANNEL_READER_0, amask);
    csrw_ss(S_STRIDE_READER_0_0, 8);
    csrw_ss(T_BOUND_READER_0_0, kt);
    csrw_ss(T_STRIDE_READER_0_0, blk);
    csrw_ss(T_BOUND_READER_0_1, nb);
    csrw_ss(T_STRIDE_READER_0_1, 0);
    csrw_ss(T_BOUND_READER_0_2, groups);
    csrw_ss(T_STRIDE_READER_0_2, a_step);
    csrw_ss(T_BOUND_READER_0_3, 1);
    csrw_ss(T_STRIDE_READER_0_3, 0);
    csrw_ss(T_BOUND_READER_0_4, 1);
    csrw_ss(T_STRIDE_READER_0_4, 0);
    csrw_ss(T_BOUND_READER_0_5, 1);
    csrw_ss(T_STRIDE_READER_0_5, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0);
    // B -- k inner, n, then the group; one 4 x 16 block a pass, channels 0..7. INT4: 32 bytes a
    // pass on channels 0..3 -- k inner (a pair's 64 B apart), the pair's two halves, the pairs.
    csrw_ss(ENABLED_CHANNEL_READER_1, w4 ? 0x0Fu : 0xFFu);
    csrw_ss(S_STRIDE_READER_1_0, 8);
    csrw_ss(S_STRIDE_READER_1_1, 0);
    csrw_ss(T_BOUND_READER_1_0, kt);
    csrw_ss(T_STRIDE_READER_1_0, blk);
    csrw_ss(T_BOUND_READER_1_1, w4 ? 2u : nb);
    csrw_ss(T_STRIDE_READER_1_1, w4 ? blk / 2u : kt * blk);
    csrw_ss(T_BOUND_READER_1_2, w4 ? nb / 2u : groups);
    csrw_ss(T_STRIDE_READER_1_2, w4 ? kt * blk : kt * blk * nb);
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0);
    dsv2_b_int4(w4);
    // C -- masked; the bounds still count the eight INT32 beats of every output block.
    csrw_ss(S_STRIDE_READER_WRITER_0_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_0_1, slot);
    csrw_ss(T_BOUND_READER_WRITER_0_0, DSV2_MR * DSV2_NU * 32u / 1024u);
    csrw_ss(T_STRIDE_READER_WRITER_0_0, 128u * nb);
    csrw_ss(T_BOUND_READER_WRITER_0_1, nb);
    csrw_ss(T_STRIDE_READER_WRITER_0_1, DSV2_NU * 2u);
    csrw_ss(T_BOUND_READER_WRITER_0_2, groups);
    csrw_ss(T_STRIDE_READER_WRITER_0_2, slot);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, 0);
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_0, 0);
    // D -- channel (i % 4) = 4 columns, (i / 4) = a row of the beat; packed, pitch = one slot.
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_1, 0xFFFFFFFFu);
    csrw_ss(S_STRIDE_READER_WRITER_1_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_1_1, slot);
    csrw_ss(T_BOUND_READER_WRITER_1_0, DSV2_MR / 4u);
    csrw_ss(T_STRIDE_READER_WRITER_1_0, 4u * slot);
    csrw_ss(T_BOUND_READER_WRITER_1_1, nb);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, DSV2_NU * 2u);
    csrw_ss(T_BOUND_READER_WRITER_1_2, groups);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, slot);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, 0);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 0, 1u);  // Int32ToFp16 on
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 0u);  // extra-loop 0: the 2:1 narrowing
    (void)set_versacore_d_shift(k);                        // RNE(acc * 2^-k)
    // The array: output-stationary, kt passes per block, nb blocks per group.
    csrw_ss(OVERWRITE_ACCUM, 1);
    csrw_ss(ACCUM_BOUND, kt);
    csrw_ss(OUTPUT_BOUND, nb * groups);
    csrw_ss(SUBTRACTIONS, 0);
    csrw_ss(ARRAY_SHAPE_CFG, DSV2_SHAPE_GEMM);
    csrw_ss(DATA_TYPE_CFG, 0);
}

// Arm a one-row GEMV (shape 1) for a run of tasks: w = 2 column blocks a pass, or 1 for an odd
// nb (half width). w4: INT4 weights, nb even.
__attribute__((always_inline)) static inline void dsv2_gemv1_arm(uint32_t kt, uint32_t nb,
                                                                 uint32_t groups,
                                                                 uint32_t a_step, uint32_t k,
                                                                 uint32_t w4) {
    const uint32_t blk = DSV2_MR * DSV2_KU;  // one A or B block = 64 B
    const uint32_t w = dsv2_gemv_blocks_per_pass(nb);
    const uint32_t nbp = nb / w;               // output blocks per group
    const uint32_t slot = DSV2_SLOT(nb);
    // A -- row 0 of each block, channel 0 (rows 0 and 1; the array takes row 0): k inner, n
    // broadcast, then the group.
    csrw_ss(ENABLED_CHANNEL_READER_0, 0x1u);
    csrw_ss(S_STRIDE_READER_0_0, 8);
    csrw_ss(T_BOUND_READER_0_0, kt);
    csrw_ss(T_STRIDE_READER_0_0, blk);
    csrw_ss(T_BOUND_READER_0_1, nbp);
    csrw_ss(T_STRIDE_READER_0_1, 0);
    csrw_ss(T_BOUND_READER_0_2, groups);
    csrw_ss(T_STRIDE_READER_0_2, a_step);
    csrw_ss(T_BOUND_READER_0_3, 1);
    csrw_ss(T_STRIDE_READER_0_3, 0);
    csrw_ss(T_BOUND_READER_0_4, 1);
    csrw_ss(T_STRIDE_READER_0_4, 0);
    csrw_ss(T_BOUND_READER_0_5, 1);
    csrw_ss(T_STRIDE_READER_0_5, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0);
    // B -- w column blocks at the same k a pass: channels 8..15 read the block kt * 64 bytes on,
    // or are masked at half width. INT4: the pair is one 64-byte run on channels 0..7, the next
    // pass's 64 bytes on, and the next pair's kt * 64.
    csrw_ss(ENABLED_CHANNEL_READER_1, w4 || w == 1u ? 0xFFu : 0xFFFFu);
    csrw_ss(S_STRIDE_READER_1_0, 8);
    csrw_ss(S_STRIDE_READER_1_1, w4 ? 0u : kt * blk);
    csrw_ss(T_BOUND_READER_1_0, kt);
    csrw_ss(T_STRIDE_READER_1_0, blk);
    csrw_ss(T_BOUND_READER_1_1, nbp);
    csrw_ss(T_STRIDE_READER_1_1, DSV2_WB(w * kt * blk, w4));
    csrw_ss(T_BOUND_READER_1_2, groups);
    csrw_ss(T_STRIDE_READER_1_2, DSV2_WB(kt * blk * nb, w4));
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0);
    dsv2_b_int4(w4);
    // C -- masked, one beat per output block.
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_0, 0);
    csrw_ss(S_STRIDE_READER_WRITER_0_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_0_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_0_0, 1);
    csrw_ss(T_STRIDE_READER_WRITER_0_0, 0);
    csrw_ss(T_BOUND_READER_WRITER_0_1, nbp);
    csrw_ss(T_STRIDE_READER_WRITER_0_1, 32u * w);
    csrw_ss(T_BOUND_READER_WRITER_0_2, groups);
    csrw_ss(T_STRIDE_READER_WRITER_0_2, slot);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, 0);
    // D -- one beat per block, its 16 w FP16 values on channels 0 .. 4w - 1: 32 w bytes a block,
    // contiguous.
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_1, w == 2u ? 0xFFu : 0xFu);
    csrw_ss(S_STRIDE_READER_WRITER_1_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_1_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_1_0, nbp);
    csrw_ss(T_STRIDE_READER_WRITER_1_0, 32u * w);
    csrw_ss(T_BOUND_READER_WRITER_1_1, groups);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, slot);
    csrw_ss(T_BOUND_READER_WRITER_1_2, 1);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, 0);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 0, 1u);  // Int32ToFp16 on
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 3u);  // single-beat: a block is one beat
    (void)set_versacore_d_shift(k);                        // RNE(acc * 2^-k)
    // The array: output-stationary, kt passes per block, nbp blocks per group.
    csrw_ss(OVERWRITE_ACCUM, 1);
    csrw_ss(ACCUM_BOUND, kt);
    csrw_ss(OUTPUT_BOUND, nbp * groups);
    csrw_ss(SUBTRACTIONS, 0);
    csrw_ss(ARRAY_SHAPE_CFG, DSV2_SHAPE_GEMV);
    csrw_ss(DATA_TYPE_CFG, 0);
}

// Arm a one-token GEMV of nb 16-column blocks, in the shape DSV2_GEMV picks. w4: INT4 weights.
__attribute__((always_inline)) static inline void dsv2_gemv_arm(uint32_t kt, uint32_t nb,
                                                                uint32_t groups,
                                                                uint32_t a_step, uint32_t k,
                                                                uint32_t w4) {
    if (DSV2_GEMV)
        dsv2_gemv1_arm(kt, nb, groups, a_step, k, w4);
    else
        dsv2_gemm16_arm(kt, nb, groups, a_step, k, 0x01u, w4);
}

// Arm a GEMV for ntok tokens: one token as dsv2_gemv_arm; two on shape 0, rows 0 and 4, after
// which the caller replaces the D walk with dsv2_gemv_two_tokens.
__attribute__((always_inline)) static inline void dsv2_gemv_arm_ntok(uint32_t kt, uint32_t nb,
                                                                     uint32_t groups,
                                                                     uint32_t a_step,
                                                                     uint32_t k,
                                                                     uint32_t ntok, uint32_t w4) {
    if (ntok == 2u)
        dsv2_gemm16_arm(kt, nb, groups, a_step, k, 0x05u, w4);
    else
        dsv2_gemv_arm(kt, nb, groups, a_step, k, w4);
}

// Two tokens per GEMV (see the header): after dsv2_gemv_arm_ntok(kt, nb, 1, 0, k, 2), replace
// the D walk.
__attribute__((always_inline)) static inline void dsv2_gemv_two_tokens(uint32_t nb, uint32_t r) {
    csrw_ss(T_BOUND_READER_WRITER_1_0, 2u);  // beats 0, 1: token 0, token 1, r apart
    csrw_ss(T_STRIDE_READER_WRITER_1_0, r);
    csrw_ss(T_BOUND_READER_WRITER_1_1, 2u);  // beats 2, 3: rows 8..15, one slot further
    csrw_ss(T_STRIDE_READER_WRITER_1_1, DSV2_SLOT(nb));
    csrw_ss(T_BOUND_READER_WRITER_1_2, nb);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, DSV2_NU * 2u);
}

// Point the armed descriptor at its operands and output, and submit. The descriptor is
// sticky, so only the bases change; a start is a write EVENT, so back-to-back fires queue.
__attribute__((always_inline)) static inline void dsv2_gemv_fire(const void *a, const void *b,
                                                                 void *d) {
    csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)a);
    csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)b);
    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, (uint32_t)d);
    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, (uint32_t)d);
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(GEMMX_START, 1);
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
}

// Wait for the array to retire task `id`: its retired-task counter is the only signal that
// separates two queued dispatches. Bounded; returns 1 on timeout.
__attribute__((always_inline)) static inline uint32_t dsv2_gemm_wait(uint32_t id) {
    uint32_t spins = 0;
    while ((int32_t)(csrr_ss(GEMMX_FINISHED_TASK) - id) < 0)
        if (++spins > DSV2_SPIN_LIMIT) return 1u;
    csrw_ss(GEMMX_START, 0);
    return 0u;
}

// ============================================================ handoffs
//
// Engines hand off through monotonic counters in L1, never a write-barrier-read: a hardware
// barrier does not wait for a posted TCDM store. Bounded; a timeout bumps *tmo.
static inline void dsv2_spin_ge(volatile uint32_t *w, uint32_t v, volatile uint32_t *tmo) {
    uint32_t n = 0;
    while (*w < v)
        if (++n > DSV2_SPIN_LIMIT) {
            (*tmo)++;
            return;
        }
}

// dsv2_spin_ge, adding the cycles it waited to *acc: an engine's time lost to one kind of wait.
static inline void dsv2_wait_acc(volatile uint32_t *w, uint32_t v, volatile uint32_t *tmo,
                                 uint32_t *acc) {
    if (*w >= v) return;
    const uint32_t t0 = snrt_mcycle();
    dsv2_spin_ge(w, v, tmo);
    *acc += snrt_mcycle() - t0;
}

// ============================================================ the iDMA's transfer counters
//
// dmstati status 0 is the frontend's completed id, which counts retired transfers; status 1 is
// the id the next transfer gets. The id a dmcpy returns cannot tell two transfers apart while
// the first still waits in the frontend's FIFO (both get the next id; it advances only when a
// transfer leaves the FIFO), so a stream that keeps two transfers in flight counts retirements
// from a quiet point instead: transfers retire in order.
static inline uint32_t dsv2_dma_retired(void) {
    register uint32_t r asm("t0");
    asm volatile(".word (0b0000100 << 25) | (0b00000 << 20) | (0b000 << 12) | (5 << 7) | 0b0101011"
                 : "=r"(r));
    return r;
}

static inline uint32_t dsv2_dma_next_id(void) {
    register uint32_t r asm("t0");
    asm volatile(".word (0b0000100 << 25) | (0b00001 << 20) | (0b000 << 12) | (5 << 7) | 0b0101011"
                 : "=r"(r));
    return r;
}

// Wait until every transfer has retired and been counted; returns the count, the base the
// transfers issued next are counted from.
static inline uint32_t dsv2_dma_quiet(void) {
    snrt_dma_wait_all();
    uint32_t c;
    while ((c = dsv2_dma_retired()) + 1u != dsv2_dma_next_id()) {
    }
    return c;
}

// ============================================================ the xDMA

static inline void dsv2_xdma_fill(void *dst, uint32_t beats, uint32_t pattern);

// Zero [dst, dst + beats * 64) with the writer memset (reader channels off, nothing read).
// TCDM has no reset: on RTL an unwritten word reads X, which Verilator's zeroes hide.
static inline void dsv2_xdma_fill_zero(void *dst, uint32_t beats) {
    dsv2_xdma_fill(dst, beats, 0u);
}

// The same memset with a 32-bit pattern: 0xFBFFFBFF is the FP16 -65504 in every lane, the
// running max's seed (no single byte repeats into it).
static inline void dsv2_xdma_fill(void *dst, uint32_t beats, uint32_t pattern) {
    uint32_t hi = (uint32_t)snrt_cluster_base_addrh();
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, (uint32_t)dst);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, hi);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, (uint32_t)dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, hi);
    snax_write_xdma_cfg_reg(XDMA_SRC_SPATIAL_STRIDE_PTR, 8);
    snax_write_xdma_cfg_reg(XDMA_DST_SPATIAL_STRIDE_PTR, 8);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 0, beats);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 1, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 2, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 0, DSV2_BEAT);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 1, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 2, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 4, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 0, beats);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 1, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 2, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 0, DSV2_BEAT);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 1, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 2, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 4, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, 0xFFFFFFFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, 0xFFFFFFFFu);
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLE_PTR, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLE_PTR, 1u << WRITER_EXT_VERILOGMEMSET);
    snax_write_xdma_cfg_reg(XDMA_DST_EXT_CSR_PTR, pattern);
    snax_xdma_local_wait(snax_xdma_start());
}

// FP16 transpose of a [rows, cols] sub-matrix (row pitch src_pitch bytes) into [cols, rows]
// (row pitch dst_pitch), through the reader's 8 x 8 transposer in 16-bit mode. rows and cols
// multiples of 8. Two beats per 8 x 8 block: the reader's lanes are 8 source rows, its beat
// four elements of each; the block grid is walked so each block lands transposed.
static inline void dsv2_xdma_transpose16(const void *src, uint32_t src_pitch, void *dst,
                                         uint32_t dst_pitch, uint32_t rows, uint32_t cols) {
    uint32_t hi = (uint32_t)snrt_cluster_base_addrh();
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, (uint32_t)src);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, hi);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, (uint32_t)dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, hi);
    snax_write_xdma_cfg_reg(XDMA_SRC_SPATIAL_STRIDE_PTR, src_pitch);
    snax_write_xdma_cfg_reg(XDMA_DST_SPATIAL_STRIDE_PTR, dst_pitch);
    // src: the two halves of a block, then block columns, then block rows
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 0, 2);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 1, cols / 8u);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 2, rows / 8u);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 0, 8u);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 1, 16u);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 2, 8u * src_pitch);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 4, 0);
    // dst: block column c becomes block row c, block row r block column r
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 0, 2);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 1, cols / 8u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 2, rows / 8u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 3, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 4, 1);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 0, 8u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 1, 8u * dst_pitch);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 2, 16u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 3, 0);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 4, 0);
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, 0xFFFFFFFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, 0xFFFFFFFFu);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, 0xFFFFFFFFu);
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLE_PTR, 1u << READER_EXT_TRANSPOSERROW8_8COL8_8BIT8_16);
    snax_write_xdma_cfg_reg(XDMA_SRC_EXT_CSR_PTR, 1u);  // 16-bit elements
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLE_PTR, 0u);
    snax_xdma_local_wait(snax_xdma_start());
}

// ============================================================ the SIMD

#if !defined(SIMD_EXT_STREAMELEMENTWISE_0_HAS_MUL)
#error "the dequantisation needs StreamElementwise 0 with MUL"
#endif

// A 2-operand read of a COMMUTATIVE op, from wherever the two operands sit: the reader's
// operand stride is unsigned (an operand below the base would wrap the address and read
// garbage while the task still completes), so the read starts at the LOWER of the two.
static inline void dsv2_shape_pair(snax_simd_shape_t *in, const void *a, const void *b,
                                   uint32_t beats) {
    uint32_t ua = (uint32_t)a, ub = (uint32_t)b;
    snax_simd_shape_2d(in, (void *)(ua < ub ? ua : ub), 2u, ua < ub ? ub - ua : ua - ub, beats,
                       DSV2_BEAT);
}

// yd = y (.) s over n FP16 (a multiple of 32): EW0 MUL on interleaved beats. Programs the bank
// only; a later snax_simd_fire() submits it (the bank is snapshotted on the fire).
static inline void dsv2_dequant_prep(void *y, void *s, void *yd, uint32_t n) {
    snax_simd_shape_t in, out;
    dsv2_shape_pair(&in, y, s, n / 32u);
    snax_simd_shape_flat(&out, yd, n / 32u);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_0, SIMD_EXT_STREAMELEMENTWISE_0_CSR, 2u,
                   SIMD_EW_MUL);
    snax_simd_program_fast(&in, &out);
}

// ---- V1: RMSNorm of one row, no gain (the gain is folded into the weights that follow) ----
//
// Three tasks on a row x of n FP16 (n = 32 beats, a power of two so 1/n is exact), with a
// one-beat SEED slot immediately below x:
//
//   reduce   SUMSQ over the row: FP32 accumulate, one splatted FP16 beat          -> ssq
//   map      RSQRT(ssq / n), the 1/rms beat                                       -> seed
//   ew1      MUL | STICKY_B over [seed, x_0 .. x_last]: the seed is latched and
//            multiplies every later beat, so no plane of 1/rms is ever written     -> y
//
// FP16 caps the sum of squares at 65,504: rms <= sqrt(65504 / n) (5.66 at n = 2,048).
#if !defined(SIMD_EXT_STREAMREDUCE_HAS_SUMSQ) || !defined(SIMD_EXT_STREAMMAP_HAS_RSQRT) || \
    !defined(SIMD_EXT_STREAMELEMENTWISE_1_HAS_MUL)
#error "the RMSNorm needs StreamReduce SUMSQ, StreamMap RSQRT and a post-map MUL"
#endif

// All three tasks, queued back to back; the block runs them in order. `log2n` = log2(n).
static inline void dsv2_rmsnorm_row(void *x, void *ssq, void *y, uint32_t log2n) {
    const uint32_t beats = (1u << log2n) / 32u;
    uint8_t *seed = (uint8_t *)x - DSV2_BEAT;
    snax_simd_shape_t in, out;
    snax_simd_shape_rows(&in, x, 1u, beats, beats * DSV2_BEAT);
    snax_simd_shape_flat(&out, ssq, 1u);
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, beats, SIMD_RED_SUMSQ);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    snax_simd_shape_flat(&in, ssq, 1u);
    snax_simd_shape_flat(&out, seed, 1u);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, 0x3F800000u - (log2n << 23), 0u,
                   SIMD_FUNC_RSQRT);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    snax_simd_shape_flat(&in, seed, beats + 1u);
    snax_simd_shape_flat(&out, y, beats);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1u,
                   SIMD_EW_MUL | SIMD_EW_STICKY_B);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
}

// ---- V2: quantise with one static scale -----------------------------------------------------
//
// Fp16ToInt8: sat_127(rne(x * inv_scale)), FP32 product. inv_scale is the FP32 bit pattern.
#if !defined(SIMD_EXT_FP16TOINT8)
#error "the quantiser needs Fp16ToInt8"
#endif


#define DSV2_ARM_QUANT(inv_bits)                                          \
    do {                                                                  \
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 0, (inv_bits));     \
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 1, 0u);             \
    } while (0)

// n FP16 -> n INT8, in order (a cache row, say). n a multiple of 64.
static inline void dsv2_quant_flat(void *x, void *q, uint32_t n, uint32_t inv_bits) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, x, n / 32u);
    snax_simd_shape_flat(&out, q, n / 64u);
    snax_simd_use0(SIMD_EXT_FP16TOINT8);
    DSV2_ARM_QUANT(inv_bits);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
}

// ---- the GEMV's A operand, row by row ------------------------------------------------------
//
// A GEMV's outputs come from row 0 of its A operand (row 4 for a second token), so the
// quantiser writes those rows and nothing else. A-block k is 16 rows of four values at 64 k;
// row r of it is four bytes at 64 k + 4 r.
//   read   a beat of 16 values: channels 0, 2, 4, 6, four values each, 8 bytes apart (lane
//          stride 4, the odd channels disabled: they read nothing and present zero)
//   pack   two beats into one output beat, whose 8-byte words hold 4 values each in their
//          low half
//   write  word c to A-block c of the beat's eight (lane stride 64), low half only (byte mask)
// so one output beat is 32 values in 8 blocks: n / 32 beats for n values, a quarter of what a
// fully replicated operand costs the quantiser, which is the SIMD's limit on these tasks.
#define DSV2_A_READ 0x55u   // reader channels 0, 2, 4, 6
#define DSV2_A_WORD 0x0Fu   // a written word's low four bytes: one A row

// The shapes of a row write: n values from x (n a multiple of 32) into row `row` of A at a. `ops`
// operands `od` bytes apart feed each value (1 for a plain quantise, 2 for a fused MUL); a plain
// quantise leaves the reader's third loop free for the caller.
static inline void dsv2_a_row_shapes(snax_simd_shape_t *in, snax_simd_shape_t *out, void *x,
                                     uint32_t ops, uint32_t od, void *a, uint32_t n,
                                     uint32_t row) {
    snax_simd_shape_flat(in, x, 1u);
    in->lane_stride = 4u;
    in->lane_mask = DSV2_A_READ;
    uint32_t d = 0;
    if (ops > 1u) {  // the operands of one value
        in->bound[d] = ops;
        in->stride[d++] = od;
    }
    in->bound[d] = 2u;  // the two beats of one output beat
    in->stride[d++] = 32u;
    in->bound[d] = n / 32u;
    in->stride[d++] = 64u;
    in->dim = d;
    snax_simd_shape_flat(out, (uint8_t *)a + 4u * row, n / 32u);
    out->lane_stride = 64u;
    out->stride[0] = 8u * 64u;
    out->byte_mask = DSV2_A_WORD;
}

// n FP16 -> row `row` of a GEMV's A operand (16 n bytes; 0 for token 0, 4 for token 1).
static inline void dsv2_quant_a_row(void *x, void *a, uint32_t n, uint32_t row,
                                    uint32_t inv_bits) {
    snax_simd_shape_t in, out;
    dsv2_a_row_shapes(&in, &out, x, 1u, 0u, a, n, row);
    snax_simd_use0(SIMD_EXT_FP16TOINT8);
    DSV2_ARM_QUANT(inv_bits);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
}

// The A operand of one token (x1 == x0) or two (x1: token 1, into row 4).
static inline void dsv2_quant_a2(void *x0, void *x1, void *a, uint32_t n, uint32_t inv_bits) {
    dsv2_quant_a_row(x0, a, n, 0u, inv_bits);
    if (x1 != x0) dsv2_quant_a_row(x1, a, n, 4u, inv_bits);
}

static inline void dsv2_quant_a(void *x, void *a, uint32_t n, uint32_t inv_bits) {
    dsv2_quant_a_row(x, a, n, 0u, inv_bits);
}

// ---- V3: RoPE, adjacent pairs, one task -----------------------------------------------------
//
// out = x (.) cos + swap(x) (.) sin over n FP16, with cos repeated per pair and sin carrying
// the pair's sign ([-s0, +s0, -s1, +s1, ...]); swap(x) exchanges the two halves of every
// pair. The four operands sit EQUALLY SPACED, `ostride` apart from `x`: x, cos, swap(x), sin.
//
//   read   [ x_b , cos_b , swap_b , sin_b ]    operand dim, 4 deep
//   EW0    MUL, 2 operands ->  [ x*cos , swap*sin ]
//   EW1    ADD, 2 operands ->  [ x*cos + swap*sin ]
//
// Each stage narrows to FP16, so the result is RNE(RNE(x*cos) + RNE(swap*sin)), bit for bit.
#if !defined(SIMD_EXT_STREAMELEMENTWISE_1_HAS_ADD)
#error "the fused RoPE needs a post-map StreamElementwise ADD"
#endif

static inline void dsv2_rope(void *x, uint32_t ostride, void *y, uint32_t n) {
    snax_simd_shape_t in, out;
    snax_simd_shape_2d(&in, x, 4u, ostride, n / 32u, DSV2_BEAT);
    snax_simd_shape_flat(&out, y, n / 32u);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                                     (1u << SIMD_EXT_STREAMELEMENTWISE_1));
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 0, 2u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 1, SIMD_EW_MUL);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 0, 2u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1, SIMD_EW_ADD);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
}

// ---- V6: SwiGLU -----------------------------------------------------------------------------
//
// g = [gate | up], 2 i FP16 (the fused gate|up GEMV's dequantised output). Two tasks:
//   map   SILU on gate                                      -> sg (i FP16)
//   ew1   MUL over interleaved [sg, up]                      -> a  (i FP16)       (a16 != 0)
//   ew1   MUL, then the quantiser into row 0 of the down GEMV's A operand (a8 != 0; the
//         row write of dsv2_quant_a_row, each value from its two operands)
// The products commute, so sg may sit on either side of g. tok: 0 for one token; for two, the
// distance from token 0's g and sg to token 1's (the same for both), whose SwiGLU goes into
// row 4.
#if !defined(SIMD_EXT_STREAMMAP_HAS_SILU)
#error "the SwiGLU needs StreamMap SILU"
#endif

static inline void dsv2_swiglu(void *g, void *sg, void *a16, void *a8, uint32_t i,
                               uint32_t inv_bits, uint32_t tok) {
    uint8_t *up = (uint8_t *)g + 2u * i;
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, g, i / 32u);
    snax_simd_shape_flat(&out, sg, i / 32u);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, 0x3F800000u, 0u, SIMD_FUNC_SILU);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    if (a16) {
        dsv2_shape_pair(&in, sg, up, i / 32u);
        snax_simd_shape_flat(&out, a16, i / 32u);
        snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 2u,
                       SIMD_EW_MUL);
        snax_simd_program_fast(&in, &out);
        snax_simd_fire();
    }
    if (a8) {
        const uint32_t us = (uint32_t)sg, uu = (uint32_t)up;
        for (uint32_t t = 0; t < (tok ? 2u : 1u); t++) {
            if (t) {  // token 1's silu
                snax_simd_shape_flat(&in, (uint8_t *)g + tok, i / 32u);
                snax_simd_shape_flat(&out, (uint8_t *)sg + tok, i / 32u);
                snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, 0x3F800000u, 0u,
                               SIMD_FUNC_SILU);
                snax_simd_program_fast(&in, &out);
                snax_simd_fire();
            }
            // sg and up, the lower first (MUL commutes)
            dsv2_a_row_shapes(&in, &out, (void *)((us < uu ? us : uu) + t * tok), 2u,
                              us < uu ? uu - us : us - uu, a8, i, 4u * t);
            snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_1) |
                                                             (1u << SIMD_EXT_FP16TOINT8));
            snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 0, 2u);
            snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1, SIMD_EW_MUL);
            DSV2_ARM_QUANT(inv_bits);
            snax_simd_program_fast(&in, &out);
            snax_simd_fire();
        }
    }
}

// ---- V5: softmax over one row, and the top k ------------------------------------------------
//
// The router's softmax over `beats` beats of FP16 logits (64 experts = 2 beats). The row's max
// and sum fold ACROSS lanes (the reduce's row mode), so each comes out splatted. Buffers:
// x with a free latch beat below it; e with a free latch beat below it and room for the sum
// beat after it; tmp one beat; p `beats` beats. Five tasks, queued back to back:
//
//   1  Reduce MAX over x                                   -> m, splatted      (tmp)
//   2  Map LINEAR a = -1                                   -> -m               (x's latch)
//   3  EW0 ADD|STICKY_B [-m][x] -> Map EXP -> Reduce ADD|TAP -> [e][s]         (e)
//   4  [s][s] at stride 0: EW0 MUL|STICKY_B -> Map RSQRT   -> 1/s = rsqrt(s*s) (e's latch)
//   5  EW1 MUL|STICKY_B [1/s][e]                           -> p
//
// 1/s = rsqrt(s * s) is exact for any positive s whose square fits FP16 (s <= 64 here).
#if !defined(SIMD_EXT_STREAMREDUCE_HAS_MAX) || !defined(SIMD_EXT_STREAMREDUCE_HAS_ADD) || \
    !defined(SIMD_EXT_STREAMMAP_HAS_EXP) || !defined(SIMD_EXT_STREAMELEMENTWISE_0_HAS_ADD)
#error "the softmax needs StreamReduce MAX and ADD, StreamMap EXP and a pre-map ADD"
#endif

static inline void dsv2_softmax_row(void *x, void *tmp, void *e, void *p, uint32_t beats) {
    uint8_t *xl = (uint8_t *)x - DSV2_BEAT, *el = (uint8_t *)e - DSV2_BEAT;
    uint8_t *s = (uint8_t *)e + beats * DSV2_BEAT;
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, x, beats);
    snax_simd_shape_flat(&out, tmp, 1);
    snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, beats, SIMD_RED_MAX);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    snax_simd_shape_flat(&in, tmp, 1);
    snax_simd_shape_flat(&out, xl, 1);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, snax_simd_f32_neg(SIMD_F32_ONE),
                   0u, SIMD_FUNC_LINEAR);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    snax_simd_shape_flat(&in, xl, 1 + beats);
    snax_simd_shape_flat(&out, e, beats + 1);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                                     (1u << SIMD_EXT_STREAMMAP) |
                                                     (1u << SIMD_EXT_STREAMREDUCE));
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 0, 1u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 1, SIMD_EW_ADD | SIMD_EW_STICKY_B);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 0, SIMD_F32_ONE);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 1, 0u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 2, SIMD_FUNC_EXP);
    snax_simd_set_op_csr(SIMD_EXT_STREAMREDUCE_CSR, 0, beats);
    snax_simd_set_op_csr(SIMD_EXT_STREAMREDUCE_CSR, 1, SIMD_RED_ADD | SIMD_RED_TAP);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    snax_simd_shape_broadcast(&in, s, 2);
    snax_simd_shape_flat(&out, el, 1);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                                     (1u << SIMD_EXT_STREAMMAP));
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 0, 1u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 1, SIMD_EW_MUL | SIMD_EW_STICKY_B);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 0, SIMD_F32_ONE);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 1, 0u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 2, SIMD_FUNC_RSQRT);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    snax_simd_shape_flat(&in, el, 1 + beats);
    snax_simd_shape_flat(&out, p, beats);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1u,
                   SIMD_EW_MUL | SIMD_EW_STICKY_B);
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
}

static inline int32_t dsv2_mono16(uint16_t h);

// The k largest of n <= 64 FP16 values, largest first, in core code (no FPU on any hart): FP16
// bit patterns map to integers in value order, and equal values go to the LOWER index.
static inline void dsv2_top_k16(const uint16_t *v, uint32_t n, uint32_t k, uint32_t *ids) {
    uint64_t taken = 0;
    for (uint32_t r = 0; r < k; r++) {
        int32_t best = 0;
        uint32_t bi = n;
        for (uint32_t i = 0; i < n; i++) {
            if (taken & (1ull << i)) continue;
            int32_t key = dsv2_mono16(v[i]);
            if (bi == n || key > best) {
                best = key;
                bi = i;
            }
        }
        ids[r] = bi;
        taken |= 1ull << bi;
    }
}

// ============================================================ D1: the latent cache
//
// Two copies in DRAM, both INT8, `cap` tokens (a multiple of 16):
//   KEY    A-layout of [cap, 576] = tokens x [c | k_pe], blocks of 16 tokens x 4 values. A
//          tile of Bc tokens is contiguous, Bc * 576 bytes: the scores' A operand as is.
//   VALUE  A-layout of V^T = [512, cap] = latents x tokens, blocks of 16 latents x 4 tokens.
//          A tile is 32 runs of 16 Bc bytes, cap * 16 apart: the weighted sum's A operand.
// sw/apps/dsv2/util/layout.py key_copy / value_copy build the same bytes.
#define DSV2_KV_ROW 576u  // [c 512 | k_pe 64]
#define DSV2_KV_RANK 512u

// Append the row `row` (L1, 576 INT8) for token t, and wait for it: into the key copy as 144
// runs of 4 bytes 64 apart, into the value copy as 32 x 16 runs of one byte 4 apart. On the
// DM core; the iDMA is the only engine here that reaches all of DRAM.
static inline void dsv2_cache_append(int8_t *key, int8_t *val, uint32_t cap, uint32_t t,
                                     const int8_t *row) {
    int8_t *k0 = key + (t / 16u) * (DSV2_KV_ROW / 4u) * 64u + (t % 16u) * 4u;
    snrt_dma_start_2d(k0, row, 4, 64, 4, DSV2_KV_ROW / 4u);
    int8_t *v0 = val + (t / 4u) * 64u + (t % 4u);
    for (uint32_t m = 0; m < DSV2_KV_RANK / 16u; m++)
        snrt_dma_start_2d(v0 + m * cap * 16u, row + 16u * m, 1, 4, 1, 16);
    snrt_dma_wait_all();
}

// ============================================================ checks

// The terms of an n-term tensor a check compares (DSV2_CHECK_TERMS). Every helper below applies
// it, and a caller that prints the count applies it too, so the report says what was compared.
static inline uint32_t dsv2_check_terms(uint32_t n) {
    return (DSV2_CHECK_TERMS && n > DSV2_CHECK_TERMS) ? DSV2_CHECK_TERMS : n;
}

// The app's L1 footprint, and what its checks compare when that is not every term.
static inline void dsv2_print_l1(uint32_t used, uint32_t avail) {
    printf("[L1] %u of %u bytes\n", used, avail);
    if (DSV2_CHECK_TERMS)
        printf("[CHECK] the first %u terms of each tensor (DSV2_CHECK_TERMS)\n",
               DSV2_CHECK_TERMS);
}

// Count the FP16 (or any 16-bit) words that differ; print the first few under `tag`.
static inline uint32_t dsv2_cmp16(const char *tag, const char *what, const uint16_t *got,
                                  const uint16_t *want, uint32_t n) {
    n = dsv2_check_terms(n);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; i++)
        if (got[i] != want[i]) {
            if (bad < 3)
                printf("%s     %s[%u] = %04x, want %04x\n", tag, what, i, got[i], want[i]);
            bad++;
        }
    return bad;
}

// FP16 bit patterns -> integers in value order, so a difference counts ULPs.
static inline int32_t dsv2_mono16(uint16_t h) {
    int32_t mag = (int32_t)(h & 0x7FFFu);
    return (h & 0x8000u) ? -mag : mag;
}

// The byte offset, from `from` on, of the first 32-byte block in which two arrays differ, or
// `bytes` if none does. A check spends its time on loads: a hart has one load in flight, so each
// costs a full TCDM round trip. A block that matches whole (the common case) is 16 loads and one
// branch, about 60% of the cycles of comparing it word by word. Both arrays 4-byte aligned;
// `from` and `bytes` multiples of 32.
static inline uint32_t dsv2_diff_block(const void *a, const void *b, uint32_t from,
                                       uint32_t bytes) {
    const uint32_t *x = (const uint32_t *)a, *y = (const uint32_t *)b;
    for (uint32_t o = from; o < bytes; o += 32u) {
        const uint32_t w = o / 4u;
        if ((x[w] ^ y[w]) | (x[w + 1] ^ y[w + 1]) | (x[w + 2] ^ y[w + 2]) |
            (x[w + 3] ^ y[w + 3]) | (x[w + 4] ^ y[w + 4]) | (x[w + 5] ^ y[w + 5]) |
            (x[w + 6] ^ y[w + 6]) | (x[w + 7] ^ y[w + 7]))
            return o;
    }
    return bytes;
}

// FP16: how many elements differ, and the largest distance in ULPs (+0 and -0 are 0 apart).
// Blocks that match whole are skipped (dsv2_diff_block); the elements of a block that does not,
// and the tail, are compared one by one. Both arrays 4-byte aligned.
static inline uint32_t dsv2_ulp16(const uint16_t *got, const uint16_t *want, uint32_t n,
                                  uint32_t *worst) {
    n = dsv2_check_terms(n);
    uint32_t bad = 0;
    *worst = 0;
    const uint32_t nb = n & ~15u;  // the elements in whole blocks
    for (uint32_t i = 0; i < n;) {
        uint32_t end = n;
        if (i < nb) {
            i = dsv2_diff_block(got, want, 2u * i, 2u * nb) / 2u;
            if (i == nb) continue;
            end = i + 16u;
        }
        for (; i < end; i++) {
            int32_t d = dsv2_mono16(got[i]) - dsv2_mono16(want[i]);
            uint32_t u = (uint32_t)(d < 0 ? -d : d);
            if (u) bad++;
            if (u > *worst) *worst = u;
        }
    }
    return bad;
}

// INT8: how many elements differ, and the largest difference in LSBs. Blocks that match whole
// are skipped as in dsv2_ulp16. Both arrays 4-byte aligned.
static inline uint32_t dsv2_lsb8(const int8_t *got, const int8_t *want, uint32_t n,
                                 uint32_t *worst) {
    n = dsv2_check_terms(n);
    uint32_t bad = 0;
    *worst = 0;
    const uint32_t nb = n & ~31u;  // the elements in whole blocks
    for (uint32_t i = 0; i < n;) {
        uint32_t end = n;
        if (i < nb) {
            i = dsv2_diff_block(got, want, i, nb);
            if (i == nb) continue;
            end = i + 32u;
        }
        for (; i < end; i++) {
            int32_t d = (int32_t)got[i] - (int32_t)want[i];
            uint32_t u = (uint32_t)(d < 0 ? -d : d);
            if (u) bad++;
            if (u > *worst) *worst = u;
        }
    }
    return bad;
}

// FP16 -> Q20 fixed point (value * 2^20, truncated), in integer code: the harts have no FPU.
// Exact for |value| < 2^11 down to 2^-20; saturates beyond, Inf and NaN to the rails.
static inline int32_t dsv2_f16_q20(uint16_t h) {
    int32_t e = (h >> 10) & 0x1F, m = h & 0x3FF, v;
    if (e == 0x1F) return (h & 0x8000u) ? -0x7FFFFFFF : 0x7FFFFFFF;
    if (e == 0)
        v = m >> 4;  // subnormal: m * 2^-24
    else if (e >= 26)
        v = 0x7FFFFFFF;
    else
        v = e >= 5 ? (m | 0x400) << (e - 5) : (m | 0x400) >> (5 - e);
    return (h & 0x8000u) ? -v : v;
}

// FP16 against a golden with a per-element FP16 tolerance: how many exceed it, and the
// largest |got - want| / tol seen, in percent. An element equal to its golden is 0% off, so
// blocks that match whole are skipped as in dsv2_ulp16. Both arrays 4-byte aligned.
static inline uint32_t dsv2_tol16(const uint16_t *got, const uint16_t *want,
                                  const uint16_t *tol, uint32_t n, uint32_t *worst_pct) {
    n = dsv2_check_terms(n);
    uint32_t bad = 0;
    *worst_pct = 0;
    const uint32_t nb = n & ~15u;  // the elements in whole blocks
    for (uint32_t i = 0; i < n;) {
        uint32_t end = n;
        if (i < nb) {
            i = dsv2_diff_block(got, want, 2u * i, 2u * nb) / 2u;
            if (i == nb) continue;
            end = i + 16u;
        }
        for (; i < end; i++) {
            int32_t d = dsv2_f16_q20(got[i]) - dsv2_f16_q20(want[i]);
            uint32_t ad = (uint32_t)(d < 0 ? -d : d), t = (uint32_t)dsv2_f16_q20(tol[i]);
            if (ad > t) bad++;
            uint32_t pct =
                t ? (ad > 0x1FFFFFFu ? 100000u : (100u * ad) / t) : (ad ? 100000u : 0u);
            if (pct > *worst_pct) *worst_pct = pct;
        }
    }
    return bad;
}

// A GEMV A operand against the INT8 values of the token in its row `row` (0 or 4): the count of
// the n values that differ. Rows the quantiser does not write are not compared.
static inline uint32_t dsv2_a_row_diff(const int8_t *a, const int8_t *x, uint32_t n,
                                       uint32_t row) {
    n = dsv2_check_terms(n);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; i++) bad += a[(i / 4u) * 64u + 4u * row + (i % 4u)] != x[i];
    return bad;
}

static inline uint32_t dsv2_count_inf16(const uint16_t *v, uint32_t n) {
    n = dsv2_check_terms(n);
    uint32_t c = 0;
    for (uint32_t i = 0; i < n; i++) c += (v[i] & 0x7FFFu) == 0x7C00u;
    return c;
}

// Print one named check and return 1 if it failed.
static inline uint32_t dsv2_check(const char *tag, int ok, const char *who, const char *what) {
    printf("%s   %s %s: %s\n", tag, ok ? "PASS" : "FAIL", who, what);
    return ok ? 0u : 1u;
}

// The top of the usable L1: the hart stacks are carved downward from the end of TCDM.
#define DSV2_L1_TOP                              \
    (SNRT_TCDM_START_ADDR + SNRT_TCDM_SIZE -    \
     SNRT_CLUSTER_CORE_NUM * (1u << SNRT_LOG2_STACK_SIZE))
