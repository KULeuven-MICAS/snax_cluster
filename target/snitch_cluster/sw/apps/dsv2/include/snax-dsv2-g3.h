// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// VersaCore array shape 3, (Mu, Ku, Nu) = (8, 8, 16), the shape of HeMAiA's MoE prefill GEMMs,
// with INT4 weights and a ROW-MAJOR A operand:
//
//     D (8 mt x 16 nt, FP16) = RNE(A . B * 2^-k)
//
//   A  mt*8 rows of kt*8 INT8, row r at a + r * a_pitch (a_pitch = 32 mod 64, so the eight rows
//      of a pass sit in eight banks). A pass reads one 8-byte word of each of eight rows: channel
//      i is row i, the next pass is the next word of the rows.
//   B  INT4 weights, 64 bytes of nibbles a pass (16 columns x 8 k), widened by B's converter.
//      Where pass p of column tile n sits is the layout (g3_lay_t):
//        G3_LAY_B       tile n at n * kt * 64, pass p at p * 64.
//        G3_LAY_PAIRED  blocks of two column tiles: tile n at (n / 2) * kt * 128 + (n % 2) * 64,
//                       pass p at p * 128 (B advances 128 bytes a pass).
//        G3_LAY_PARITY  passes in pairs, each pair one 128-byte run, word w of pass 2j + i at
//                       16 w + 8 i, so a pass's eight words are 16 bytes apart; B starts one word
//                       (G3_PARITY_OFF) into its slab.
//   D  tile (m, n) at d + 256 (m nt + n), row r of it 32 bytes on.
//
// THREE WAYS TO PROGRAM A GEMM (the cost of a launch):
//   g3_arm          the lean arm: every descriptor register, straight-line, compile-time CSR
//                   addresses (one csrw each).
//   g3_fire         the sticky descriptor's launch: only A, B and D's bases, then the start. Cheap
//                   enough to follow an arm done once per geometry.
//   g3_arm_generic  the generic path HeMAiA's gemm_start takes (offload_hw_kernels/gemm.h,
//                   __bingo_gemm_run): the shape's parameters from a table, the channel masks built
//                   from the operand widths, every stream's registers written by the shared
//                   library (snax-versacore-to-lib.c), whose loops address CSRs by a run-time
//                   index.

#pragma once

#include "snax-dsv2.h"

#if !defined(READER_EXTENSION_1_CSR_BASE)
#error "INT4 weights need B's HasIntlowToInthighConverter (cfg/snax_split_cluster.hjson)"
#endif

#define G3_SHAPE 3u
#define G3_MR 8u
#define G3_KU 8u
#define G3_NU 16u
#define G3_BLK 64u         // INT4 B bytes a pass: 8 x 16 nibbles
#define G3_ABLK 64u        // A bytes a pass: 8 rows x 8 INT8
#define G3_DTILE 256u      // one FP16 D tile: 8 x 16
#define G3_PARITY_OFF 8u   // a parity-layout chunk starts one word into its slab

typedef enum { G3_LAY_B = 1, G3_LAY_PAIRED = 2, G3_LAY_PARITY = 3 } g3_lay_t;

// A's layout. Row-major is what an expert's gathered X rows are; the other two are A-layout tiles
// (8 rows x 8 INT8 = one 64-byte pass, K-pass p of row tile m at 64 (kt m + p)), the second with
// the passes interleaved in two K halves (pass p at 128 (p % H) + 64 (p / H), H = kt / 2), so A
// advances 128 bytes a pass. Two streams advancing at one stride keep a fixed bank distance.
typedef enum { G3_A_ROW = 0, G3_A_BLOCKED = 1, G3_A_ILV = 2 } g3_alay_t;

// A row-major operand's pitch for a row of `bytes`: a row and 16 more (a quantiser writes 16
// values past the end), beat-aligned and 32 mod 64.
#define G3_PITCH(bytes) ((((bytes) + 16u) + (32u - (((bytes) + 16u) % 64u)) % 64u))

#if T_BOUND_NUM_READER_0 != 6 || T_BOUND_NUM_READER_1 != 6
#error "the arms program six loops on A and on B (cfg/snax_split_cluster.hjson temporal_dim)"
#endif

// ---------------------------------------------------------------------------- the lean arm
//
// Arm the whole descriptor for a run of tasks (the bases come with g3_fire). mt, kt, nt: tiles of
// 8 rows, passes of 8 k, tiles of 16 columns. nt even for the paired layout, kt even for parity.
__attribute__((always_inline)) static inline void g3_arm(uint32_t mt, uint32_t kt, uint32_t nt,
                                                         uint32_t k, uint32_t lay, uint32_t alay,
                                                         uint32_t a_pitch) {
    // A -- k inner, n broadcast, then the next eight rows (row tile).
    csrw_ss(ENABLED_CHANNEL_READER_0, 0xFFu);
    if (alay == G3_A_ROW) {  // row-major: the next word of the rows
        csrw_ss(S_STRIDE_READER_0_0, a_pitch);
        csrw_ss(T_BOUND_READER_0_0, kt);
        csrw_ss(T_STRIDE_READER_0_0, 8);
        csrw_ss(T_BOUND_READER_0_1, nt);
        csrw_ss(T_STRIDE_READER_0_1, 0);
        csrw_ss(T_BOUND_READER_0_2, mt);
        csrw_ss(T_STRIDE_READER_0_2, G3_MR * a_pitch);
        csrw_ss(T_BOUND_READER_0_3, 1);
        csrw_ss(T_STRIDE_READER_0_3, 0);
    } else if (alay == G3_A_BLOCKED) {  // 64 bytes a pass
        csrw_ss(S_STRIDE_READER_0_0, 8);
        csrw_ss(T_BOUND_READER_0_0, kt);
        csrw_ss(T_STRIDE_READER_0_0, G3_ABLK);
        csrw_ss(T_BOUND_READER_0_1, nt);
        csrw_ss(T_STRIDE_READER_0_1, 0);
        csrw_ss(T_BOUND_READER_0_2, mt);
        csrw_ss(T_STRIDE_READER_0_2, kt * G3_ABLK);
        csrw_ss(T_BOUND_READER_0_3, 1);
        csrw_ss(T_STRIDE_READER_0_3, 0);
    } else {  // interleaved in two K halves: 128 bytes a pass
        csrw_ss(S_STRIDE_READER_0_0, 8);
        csrw_ss(T_BOUND_READER_0_0, kt / 2u);
        csrw_ss(T_STRIDE_READER_0_0, 2u * G3_ABLK);
        csrw_ss(T_BOUND_READER_0_1, 2);
        csrw_ss(T_STRIDE_READER_0_1, G3_ABLK);
        csrw_ss(T_BOUND_READER_0_2, nt);
        csrw_ss(T_STRIDE_READER_0_2, 0);
        csrw_ss(T_BOUND_READER_0_3, mt);
        csrw_ss(T_STRIDE_READER_0_3, kt * G3_ABLK);
    }
    csrw_ss(T_BOUND_READER_0_4, 1);
    csrw_ss(T_STRIDE_READER_0_4, 0);
    csrw_ss(T_BOUND_READER_0_5, 1);
    csrw_ss(T_STRIDE_READER_0_5, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0);
    // B -- eight 8-byte channels; the walk is the layout's. The row tile is last, at stride 0.
    csrw_ss(ENABLED_CHANNEL_READER_1, 0xFFu);
    csrw_ss(S_STRIDE_READER_1_0, lay == G3_LAY_PARITY ? 16u : 8u);
    csrw_ss(S_STRIDE_READER_1_1, 64);
    if (lay == G3_LAY_PAIRED) {
        csrw_ss(T_BOUND_READER_1_0, kt);
        csrw_ss(T_STRIDE_READER_1_0, 2u * G3_BLK);
        csrw_ss(T_BOUND_READER_1_1, 2);
        csrw_ss(T_STRIDE_READER_1_1, G3_BLK);
        csrw_ss(T_BOUND_READER_1_2, nt / 2u);
        csrw_ss(T_STRIDE_READER_1_2, 2u * G3_BLK * kt);
        csrw_ss(T_BOUND_READER_1_3, mt);
        csrw_ss(T_STRIDE_READER_1_3, 0);
    } else if (lay == G3_LAY_PARITY) {
        csrw_ss(T_BOUND_READER_1_0, 2);
        csrw_ss(T_STRIDE_READER_1_0, 8);
        csrw_ss(T_BOUND_READER_1_1, kt / 2u);
        csrw_ss(T_STRIDE_READER_1_1, 2u * G3_BLK);
        csrw_ss(T_BOUND_READER_1_2, nt);
        csrw_ss(T_STRIDE_READER_1_2, G3_BLK * kt);
        csrw_ss(T_BOUND_READER_1_3, mt);
        csrw_ss(T_STRIDE_READER_1_3, 0);
    } else {
        csrw_ss(T_BOUND_READER_1_0, kt);
        csrw_ss(T_STRIDE_READER_1_0, G3_BLK);
        csrw_ss(T_BOUND_READER_1_1, nt);
        csrw_ss(T_STRIDE_READER_1_1, G3_BLK * kt);
        csrw_ss(T_BOUND_READER_1_2, mt);
        csrw_ss(T_STRIDE_READER_1_2, 0);
        csrw_ss(T_BOUND_READER_1_3, 1);
        csrw_ss(T_STRIDE_READER_1_3, 0);
    }
    csrw_ss(T_BOUND_READER_1_4, 1);
    csrw_ss(T_STRIDE_READER_1_4, 0);
    csrw_ss(T_BOUND_READER_1_5, 1);
    csrw_ss(T_STRIDE_READER_1_5, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0);
    dsv2_b_int4(1u);
    // C -- masked; the bounds still count the four INT32 beats of every output tile.
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_0, 0);
    csrw_ss(S_STRIDE_READER_WRITER_0_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_0_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_0_0, G3_MR * G3_NU * 32u / 1024u);
    csrw_ss(T_STRIDE_READER_WRITER_0_0, 0);
    csrw_ss(T_BOUND_READER_WRITER_0_1, nt);
    csrw_ss(T_STRIDE_READER_WRITER_0_1, 0);
    csrw_ss(T_BOUND_READER_WRITER_0_2, mt);
    csrw_ss(T_STRIDE_READER_WRITER_0_2, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, 0);
    // D -- a tile leaves as two beats of four rows (channel i % 4 = 4 columns, i / 4 = the row of
    // the beat, rows 32 bytes apart); tiles in (m, n) order, 256 bytes each.
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_1, 0xFFFFu);
    csrw_ss(S_STRIDE_READER_WRITER_1_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_1_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_1_0, G3_DTILE / 128u);
    csrw_ss(T_STRIDE_READER_WRITER_1_0, 128);
    csrw_ss(T_BOUND_READER_WRITER_1_1, nt);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, G3_DTILE);
    csrw_ss(T_BOUND_READER_WRITER_1_2, mt);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, G3_DTILE * nt);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, 0);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 0, 1u);  // Int32ToFp16 on
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 0u);  // extra-loop 0: the 2:1 narrowing
    (void)set_versacore_d_shift(k);                        // RNE(acc * 2^-k)
    // The array: output-stationary, kt passes per tile, mt * nt tiles.
    csrw_ss(OVERWRITE_ACCUM, 1);
    csrw_ss(ACCUM_BOUND, kt);
    csrw_ss(OUTPUT_BOUND, mt * nt);
    csrw_ss(SUBTRACTIONS, 0);
    csrw_ss(ARRAY_SHAPE_CFG, G3_SHAPE);
    csrw_ss(DATA_TYPE_CFG, 0);
}

// The sticky descriptor's launch: A, B and D's bases and the start. `b` is the B operand's first
// byte (for the parity layout, G3_PARITY_OFF into its slab). A start is a write EVENT, so
// back-to-back fires queue (snax_cfg_queue_depth).
__attribute__((always_inline)) static inline void g3_fire(const void *a, const void *b, void *d) {
    dsv2_gemv_fire(a, b, d);
}

// ------------------------------------------------------------------------ the generic arm
//
// The shape's row of HeMAiA's bingo_gemm_shape_params, kept in L1 (params gemm_shape_l1): a table in
// .rodata would be read from main memory a dozen times a launch.
typedef struct {
    uint32_t meshRow, tileSize, meshCol;
    uint32_t Ctlbound0, Ctlstride0, D32tlbound0, D32tlstride0;
    uint32_t channel_en_A, channel_en_B, channel_en_C, channel_en_D32;
} g3_shape_t;

#define G3_BANK_WIDTH 64u        // bits
#define G3_SERIAL_CD_WIDTH 1024u // bits
#define G3_CD_SPATIAL_BOUND0 4u

// The shape-3 row, to be copied into L1 once.
static const g3_shape_t g3_shape_rom = {8u, 8u, 16u, 4u, 128u, 4u, 128u, 0xFFu, 0xFFFFu,
                                        0xFFFFFFFFu, 0xFFFFFFFFu};

// Arm one task the way the generic kernel does and leave the start to g3_fire's caller (g3_start).
// accumPrevC = 0 and no C: D = A . B, converted to FP16 at shift k. The addresses are absolute.
static void g3_arm_generic(const g3_shape_t *shape, uint32_t A_addr, uint32_t B_addr,
                           uint32_t D_addr, uint32_t M, uint32_t K, uint32_t N, uint32_t d_shift,
                           uint32_t lay, uint32_t alay, uint32_t a_pitch) {
    const uint32_t meshRow = shape->meshRow, tileSize = shape->tileSize, meshCol = shape->meshCol;
    const uint32_t b_paired = lay == G3_LAY_PAIRED, b_parity = lay == G3_LAY_PARITY;
    const uint32_t a_elem_len = 8u, b_elem_len = 4u, d_elem_len = 16u;
    const uint32_t one_output_tile_bits = meshRow * meshCol * d_elem_len;
    uint32_t channel_en_A_dyn[1] = {0}, channel_en_B_dyn[1] = {0}, channel_en_D32_dyn[1] = {0};
    const uint32_t channel_en_C_null[1] = {0};
    const uint32_t *channel_en_A_ptr = &shape->channel_en_A;
    // B: the channel mask from the operand's width
    {
        uint32_t bits = (meshCol * tileSize * b_elem_len + G3_BANK_WIDTH - 1) / G3_BANK_WIDTH;
        if (bits == 0) bits = 1;
        for (uint32_t i = 0; i < bits; i++) channel_en_B_dyn[0] |= (uint32_t)(1u << (i % 32));
    }
    // D: the converter's serial width
    {
        uint32_t bits = (G3_SERIAL_CD_WIDTH + G3_BANK_WIDTH - 1) / G3_BANK_WIDTH;
        for (uint32_t i = 0; i < bits; i++) channel_en_D32_dyn[0] |= (uint32_t)(1u << (i % 32));
    }
    // A
    const uint32_t row = alay == G3_A_ROW;
    uint32_t Aslstride0 = row ? a_pitch : G3_BANK_WIDTH / 8;
    uint32_t Atlbound[6] = {K, N, M, 1, 1, 1};
    uint32_t Atlstride[6];
    uint32_t a_tile_banks = (a_elem_len * meshRow * tileSize + G3_BANK_WIDTH - 1) / G3_BANK_WIDTH;
    if (a_tile_banks == 0) a_tile_banks = 1;
    Atlstride[0] = a_tile_banks * (G3_BANK_WIDTH / 8);
    if (row) Atlstride[0] = G3_BANK_WIDTH / 8;
    Atlstride[1] = 0;
    Atlstride[2] = row ? meshRow * a_pitch : Atlstride[0] * K;
    Atlstride[3] = 0;
    Atlstride[4] = 0;
    Atlstride[5] = 0;
    if (alay == G3_A_ILV) {  // the two K halves
        Atlbound[0] = K / 2u;
        Atlstride[0] = 2u * Atlstride[0];
        Atlbound[1] = 2;
        Atlstride[1] = G3_ABLK;
        Atlbound[2] = N;
        Atlstride[2] = 0;
        Atlbound[3] = M;
        Atlstride[3] = K * G3_ABLK;
    }
    // B
    uint32_t Bslstride[2];
    Bslstride[0] = (b_parity ? 2u : 1u) * (G3_BANK_WIDTH / 8);
    Bslstride[1] = (G3_BANK_WIDTH / 8) * 8u;
    uint32_t Btlbound[6], Btlstride[6];
    for (uint32_t i = 0; i < 6; i++) {
        Btlbound[i] = 1;
        Btlstride[i] = 0;
    }
    const uint32_t b_blk = b_elem_len * tileSize * meshCol / 8;
    if (b_paired) {
        Btlbound[0] = K;
        Btlstride[0] = 2u * b_blk;
        Btlbound[1] = 2;
        Btlstride[1] = b_blk;
        Btlbound[2] = N / 2u;
        Btlstride[2] = 2u * b_blk * K;
        Btlbound[3] = M;
        Btlstride[3] = 0;
    } else if (b_parity) {
        Btlbound[0] = 2;
        Btlstride[0] = G3_BANK_WIDTH / 8;
        Btlbound[1] = K / 2u;
        Btlstride[1] = 2u * b_blk;
        Btlbound[2] = N;
        Btlstride[2] = b_blk * K;
        Btlbound[3] = M;
        Btlstride[3] = 0;
    } else {
        Btlbound[0] = K;
        Btlstride[0] = b_blk;
        Btlbound[1] = N;
        Btlstride[1] = b_blk * K;
        Btlbound[2] = M;
        Btlstride[2] = 0;
    }
    // C: masked
    uint32_t Cslstride[2];
    Cslstride[0] = G3_BANK_WIDTH / 8;
    Cslstride[1] = (G3_BANK_WIDTH / 8) * G3_CD_SPATIAL_BOUND0;
    uint32_t Ctlbound[4] = {shape->Ctlbound0, N, M, 1};
    uint32_t Ctlstride[4];
    Ctlstride[0] = shape->Ctlstride0;
    Ctlstride[1] = 32u * meshRow * meshCol / 8;
    Ctlstride[2] = N * 32u * meshRow * meshCol / 8;
    Ctlstride[3] = 0;
    // D
    uint32_t D32slstride[2];
    D32slstride[0] = G3_BANK_WIDTH / 8;
    D32slstride[1] = (G3_BANK_WIDTH / 8) * G3_CD_SPATIAL_BOUND0;
    uint32_t D32tlbound[4], D32tlstride[4];
    if (one_output_tile_bits > G3_SERIAL_CD_WIDTH) {
        D32tlbound[0] = one_output_tile_bits / G3_SERIAL_CD_WIDTH;
        D32tlbound[1] = N * M;
        D32tlbound[2] = 1;
        D32tlbound[3] = 1;
        D32tlstride[0] = G3_SERIAL_CD_WIDTH / 8;
        D32tlstride[1] = d_elem_len * meshRow * meshCol / 8;
        D32tlstride[2] = 0;
        D32tlstride[3] = 0;
    } else {
        const uint32_t output_matrix_per_store = G3_SERIAL_CD_WIDTH / one_output_tile_bits;
        D32tlbound[0] = 1;
        D32tlbound[1] = N * M / output_matrix_per_store;
        D32tlbound[2] = 1;
        D32tlbound[3] = 1;
        D32tlstride[0] = G3_SERIAL_CD_WIDTH / 8;
        D32tlstride[1] = G3_SERIAL_CD_WIDTH / 8;
        D32tlstride[2] = 0;
        D32tlstride[3] = 0;
    }
    const int32_t l1 = (int32_t)snrt_l1_next();  // the library takes bases as offsets from L1's start
    set_versacore_streamer_csr(
        (int32_t)A_addr - l1, (int32_t *)&Aslstride0, (int32_t *)Atlbound, (int32_t *)Atlstride, 0,
        0, (int32_t *)channel_en_A_ptr,
        (int32_t)B_addr - l1, (int32_t *)Bslstride, (int32_t *)Btlbound, (int32_t *)Btlstride, 0,
        0, (int32_t *)channel_en_B_dyn,
        0 - l1, (int32_t *)Cslstride, (int32_t *)Ctlbound, (int32_t *)Ctlstride, 0,
        (int32_t *)channel_en_C_null,
        (int32_t)D_addr - l1, (int32_t *)D32slstride, (int32_t *)D32tlbound,
        (int32_t *)D32tlstride, 0, (int32_t *)channel_en_D32_dyn, G3_SHAPE, 0u, 0u, 0u, 0, 0, 1, 0,
        1);
    (void)set_versacore_d_shift(d_shift);
    set_versacore_csr(1u, K, N * M, 0, G3_SHAPE, 0);
}

// The launch of a task the generic arm set up (the bases were in its descriptor).
__attribute__((always_inline)) static inline void g3_start(void) {
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(GEMMX_START, 1);
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
}
