// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// One cluster's share of HeMAiA's MoE prefill (DeepSeek-V2-Lite layer 1, 128 tokens): the experts'
// gate|up phase of workloads/moe/common/moe_prefill_mc.py, on one cluster, checked.
//
// A cluster runs MP_NSLOT expert slots. A slot is an expert's rows of X (padded to whole 8-row
// tiles, gathered into L1) and MP_NE column blocks of the expert's gate|up weights, streamed in
// chunks of three gate blocks and as many up blocks (~98 KiB of INT4 weights and the chunk's scale
// tile). Per CHUNK, three engines in the order HeMAiA's per-core task streams run them:
//
//   iDMA   the ring load: the chunk from DRAM (HeMAiA: the chip's L3 ring) into the next of
//          MP_SLABS slabs, after the slab is free
//   GEMM   gemm_start: D = X . W on array shape 3, (8, 8, 16), FP16 into a D buffer (two of them);
//          gemm_wait: the array has retired it
//   SIMD   deq_swiglu: g (.) scale, silu(gate), silu(gate) (.) up into the slot's A16; after the
//          slot's last chunk, quant_rows: A16 into the slot's row-major INT8 A operand
//
// THE DEPENDENCIES (the BINGO graph's edges, as counters in L1):
//   GEMM(i)   needs the slot's X rows, chunk i landed, and the D buffer free: SwiGLU(i - 2) done
//   SwiGLU(i) needs GEMM(i) retired
//   Ld(i)     needs slab i % MP_SLABS free: SwiGLU(i - MP_SLABS) done -- the scale tile is read
//             from the slab; with MP_SCALE=sep the scale tile is not in the slab and the slab is
//             free when GEMM(i - MP_SLABS) has retired
//
// THE LEVERS (build knobs, the Makefile; the snax-level levers of the MoE prefill tracker):
//   MP_LAYOUT  the INT4 weights' layout: parity (HeMAiA today), paired, b            [L09]
//   MP_LAUNCH  how the GEMM is launched: generic (HeMAiA's gemm_start path), lean arm every
//              chunk, delta (arm once per geometry, then only the bases)              [L10]
//   MP_ASYNC   1: the next GEMM is launched while the previous runs (the config queue)
//   MP_SCALE   slab: the scale tile rides in the weight slab (today); sep: its own buffer  [L11]
//   MP_SLABS   weight slabs (HeMAiA: 4)
//   MP_PACE    cycles between chunk arrivals (0: the DRAM stream's own rate); a model of the
//              D2D push that feeds HeMAiA's rings
//
// The report: the window (every engine's start to the last quantiser), the array's passes, the
// time each engine spent waiting and working, and one line a chunk. main returns the number of
// a8 values that differ from the golden, plus one if a wait ran out.

#include "data.h"
#include "snax-core-roles.h"
#include "snax-dsv2-g3.h"
#include "snax-perf-census.h"

#if !SNAX_HAS_GEMM_CORE || !SNAX_HAS_SIMD_CORE || !SNAX_HAS_IDMA_CORE
#error "this app needs the gemm, simd and idma cores"
#endif

#ifndef MP_LAYOUT
#define MP_LAYOUT 3
#endif
#ifndef MP_ALAYOUT
#define MP_ALAYOUT 0  // A's layout: 0 row-major (gathered X rows), 1 A-layout tiles, 2 interleaved
#endif
#ifndef MP_LAUNCH
#define MP_LAUNCH 0
#endif
#ifndef MP_ASYNC
#define MP_ASYNC 1
#endif
#ifndef MP_SCALE
#define MP_SCALE 0
#endif
#ifndef MP_SLABS
#define MP_SLABS 4
#endif
#ifndef MP_PACE
#define MP_PACE 0
#endif
#ifndef MP_TIMELINE
#define MP_TIMELINE 1
#endif
#ifndef MP_SCALE_SKEW
#define MP_SCALE_SKEW 0  // MP_SCALE=sep: bytes past a 256-byte boundary the scale slots start
#endif
#ifndef MP_CENSUS
#define MP_CENSUS 0  // MP_ASYNC=0 only: the array's stall census of every GEMM, summed
#endif
#ifndef MP_PERF
#define MP_PERF 0    // the cluster's contention census over the window
#endif
#if MP_CENSUS && MP_ASYNC
#error "MP_CENSUS reads the array's per-task counters: MP_ASYNC=0"
#endif
#if MP_SLABS < 2
#error "MP_SLABS: at least 2 (the GEMM's wait trails the next start)"
#endif

#define TAG "[MP]"
#define MP_GEN 0
#define MP_LEAN 1
#define MP_DELTA 2
#define MP_NSC 4u                 // scale tiles in flight (MP_SCALE=sep)
#define MP_DBLK G3_DTILE          // one D block: 8 rows x 16 FP16
#if MP_DBLK != 8u * SIMD_BEAT_BYTES
#error "a D block is eight SIMD beats of one row each"
#endif
#define MP_TMO_LIMIT 4000000u

static const g3_lay_t mp_lay = (g3_lay_t)MP_LAYOUT;
#if MP_LAYOUT == 1
#define MP_CHUNKS mp_chunks_b
#define MP_BOFF MP_BOFF_B
#elif MP_LAYOUT == 2
#define MP_CHUNKS mp_chunks_paired
#define MP_BOFF MP_BOFF_PAIRED
#else
#define MP_CHUNKS mp_chunks_parity
#define MP_BOFF MP_BOFF_PARITY
#endif

// ---- shared words ----------------------------------------------------------------------
#define XE_CNT sy[0]    // slots whose X rows are in L1                (iDMA -> GEMM)
#define LD_CNT sy[1]    // chunks landed in their slabs                (iDMA -> GEMM)
#define G_CNT sy[2]     // GEMMs retired                               (GEMM -> SIMD, iDMA)
#define SW_CNT sy[3]    // chunks dequantised and SwiGLU'd             (SIMD -> GEMM, iDMA)
#define DONE sy[4]      // the last quantiser has drained              (SIMD -> iDMA)
#define GOLD_OK sy[5]   // the golden is in L1                         (iDMA -> SIMD)
#define TMO sy[6]       // bounded waits that ran out
#define T_ORG sy[7]     // the cycle origin
#define T_END sy[8]     // the last quantiser's end
#define W_XE sy[9]      // GEMM: waiting for a slot's X rows
#define W_LD sy[10]     //   for a chunk to land
#define W_GB sy[11]     //   for a D buffer to be free
#define W_RET sy[12]    //   for the array to retire (trailing the next start under MP_ASYNC)
#define T_ARM sy[13]    //   launching: the arm and the start
#define W_SLAB sy[14]   // iDMA: waiting for a slab (and a scale slot) to be free
#define W_PACE sy[15]   //   for the pace
#define T_XFER sy[16]   //   in transfers (X rows and chunks)
#define W_G sy[17]      // SIMD: waiting for a GEMM
#define T_SW sy[18]     //   in deq_swiglu
#define T_Q sy[19]      //   in quant_rows
#define C_BUSY sy[20]   // MP_CENSUS: the array's busy cycles, summed over the GEMMs
#define C_WA sy[21]     //   of which waiting for A
#define C_WB sy[22]     //   for B
#define C_WD sy[23]     //   for the array / the D drain
#define MP_SY_TS 32     // the per-chunk timestamps: 6 arrays of MP_NCHUNK words
#define TS(a, i) sy[MP_SY_TS + (a) * MP_NCHUNK + (i)]
#define TS_LD 0         //   chunk landed
#define TS_GF 1         //   GEMM launched (the start returned)
#define TS_GD 2         //   GEMM retired (observed)
#define TS_SB 3         //   SwiGLU begun
#define TS_SE 4         //   SwiGLU done
#define TS_ARM 5        //   cycles the launch took
#define MP_SY_BYTES ((((MP_SY_TS + 6 * MP_NCHUNK + MP_NSLOT * 2) * 4u) + 255u) & ~255u)
#define TQ(s, e) sy[MP_SY_TS + 6 * MP_NCHUNK + 2 * (s) + (e)]  // slot s's quantiser begin / end

typedef struct {
    volatile uint32_t *sy;
    uint8_t *slab[MP_SLABS];
    uint8_t *scale[MP_NSC];
    uint8_t *xe[2];
    uint8_t *gb[2];
    uint8_t *sg;
    uint8_t *a16;
    uint8_t *a8[MP_NSLOT];
    int8_t *gold;
    uint32_t gold_bytes;
    g3_shape_t *shape;
} mp_t;

// Bytes of a rounded-up multiple of n.
#define MP_UP(x, n) ((((x) + (n) - 1u) / (n)) * (n))

// Wait until *w >= v, adding the cycles waited to *acc.
static inline void mp_w(volatile uint32_t *w, uint32_t v, volatile uint32_t *tmo,
                        volatile uint32_t *acc) {
    if (*w >= v) return;
    const uint32_t t0 = snrt_mcycle();
    dsv2_spin_ge(w, v, tmo);
    *acc += snrt_mcycle() - t0;
}

// ============================================================================ the SIMD passes
//
// Ports of HeMAiA's simd_deq_swiglu and simd_quant_rows (offload_hw_kernels/moe_prefill.h) onto
// this tree's SIMD library. A D-layout matrix is row tiles of 8 rows; a block is 8 x 16 FP16, 256
// bytes, one row of it one SIMD beat.

static inline uint32_t mp_drain(void) {
    return snax_simd_wait_all_checked("mp", SIMD_WAIT_BUDGET);
}

// Program one EW1 pass over the pairs (a[j], b[j]), j < beats, beats `beat_stride` apart in both
// operands, written to dst as an (ob0, os0) x (ob1, os1) sweep. The reader starts at the lower
// operand; the operand stride is unsigned.
static inline void mp_ew2_program(void *a, void *b, uint32_t beats, uint32_t beat_stride,
                                  void *dst, uint32_t ob0, uint32_t os0, uint32_t ob1,
                                  uint32_t os1) {
    const uint32_t ua = (uint32_t)a, ub = (uint32_t)b;
    snax_simd_shape_t in, out;
    snax_simd_shape_2d(&in, ua < ub ? a : b, 2u, ua < ub ? ub - ua : ua - ub, beats, beat_stride);
    snax_simd_shape_2d(&out, dst, ob0, os0, ob1, os1);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 2u,
                   SIMD_EW_MUL);
    snax_simd_program_fast(&in, &out);
}

// The programmed pass moved to the operands a, b and the output dst.
static inline void mp_ew2_repoint(uint32_t a, uint32_t b, uint32_t dst) {
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_LSB, a < b ? a : b);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 0u, a < b ? b - a : a - b);
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_LSB, dst);
}

// A gate|up GEMM output g ([tiles][2 nb][8][16] FP16: each row tile's nb gate blocks, then its nb up
// blocks) times its scale tile s (2 nb blocks, one row tile's), in place; then silu(gate) * up of
// every row tile into out + m * out_tile_stride (nb blocks). sg: scratch as large as g.
static uint32_t mp_deq_swiglu(uint8_t *g, uint8_t *s, uint8_t *sg, uint8_t *out, uint32_t tiles,
                              uint32_t nb, uint32_t out_tile_stride) {
    const uint32_t half = nb * MP_DBLK, tile = 2u * half, hb = nb * 8u;  // hb: beats in a half
    uint32_t bad = 0;
    // 1) g (.) s, row tile by row tile, in place: programmed once, each tile's pass moved and queued
    for (uint32_t m = 0; m < tiles; m++) {
        uint8_t *gm = g + m * tile;
        if (m == 0u)
            mp_ew2_program(gm, s, 2u * hb, SIMD_BEAT_BYTES, gm, 2u * hb, SIMD_BEAT_BYTES, 1u, 0u);
        else
            mp_ew2_repoint((uint32_t)gm, (uint32_t)s, (uint32_t)gm);
        snax_simd_fire();
    }
    bad += mp_drain();
    // 2) silu over the whole of g (the up halves' silu is never read)
    {
        snax_simd_shape_t in, o;
        snax_simd_shape_flat(&in, g, tiles * 2u * hb);
        snax_simd_shape_flat(&o, sg, tiles * 2u * hb);
        snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_ONE, 0u,
                       SIMD_FUNC_SILU);
        snax_simd_program_fast(&in, &o);
        snax_simd_fire();
        bad += mp_drain();
    }
    // 3) silu(gate) (.) up: in every row tile the gate half of sg against the up half of g
    {
        const uint32_t ua = (uint32_t)sg, ub = (uint32_t)(g + half);
        snax_simd_shape_t in, o;
        snax_simd_shape_2d(&in, ua < ub ? (void *)sg : (void *)(g + half), 2u,
                           ua < ub ? ub - ua : ua - ub, hb, SIMD_BEAT_BYTES);
        in.bound[2] = tiles;
        in.stride[2] = tile;
        in.dim = 3u;
        snax_simd_shape_2d(&o, out, hb, SIMD_BEAT_BYTES, tiles, out_tile_stride);
        snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 2u,
                       SIMD_EW_MUL);
        snax_simd_program_fast(&in, &o);
        snax_simd_fire();
        bad += mp_drain();
    }
    return bad;
}

// Rows i < rows of a D-layout A16 (row i's nb beats 256 B apart from src + (i / 8) *
// src_tile_stride + (i % 8) * 32), quantised with inv_bits (the static 1 / scale) into row i of a
// row-major INT8 matrix at dst + i * dst_pitch. One pass a row, all of one geometry: programmed
// once, each row then sets its two bases and is queued.
static uint32_t mp_quant_rows(uint8_t *src, uint32_t rows, uint32_t nb, uint32_t src_tile_stride,
                              uint32_t inv_bits, uint8_t *dst, uint32_t dst_pitch) {
    const uint32_t nb2 = (nb + 1u) & ~1u;  // the quantiser packs two beats into one
    snax_simd_shape_t in, out;
    snax_simd_shape_2d(&in, src, nb2, MP_DBLK, 1u, 0u);
    snax_simd_shape_flat(&out, dst, nb2 / 2u);
    snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_ONE, 0u, SIMD_FUNC_LINEAR);
    snax_simd_arm2(SIMD_EXT_FP16TOINT8, SIMD_EXT_FP16TOINT8_CSR, inv_bits, SIMD_QUANT_TAIL(0));
    snax_simd_program_fast(&in, &out);
    snax_simd_fire();
    for (uint32_t i = 1; i < rows; i++) {
        snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_LSB, (uint32_t)src + (i / 8u) * src_tile_stride +
                                                           (i % 8u) * SIMD_BEAT_BYTES);
        snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_LSB, (uint32_t)dst + i * dst_pitch);
        snax_simd_fire();
    }
    return mp_drain();
}

// ============================================================================ the iDMA hart
// The ring loads, in the order the DM core's program runs them.

static void mp_loader(mp_t *m) {
    volatile uint32_t *sy = m->sy;
    uint32_t t_last = snrt_mcycle();
    for (uint32_t i = 0; i < MP_NCHUNK; i++) {
        const uint32_t s = i / MP_CPS, j = i % MP_CPS;
        const mp_chunk_t *ch = &MP_CHUNKS[i];
        if (j == 0u) {  // the slot's rows of X, into the buffer slot s - 2 has finished with
            if (s >= 2u) mp_w(&G_CNT, (s - 1u) * MP_CPS, &TMO, &W_SLAB);
            const uint32_t t0 = snrt_mcycle();
#if MP_ALAYOUT == 0
            snrt_dma_start_2d(m->xe[s & 1u], mp_slots[s].x, MP_K, MP_PX, MP_K, mp_slots[s].rows);
#else  // X already in A-layout tiles (HeMAiA's gather writes rows; a tile layout is a reshuffle)
            snrt_dma_start_1d(m->xe[s & 1u], MP_ALAYOUT == 1 ? mp_slots[s].xb : mp_slots[s].xi,
                              mp_slots[s].rows * MP_K);
#endif
            snrt_dma_wait_all();
            T_XFER += snrt_mcycle() - t0;
            XE_CNT = s + 1u;
        }
        // the slab: free when its last user is done with it
        if (i >= MP_SLABS) mp_w(MP_SCALE ? &G_CNT : &SW_CNT, i - MP_SLABS + 1u, &TMO, &W_SLAB);
        if (MP_SCALE && i >= MP_NSC) mp_w(&SW_CNT, i - MP_NSC + 1u, &TMO, &W_SLAB);
#if MP_PACE
        {
            const uint32_t t0 = snrt_mcycle();
            while (snrt_mcycle() - t_last < MP_PACE) {
            }
            W_PACE += snrt_mcycle() - t0;
        }
#endif
        const uint32_t t0 = snrt_mcycle();
#if MP_SCALE
        snrt_dma_start_1d(m->slab[i % MP_SLABS], ch->img, ch->soff);
        snrt_dma_start_1d(m->scale[i % MP_NSC], ch->img + ch->soff, 2u * ch->nb * MP_DBLK);
#else
        snrt_dma_start_1d(m->slab[i % MP_SLABS], ch->img, ch->bytes);
#endif
        snrt_dma_wait_all();
        t_last = snrt_mcycle();
        T_XFER += t_last - t0;
        TS(TS_LD, i) = t_last;
        LD_CNT = i + 1u;
    }
    // the golden in, once the last quantiser has drained
    volatile uint32_t unused = 0;
    mp_w(&DONE, 1u, &TMO, &unused);
    snrt_dma_start_1d(m->gold, mp_gold_a8, m->gold_bytes);
    snrt_dma_wait_all();
    GOLD_OK = 1u;
}

// ============================================================================ the GEMM hart

static void mp_gemm(mp_t *m) {
    volatile uint32_t *sy = m->sy;
    uint32_t id = csrr_ss(GEMMX_FINISHED_TASK);
    uint32_t armed_mt = 0, armed_nt = 0;
    for (uint32_t i = 0; i < MP_NCHUNK; i++) {
        const uint32_t s = i / MP_CPS, j = i % MP_CPS;
        const mp_chunk_t *ch = &MP_CHUNKS[i];
        const uint32_t mt = mp_slots[s].rows / 8u, nt = 2u * ch->nb;
        if (j == 0u) mp_w(&XE_CNT, s + 1u, &TMO, &W_XE);
        mp_w(&LD_CNT, i + 1u, &TMO, &W_LD);
        if (i >= 2u) mp_w(&SW_CNT, i - 1u, &TMO, &W_GB);  // gb[i % 2]: SwiGLU(i - 2) is done
        const uint32_t ta = snrt_mcycle();
        const uint32_t a = (uint32_t)m->xe[s & 1u];
        const uint32_t b = (uint32_t)m->slab[i % MP_SLABS] + MP_BOFF;
        const uint32_t d = (uint32_t)m->gb[i & 1u];
#if MP_LAUNCH == MP_GEN
        g3_arm_generic(m->shape, a, b, d, mt, MP_KT, nt, MP_KSHIFT, mp_lay, MP_ALAYOUT, MP_PX);
        g3_start();
#elif MP_LAUNCH == MP_LEAN
        g3_arm(mt, MP_KT, nt, MP_KSHIFT, mp_lay, MP_ALAYOUT, MP_PX);
        g3_fire((const void *)a, (const void *)b, (void *)d);
#else
        if (mt != armed_mt || nt != armed_nt) {
            g3_arm(mt, MP_KT, nt, MP_KSHIFT, mp_lay, MP_ALAYOUT, MP_PX);
            armed_mt = mt;
            armed_nt = nt;
        }
        g3_fire((const void *)a, (const void *)b, (void *)d);
#endif
        const uint32_t tb = snrt_mcycle();
        T_ARM += tb - ta;
        TS(TS_ARM, i) = tb - ta;
        TS(TS_GF, i) = tb;
        ++id;
#if MP_ASYNC
        if (i) {  // the previous GEMM's wait trails this start
            const uint32_t t0 = snrt_mcycle();
            TMO += dsv2_gemm_wait(id - 1u);
            const uint32_t t1 = snrt_mcycle();
            W_RET += t1 - t0;
            TS(TS_GD, i - 1u) = t1;
            G_CNT = i;
        }
#else
        {
            const uint32_t t0 = snrt_mcycle();
            TMO += dsv2_gemm_wait(id);
            const uint32_t t1 = snrt_mcycle();
            W_RET += t1 - t0;
            TS(TS_GD, i) = t1;
#if MP_CENSUS
            C_BUSY += csrr_ss(GEMMX_PERFORMANCE_COUNTER);
            C_WA += csrr_ss(GEMMX_STALL_A);
            C_WB += csrr_ss(GEMMX_STALL_B);
            C_WD += csrr_ss(GEMMX_STALL_D);
#endif
            G_CNT = i + 1u;
        }
#endif
    }
#if MP_ASYNC
    {
        const uint32_t t0 = snrt_mcycle();
        TMO += dsv2_gemm_wait(id);
        const uint32_t t1 = snrt_mcycle();
        W_RET += t1 - t0;
        TS(TS_GD, MP_NCHUNK - 1u) = t1;
        G_CNT = MP_NCHUNK;
    }
#endif
}

// ============================================================================ the SIMD hart

static void mp_simd(mp_t *m) {
    volatile uint32_t *sy = m->sy;
    for (uint32_t i = 0; i < MP_NCHUNK; i++) {
        const uint32_t s = i / MP_CPS, j = i % MP_CPS;
        const mp_chunk_t *ch = &MP_CHUNKS[i];
        mp_w(&G_CNT, i + 1u, &TMO, &W_G);
        uint8_t *scale = MP_SCALE ? m->scale[i % MP_NSC] : m->slab[i % MP_SLABS] + ch->soff;
        uint8_t *out = m->a16 + ch->b0 * MP_DBLK;
        const uint32_t tiles = mp_slots[s].rows / 8u, nb = ch->nb;
        const uint32_t t0 = snrt_mcycle();
        TS(TS_SB, i) = t0;
        TMO += mp_deq_swiglu(m->gb[i & 1u], scale, m->sg, out, tiles, nb, MP_NE * MP_DBLK);
        const uint32_t t1 = snrt_mcycle();
        T_SW += t1 - t0;
        TS(TS_SE, i) = t1;
        SW_CNT = i + 1u;
        if (j == MP_CPS - 1u) {  // the slot's last chunk: its A16 into the row-major A8
            TQ(s, 0) = t1;
            TMO += mp_quant_rows(m->a16, mp_slots[s].rows, MP_NE, MP_NE * MP_DBLK, MP_INV_BITS,
                                 m->a8[s], MP_PE);
            const uint32_t t2 = snrt_mcycle();
            T_Q += t2 - t1;
            TQ(s, 1) = t2;
        }
    }
    T_END = snrt_mcycle();
    DONE = 1u;
}

// ============================================================================ main

int main() {
    // ---- L1, derived identically on every hart -------------------------------------------------
    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 255u) & ~255u);
    mp_t mp;
    mp_t *m = &mp;
#define TAKE(n) (p += MP_UP((n), 256u), p - MP_UP((n), 256u))
    m->sy = (volatile uint32_t *)TAKE(MP_SY_BYTES);
    volatile uint32_t *sy = m->sy;
    const uint32_t slab_bytes = MP_UP(MP_IMGMAX, 256u);
    for (uint32_t i = 0; i < MP_SLABS / 2u; i++) m->slab[i] = TAKE(slab_bytes);
    const uint32_t gb_bytes = MP_ROWMAX / 8u * 2u * MP_NBMAX * MP_DBLK;
    m->gb[0] = TAKE(gb_bytes);
    m->gb[1] = TAKE(gb_bytes);
    m->sg = TAKE(gb_bytes);
    for (uint32_t i = 0; i < MP_NSC; i++)
        m->scale[i] = TAKE(2u * MP_NBMAX * MP_DBLK + 256u) + MP_SCALE_SKEW;
    for (uint32_t i = MP_SLABS / 2u; i < MP_SLABS; i++) m->slab[i] = TAKE(slab_bytes);
    m->xe[0] = TAKE(MP_ROWMAX * MP_PX);
    m->xe[1] = TAKE(MP_ROWMAX * MP_PX);
    m->a16 = TAKE(MP_ROWMAX / 8u * MP_NE * MP_DBLK + MP_DBLK);
    uint32_t goldb = 0;
    for (uint32_t s = 0; s < MP_NSLOT; s++) {
        m->a8[s] = TAKE(mp_slots[s].rows * MP_PE);
        goldb += mp_slots[s].rows * MP_NE * 16u;
    }
    m->gold_bytes = goldb;
    m->gold = (int8_t *)TAKE(goldb);
    m->shape = (g3_shape_t *)TAKE(64);
#undef TAKE

    const int isG = snax_is_gemm_core(), isS = snax_is_simd_core(), isD = snax_is_idma_core();
    if (isS) {
        for (uint32_t i = 0; i < MP_SY_BYTES / 4u; i++) sy[i] = 0;
        dsv2_tcdm_prio_install(DSV2_TCDM_PRIO);
        dsv2_print_l1((uint32_t)(p - (uint8_t *)sy), DSV2_L1_TOP - (uint32_t)sy);
    }
    if ((uint32_t)p > DSV2_L1_TOP) {
        if (isS) printf(TAG " FAIL: the arena ends at 0x%08x, past the stacks at 0x%08x\n",
                        (uint32_t)p, DSV2_L1_TOP);
        return isS ? 1 : 0;
    }
    if (isG) {  // the generic arm's table, in L1
        for (uint32_t i = 0; i < sizeof(g3_shape_t) / 4u; i++)
            ((volatile uint32_t *)m->shape)[i] = ((const uint32_t *)&g3_shape_rom)[i];
    }
    snrt_cluster_hw_barrier();
    if (isG) T_ORG = snrt_mcycle();
#if MP_PERF
    if (isS) snax_perf_arm();  // the contention census covers the window
#endif
    snrt_cluster_hw_barrier();
    while (T_ORG == 0u) {  // the barrier does not order the store
    }

    if (isD) {
        mp_loader(m);
        return 0;
    }
    if (isG) {
        mp_gemm(m);
        return 0;
    }
    if (!isS) return 0;

    // ============================== the SIMD
    mp_simd(m);
#if MP_PERF
    snax_perf_snapshot_t census;
    snax_perf_read(&census);
#endif
    dsv2_spin_ge(&GOLD_OK, 1u, &TMO);
    uint32_t bad = 0, worst = 0;
    for (uint32_t s = 0; s < MP_NSLOT; s++) {
        for (uint32_t r = 0; r < mp_slots[s].rows; r++) {
            uint32_t w;
            bad += dsv2_lsb8((const int8_t *)m->a8[s] + r * MP_PE,
                             m->gold + mp_slots[s].gold + r * MP_NE * 16u, MP_NE * 16u, &w);
            if (w > worst) worst = w;
        }
    }
    const uint32_t org = T_ORG, win = T_END - org;
    uint32_t passes = 0;
    for (uint32_t i = 0; i < MP_NCHUNK; i++)
        passes += mp_slots[i / MP_CPS].rows / 8u * MP_KT * 2u * MP_CHUNKS[i].nb;
    printf(TAG " layout %u launch %u async %u scale %u slabs %u pace %u alay %u boff %u: %u slots, %u chunks\n",
           MP_LAYOUT, MP_LAUNCH, MP_ASYNC, MP_SCALE, MP_SLABS, MP_PACE, MP_ALAYOUT, MP_BOFF,
           MP_NSLOT, MP_NCHUNK);
    const uint32_t um = passes * 1000u / win;
    printf(TAG " window %u cc  array passes %u  util %u.%03u\n", win, passes, um / 1000u,
           um % 1000u);
    printf(TAG " gemm: wait x %u ld %u gb %u ret %u  launching %u  | idma: wait slab %u pace %u "
              "xfer %u  | simd: wait gemm %u swiglu %u quant %u\n",
           W_XE, W_LD, W_GB, W_RET, T_ARM, W_SLAB, W_PACE, T_XFER, W_G, T_SW, T_Q);
#if MP_CENSUS
    printf(TAG " array census (sum over the GEMMs): busy %u wait A %u B %u D %u  (%u passes)\n",
           C_BUSY, C_WA, C_WB, C_WD, passes);
#endif
#if MP_PERF
    snax_perf_report(&census, win);
#endif
#if MP_TIMELINE
    for (uint32_t i = 0; i < MP_NCHUNK; i++)
        printf(TAG " c%02u s%u ld %6u gf %6u gd %6u sb %6u se %6u arm %4u\n", i, i / MP_CPS,
               TS(TS_LD, i) - org, TS(TS_GF, i) - org, TS(TS_GD, i) - org, TS(TS_SB, i) - org,
               TS(TS_SE, i) - org, TS(TS_ARM, i));
    for (uint32_t s = 0; s < MP_NSLOT; s++)
        printf(TAG " q%u %6u %6u\n", s, TQ(s, 0) - org, TQ(s, 1) - org);
#endif
    if (TMO) printf(TAG " FAIL: %u bounded waits ran out\n", TMO);
    printf(bad || TMO ? TAG " FAIL (%u a8 values differ, worst %u LSB)\n" : TAG " PASS (a8 bit-exact)\n",
           bad, worst);
    return (int)(bad + (TMO ? 1u : 0u));
}
