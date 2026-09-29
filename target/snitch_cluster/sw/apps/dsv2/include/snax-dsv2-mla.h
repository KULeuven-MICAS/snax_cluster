// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// DeepSeek-V2-Lite layer 1's MLA block for NTOK = 1 or 2 consecutive tokens, x -> h = x + MLA(x),
// with each token's row appended to the latent cache, on the four-engine split cluster:
//
//      1  xn  = rmsnorm(x)                          V1, V2 into the GEMV A operand
//      3  ckv = xn . W_DKV    [c 512 | k_pe 64]      G1 + G3
//      2  q   = xn . W_Q      16 x [q_nope 128 | q_pe 64]
//      4  cn  = rmsnorm(c)                          V1, V2 -> c8, the row's latent
//      5  RoPE at pos on the 16 q_pe and the k_pe    V3, V2 -> kpe8, the row's key tail
//      6  q~  = q_nope_h . W_UK_h, per head          G4
//      7  append [c8 | kpe8] to both cache copies    D1
//   8-10  scores, softmax, weighted sum -> o~        D2, A1, V4
//     11  o_h = o~_h . W_UV_h, per head              G4
//     12  attn = o . W_O                             G1 + G3
//     13  h = x + attn                               EW ADD
//
// The app includes data.h first, for the model's dimensions (HID, HEADS, Q_HEAD, Q_NOPE,
// ROPE_DIM, KV_RANK, V_HEAD), the tile BC and cache CAP, the static scales (INV_*, A_EXP,
// P8_INV), the D-port shifts (K_X, K_UK, K_S, K_UV) and the blobs this block streams: wdkv, wq,
// wuk, wuv, wo, their factors s_kv .. s_o, and the cache copies key and val.
//
// ======================================================================================
// ONE STREAM OF LOADS, ONE STREAM OF GEMM TASKS
// ======================================================================================
//
// Every GEMM task of the block has a global index, and so does every load into L1 the iDMA
// makes for one. The two sequences are
//
//   tasks   W_DKV chunks | W_Q chunks | W_UK heads | QK0 QK1 PV0 QK2 PV1 .. PV(nt-1) | W_UV heads | W_O chunks
//   loads   W_DKV chunks | W_Q chunks | W_UK heads | K0  V0  K1  V1  ..       V(nt-1) | W_UV heads | W_O chunks
//
// (a chunk is 32 output columns, K x 32 bytes = 64 KiB at K = 2,048; a head's absorbed weight
// is 64 KiB too; both half that with INT4 weights, DSV2_WBITS), so outside attention load t is
// what task t reads. Two SLABS of 68 KiB hold
// them: a GEMV load goes to slab (t & 1); attention tile j goes to slab (j & 1), K in its
// first 36 KiB and V behind it. Two monotonic counters carry every handoff between the
// iDMA and the GEMM: LOADED (loads landed) and RETIRED (tasks retired). A load waits until
// the task that last read its region has retired; a task waits until its load has landed.
// The SIMD watches RETIRED for the outputs it post-processes and publishes one flag per
// operand it produces.
//
// TWO WAYS IN (DSV2_DUAL_LOAD, sw/apps/dsv2/dsv2.mk; on by default). A streamed chunk of W_DKV, W_Q
// or W_O loads in two parts at once: its head on the iDMA, which reads main memory with AXI
// reads, and its last DSV2_DUAL_XBYTES on the xDMA, whose main-memory endpoint pushes them into
// the cluster as AXI writes (snax-xdma-lib.h). The two routes share no channel and no TCDM
// port, so a chunk lands in about half the time. The iDMA still runs the stream: once a slab is
// free it raises XREQ, and it publishes LOADED when its part has landed and XLOADED says the
// xDMA's has. W_UK, W_UV and the attention tiles stay on the iDMA.
//
// One token's GEMV reads 128 B of weights and 8 B of A a cycle, so the TCDM has room for the second
// writer: with the TCDM arbitrating on every requester's urgency (snax-tcdm-priority.h) the
// xDMA's part lands at 59 B a cycle, the iDMA's at 61, and W_Q streams at 82 B a cycle against 57
// on the iDMA alone. Two tokens run the (16, 4, 16) shape at 64 B a cycle, which the iDMA feeds
// alone, so they load on the iDMA only (MLA_DUAL).
//
//   iDMA (hart 3)   x, the factors, the RoPE tables; the load stream; the cache append
//   GEMM (hart 0)   the task stream
//   SIMD (hart 1)   every SIMD stage, in order
//   xDMA (hart 2)   seeds m; stages the RoPE row and its pair swap (core code: a pair of FP16
//                   is one word, so the swap is a 16-bit rotate); transposes o~; with
//                   MLA_DUAL, its part of every W_DKV, W_Q and W_O chunk
//
// THE PASS. Token t sits at position pos + t; pos + NTOK keys run in nt = ceil((pos + NTOK) /
// BC) tiles. A last tile with fewer keys runs whole: the cache's rows past the pass are zero, and
// before its softmax the SIMD overwrites their scores with -65504 (one Map task), so their P is
// exactly 0 (hwmodel.mla_attention).
//
// TWO TOKENS (NTOK = 2, from data.h; 1 by default). Every GEMV carries both: A holds token 0 in
// row 0 and token 1 in row 4, and the D walk splits them into two outputs a pitch apart
// (snax-dsv2.h dsv2_gemv_two_tokens), so the weights stream once for two tokens. Per-token
// buffers sit at fixed pitches, so a SIMD task reads both through one extra loop. The attention
// is one pass over the keys: Q8's 32 rows are token 0's 16 heads and token 1's, and token 0 must
// not see token 1's key -- the last one -- whose score its 16 lanes get as -65504 from a Map
// task that writes only those lanes.
//
// USE. The whole L1 the block carves (dsv2_mla_carve) must be zero when a pass starts (TCDM
// has no reset, and Q8's rows 16..31 and l must start at zero), and its sync words fresh: the
// caller zeroes the region, then every hart runs its role. The output h (NTOK rows MLA_HP apart,
// each with a free beat below it, for a norm on h) and the sync words are the caller's.

#pragma once

#include "snax-dsv2.h"
#include "snax-dsv2-trace.h"

#ifndef DSV2_DUAL_LOAD
#define DSV2_DUAL_LOAD 1
#endif
// The bytes of each streamed chunk the xDMA carries (its tail); the iDMA carries the rest.
#ifndef DSV2_DUAL_XBYTES
#define DSV2_DUAL_XBYTES (MLA_CHW / 2u)
#endif

#if !SNAX_HAS_GEMM_CORE || !SNAX_HAS_SIMD_CORE || !SNAX_HAS_XDMA_CORE || !SNAX_HAS_IDMA_CORE
#error "the MLA block needs the four-engine cluster"
#endif
#if !defined(READER_WRITER_EXTENSION_0_CSR_BASE) || READER_WRITER_EXTENSION_0_CSR_NUM != 18
#error "the O rescale needs the C read path's Int32ColumnScale (18 CSRs)"
#endif
#define MLA_COLSCALE READER_WRITER_EXTENSION_0_CSR_BASE

#ifndef NTOK
#define NTOK 1
#endif
#if NTOK != 1 && NTOK != 2
#error "a pass holds one or two tokens: 16 heads each in the 32 query lanes"
#endif

// Whether the streamed chunks load in two parts (DSV2_DUAL_LOAD): with one token only. Two
// tokens run the (16, 4, 16) shape, 64 weight bytes a cycle, which the iDMA's stream feeds alone.
#if DSV2_DUAL_LOAD && NTOK == 1
#define MLA_DUAL 1
#else
#define MLA_DUAL 0
#endif

#define MLA_BR 32                            // query lanes: 16 heads per token, zero rows past them
#define MLA_DQK 576                          // [c | k_pe]
#define MLA_DV KV_RANK                       // 512
#define MLA_Q_N (HEADS * Q_HEAD)             // 3072
#define MLA_ROPE_N ((HEADS + 1) * ROPE_DIM)  // 1,088: 16 q_pe and one k_pe
#define MLA_ROPE_B (2u * MLA_ROPE_N)         // one RoPE operand, 34 beats
#define MLA_CH 32u                           // GEMV chunk: output columns per task
#define MLA_CHB (HID * MLA_CH)               // 64 KiB of INT8 weights
#define MLA_CHW DSV2_WB(MLA_CHB, DSV2_W4)    // a chunk's bytes at the weights' width
#define MLA_KT_S (MLA_DQK / DSV2_KU)         // 144: QK's contraction blocks
#define MLA_M_S (BC / DSV2_MR)
#define MLA_N_Q (MLA_BR / DSV2_NU)           // 2
#define MLA_KT_P (BC / DSV2_KU)
#define MLA_M_P (MLA_DV / DSV2_MR)           // 32
#define MLA_SBEATS BC
#define MLA_PBEATS (BC / 2)
#define MLA_KTILE (BC * MLA_DQK)             // 36 KiB
#define MLA_VTILE (MLA_DV * BC)              // 32 KiB
#define MLA_SLAB (MLA_KTILE + MLA_VTILE)
#define MLA_QT_PITCH (2u * MLA_DV + DSV2_BEAT)  // q~ rows: 17 beats, so A-order reads meet 2-way bank conflicts
#define MLA_UK_BYTES DSV2_WB(Q_NOPE * MLA_DV, DSV2_W4)  // one head's W_UK: 64 KiB at INT8
#define MLA_UV_BYTES DSV2_WB(MLA_DV * V_HEAD, DSV2_W4)  // one head's W_UV: 64 KiB at INT8
#define MLA_SCORE_MASK 0xC77FE000u           // FP32 -65504: a padding key's score
#define MLA_TOK0_LANES 0x0Fu                 // writer channels 0..3: FP16 lanes 0..15, token 0's heads

// Per-token pitches (bytes). A row with a norm on it keeps a free beat below it.
#define MLA_XP (DSV2_BEAT + 2u * HID)        // x, and the caller's h
#define MLA_HP MLA_XP
#define MLA_CKP (DSV2_BEAT + 2u * MLA_DQK)   // ckv
#define MLA_ROPE_BT (NTOK * MLA_ROPE_B)      // one RoPE operand, all tokens
#define MLA_XTP (2u * HEADS * MLA_DV)        // o~, [16, 512] per token
// A GEMV output of `bytes` per token with n-blocks nb per chunk: one token's buffer ends in the
// packed output's 15 slots of spill; two tokens' outputs sit a pitch apart, 4 slots each
// (snax-dsv2.h, two tokens per GEMV).
#if NTOK == 1
#define MLA_YP(bytes, nb) 0u
#define MLA_YBUF(bytes, nb) ((bytes) + DSV2_SPILL(nb))
#else
#define MLA_YP(bytes, nb) ((((bytes) + DSV2_SPILL2(nb)) + 63u) & ~63u)
#define MLA_YBUF(bytes, nb) (2u * MLA_YP(bytes, nb))
#endif
#define MLA_YP_KV MLA_YP(2u * MLA_DQK, MLA_CH / DSV2_NU)
#define MLA_YP_Q MLA_YP(2u * MLA_Q_N, MLA_CH / DSV2_NU)
#define MLA_YP_UK MLA_YP(HEADS * DSV2_SLOT(MLA_DV / DSV2_NU), MLA_DV / DSV2_NU)
#define MLA_YP_UV MLA_YP(HEADS * DSV2_SLOT(V_HEAD / DSV2_NU), V_HEAD / DSV2_NU)
#define MLA_YP_A MLA_YP(2u * HID, MLA_CH / DSV2_NU)

// ---- the global task / load index -----------------------------------------------------------
#define MLA_NC_KV (MLA_DQK / MLA_CH)         // 18
#define MLA_NC_Q (MLA_Q_N / MLA_CH)          // 96
#define MLA_NC_O (HID / MLA_CH)              // 64
#define MLA_T_KV0 0u
#define MLA_T_Q0 (MLA_T_KV0 + MLA_NC_KV)
#define MLA_T_UK0 (MLA_T_Q0 + MLA_NC_Q)
#define MLA_T_ATT (MLA_T_UK0 + HEADS)
#define MLA_L_K(j) (MLA_T_ATT + 2u * (j))
#define MLA_L_V(j) (MLA_T_ATT + 2u * (j) + 1u)
// the GEMM runs QK(j) then PV(j-1): QK0 QK1 PV0 QK2 PV1 .. QK(nt-1) PV(nt-2) PV(nt-1)
static inline uint32_t mla_t_qk(uint32_t j) { return MLA_T_ATT + (j ? 2u * j - 1u : 0u); }
static inline uint32_t mla_t_pv(uint32_t nt, uint32_t j) {
    return MLA_T_ATT + (j == nt - 1u ? 2u * nt - 1u : 2u * j + 2u);
}
static inline uint32_t mla_t_uv0(uint32_t nt) { return MLA_T_ATT + 2u * nt; }
static inline uint32_t mla_t_o0(uint32_t nt) { return mla_t_uv0(nt) + HEADS; }
static inline uint32_t mla_t_end(uint32_t nt) { return mla_t_o0(nt) + MLA_NC_O; }

// ---- the sync words (1 KiB, fresh per pass) ---------------------------------------------------
#define MLA_LOADED(m) (m)->sy[0]    // loads landed                       (iDMA -> GEMM)
#define MLA_RETIRED(m) (m)->sy[1]   // GEMM tasks retired                 (GEMM -> iDMA, SIMD)
#define MLA_IN_OK(m) (m)->sy[2]     // x, the factors, the RoPE tables    (iDMA -> SIMD)
#define MLA_XA_OK(m) (m)->sy[3]     // xn's A operand                     (SIMD -> GEMM)
#define MLA_Q_DQ(m) (m)->sy[4]      // q and ckv dequantised              (SIMD -> xDMA)
#define MLA_QN_OK(m) (m)->sy[5]     // W_UK's 16 A operands               (SIMD -> GEMM)
#define MLA_ROPE_IN(m) (m)->sy[6]   // the RoPE row and its swap          (xDMA -> SIMD)
#define MLA_ROW_OK(m) (m)->sy[7]    // the cache row [c8 | kpe8]          (SIMD -> iDMA)
#define MLA_APPENDED(m) (m)->sy[8]  // the row is in both copies          (iDMA -> SIMD)
#define MLA_Q_OUT(m) (m)->sy[9]     // Q8 assembled                       (SIMD -> GEMM)
#define MLA_M_INIT(m) (m)->sy[10]   // m seeded to -65504                 (xDMA -> SIMD)
#define MLA_P_OUT(m) (m)->sy[11]    // P8 tiles published                 (SIMD -> GEMM)
#define MLA_N_OUT(m) (m)->sy[12]    // o~^T written                       (SIMD -> xDMA)
#define MLA_X_OUT(m) (m)->sy[13]    // o~ transposed                      (xDMA -> SIMD)
#define MLA_AUV_OK(m) (m)->sy[14]   // W_UV's 16 A operands               (SIMD -> GEMM)
#define MLA_OA_OK(m) (m)->sy[15]    // W_O's A operand                    (SIMD -> GEMM)
#define MLA_DONE(m) (m)->sy[16]     // h computed                         (SIMD -> the caller)
#define MLA_TMO(m) (m)->sy[18]      // bounded waits that ran out
#define MLA_XREQ(m) (m)->sy[24]     // streamed loads whose slab is free  (iDMA -> xDMA hart)
#define MLA_XLOADED(m) (m)->sy[25]  // their second halves landed         (xDMA hart -> iDMA)
#define MLA_T_ORG(m) (m)->sy[19]    // the pass's cycle origin
#define MLA_FIN(m, r) (m)->sy[20 + (r)]  // role r (0 GEMM, 1 iDMA, 2 xDMA hart) wrote its profile words
#define MLA_T_PH(m, i) (m)->sy[32 + (i)]  // GEMM phase i done, cycles from the origin
#define MLA_T_SS(m, i) (m)->sy[48 + (i)]  // SIMD stage i done
#define MLA_D_WAIT(m) (m)->sy[64]         // cycles the iDMA waited for a free region
#define MLA_T_APPEND(m) (m)->sy[65]       // cycles the append took
#define MLA_SPIN(m, w, v) dsv2_spin_ge(&(w), (v), &MLA_TMO(m))
// Where each engine's time went (dsv2_mla_report): per GEMM phase p (the five GEMV phases and
// attention, in order) the cycles its core waited for a load, for the SIMD and for the array
#define MLA_P(m, i) (m)->sy[128 + (i)]
#define MLA_PG(m, ph, k) MLA_P(m, 3 * (ph) + (k))  // k: 0 load, 1 SIMD, 2 array
#define MLA_PGW(m, ph) ((uint32_t *)&MLA_PG(m, ph, 0))  // a phase's three accumulators
#define MLA_P_DXFER 18   // iDMA: transferring
#define MLA_P_DFLAG 19   // iDMA: waiting for the cache row
#define MLA_P_SGEMM 20   // SIMD core: waiting for the GEMM
#define MLA_P_SFLAG 21   // SIMD core: waiting for the iDMA / xDMA hart
#define MLA_P_SDRAIN 22  // SIMD core: waiting for its own tasks to retire
#define MLA_P_SBUSY 23   // the SIMD datapath busy (SIMD_BUSY_CYCLES)
#define MLA_P_XROPE 24   // xDMA hart: staging the RoPE rows
#define MLA_P_XTRANS 25  // xDMA hart: the o~ transpose
#define MLA_P_XLOAD 26   // xDMA hart: transferring its parts
#define MLA_P_XFREE 27   // xDMA hart: waiting for a free slab
#define MLA_P_DXWAIT 28  // iDMA: waiting for the xDMA's part
#define MLA_XDMA_SPIN 200000u  // CSR polls per xDMA start or wait, about 5 cycles each
#define MLA_W(m, w, v, acc) dsv2_wait_acc(&(w), (v), &MLA_TMO(m), (acc))

// One pass: the tokens, where they go, and what their depth sets.
typedef struct {
    const uint16_t *x16;                  // the NTOK tokens' hidden states, [NTOK, HID] (DRAM)
    const uint16_t *rope_cos, *rope_sin;  // their positions' RoPE tables, one row per token (DRAM)
    uint32_t pos;                         // token 0's position = its cache row; pos + NTOK keys
    uint32_t k_o;                         // O's D-port shift for depth pos + NTOK
    uint32_t a_n;                         // FP32 bits of 1 / c, c = 2^k_o s_c / 127
} dsv2_mla_pass_t;

// The block's L1 and state. Every pointer is set by dsv2_mla_carve; a per-token buffer holds
// token t at its pointer + t times its pitch.
typedef struct {
    volatile uint32_t *sy;
    uint32_t nt, nvalid;  // the pass's key tiles, and the keys in the last one
    dsv2_trace_t *tr;     // spans for a Gantt chart, or 0 (snax-dsv2-trace.h)
    uint8_t *slab[2];
    // persistent over the pass: inputs, the dequant factors, every intermediate a check reads
    uint16_t *x, *xn, *f_kv, *f_q, *f_uk, *f_uv, *yoh, *f_o, *ya, *ykv, *ckv, *cn, *q16, *rot;
    uint16_t *xt, *oh, *attn, *h;
    uint8_t *ssq, *qt, *sbuf, *rmax, *mrun, *mnew, *corr, *lrun, *lnew, *p8[2], *rsc;
    int8_t *row, *q8;
    // one region, three views in turn
    int8_t *xa, *qn8, *auv, *oa;
    uint16_t *yqt, *yq, *o16, *ot16;
    uint8_t *rope, *olatch, *region;  // q~'s padded rows (qt) live in the region too
    int32_t *oacc;
} dsv2_mla_t;

// Carve the block's L1 from `p` (64-byte aligned); h is the caller's. Returns the end.
static inline uint8_t *dsv2_mla_carve(dsv2_mla_t *m, uint8_t *p, volatile uint32_t *sy,
                                      uint16_t *h) {
#define MLA_TAKE(n) (p += (((n) + 63u) & ~63u), p - (((n) + 63u) & ~63u))
    m->sy = sy;
    m->h = h;
    m->slab[0] = MLA_TAKE(MLA_SLAB);
    m->slab[1] = MLA_TAKE(MLA_SLAB);
    m->tr = (dsv2_trace_t *)0;
    // Adjacency is part of the layout: a norm's input has its seed beat right below it, a
    // sticky task reads its latch as the beat in front of its data, and a 2-beat lanewise
    // reduce reads its pair as consecutive beats
    m->x = (uint16_t *)(MLA_TAKE(NTOK * MLA_XP) + DSV2_BEAT);        // [seed][x] per token
    m->ssq = MLA_TAKE(NTOK * DSV2_BEAT);
    m->xn = (uint16_t *)MLA_TAKE(NTOK * 2 * HID);
    m->f_kv = (uint16_t *)MLA_TAKE(2 * MLA_DQK);
    m->f_q = (uint16_t *)MLA_TAKE(2 * MLA_Q_N);
    m->f_uk = (uint16_t *)MLA_TAKE(2 * HEADS * MLA_DV);
    m->f_uv = (uint16_t *)MLA_TAKE(2 * HEADS * V_HEAD);
    m->yoh = (uint16_t *)MLA_TAKE(MLA_YBUF(HEADS * DSV2_SLOT(V_HEAD / DSV2_NU), V_HEAD / DSV2_NU));
    m->f_o = (uint16_t *)MLA_TAKE(2 * HID);
    m->ya = (uint16_t *)MLA_TAKE(MLA_YBUF(2 * HID, MLA_CH / DSV2_NU));
    m->ykv = (uint16_t *)MLA_TAKE(MLA_YBUF(2 * MLA_DQK, MLA_CH / DSV2_NU));
    m->ckv = (uint16_t *)(MLA_TAKE(NTOK * MLA_CKP) + DSV2_BEAT);        // [seed][ckv] per token
    m->cn = (uint16_t *)MLA_TAKE(NTOK * 2 * MLA_DV);
    m->row = (int8_t *)MLA_TAKE(NTOK * MLA_DQK);                        // [c8 | kpe8] per token
    m->q16 = (uint16_t *)MLA_TAKE(NTOK * 2 * MLA_Q_N);
    m->rot = (uint16_t *)MLA_TAKE(MLA_ROPE_BT);                         // [q_pe x 16 | k_pe] rotated
    m->q8 = (int8_t *)MLA_TAKE(MLA_BR * MLA_DQK);                       // Q^T operand, 16 rows per token
    m->sbuf = MLA_TAKE(2 * (1 + MLA_SBEATS) * DSV2_BEAT);               // [latch][S^T], by tile parity
    m->rmax = MLA_TAKE(DSV2_BEAT);
    m->mrun = MLA_TAKE(DSV2_BEAT);
    m->mnew = MLA_TAKE(DSV2_BEAT);
    m->corr = MLA_TAKE(2 * DSV2_BEAT);
    m->lrun = MLA_TAKE(DSV2_BEAT);
    m->lnew = MLA_TAKE(DSV2_BEAT);
    m->p8[0] = MLA_TAKE((MLA_PBEATS + 2) * DSV2_BEAT);                  // [P8][rowsum][corr*l]
    m->p8[1] = MLA_TAKE((MLA_PBEATS + 2) * DSV2_BEAT);
    m->rsc = MLA_TAKE(DSV2_BEAT);                                       // c / l, for a check
    m->xt = (uint16_t *)MLA_TAKE(NTOK * MLA_XTP);                       // o~, [16, 512] per token
    m->oh = (uint16_t *)MLA_TAKE(NTOK * 2 * HID);                       // the heads' outputs
    m->attn = (uint16_t *)MLA_TAKE(NTOK * 2 * HID);
    uint8_t *R = m->region = MLA_TAKE(0);
    //   up to Q8: xn's A operand (then W_UK's outputs over it), q, the RoPE operands, W_UK's A;
    //   q~'s padded rows behind W_UK's outputs, over q, the RoPE operands and W_UK's A, all dead
    //   by the time the W_UK outputs are dequantised
    const uint32_t yqt_bytes = MLA_YBUF(HEADS * DSV2_SLOT(MLA_DV / DSV2_NU), MLA_DV / DSV2_NU);
    m->xa = (int8_t *)R;
    m->yqt = (uint16_t *)R;
    m->yq = (uint16_t *)(R + DSV2_MR * HID);
    m->rope = (uint8_t *)m->yq + ((MLA_YBUF(2 * MLA_Q_N, MLA_CH / DSV2_NU) + 63u) & ~63u);
    m->qn8 = (int8_t *)(m->rope + 4 * MLA_ROPE_BT);
    m->qt = R + ((yqt_bytes + 63u) & ~63u);
    uint8_t *r1_end = (uint8_t *)m->qn8 + HEADS * DSV2_MR * Q_NOPE;
    if (m->qt + NTOK * HEADS * MLA_QT_PITCH > r1_end) r1_end = m->qt + NTOK * HEADS * MLA_QT_PITCH;
    //   attention: O16 behind its latch (c / l), o~^T, O^T in INT32; then W_UV's A operands
    m->olatch = R;
    m->o16 = (uint16_t *)(R + DSV2_BEAT);
    m->ot16 = m->o16 + MLA_DV * MLA_BR;
    m->oacc = (int32_t *)(m->ot16 + MLA_DV * MLA_BR);
    m->auv = (int8_t *)R;
    uint8_t *r2_end = R + DSV2_BEAT + 2 * MLA_DV * MLA_BR * 2 + MLA_DV * MLA_BR * 4;
    //   behind them: W_O's A operand, written while W_UV's are still being read
    m->oa = (int8_t *)(r1_end > r2_end ? r1_end : r2_end);
    return (uint8_t *)m->oa + DSV2_MR * HID;
#undef MLA_TAKE
}

// The pass's tile count; every hart calls this before its role.
static inline void dsv2_mla_begin(dsv2_mla_t *m, const dsv2_mla_pass_t *ps) {
    const uint32_t keys = ps->pos + NTOK;
    m->nt = (keys + BC - 1u) / BC;
    m->nvalid = keys - (m->nt - 1u) * BC;
}

// ============================================================ the GEMM (hart 0)

// The attention descriptors: C and D share their INT32 walk (O^T accumulates in place); the
// FP16 walk halves the beats per block and the block stride.
__attribute__((always_inline)) static inline void mla_cd_walk(uint32_t mb, uint32_t c_on,
                                                              uint32_t d_fp16, uint32_t k) {
    csrw_ss(S_STRIDE_READER_WRITER_0_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_0_1, 64);
    csrw_ss(T_BOUND_READER_WRITER_0_0, 8);
    csrw_ss(T_STRIDE_READER_WRITER_0_0, 256);
    csrw_ss(T_BOUND_READER_WRITER_0_1, MLA_N_Q);
    csrw_ss(T_STRIDE_READER_WRITER_0_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_0_2, mb);
    csrw_ss(T_STRIDE_READER_WRITER_0_2, 2048);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, 0);
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_0, c_on ? 0xFFFFFFFFu : 0u);
    csrw_ss(S_STRIDE_READER_WRITER_1_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_1_1, 64);
    csrw_ss(T_BOUND_READER_WRITER_1_0, d_fp16 ? 4u : 8u);
    csrw_ss(T_STRIDE_READER_WRITER_1_0, 256);
    csrw_ss(T_BOUND_READER_WRITER_1_1, MLA_N_Q);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_1_2, mb);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, d_fp16 ? 1024u : 2048u);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, 0);
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_1, 0xFFFFFFFFu);  // shape 0: full output beats
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 0, d_fp16 ? 1u : 0u);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 0u);
    (void)set_versacore_d_shift(d_fp16 ? k : 0u);
}

// A and B walks: k innermost, then n, then m. A broadcasts over n; B over m.
__attribute__((always_inline)) static inline void mla_ab_walk(uint32_t kt, uint32_t mb,
                                                              uint32_t a_ms, uint32_t b_ks,
                                                              uint32_t b_ns) {
    csrw_ss(S_STRIDE_READER_0_0, 8);
    csrw_ss(T_BOUND_READER_0_0, kt);
    csrw_ss(T_STRIDE_READER_0_0, 64);
    csrw_ss(T_BOUND_READER_0_1, MLA_N_Q);
    csrw_ss(T_STRIDE_READER_0_1, 0);
    csrw_ss(T_BOUND_READER_0_2, mb);
    csrw_ss(T_STRIDE_READER_0_2, a_ms);
    csrw_ss(T_BOUND_READER_0_3, 1);
    csrw_ss(T_STRIDE_READER_0_3, 0);
    csrw_ss(T_BOUND_READER_0_4, 1);
    csrw_ss(T_STRIDE_READER_0_4, 0);
    csrw_ss(T_BOUND_READER_0_5, 1);
    csrw_ss(T_STRIDE_READER_0_5, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0);
    csrw_ss(ENABLED_CHANNEL_READER_0, 0xFFFFFFFFu);  // shape 0: every A channel
    csrw_ss(S_STRIDE_READER_1_0, 8);
    csrw_ss(T_BOUND_READER_1_0, kt);
    csrw_ss(T_STRIDE_READER_1_0, b_ks);
    csrw_ss(T_BOUND_READER_1_1, MLA_N_Q);
    csrw_ss(T_STRIDE_READER_1_1, b_ns);
    csrw_ss(T_BOUND_READER_1_2, mb);
    csrw_ss(T_STRIDE_READER_1_2, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0);
    csrw_ss(S_STRIDE_READER_1_1, 0);
    csrw_ss(ENABLED_CHANNEL_READER_1, 0xFFu);  // shape 0: one 4 x 16 block, channels 0..7
    dsv2_b_int4(0u);                           // Q8 and P8 are INT8 operands
    csrw_ss(OVERWRITE_ACCUM, 1);
    csrw_ss(ACCUM_BOUND, kt);
    csrw_ss(OUTPUT_BOUND, mb * MLA_N_Q);
    csrw_ss(SUBTRACTIONS, 0);
    csrw_ss(ARRAY_SHAPE_CFG, 0);
    csrw_ss(DATA_TYPE_CFG, 0);
}

// QK: S^T [Bc, 32] = K [Bc, 576] . Q^T, D packed FP16, one 64 B key row per beat. B(Q^T) is
// A(Q)'s bytes (meshRow == meshCol), walked n-major.
__attribute__((always_inline)) static inline void mla_qk_arm(void) {
    mla_ab_walk(MLA_KT_S, MLA_M_S, MLA_KT_S * 64u, 64u, MLA_KT_S * 64u);
    mla_cd_walk(MLA_M_S, 0u, 1u, K_S);
    csrw_ss(MLA_COLSCALE + 1, MLA_N_Q);
    csrw_ss(MLA_COLSCALE + 0, 0u);
}

// PV: O^T [512, 32] += V^T [512, Bc] . P^T, P8 k-major as the interleaving quantiser writes
// it; C scaled per query by corr_j (masked at j = 0); the last tile's D leaves as FP16 at k_o.
__attribute__((always_inline)) static inline void mla_pv_arm(uint32_t first, uint32_t last,
                                                             uint32_t k_o) {
    mla_ab_walk(MLA_KT_P, MLA_M_P, MLA_KT_P * 64u, MLA_N_Q * 64u, 64u);
    mla_cd_walk(MLA_M_P, !first, last, k_o);
    csrw_ss(MLA_COLSCALE + 1, MLA_N_Q);
    csrw_ss(MLA_COLSCALE + 0, first ? 0u : 1u);
}

// corr_j's 32 FP16 factors, two per word: the 64 B beat the softmax wrote, copied verbatim.
__attribute__((always_inline)) static inline void mla_pv_factors(const volatile uint32_t *f) {
    csrw_ss(MLA_COLSCALE + 2, f[0]);    csrw_ss(MLA_COLSCALE + 3, f[1]);
    csrw_ss(MLA_COLSCALE + 4, f[2]);    csrw_ss(MLA_COLSCALE + 5, f[3]);
    csrw_ss(MLA_COLSCALE + 6, f[4]);    csrw_ss(MLA_COLSCALE + 7, f[5]);
    csrw_ss(MLA_COLSCALE + 8, f[6]);    csrw_ss(MLA_COLSCALE + 9, f[7]);
    csrw_ss(MLA_COLSCALE + 10, f[8]);   csrw_ss(MLA_COLSCALE + 11, f[9]);
    csrw_ss(MLA_COLSCALE + 12, f[10]);  csrw_ss(MLA_COLSCALE + 13, f[11]);
    csrw_ss(MLA_COLSCALE + 14, f[12]);  csrw_ss(MLA_COLSCALE + 15, f[13]);
    csrw_ss(MLA_COLSCALE + 16, f[14]);  csrw_ss(MLA_COLSCALE + 17, f[15]);
}

__attribute__((always_inline)) static inline void mla_gemm_go(const void *a, const void *b,
                                                              void *c, void *d) {
    csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)a);
    csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)b);
    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, (uint32_t)c);
    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, (uint32_t)d);
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(GEMMX_START, 1);
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
}

// Wait for task `id` to retire, adding the wait to *acc.
static inline void mla_retire(dsv2_mla_t *m, uint32_t id, uint32_t *acc) {
    const uint32_t t0 = snrt_mcycle();
    MLA_TMO(m) += dsv2_gemm_wait(id);
    *acc += snrt_mcycle() - t0;
}

// Tasks [t0, t0 + n) of one armed GEMV, each once its load has landed, two in flight; group i
// reads A at a + i * a_step and writes y + i * y_step. RETIRED follows each retirement. w: the
// phase's waits (load, SIMD, array).
static void mla_gemv_run(dsv2_mla_t *m, uint32_t t0, uint32_t n, const int8_t *a,
                         uint32_t a_step, uint8_t *y, uint32_t y_step, uint32_t *id, uint32_t *w) {
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t t = t0 + i;
        MLA_W(m, MLA_LOADED(m), t + 1u, &w[0]);
        dsv2_gemv_fire(a + i * a_step, m->slab[t & 1u], y + i * y_step);
        ++*id;
        if (i) {
            mla_retire(m, *id - 1u, &w[2]);
            MLA_RETIRED(m) = t;
        }
    }
    mla_retire(m, *id, &w[2]);
    MLA_RETIRED(m) = t0 + n;
}

// A phase's TCDM priority policy (DSV2_TCDM_PRIO_*, snax-dsv2.h); nothing to do when every
// phase runs the app's one policy.
static inline void mla_prio(uint32_t policy) {
    if (DSV2_TCDM_PRIO_PER_KERNEL) dsv2_tcdm_prio_install(policy);
}

static void dsv2_mla_gemm(dsv2_mla_t *m, const dsv2_mla_pass_t *ps) {
    const uint32_t nt = m->nt, org = MLA_T_ORG(m);
    uint32_t id = csrr_ss(GEMMX_FINISHED_TASK);
    uint32_t s0;  // a span's start
    MLA_W(m, MLA_XA_OK(m), 1u, MLA_PGW(m, 0) + 1);
    mla_prio(DSV2_TCDM_PRIO_STREAM);
    s0 = snrt_mcycle();
    dsv2_gemv_arm_ntok(HID / DSV2_KU, MLA_CH / DSV2_NU, 1u, 0u, K_X, NTOK, DSV2_W4);
    if (NTOK == 2) dsv2_gemv_two_tokens(MLA_CH / DSV2_NU, MLA_YP_KV);
    mla_gemv_run(m, MLA_T_KV0, MLA_NC_KV, m->xa, 0u, (uint8_t *)m->ykv,
                 DSV2_SLOT(MLA_CH / DSV2_NU), &id, MLA_PGW(m, 0));
    MLA_T_PH(m, 0) = snrt_mcycle() - org;
    dsv2_span(m->tr, DSV2_TR_GEMM, "3 W_DKV", s0, snrt_mcycle());
    s0 = snrt_mcycle();
    if (NTOK == 2) dsv2_gemv_two_tokens(MLA_CH / DSV2_NU, MLA_YP_Q);
    mla_gemv_run(m, MLA_T_Q0, MLA_NC_Q, m->xa, 0u, (uint8_t *)m->yq, DSV2_SLOT(MLA_CH / DSV2_NU),
                 &id, MLA_PGW(m, 1));
    MLA_T_PH(m, 1) = snrt_mcycle() - org;
    dsv2_span(m->tr, DSV2_TR_GEMM, "2 W_Q", s0, snrt_mcycle());
    MLA_W(m, MLA_QN_OK(m), 1u, MLA_PGW(m, 2) + 1);
    mla_prio(DSV2_TCDM_PRIO_HEAD);
    s0 = snrt_mcycle();
    dsv2_gemv_arm_ntok(Q_NOPE / DSV2_KU, MLA_DV / DSV2_NU, 1u, 0u, K_UK, NTOK, DSV2_W4);
    if (NTOK == 2) dsv2_gemv_two_tokens(MLA_DV / DSV2_NU, MLA_YP_UK);
    mla_gemv_run(m, MLA_T_UK0, HEADS, m->qn8, DSV2_MR * Q_NOPE, (uint8_t *)m->yqt,
                 DSV2_SLOT(MLA_DV / DSV2_NU), &id, MLA_PGW(m, 2));
    MLA_T_PH(m, 2) = snrt_mcycle() - org;
    dsv2_span(m->tr, DSV2_TR_GEMM, "6 W_UK", s0, snrt_mcycle());
    // attention: QK(j), then PV(j-1), one task at a time
    MLA_W(m, MLA_Q_OUT(m), 1u, MLA_PGW(m, 3) + 1);
    mla_prio(DSV2_TCDM_PRIO_ATTN);
    uint32_t t = MLA_T_ATT;
    for (uint32_t j = 0; j <= nt; j++) {
        if (j < nt) {
            MLA_W(m, MLA_LOADED(m), MLA_L_K(j) + 1u, MLA_PGW(m, 3));
            s0 = snrt_mcycle();
            mla_qk_arm();
            mla_gemm_go(m->slab[j & 1u], m->q8, m->oacc,
                        m->sbuf + (j & 1u) * (1 + MLA_SBEATS) * DSV2_BEAT + DSV2_BEAT);
            mla_retire(m, ++id, MLA_PGW(m, 3) + 2);
            MLA_RETIRED(m) = ++t;
            dsv2_span(m->tr, DSV2_TR_GEMM, "8 QK", s0, snrt_mcycle());
        }
        if (j > 0) {
            const uint32_t tt = j - 1u, last = (tt == nt - 1u);
            MLA_W(m, MLA_P_OUT(m), tt + 1u, MLA_PGW(m, 3) + 1);
            MLA_W(m, MLA_LOADED(m), MLA_L_V(tt) + 1u, MLA_PGW(m, 3));
            s0 = snrt_mcycle();
            mla_pv_arm(tt == 0u, last, ps->k_o);
            if (tt) mla_pv_factors((const volatile uint32_t *)(m->corr + (tt & 1u) * DSV2_BEAT));
            mla_gemm_go(m->slab[tt & 1u] + MLA_KTILE, m->p8[tt & 1u], m->oacc,
                        last ? (void *)m->o16 : (void *)m->oacc);
            mla_retire(m, ++id, MLA_PGW(m, 3) + 2);
            MLA_RETIRED(m) = ++t;
            dsv2_span(m->tr, DSV2_TR_GEMM, "10 PV", s0, snrt_mcycle());
        }
    }
    csrw_ss(MLA_COLSCALE + 0, 0u);  // no dispatch inherits an armed scaler
    MLA_T_PH(m, 3) = snrt_mcycle() - org;
    MLA_W(m, MLA_AUV_OK(m), 1u, MLA_PGW(m, 4) + 1);
    mla_prio(DSV2_TCDM_PRIO_HEAD);
    s0 = snrt_mcycle();
    dsv2_gemv_arm_ntok(MLA_DV / DSV2_KU, V_HEAD / DSV2_NU, 1u, 0u, K_UV, NTOK, DSV2_W4);
    if (NTOK == 2) dsv2_gemv_two_tokens(V_HEAD / DSV2_NU, MLA_YP_UV);
    mla_gemv_run(m, mla_t_uv0(nt), HEADS, m->auv, DSV2_MR * MLA_DV, (uint8_t *)m->yoh,
                 DSV2_SLOT(V_HEAD / DSV2_NU), &id, MLA_PGW(m, 4));
    MLA_T_PH(m, 4) = snrt_mcycle() - org;
    dsv2_span(m->tr, DSV2_TR_GEMM, "11 W_UV", s0, snrt_mcycle());
    MLA_W(m, MLA_OA_OK(m), 1u, MLA_PGW(m, 5) + 1);
    mla_prio(DSV2_TCDM_PRIO_STREAM);
    s0 = snrt_mcycle();
    dsv2_gemv_arm_ntok(HID / DSV2_KU, MLA_CH / DSV2_NU, 1u, 0u, K_X, NTOK, DSV2_W4);
    if (NTOK == 2) dsv2_gemv_two_tokens(MLA_CH / DSV2_NU, MLA_YP_A);
    mla_gemv_run(m, mla_t_o0(nt), MLA_NC_O, m->oa, 0u, (uint8_t *)m->ya,
                 DSV2_SLOT(MLA_CH / DSV2_NU), &id, MLA_PGW(m, 5));
    MLA_T_PH(m, 5) = snrt_mcycle() - org;
    dsv2_span(m->tr, DSV2_TR_GEMM, "12 W_O", s0, snrt_mcycle());
    MLA_FIN(m, 0) = 1u;
}

// ============================================================ the iDMA (hart 3)

static void dsv2_mla_idma(dsv2_mla_t *m, const dsv2_mla_pass_t *ps) {
    const uint32_t nt = m->nt, i0 = snrt_mcycle();
    for (uint32_t t = 0; t < NTOK; t++)
        snrt_dma_start_1d((uint8_t *)m->x + t * MLA_XP, ps->x16 + t * HID, 2 * HID);
    snrt_dma_start_1d(m->f_kv, s_kv, 2 * MLA_DQK);
    snrt_dma_start_1d(m->f_q, s_q, 2 * MLA_Q_N);
    snrt_dma_start_1d(m->f_uk, s_uk, 2 * HEADS * MLA_DV);
    snrt_dma_start_1d(m->f_uv, s_uv, 2 * HEADS * V_HEAD);
    snrt_dma_start_1d(m->f_o, s_o, 2 * HID);
    snrt_dma_start_1d(m->rope + MLA_ROPE_BT, ps->rope_cos, MLA_ROPE_BT);
    snrt_dma_start_1d(m->rope + 3 * MLA_ROPE_BT, ps->rope_sin, MLA_ROPE_BT);
    snrt_dma_wait_all();
    MLA_IN_OK(m) = 1u;
    dsv2_span(m->tr, DSV2_TR_IDMA, "0 inputs", i0, snrt_mcycle());
    // need[s]: RETIRED must reach this before slab s is written again
    uint32_t need[2] = {0u, 0u}, waited = 0, xfer = 0, wflag = 0, xwait = 0;
#define MLA_LOAD(t, src, n, label)                           \
    do {                                                     \
        const uint32_t s_ = (t) & 1u;                        \
        MLA_W(m, MLA_RETIRED(m), need[s_], &waited);         \
        const uint32_t x0_ = snrt_mcycle();                  \
        snrt_dma_start_1d(m->slab[s_], (src), (n));          \
        snrt_dma_wait_all();                                 \
        const uint32_t x1_ = snrt_mcycle();                  \
        xfer += x1_ - x0_;                                   \
        need[s_] = (t) + 1u;                                 \
        MLA_LOADED(m) = (t) + 1u;                            \
        dsv2_span(m->tr, DSV2_TR_IDMA, (label), x0_, x1_);   \
    } while (0)
#if MLA_DUAL
// a streamed chunk: its head here, its tail on the xDMA hart (mla_xstream)
#define MLA_STREAM(t, src, label)                               \
    do {                                                        \
        const uint32_t s_ = (t) & 1u;                           \
        MLA_W(m, MLA_RETIRED(m), need[s_], &waited);            \
        const uint32_t x0_ = snrt_mcycle();                     \
        MLA_XREQ(m) = (t) + 1u;                                 \
        snrt_dma_start_1d(m->slab[s_], (src), MLA_CHW - DSV2_DUAL_XBYTES); \
        snrt_dma_wait_all();                                    \
        const uint32_t x1_ = snrt_mcycle();                     \
        xfer += x1_ - x0_;                                      \
        MLA_W(m, MLA_XLOADED(m), (t) + 1u, &xwait);             \
        need[s_] = (t) + 1u;                                    \
        MLA_LOADED(m) = (t) + 1u;                               \
        dsv2_span(m->tr, DSV2_TR_IDMA, (label), x0_, x1_);      \
    } while (0)
#else
#define MLA_STREAM(t, src, label) MLA_LOAD(t, src, MLA_CHW, label)
#endif
    for (uint32_t c = 0; c < MLA_NC_KV; c++)
        MLA_STREAM(MLA_T_KV0 + c, wdkv + c * MLA_CHW, "3 W_DKV");
    for (uint32_t c = 0; c < MLA_NC_Q; c++)
        MLA_STREAM(MLA_T_Q0 + c, wq + c * MLA_CHW, "2 W_Q");
    for (uint32_t hh = 0; hh < HEADS; hh++)
        MLA_LOAD(MLA_T_UK0 + hh, wuk + hh * MLA_UK_BYTES, MLA_UK_BYTES, "6 W_UK");
    // the tokens' rows, into both copies, before any key tile is read
    MLA_W(m, MLA_ROW_OK(m), 1u, &wflag);
    uint32_t t_app = snrt_mcycle();
    for (uint32_t t = 0; t < NTOK; t++)
        dsv2_cache_append(key, val, CAP, ps->pos + t, m->row + t * MLA_DQK);
    MLA_T_APPEND(m) = snrt_mcycle() - t_app;
    dsv2_span(m->tr, DSV2_TR_IDMA, "7 cache append", t_app, snrt_mcycle());
    MLA_APPENDED(m) = 1u;
    // attention: K(j) frees after QK(j), V(j) after PV(j)
    uint32_t kneed[2] = {need[0], need[1]}, vneed[2] = {need[0], need[1]};
    for (uint32_t j = 0; j < nt; j++) {
        const uint32_t s = j & 1u;
        MLA_W(m, MLA_RETIRED(m), kneed[s], &waited);
        uint32_t x0 = snrt_mcycle();
        snrt_dma_start_1d(m->slab[s], key + j * MLA_KTILE, MLA_KTILE);
        snrt_dma_wait_all();
        xfer += snrt_mcycle() - x0;
        dsv2_span(m->tr, DSV2_TR_IDMA, "8 key tile", x0, snrt_mcycle());
        kneed[s] = mla_t_qk(j) + 1u;
        MLA_LOADED(m) = MLA_L_K(j) + 1u;
        MLA_W(m, MLA_RETIRED(m), vneed[s], &waited);
        x0 = snrt_mcycle();
        snrt_dma_start_2d(m->slab[s] + MLA_KTILE, val + j * (16u * BC), 16u * BC, 16u * BC,
                          16u * CAP, MLA_DV / DSV2_MR);
        snrt_dma_wait_all();
        xfer += snrt_mcycle() - x0;
        dsv2_span(m->tr, DSV2_TR_IDMA, "10 value tile", x0, snrt_mcycle());
        vneed[s] = mla_t_pv(nt, j) + 1u;
        MLA_LOADED(m) = MLA_L_V(j) + 1u;
    }
    need[0] = vneed[0];
    need[1] = vneed[1];
    for (uint32_t hh = 0; hh < HEADS; hh++)
        MLA_LOAD(mla_t_uv0(nt) + hh, wuv + hh * MLA_UV_BYTES, MLA_UV_BYTES, "11 W_UV");
    for (uint32_t c = 0; c < MLA_NC_O; c++)
        MLA_STREAM(mla_t_o0(nt) + c, wo + c * MLA_CHW, "12 W_O");
#undef MLA_STREAM
#undef MLA_LOAD
    MLA_D_WAIT(m) = waited;
    MLA_P(m, MLA_P_DXFER) = xfer;
    MLA_P(m, MLA_P_DFLAG) = wflag;
    MLA_P(m, MLA_P_DXWAIT) = xwait;
    MLA_FIN(m, 1) = 1u;
}

// ============================================================ the xDMA hart (hart 2)

#if MLA_DUAL
// The last DSV2_DUAL_XBYTES of streamed chunks t0 .. t0 + n - 1 of `src`: each when XREQ says its
// slab is free, published on XLOADED. The descriptor is armed once per run of chunks, since the
// fill and the transpose leave other settings behind, and re-pointed per chunk.
static void mla_xstream(dsv2_mla_t *m, uint32_t t0, uint32_t n, const int8_t *src,
                        const char *label, uint32_t *busy, uint32_t *free_wait) {
    const uint32_t off = MLA_CHW - DSV2_DUAL_XBYTES, beats = DSV2_DUAL_XBYTES / DSV2_BEAT;
    snax_xdma_read_arm((uint32_t)(m->slab[t0 & 1u] + off), (uint32_t)(src + off), beats);
    for (uint32_t c = 0; c < n; c++) {
        const uint32_t t = t0 + c;
        MLA_W(m, MLA_XREQ(m), t + 1u, free_wait);
        const uint32_t x0 = snrt_mcycle();
        if (c)
            snax_xdma_read_retask((uint32_t)(m->slab[t & 1u] + off),
                                  (uint32_t)(src + c * MLA_CHW + off), beats);
        int to = 0;
        const snax_xdma_task_t tk = snax_xdma_start_bounded(MLA_XDMA_SPIN, &to);
        if (!to) to = snax_xdma_wait_bounded(tk, MLA_XDMA_SPIN);
        MLA_TMO(m) += (uint32_t)to;
        MLA_XLOADED(m) = t + 1u;
        const uint32_t x1 = snrt_mcycle();
        *busy += x1 - x0;
        dsv2_span(m->tr, DSV2_TR_XDMA, label, x0, x1);
    }
}
#endif

static void dsv2_mla_xdma(dsv2_mla_t *m) {
    uint32_t xbusy = 0, xfree = 0;
    const uint32_t f0 = snrt_mcycle();
    dsv2_xdma_fill(m->mrun, 1u, 0xFBFFFBFFu);
    MLA_M_INIT(m) = 1u;
    dsv2_span(m->tr, DSV2_TR_XDMA, "9 m = -65504", f0, snrt_mcycle());
#if MLA_DUAL
    mla_xstream(m, MLA_T_KV0, MLA_NC_KV, wdkv, "3 W_DKV", &xbusy, &xfree);
    mla_xstream(m, MLA_T_Q0, MLA_NC_Q, wq, "2 W_Q", &xbusy, &xfree);
#endif
    // The RoPE rows, one per token: the heads' q_pe (strided in q) and k_pe (behind c in ckv),
    // and their pair swap. A pair of FP16 is one word, so the swap is a 16-bit rotate.
    MLA_SPIN(m, MLA_Q_DQ(m), 1u);
    uint32_t x0 = snrt_mcycle();
    for (uint32_t t = 0; t < NTOK; t++) {
        uint32_t *dst = (uint32_t *)(m->rope + t * MLA_ROPE_B);
        uint32_t *swp = (uint32_t *)(m->rope + 2 * MLA_ROPE_BT + t * MLA_ROPE_B);
        const uint16_t *q = (const uint16_t *)((const uint8_t *)m->q16 + t * 2 * MLA_Q_N);
        const uint16_t *kv = (const uint16_t *)((const uint8_t *)m->ckv + t * MLA_CKP);
        for (uint32_t r = 0; r <= HEADS; r++) {
            const uint32_t *src =
                (const uint32_t *)(r < HEADS ? q + r * Q_HEAD + Q_NOPE : kv + KV_RANK);
            for (uint32_t i = 0; i < ROPE_DIM / 2u; i++) {
                const uint32_t w = src[i];
                dst[r * (ROPE_DIM / 2u) + i] = w;
                swp[r * (ROPE_DIM / 2u) + i] = (w >> 16) | (w << 16);
            }
        }
    }
    MLA_ROPE_IN(m) = 1u;
    MLA_P(m, MLA_P_XROPE) = snrt_mcycle() - x0;
    dsv2_span(m->tr, DSV2_TR_XDMA, "5 RoPE rows (core)", x0, snrt_mcycle());
    // o~^T is [512, 32] FP16; its first 16 NTOK columns are the tokens' heads
    MLA_SPIN(m, MLA_N_OUT(m), 1u);
    x0 = snrt_mcycle();
    dsv2_xdma_transpose16(m->ot16, 2u * MLA_BR, m->xt, 2u * MLA_DV, MLA_DV, NTOK * HEADS);
    MLA_P(m, MLA_P_XTRANS) = snrt_mcycle() - x0;
    dsv2_span(m->tr, DSV2_TR_XDMA, "10 o~ transpose", x0, snrt_mcycle());
    MLA_X_OUT(m) = 1u;
#if MLA_DUAL
    mla_xstream(m, mla_t_o0(m->nt), MLA_NC_O, wo, "12 W_O", &xbusy, &xfree);
#endif
    MLA_P(m, MLA_P_XLOAD) = xbusy;
    MLA_P(m, MLA_P_XFREE) = xfree;
    MLA_FIN(m, 2) = 1u;
}

// ============================================================ the SIMD (hart 1)

static inline void mla_task(const snax_simd_shape_t *in, const snax_simd_shape_t *out) {
    snax_simd_program_fast(in, out);
    snax_simd_fire();
}

static inline void mla_drain(dsv2_mla_t *m, const char *what) {
    const uint32_t t0 = snrt_mcycle();
    if (snax_simd_wait_all_checked(what, SIMD_WAIT_BUDGET)) MLA_TMO(m)++;
    MLA_P(m, MLA_P_SDRAIN) += snrt_mcycle() - t0;
}

// D2: one segment of Q8 from FP16 rows `pitch` apart, read in A-order: a beat is 8 rows x 4
// values (lane stride = pitch); two such beats (rows 0-7, 8-15) pack into one A block.
static void mla_quant_q_segment(const void *rows, uint32_t pitch, void *q8, uint32_t kt,
                                uint32_t inv) {
    snax_simd_shape_t in, out;
    snax_simd_shape_flat(&in, (void *)rows, 1);
    in.lane_stride = pitch;
    in.bound[0] = 2u;
    in.stride[0] = 8u * pitch;
    in.bound[1] = kt;
    in.stride[1] = 8u;
    snax_simd_shape_flat(&out, q8, kt);
    snax_simd_use0(SIMD_EXT_FP16TOINT8);
    DSV2_ARM_QUANT(inv);
    mla_task(&in, &out);
}

// Per-head A operands of an absorbed GEMV, one task per token, in snax-dsv2.h's row write: x_h
// is `k` values at x + h * pitch, head h's operand 16 k bytes at a + 16 k h; token t's rows
// (tok bytes further in x) go to row 4 t.
static void mla_quant_heads(const void *x, uint32_t pitch, uint32_t k, void *a, uint32_t inv,
                            uint32_t tok) {
    snax_simd_shape_t in, out;
    for (uint32_t t = 0; t < NTOK; t++) {
        dsv2_a_row_shapes(&in, &out, (uint8_t *)x + t * tok, 1u, 0u, a, HEADS * k, 4u * t);
        in.bound[1] = k / 32u;  // per head: its k values, then the next head's
        in.dim = 3u;
        in.bound[2] = HEADS;
        in.stride[2] = pitch;
        snax_simd_use0(SIMD_EXT_FP16TOINT8);
        DSV2_ARM_QUANT(inv);
        mla_task(&in, &out);
    }
}

static void dsv2_mla_simd(dsv2_mla_t *m, const dsv2_mla_pass_t *ps) {
    const uint32_t nt = m->nt, org = MLA_T_ORG(m), busy0 = snax_simd_busy_cycles();
    uint32_t wg = 0, wf = 0, s0;  // waited for the GEMM, for a flag; a span's start
    snax_simd_shape_t in, out;
#define MLA_TOK(p, t, pitch) ((void *)((uint8_t *)(p) + (t) * (pitch)))
#define MLA_SPAN(label) dsv2_span(m->tr, DSV2_TR_SIMD, (label), s0, snrt_mcycle())
    // 1: the input norms and the W_DKV / W_Q A operand
    MLA_W(m, MLA_IN_OK(m), 1u, &wf);
    s0 = snrt_mcycle();
    for (uint32_t t = 0; t < NTOK; t++)
        dsv2_rmsnorm_row(MLA_TOK(m->x, t, MLA_XP), m->ssq + t * DSV2_BEAT,
                         MLA_TOK(m->xn, t, 2 * HID), 11);
    dsv2_quant_a2(m->xn, MLA_TOK(m->xn, NTOK - 1u, 2 * HID), m->xa, HID, INV_X);
    mla_drain(m, "input norm");
    MLA_XA_OK(m) = 1u;
    MLA_SPAN("1 norm, A operand");
    MLA_T_SS(m, 0) = snrt_mcycle() - org;
    // 3-4: ckv, the latent norm, the row's c8
    MLA_W(m, MLA_RETIRED(m), MLA_T_Q0, &wg);
    s0 = snrt_mcycle();
    for (uint32_t t = 0; t < NTOK; t++) {
        dsv2_dequant_prep(MLA_TOK(m->ykv, t, MLA_YP_KV), m->f_kv, MLA_TOK(m->ckv, t, MLA_CKP),
                          MLA_DQK);
        snax_simd_fire();
        dsv2_rmsnorm_row(MLA_TOK(m->ckv, t, MLA_CKP), m->ssq + t * DSV2_BEAT,
                         MLA_TOK(m->cn, t, 2 * MLA_DV), 9);
        dsv2_quant_flat(MLA_TOK(m->cn, t, 2 * MLA_DV), m->row + t * MLA_DQK, MLA_DV, INV_C);
    }
    mla_drain(m, "latent");
    MLA_T_SS(m, 1) = snrt_mcycle() - org;
    MLA_SPAN("4 ckv, latent norm, c8");
    // 2: q, then W_UK's A operands from the heads' q_nope
    MLA_W(m, MLA_RETIRED(m), MLA_T_UK0, &wg);
    s0 = snrt_mcycle();
    for (uint32_t t = 0; t < NTOK; t++) {
        dsv2_dequant_prep(MLA_TOK(m->yq, t, MLA_YP_Q), m->f_q, MLA_TOK(m->q16, t, 2 * MLA_Q_N),
                          MLA_Q_N);
        snax_simd_fire();
    }
    mla_drain(m, "q");
    MLA_Q_DQ(m) = 1u;
    mla_quant_heads(m->q16, 2u * Q_HEAD, Q_NOPE, m->qn8, INV_QN, (NTOK - 1u) * 2 * MLA_Q_N);
    mla_drain(m, "W_UK operands");
    MLA_QN_OK(m) = 1u;
    MLA_SPAN("2 q, W_UK operands");
    MLA_T_SS(m, 2) = snrt_mcycle() - org;
    // 5: RoPE; the row's kpe8; Q8's q_pe segment
    MLA_W(m, MLA_ROPE_IN(m), 1u, &wf);
    s0 = snrt_mcycle();
    dsv2_rope(m->rope, MLA_ROPE_BT, m->rot, NTOK * MLA_ROPE_N);
    for (uint32_t t = 0; t < NTOK; t++)
        dsv2_quant_flat((uint8_t *)m->rot + t * MLA_ROPE_B + 2 * HEADS * ROPE_DIM,
                        m->row + t * MLA_DQK + MLA_DV, ROPE_DIM, INV_KPE);
    mla_drain(m, "rope");
    MLA_ROW_OK(m) = 1u;
    for (uint32_t t = 0; t < NTOK; t++)  // Q8's m-block t is token t's 16 heads
        mla_quant_q_segment((uint8_t *)m->rot + t * MLA_ROPE_B, 2u * ROPE_DIM,
                            m->q8 + (t * MLA_KT_S + MLA_DV / DSV2_KU) * DSV2_BEAT,
                            ROPE_DIM / DSV2_KU, INV_QPE);
    MLA_T_SS(m, 3) = snrt_mcycle() - org;
    MLA_SPAN("5 RoPE, kpe8, q_pe");
    // 6: q~ into padded rows, then Q8's q~ segment
    MLA_W(m, MLA_RETIRED(m), MLA_T_ATT, &wg);
    s0 = snrt_mcycle();
    for (uint32_t t = 0; t < NTOK; t++) {
        uint8_t *qt = m->qt + t * HEADS * MLA_QT_PITCH;
        dsv2_shape_pair(&in, MLA_TOK(m->yqt, t, MLA_YP_UK), m->f_uk, HEADS * MLA_DV / 32u);
        snax_simd_shape_2d(&out, qt, MLA_DV / 32u, DSV2_BEAT, HEADS, MLA_QT_PITCH);
        snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_0, SIMD_EXT_STREAMELEMENTWISE_0_CSR, 2u,
                       SIMD_EW_MUL);
        mla_task(&in, &out);
        mla_quant_q_segment(qt, MLA_QT_PITCH, m->q8 + t * MLA_KT_S * DSV2_BEAT, MLA_DV / DSV2_KU,
                            INV_QT);
    }
    mla_drain(m, "Q8");
    MLA_Q_OUT(m) = 1u;
    MLA_SPAN("6 q~, Q8");
    MLA_T_SS(m, 4) = snrt_mcycle() - org;

    // 8-10: the online softmax per key tile, eight tasks
    for (uint32_t j = 0; j < nt; j++) {
        uint8_t *latch = m->sbuf + (j & 1u) * (1 + MLA_SBEATS) * DSV2_BEAT, *st = latch + DSV2_BEAT;
        uint8_t *cj = m->corr + (j & 1u) * DSV2_BEAT;
        uint8_t *rs = m->p8[j & 1u] + MLA_PBEATS * DSV2_BEAT, *lsc = rs + DSV2_BEAT;
        MLA_W(m, MLA_RETIRED(m), mla_t_qk(j) + 1u, &wg);
        if (j == 0) MLA_W(m, MLA_M_INIT(m), 1u, &wf);
        s0 = snrt_mcycle();
        // 0 the padding keys of a partial last tile score -65504, in place: P = 0 for them; and
        //   token 0 must not see token 1's key, the last one: only its 16 lanes are written
        if (j == nt - 1u && m->nvalid < BC) {
            snax_simd_shape_flat(&in, st + m->nvalid * DSV2_BEAT, BC - m->nvalid);
            snax_simd_shape_flat(&out, st + m->nvalid * DSV2_BEAT, BC - m->nvalid);
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, 0u, MLA_SCORE_MASK,
                           SIMD_FUNC_LINEAR);
            mla_task(&in, &out);
        }
        if (NTOK == 2 && j == nt - 1u) {
            snax_simd_shape_flat(&in, st + (m->nvalid - 1u) * DSV2_BEAT, 1);
            snax_simd_shape_flat(&out, st + (m->nvalid - 1u) * DSV2_BEAT, 1);
            out.lane_mask = MLA_TOK0_LANES;
            snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, 0u, MLA_SCORE_MASK,
                           SIMD_FUNC_LINEAR);
            mla_task(&in, &out);
        }
        // 1 rowmax
        snax_simd_shape_flat(&in, st, MLA_SBEATS);
        snax_simd_shape_flat(&out, m->rmax, 1);
        snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, MLA_SBEATS,
                       SIMD_RED_MAX | SIMD_RED_LANEWISE);
        mla_task(&in, &out);
        // 2 m_new = max(rowmax, m_old)
        snax_simd_shape_flat(&in, m->rmax, 2);
        snax_simd_shape_flat(&out, m->mnew, 1);
        snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                       SIMD_RED_MAX | SIMD_RED_LANEWISE);
        mla_task(&in, &out);
        // 3 -m_new into the tile's latch and into rmax
        snax_simd_shape_broadcast(&in, m->mnew, 2);
        snax_simd_shape_2d(&out, latch, 2, (uint32_t)(m->rmax - latch), 1, 0);
        snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR,
                       snax_simd_f32_neg(SIMD_F32_ONE), 0, SIMD_FUNC_LINEAR);
        mla_task(&in, &out);
        // 4 corr = exp(a' (m_old - m_new))
        snax_simd_shape_flat(&in, m->rmax, 2);
        snax_simd_shape_flat(&out, cj, 1);
        snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                                         (1u << SIMD_EXT_STREAMMAP));
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 0, 2u);
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 1, SIMD_EW_ADD);
        snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 0, A_EXP);
        snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 1, 0u);
        snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 2, SIMD_FUNC_EXP);
        mla_task(&in, &out);
        // 5 P8 and the rowsum in one sweep over [-m_new][S^T]
        snax_simd_shape_flat(&in, latch, 1 + MLA_SBEATS);
        snax_simd_shape_flat(&out, m->p8[j & 1u], MLA_PBEATS + 1);
        snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMELEMENTWISE_0) |
                                                         (1u << SIMD_EXT_STREAMMAP) |
                                                         (1u << SIMD_EXT_STREAMREDUCE) |
                                                         (1u << SIMD_EXT_FP16TOINT8));
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 0, 1u);
        snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_0_CSR, 1, SIMD_EW_ADD | SIMD_EW_STICKY_B);
        snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 0, A_EXP);
        snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 1, 0u);
        snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 2, SIMD_FUNC_EXP);
        snax_simd_set_op_csr(SIMD_EXT_STREAMREDUCE_CSR, 0, MLA_SBEATS);
        snax_simd_set_op_csr(SIMD_EXT_STREAMREDUCE_CSR, 1,
                             SIMD_RED_ADD | SIMD_RED_LANEWISE | SIMD_RED_TAP);
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 0, P8_INV);
        snax_simd_set_op_csr(SIMD_EXT_FP16TOINT8_CSR, 1,
                             SIMD_QUANT_TAIL(MLA_SBEATS) | SIMD_QUANT_ILV4);
        mla_task(&in, &out);
        mla_drain(m, "softmax epilogue");
        MLA_P_OUT(m) = j + 1u;
        MLA_SPAN("9 softmax");
        // 6 corr * l_old
        snax_simd_shape_2d(&in, cj, 2, (uint32_t)(m->lrun - cj), 1, 0);
        snax_simd_shape_flat(&out, lsc, 1);
        snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1,
                       SIMD_EW_MUL | SIMD_EW_STICKY_B);
        mla_task(&in, &out);
        // 7 l_new = rowsum + corr * l_old
        snax_simd_shape_flat(&in, rs, 2);
        snax_simd_shape_flat(&out, m->lnew, 1);
        snax_simd_use2(SIMD_EXT_STREAMREDUCE, SIMD_EXT_STREAMREDUCE_CSR, 2,
                       SIMD_RED_ADD | SIMD_RED_LANEWISE);
        mla_task(&in, &out);
        // 8 commit: mnew -> mrun, lnew -> lrun in one strided copy
        snax_simd_shape_2d(&in, m->mnew, 2, (uint32_t)(m->lnew - m->mnew), 1, 0);
        snax_simd_shape_2d(&out, m->mrun, 2, (uint32_t)(m->lrun - m->mrun), 1, 0);
        snax_simd_use3(SIMD_EXT_STREAMMAP, SIMD_EXT_STREAMMAP_CSR, SIMD_F32_ONE, 0,
                       SIMD_FUNC_LINEAR);
        mla_task(&in, &out);
    }
    mla_drain(m, "softmax state");
    // o~ = O16 (.) c / l per query: c / l = RSQRT(l / c)^2 into O16's latch, then a sticky MUL
    // over [c/l][O16]
    MLA_W(m, MLA_RETIRED(m), mla_t_pv(nt, nt - 1u) + 1u, &wg);
    s0 = snrt_mcycle();
    snax_simd_shape_broadcast(&in, m->lrun, 2);
    snax_simd_shape_flat(&out, m->olatch, 1);
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, (1u << SIMD_EXT_STREAMMAP) |
                                                     (1u << SIMD_EXT_STREAMELEMENTWISE_1));
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 0, ps->a_n);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 1, 0u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMMAP_CSR, 2, SIMD_FUNC_RSQRT);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 0, 2u);
    snax_simd_set_op_csr(SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1, SIMD_EW_MUL);
    mla_task(&in, &out);
    snax_simd_shape_flat(&in, m->olatch, 1 + MLA_DV);
    snax_simd_shape_flat(&out, m->ot16, MLA_DV);
    snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_1, SIMD_EXT_STREAMELEMENTWISE_1_CSR, 1,
                   SIMD_EW_MUL | SIMD_EW_STICKY_B);
    mla_task(&in, &out);
    mla_drain(m, "normalise");
    // c / l, kept for a check (the region is reused); volatile, or this becomes a memcpy()
    for (uint32_t i = 0; i < DSV2_BEAT / 4u; i++)
        ((volatile uint32_t *)m->rsc)[i] = ((const volatile uint32_t *)m->olatch)[i];
    MLA_N_OUT(m) = 1u;
    MLA_SPAN("10 O / l");
    MLA_W(m, MLA_X_OUT(m), 1u, &wf);
    s0 = snrt_mcycle();
    // 11: W_UV's 16 A operands from o~
    mla_quant_heads(m->xt, 2u * MLA_DV, MLA_DV, m->auv, INV_OT, (NTOK - 1u) * MLA_XTP);
    mla_drain(m, "W_UV operands");
    MLA_AUV_OK(m) = 1u;
    MLA_SPAN("11 W_UV operands");
    MLA_T_SS(m, 5) = snrt_mcycle() - org;
    // 11-12: the heads' outputs, then W_O's A operand
    MLA_W(m, MLA_RETIRED(m), mla_t_o0(nt), &wg);
    s0 = snrt_mcycle();
    for (uint32_t t = 0; t < NTOK; t++) {
        dsv2_dequant_prep(MLA_TOK(m->yoh, t, MLA_YP_UV), m->f_uv, MLA_TOK(m->oh, t, 2 * HID), HID);
        snax_simd_fire();
    }
    dsv2_quant_a2(m->oh, MLA_TOK(m->oh, NTOK - 1u, 2 * HID), m->oa, HID, INV_O);
    mla_drain(m, "W_O operand");
    MLA_OA_OK(m) = 1u;
    MLA_SPAN("12 o, W_O operand");
    MLA_T_SS(m, 6) = snrt_mcycle() - org;
    // 12-13: attn, then h = x + attn
    MLA_W(m, MLA_RETIRED(m), mla_t_end(nt), &wg);
    s0 = snrt_mcycle();
    for (uint32_t t = 0; t < NTOK; t++) {
        dsv2_dequant_prep(MLA_TOK(m->ya, t, MLA_YP_A), m->f_o, MLA_TOK(m->attn, t, 2 * HID), HID);
        snax_simd_fire();
        dsv2_shape_pair(&in, MLA_TOK(m->x, t, MLA_XP), MLA_TOK(m->attn, t, 2 * HID), HID / 32u);
        snax_simd_shape_flat(&out, MLA_TOK(m->h, t, MLA_HP), HID / 32u);
        snax_simd_use2(SIMD_EXT_STREAMELEMENTWISE_0, SIMD_EXT_STREAMELEMENTWISE_0_CSR, 2u,
                       SIMD_EW_ADD);
        mla_task(&in, &out);
    }
    mla_drain(m, "residual");
    MLA_T_SS(m, 7) = snrt_mcycle() - org;
    MLA_SPAN("13 attn, residual");
    MLA_P(m, MLA_P_SGEMM) = wg;
    MLA_P(m, MLA_P_SFLAG) = wf;
    MLA_P(m, MLA_P_SBUSY) = snax_simd_busy_cycles() - busy0;
    MLA_DONE(m) = 1u;
#undef MLA_SPAN
#undef MLA_TOK
}

// The per-phase cycle report, from the SIMD hart once the block is done: it waits for the other
// three roles' profile words.
static void dsv2_mla_report(dsv2_mla_t *m, const char *tag) {
    for (uint32_t r = 0; r < 3u; r++) MLA_SPIN(m, MLA_FIN(m, r), 1u);
    printf("%s MLA, %u token(s), INT%u weights: %u keys in %u tiles of %u (%u in the last); h at "
           "%u cc\n", tag, NTOK, DSV2_WBITS, (m->nt - 1u) * BC + m->nvalid, m->nt, BC, m->nvalid,
           MLA_T_SS(m, 7));
    printf("%s   GEMM phases done at: W_DKV %u | W_Q %u | W_UK %u | attention %u | W_UV %u | "
           "W_O %u cc\n",
           tag, MLA_T_PH(m, 0), MLA_T_PH(m, 1), MLA_T_PH(m, 2), MLA_T_PH(m, 3), MLA_T_PH(m, 4),
           MLA_T_PH(m, 5));
    printf("%s   SIMD stages done at: norm %u | latent %u | q + W_UK operands %u | RoPE + row %u "
           "| Q8 %u | o~ + W_UV operands %u | W_O operand %u | h %u cc\n",
           tag, MLA_T_SS(m, 0), MLA_T_SS(m, 1), MLA_T_SS(m, 2), MLA_T_SS(m, 3), MLA_T_SS(m, 4),
           MLA_T_SS(m, 5), MLA_T_SS(m, 6), MLA_T_SS(m, 7));
    printf("%s   iDMA waited %u cc for a free region; the append took %u cc\n", tag,
           MLA_D_WAIT(m), MLA_T_APPEND(m));
    static const char *const phase[6] = {"W_DKV", "W_Q", "W_UK", "attention", "W_UV", "W_O"};
    printf("%s   GEMM core waited, per phase: for loads | for the SIMD | for the array (cc)\n", tag);
    for (uint32_t ph = 0; ph < 6; ph++)
        printf("%s     %-9s %7u %7u %7u\n", tag, phase[ph], MLA_PG(m, ph, 0), MLA_PG(m, ph, 1),
               MLA_PG(m, ph, 2));
    printf("%s   iDMA: transferring %u, waiting for a free region %u, for the cache row %u cc\n",
           tag, MLA_P(m, MLA_P_DXFER), MLA_D_WAIT(m), MLA_P(m, MLA_P_DFLAG));
    printf("%s   SIMD: datapath busy %u; core waited for the GEMM %u, for a flag %u, for its "
           "tasks %u cc\n", tag, MLA_P(m, MLA_P_SBUSY), MLA_P(m, MLA_P_SGEMM),
           MLA_P(m, MLA_P_SFLAG), MLA_P(m, MLA_P_SDRAIN));
    printf("%s   xDMA hart: RoPE rows %u, o~ transpose %u cc\n", tag, MLA_P(m, MLA_P_XROPE),
           MLA_P(m, MLA_P_XTRANS));
#if MLA_DUAL
    printf("%s   dual load: the xDMA moved its parts in %u cc and waited %u for a free slab; "
           "the iDMA waited %u for the xDMA's part\n", tag, MLA_P(m, MLA_P_XLOAD),
           MLA_P(m, MLA_P_XFREE), MLA_P(m, MLA_P_DXWAIT));
#endif
}
