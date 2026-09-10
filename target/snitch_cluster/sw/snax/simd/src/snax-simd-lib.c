// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snax-simd-lib.h"
#include <stdbool.h>
#include "stdint.h"

#define SIMD_DEBUG
#ifdef SIMD_DEBUG
#define SIMD_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define SIMD_DEBUG_PRINT(...)
#endif

// ============================================================== shapes

// NOTE: no `= {0}` anywhere in this file. On a 60-byte struct the compiler
// lowers it to a memset() call, and this is a freestanding runtime with no libc
// -- it links as `undefined symbol: memset`. Every field is set explicitly.
static inline void snax_simd_shape_clear(snax_simd_shape_t* s) {
    s->base = 0;
    s->lane_stride = 0;
    s->lane_mask = 0;
    s->byte_mask = 0;
    s->dim = 0;
    // bound 1 / stride 0 is the NEUTRAL loop -- one iteration, no address
    // advance -- so an unused dimension can be written to the hardware
    // unconditionally. Zero would make the whole sweep empty.
    for (uint32_t i = 0; i < SIMD_MAX_DIM; i++) {
        s->bound[i] = 1;
        s->stride[i] = 0;
    }
}

void snax_simd_shape_flat(snax_simd_shape_t* s, void* base, uint32_t beats) {
    snax_simd_shape_clear(s);
    s->base = base;
    s->lane_stride = SIMD_LANE_BYTES;
    s->lane_mask = 0xFFFFFFFFu;
    s->byte_mask = 0xFFFFFFFFu;
    s->dim = 1;
    s->bound[0] = beats;
    s->stride[0] = SIMD_BEAT_BYTES;
}

void snax_simd_shape_rows(snax_simd_shape_t* s, void* base, uint32_t rows,
                          uint32_t beats_per_row, uint32_t row_stride) {
    snax_simd_shape_clear(s);
    s->base = base;
    s->lane_stride = SIMD_LANE_BYTES;
    s->lane_mask = 0xFFFFFFFFu;
    s->byte_mask = 0xFFFFFFFFu;
    s->dim = 2;
    s->bound[0] = beats_per_row;
    s->stride[0] = SIMD_BEAT_BYTES;
    s->bound[1] = rows;
    s->stride[1] = row_stride;
}

void snax_simd_shape_2d(snax_simd_shape_t* s, void* base, uint32_t bound0,
                        uint32_t stride0, uint32_t bound1, uint32_t stride1) {
    snax_simd_shape_clear(s);
    s->base = base;
    s->lane_stride = SIMD_LANE_BYTES;
    s->lane_mask = 0xFFFFFFFFu;
    s->byte_mask = 0xFFFFFFFFu;
    s->dim = 2;
    s->bound[0] = bound0;
    s->stride[0] = stride0;
    s->bound[1] = bound1;
    s->stride[1] = stride1;
}

void snax_simd_shape_broadcast(snax_simd_shape_t* s, void* base,
                               uint32_t beats) {
    snax_simd_shape_flat(s, base, beats);
    // Stride 0 on the innermost loop is what drives the reader's repeat path:
    // one beat is fetched and presented `beats` times.
    s->stride[0] = 0;
}

uint32_t snax_simd_shape_beats(const snax_simd_shape_t* s) {
    uint32_t n = 1;
    for (uint32_t i = 0; i < s->dim; i++) n *= s->bound[i];
    return n;
}

// ============================================================== operators

void snax_simd_op_map(snax_simd_op_t* op, uint8_t ext, uint32_t func,
                      uint32_t a_bits, uint32_t b_bits) {
    // StreamMap CSR layout (StreamMap.scala:22): csr0 = a (FP32 bits),
    // csr1 = b (FP32 bits), csr2 bits[1:0] = func.
    op->csr_num = 0;
    for (uint32_t i = 0; i < SIMD_MAX_OP_CSR; i++) op->csr[i] = 0;
    op->id = ext;
    op->csr_num = 3;
    op->csr[0] = a_bits;
    op->csr[1] = b_bits;
    op->csr[2] = func;
}

void snax_simd_op_reduce(snax_simd_op_t* op, uint8_t ext, uint32_t mode,
                         uint32_t operand_beats) {
    // StreamReduce CSR layout (StreamReduce.scala:19): csr0 = operandCount,
    // csr1 = op, with bits[7:0] the fold, bit[8] tap, bit[9] fp32out.
    op->csr_num = 0;
    for (uint32_t i = 0; i < SIMD_MAX_OP_CSR; i++) op->csr[i] = 0;
    op->id = ext;
    op->csr_num = 2;
    op->csr[0] = operand_beats;
    op->csr[1] = mode;
}

void snax_simd_op_elementwise(snax_simd_op_t* op, uint8_t ext, uint32_t mode,
                              uint32_t operand_beats) {
    // StreamElementwise CSR layout (StreamElementwise.scala:22): csr0 =
    // operandCount, csr1 bits[7:0] = op.
    op->csr_num = 0;
    for (uint32_t i = 0; i < SIMD_MAX_OP_CSR; i++) op->csr[i] = 0;
    op->id = ext;
    op->csr_num = 2;
    op->csr[0] = operand_beats;
    op->csr[1] = mode;
}

void snax_simd_op_quantise(snax_simd_op_t* op, uint8_t ext,
                           uint32_t inv_scale_bits) {
    // Fp16ToInt8 CSR layout (Fp16ToInt8.scala:230): csr0 = inv_scale (FP32).
    op->csr_num = 0;
    for (uint32_t i = 0; i < SIMD_MAX_OP_CSR; i++) op->csr[i] = 0;
    op->id = ext;
    op->csr_num = 1;
    op->csr[0] = inv_scale_bits;
}

void snax_simd_op_raw(snax_simd_op_t* op, uint8_t ext, const uint32_t* csr,
                      uint32_t csr_num) {
    op->csr_num = 0;
    for (uint32_t i = 0; i < SIMD_MAX_OP_CSR; i++) op->csr[i] = 0;
    op->id = ext;
    op->csr_num = (uint8_t)(csr_num > SIMD_MAX_OP_CSR ? SIMD_MAX_OP_CSR
                                                     : csr_num);
    for (uint32_t i = 0; i < op->csr_num; i++) op->csr[i] = csr[i];
}

// ============================================================== raw operator CSRs

int32_t snax_simd_enable_ext(uint8_t ext, uint32_t* csr_value) {
#if SIMD_EXT_NUM == 0
    (void)ext;
    (void)csr_value;
    SIMD_DEBUG_PRINT("This SIMD block has no operators\n");
    return -1;
#else
    if (ext >= SIMD_EXT_NUM) return -1;
    uint8_t custom_csr_list[SIMD_EXT_NUM] = SIMD_EXT_CUSTOM_CSR_NUM;
    uint32_t csr_offset = SIMD_EXT_CSR_PTR;
    for (uint8_t i = 0; i < ext; i++) csr_offset += custom_csr_list[i];

    snax_write_simd_cfg_reg(
        SIMD_EXT_ENABLE_PTR,
        snax_read_simd_cfg_reg(SIMD_EXT_ENABLE_PTR) | (1u << ext));
    for (uint8_t i = 0; i < custom_csr_list[ext]; i++) {
        snax_write_simd_cfg_reg(csr_offset + i, csr_value[i]);
    }
    return 0;
#endif
}

int32_t snax_simd_disable_ext(uint8_t ext) {
#if SIMD_EXT_NUM == 0
    (void)ext;
    return 0;
#else
    if (ext >= SIMD_EXT_NUM) return 0;
    snax_write_simd_cfg_reg(
        SIMD_EXT_ENABLE_PTR,
        snax_read_simd_cfg_reg(SIMD_EXT_ENABLE_PTR) & ~(1u << ext));
    return 0;
#endif
}

void snax_simd_disable_all_ext(void) {
#if SIMD_EXT_NUM != 0
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0);
#endif
}

// ============================================================== task programming

// The flat form. Everything above funnels into this; it is also what
// snax-simd-compat.h binds the ported kernels' xDMA calls to.
// EVERY CSR ADDRESS BELOW IS A COMPILE-TIME CONSTANT, DELIBERATELY.
//
// csrw_ss (snRuntime/src/csr.h:461) is a switch over the CSR number, because the
// RISC-V csrw instruction takes an IMMEDIATE. When the compiler can fold the
// address it emits one `csrw imm`; when it cannot, the write becomes a
// jump-table load out of L2 plus an indirect jump -- roughly 20 cycles instead
// of 1. Writing the ~30 geometry CSRs from `for` loops therefore cost ~700
// cycles per task, and made the FlashAttention pipeline 90 % orchestration.
// Unrolled, the same task programs in a few tens of cycles.
//
// The unroll is guarded on the AGU depth the cfg actually generated.
#if SIMD_SRC_TEMP_DIM > 6 || SIMD_DST_TEMP_DIM > 6
#error "snax-simd-lib: extend the unrolled CSR writes for this AGU depth"
#endif

// bound/stride for loop level i: the caller's value, or the identity for the
// levels it does not use.
#define SIMD_BOUND_AT(i) (((i) < in_dim) ? in_bound[(i)] : 1u)
#define SIMD_STRIDE_AT(i) (((i) < in_dim) ? in_stride[(i)] : 0u)
#define SIMD_OBOUND_AT(i) (((i) < out_dim) ? out_bound[(i)] : 1u)
#define SIMD_OSTRIDE_AT(i) (((i) < out_dim) ? out_stride[(i)] : 0u)


// The hot path: nothing but CSR writes.
//
// snax_simd_program() measured 313 cycles for ~110 instructions -- 3.5 CPI,
// which is L1I miss cost, not work. Its 23 CSR writes are ~23 cycles; the rest
// is the footprint of its validation branches, two multiply loops and three
// printf call sites, 752 bytes across 12 cache lines that have to be refetched
// every time the config path is re-entered. This variant is straight-line and
// fits in a handful of lines, so it stays resident.
//
// It trusts the caller completely: every dimension is written unconditionally,
// so the shape's unused loops must hold the neutral bound=1/stride=0 that
// snax_simd_shape_clear() installs. Use snax_simd_program() once to validate a
// new shape during bring-up, then this for the steady state.
void snax_simd_program_fast(const snax_simd_shape_t* in,
                            const snax_simd_shape_t* out) {
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_LSB, (uint32_t)(uintptr_t)in->base);
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_MSB, 0);
    snax_write_simd_cfg_reg(SIMD_SRC_SPATIAL_STRIDE_PTR, in->lane_stride);
#if SIMD_SRC_TEMP_DIM > 0
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 0, in->bound[0]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 0, in->stride[0]);
#endif
#if SIMD_SRC_TEMP_DIM > 1
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 1, in->bound[1]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 1, in->stride[1]);
#endif
#if SIMD_SRC_TEMP_DIM > 2
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 2, in->bound[2]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 2, in->stride[2]);
#endif
#if SIMD_SRC_TEMP_DIM > 3
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 3, in->bound[3]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 3, in->stride[3]);
#endif
#if SIMD_SRC_TEMP_DIM > 4
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 4, in->bound[4]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 4, in->stride[4]);
#endif
#if SIMD_SRC_TEMP_DIM > 5
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 5, in->bound[5]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 5, in->stride[5]);
#endif
    snax_write_simd_cfg_reg(SIMD_SRC_ENABLED_CHAN_PTR, in->lane_mask);
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_LSB, (uint32_t)(uintptr_t)out->base);
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_MSB, 0);
    snax_write_simd_cfg_reg(SIMD_DST_SPATIAL_STRIDE_PTR, out->lane_stride);
#if SIMD_DST_TEMP_DIM > 0
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 0, out->bound[0]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 0, out->stride[0]);
#endif
#if SIMD_DST_TEMP_DIM > 1
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 1, out->bound[1]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 1, out->stride[1]);
#endif
#if SIMD_DST_TEMP_DIM > 2
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 2, out->bound[2]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 2, out->stride[2]);
#endif
#if SIMD_DST_TEMP_DIM > 3
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 3, out->bound[3]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 3, out->stride[3]);
#endif
#if SIMD_DST_TEMP_DIM > 4
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 4, out->bound[4]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 4, out->stride[4]);
#endif
#if SIMD_DST_TEMP_DIM > 5
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 5, out->bound[5]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 5, out->stride[5]);
#endif
    snax_write_simd_cfg_reg(SIMD_DST_ENABLED_CHAN_PTR, out->lane_mask);
    snax_write_simd_cfg_reg(SIMD_DST_ENABLED_BYTE_PTR, out->byte_mask);
}

int32_t snax_simd_program(void* in, void* out, uint32_t in_lane_stride,
                          uint32_t out_lane_stride, uint32_t in_dim,
                          uint32_t* in_stride, uint32_t* in_bound,
                          uint32_t out_dim, uint32_t* out_stride,
                          uint32_t* out_bound, uint32_t in_lane_mask,
                          uint32_t out_lane_mask, uint32_t out_byte_mask) {
    if (in_dim > SIMD_SRC_TEMP_DIM || out_dim > SIMD_DST_TEMP_DIM) {
        SIMD_DEBUG_PRINT("Sweep is deeper than the AGU supports\n");
        return -4;
    }
    if (((uintptr_t)in % SIMD_BEAT_BYTES) != 0 ||
        ((uintptr_t)out % SIMD_BEAT_BYTES) != 0) {
        SIMD_DEBUG_PRINT("SIMD buffers must be %d-byte aligned\n",
                         SIMD_BEAT_BYTES);
        return -6;
    }

    uint32_t in_beats = 1, out_beats = 1;
    for (uint32_t i = 0; i < in_dim; i++) in_beats *= in_bound[i];
    for (uint32_t i = 0; i < out_dim; i++) out_beats *= out_bound[i];
    if (in_beats == 0 || out_beats == 0) {
        SIMD_DEBUG_PRINT("A bound of 0 makes an empty task\n");
        return -2;
    }

    uint32_t in_lo = (uint32_t)(uintptr_t)in;
    uint32_t out_lo = (uint32_t)(uintptr_t)out;

    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_LSB, in_lo);
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_MSB, 0);
    snax_write_simd_cfg_reg(SIMD_SRC_SPATIAL_STRIDE_PTR, in_lane_stride);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 0, SIMD_BOUND_AT(0));
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 0, SIMD_STRIDE_AT(0));
#if SIMD_SRC_TEMP_DIM > 1
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 1, SIMD_BOUND_AT(1));
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 1, SIMD_STRIDE_AT(1));
#endif
#if SIMD_SRC_TEMP_DIM > 2
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 2, SIMD_BOUND_AT(2));
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 2, SIMD_STRIDE_AT(2));
#endif
#if SIMD_SRC_TEMP_DIM > 3
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 3, SIMD_BOUND_AT(3));
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 3, SIMD_STRIDE_AT(3));
#endif
#if SIMD_SRC_TEMP_DIM > 4
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 4, SIMD_BOUND_AT(4));
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 4, SIMD_STRIDE_AT(4));
#endif
#if SIMD_SRC_TEMP_DIM > 5
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 5, SIMD_BOUND_AT(5));
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 5, SIMD_STRIDE_AT(5));
#endif
    snax_write_simd_cfg_reg(SIMD_SRC_ENABLED_CHAN_PTR, in_lane_mask);

    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_LSB, out_lo);
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_MSB, 0);
    snax_write_simd_cfg_reg(SIMD_DST_SPATIAL_STRIDE_PTR, out_lane_stride);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 0, SIMD_OBOUND_AT(0));
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 0, SIMD_OSTRIDE_AT(0));
#if SIMD_DST_TEMP_DIM > 1
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 1, SIMD_OBOUND_AT(1));
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 1, SIMD_OSTRIDE_AT(1));
#endif
#if SIMD_DST_TEMP_DIM > 2
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 2, SIMD_OBOUND_AT(2));
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 2, SIMD_OSTRIDE_AT(2));
#endif
#if SIMD_DST_TEMP_DIM > 3
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 3, SIMD_OBOUND_AT(3));
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 3, SIMD_OSTRIDE_AT(3));
#endif
#if SIMD_DST_TEMP_DIM > 4
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 4, SIMD_OBOUND_AT(4));
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 4, SIMD_OSTRIDE_AT(4));
#endif
#if SIMD_DST_TEMP_DIM > 5
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 5, SIMD_OBOUND_AT(5));
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 5, SIMD_OSTRIDE_AT(5));
#endif
    snax_write_simd_cfg_reg(SIMD_DST_ENABLED_CHAN_PTR, out_lane_mask);
    snax_write_simd_cfg_reg(SIMD_DST_ENABLED_BYTE_PTR, out_byte_mask);

    return 0;
}

int32_t snax_simd_configure(const snax_simd_shape_t* in,
                            const snax_simd_shape_t* out,
                            const snax_simd_op_t* ops, uint32_t n_ops) {
    snax_simd_disable_all_ext();
    for (uint32_t i = 0; i < n_ops; i++) {
        uint32_t csr[SIMD_MAX_OP_CSR];
        for (uint32_t j = 0; j < SIMD_MAX_OP_CSR; j++) csr[j] = ops[i].csr[j];
        int32_t ret = snax_simd_enable_ext(ops[i].id, csr);
        if (ret != 0) {
            SIMD_DEBUG_PRINT("Could not arm operator %u\n", ops[i].id);
            return ret;
        }
    }
    return snax_simd_program(in->base, out->base, in->lane_stride,
                             out->lane_stride, in->dim, (uint32_t*)in->stride,
                             (uint32_t*)in->bound, out->dim,
                             (uint32_t*)out->stride, (uint32_t*)out->bound,
                             in->lane_mask, out->lane_mask, out->byte_mask);
}

uint32_t snax_simd_launch(void) {
    uint32_t submitted = snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR);
    snax_write_simd_cfg_reg(SIMD_START_PTR, 1);
    while (snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR) == submitted) {
        // The task queue was full; the write is held until it drains.
    }
    return snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR);
}

uint32_t snax_simd_launch_async(void) {
    // No poll: the caller reads the submitted counter itself, or simply waits on
    // the finished counter later. Skipping the confirm-poll is what lets task
    // r+1 be staged while task r is still running.
    snax_write_simd_cfg_reg(SIMD_START_PTR, 1);
    return snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR);
}

int32_t snax_simd_run(const snax_simd_shape_t* in, const snax_simd_shape_t* out,
                      const snax_simd_op_t* ops, uint32_t n_ops) {
    int32_t ret = snax_simd_configure(in, out, ops, n_ops);
    if (ret != 0) return ret;
    snax_simd_wait(snax_simd_launch());
    return 0;
}

int32_t snax_simd_repoint(void* in_base, void* out_base, uint32_t out_beats) {
    if (((uintptr_t)in_base % SIMD_BEAT_BYTES) != 0 ||
        ((uintptr_t)out_base % SIMD_BEAT_BYTES) != 0) {
        SIMD_DEBUG_PRINT("SIMD buffers must be %d-byte aligned\n",
                         SIMD_BEAT_BYTES);
        return -6;
    }
    uint64_t s = (uint64_t)(uintptr_t)in_base;
    uint64_t d = (uint64_t)(uintptr_t)out_base;
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_LSB, (uint32_t)s);
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_MSB, (uint32_t)(s >> 32));
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_LSB, (uint32_t)d);
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_MSB, (uint32_t)(d >> 32));
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR, out_beats);
    return 0;
}

uint32_t snax_simd_out_beats(const snax_simd_op_t* ops, uint32_t n_ops,
                             uint32_t in_beats) {
    uint32_t beats = in_beats;
    for (uint32_t i = 0; i < n_ops; i++) {
#ifdef SIMD_EXT_STREAMREDUCE
        if (ops[i].id == SIMD_EXT_STREAMREDUCE) {
            uint32_t row = ops[i].csr[0] ? ops[i].csr[0] : 1;
            uint32_t rows = beats / row;
            // tap re-emits the row and appends the scalar; otherwise only the
            // scalar survives.
            beats = (ops[i].csr[1] & SIMD_RED_TAP) ? beats + rows : rows;
            continue;
        }
#endif
#ifdef SIMD_EXT_STREAMELEMENTWISE
        if (ops[i].id == SIMD_EXT_STREAMELEMENTWISE) {
            uint32_t operands = ops[i].csr[0] ? ops[i].csr[0] : 1;
            beats = beats / operands;
            continue;
        }
#endif
#ifdef SIMD_EXT_FP16TOINT8
        if (ops[i].id == SIMD_EXT_FP16TOINT8) {
            beats = beats / 2;  // two FP16 beats pack into one INT8 beat
            continue;
        }
#endif
        // Everything else is 1:1 (StreamMap, Transposer, ElementwiseAdd).
    }
    return beats;
}

// ============================================================== convenience

int32_t snax_simd_map(uint8_t ext, void* in, void* out, uint32_t beats,
                      uint32_t func, uint32_t a_bits, uint32_t b_bits) {
    snax_simd_shape_t si, so;
    snax_simd_op_t op;
    snax_simd_shape_flat(&si, in, beats);
    snax_simd_shape_flat(&so, out, beats);
    snax_simd_op_map(&op, ext, func, a_bits, b_bits);
    return snax_simd_run(&si, &so, &op, 1);
}

int32_t snax_simd_reduce_rows(uint8_t ext, void* in, void* out, uint32_t rows,
                              uint32_t beats_per_row, uint32_t mode) {
    snax_simd_shape_t si, so;
    snax_simd_op_t op;
    snax_simd_shape_rows(&si, in, rows, beats_per_row,
                         beats_per_row * SIMD_BEAT_BYTES);
    snax_simd_op_reduce(&op, ext, mode, beats_per_row);
    snax_simd_shape_flat(&so, out,
                         snax_simd_out_beats(&op, 1, rows * beats_per_row));
    return snax_simd_run(&si, &so, &op, 1);
}

int32_t snax_simd_program_flat(void* in, void* out, uint32_t beats) {
    snax_simd_shape_t si, so;
    snax_simd_shape_flat(&si, in, beats);
    snax_simd_shape_flat(&so, out, beats);
    // Deliberately NOT snax_simd_configure(): that would clear the operator
    // chain, and this exists for kernels that armed one themselves.
    return snax_simd_program(si.base, so.base, si.lane_stride, so.lane_stride,
                             si.dim, si.stride, si.bound, so.dim, so.stride,
                             so.bound, si.lane_mask, so.lane_mask,
                             so.byte_mask);
}

int32_t snax_simd_copy(void* in, void* out, uint32_t bytes) {
    if (bytes % SIMD_BEAT_BYTES != 0) {
        SIMD_DEBUG_PRINT("Copy size must be a multiple of %d bytes\n",
                         SIMD_BEAT_BYTES);
        return -1;
    }
    snax_simd_shape_t si, so;
    snax_simd_shape_flat(&si, in, bytes / SIMD_BEAT_BYTES);
    snax_simd_shape_flat(&so, out, bytes / SIMD_BEAT_BYTES);
    return snax_simd_run(&si, &so, 0, 0);
}

// ============================================================== transpose

#if defined(SIMD_EXT_TRANSPOSERROW8_8COL8_8BIT8_16)
#define SIMD_TRANSPOSE_EXT_ID SIMD_EXT_TRANSPOSERROW8_8COL8_8BIT8_16
#elif defined(SIMD_EXT_TRANSPOSERROW8_8_8COL8_8_8BIT8_16_32)
#define SIMD_TRANSPOSE_EXT_ID SIMD_EXT_TRANSPOSERROW8_8_8COL8_8_8BIT8_16_32
#elif defined(SIMD_EXT_TRANSPOSERROW8_8_4COL8_8_4BIT8_16_32)
#define SIMD_TRANSPOSE_EXT_ID SIMD_EXT_TRANSPOSERROW8_8_4COL8_8_4BIT8_16_32
#endif

int32_t snax_simd_transpose(void* in, void* out, uint32_t rows, uint32_t cols,
                            uint32_t element_width_bits) {
#ifndef SIMD_TRANSPOSE_EXT_ID
    (void)in;
    (void)out;
    (void)rows;
    (void)cols;
    (void)element_width_bits;
    SIMD_DEBUG_PRINT("This SIMD block has no transposer\n");
    return -5;
#else
    uint32_t tile_width, transfer_count, mode;
    switch (element_width_bits) {
        case 8:
            tile_width = 8;
            transfer_count = 1;
            mode = 0;
            break;
        case 16:
            tile_width = 8;
            transfer_count = 2;
            mode = 1;
            break;
        default:
            SIMD_DEBUG_PRINT("Unsupported transpose element width %u\n",
                             element_width_bits);
            return -1;
    }
    if (rows == 0 || cols == 0) {
        SIMD_DEBUG_PRINT("Transpose dimensions must be non-zero\n");
        return -2;
    }
    if (rows % tile_width != 0 || cols % tile_width != 0) {
        SIMD_DEBUG_PRINT("Transpose needs rows/cols multiples of %u\n",
                         tile_width);
        return -3;
    }

    uint32_t bpe = element_width_bits / 8;
    uint64_t in_lane = (uint64_t)cols * bpe;
    uint64_t out_lane = (uint64_t)rows * bpe;
    uint64_t in_outer = (uint64_t)cols * tile_width * bpe;
    uint64_t out_mid = (uint64_t)rows * tile_width * bpe;
    if (in_lane > UINT32_MAX || out_lane > UINT32_MAX ||
        in_outer > UINT32_MAX || out_mid > UINT32_MAX) {
        SIMD_DEBUG_PRINT("Transpose AGU parameters overflow 32 bits\n");
        return -4;
    }

    snax_simd_shape_t si, so;
    snax_simd_shape_clear(&si);
    snax_simd_shape_clear(&so);
    si.base = in;
    so.base = out;
    si.lane_mask = so.lane_mask = 0xFFFFFFFFu;
    si.byte_mask = so.byte_mask = 0xFFFFFFFFu;

    if (transfer_count == 1 && rows == tile_width && cols == tile_width) {
        // A single tile is one beat in, one beat out.
        snax_simd_shape_flat(&si, in, 1);
        snax_simd_shape_flat(&so, out, 1);
    } else {
        si.lane_stride = (uint32_t)in_lane;
        si.dim = 3;
        si.bound[0] = transfer_count;
        si.stride[0] = SIMD_LANE_BYTES;
        si.bound[1] = cols / tile_width;
        si.stride[1] = tile_width * bpe;
        si.bound[2] = rows / tile_width;
        si.stride[2] = (uint32_t)in_outer;

        so.lane_stride = (uint32_t)out_lane;
        so.dim = 3;
        so.bound[0] = transfer_count;
        so.stride[0] = SIMD_LANE_BYTES;
        so.bound[1] = cols / tile_width;
        so.stride[1] = (uint32_t)out_mid;
        so.bound[2] = rows / tile_width;
        so.stride[2] = tile_width * bpe;
    }

    uint32_t csr[1] = {mode};
    snax_simd_op_t op;
    snax_simd_op_raw(&op, SIMD_TRANSPOSE_EXT_ID, csr, 1);
    return snax_simd_configure(&si, &so, &op, 1);
#endif
}
