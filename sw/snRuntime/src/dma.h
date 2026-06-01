// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// A DMA transfer identifier.
typedef uint32_t snrt_dma_txid_t;

// Early declaration of the functions
inline uint32_t __attribute__((const)) snrt_cluster_base_addrh();

// ---------------------------------------------------------------------------
// Snitch DMA custom instruction encoding (opcode 0b0101011)
// ---------------------------------------------------------------------------
// Each `.word` below hand-assembles one DMA instruction. The 32-bit word is
// laid out as a standard RISC-V instruction:
//
//   [31:25] funct7   selects the DMA op (DMSRC/DMDST/DMCPYI/DMSTR/...)
//   [24:20] rs2 / immediate field
//   [19:15] rs1      first source GPR
//   [14:12] funct3   = 0b000
//   [11: 7] rd       destination GPR (e.g. the returned transfer id)
//   [ 6: 0] opcode   = 0b0101011 (custom-3)
//
// The transfer "config" lives in the DMCPYI immediate field [24:20]. The iDMA
// frontend (idma_inst64_top.sv, proc_fe_inst_decode) decodes it as:
//
//   imm[1:0]  -> idma_fe_cfg   (config)
//       bit 0 : decouple_aw. R-AW coupling -- the RTL (idma_channel_coupler.sv)
//               is authoritative; the idma_pkg.sv comment is inverted.
//               0 = COUPLED  : the write address (AW) is held back until the
//                   first beat of the matching read arrives, so a data-starved
//                   write can never grab and hold a shared write mux. Use this
//                   to avoid read/write ordering deadlocks.
//               1 = DECOUPLED: AW is issued eagerly, before any read data
//                   returns -- a starved write can then hold the mux and
//                   deadlock against other writers on the same path.
//               (This bit carried `decouple_rw` in the legacy frontend.)
//       bit 1 : 2D enable. 0 = 1D transfer (reps forced to 1),
//               1 = 2D transfer (uses the dmstr strides + dmrep repetitions)
//   imm[4:2]  -> idma_fe_sel_chan (DMA channel select, also used as AXI id)
//
// So to set the config you write the immediate as:
//   imm = (chan << 2) | (twod << 1) | decouple_aw
//   1D, decouple_aw=0, ch0 -> 0b00000  ;  2D, decouple_aw=0, ch0 -> 0b00010
// ---------------------------------------------------------------------------

/// Initiate an asynchronous 1D DMA transfer with wide 64-bit pointers.
inline snrt_dma_txid_t snrt_dma_start_1d_wideptr(uint64_t dst, uint64_t src,
                                                 size_t size) {
    // Current DMA does not allow transfers with size == 0 (blocks)
    // TODO(colluca) remove this check once new DMA is integrated
    if (size > 0) {
        register uint32_t reg_dst_low asm("a0") = dst >> 0;    // 10
        register uint32_t reg_dst_high asm("a1") = dst >> 32;  // 11
        register uint32_t reg_src_low asm("a2") = src >> 0;    // 12
        register uint32_t reg_src_high asm("a3") = src >> 32;  // 13
        register uint32_t reg_size asm("a4") = size;           // 14

        // dmsrc a2, a3
        asm volatile(
            ".word (0b0000000 << 25) | \
                (     (13) << 20) | \
                (     (12) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n" ::"r"(reg_src_high),
            "r"(reg_src_low));

        // dmdst a0, a1
        asm volatile(
            ".word (0b0000001 << 25) | \
                (     (11) << 20) | \
                (     (10) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n" ::"r"(reg_dst_high),
            "r"(reg_dst_low));

        // dmcpyi a0, a4, 0b00
        // config immediate [24:20] = 0b00000: chan=0, twod=0 (1D),
        // decouple_aw=0 (R-AW coupled; see note above)
        register uint32_t reg_txid asm("a0");  // 10
        asm volatile(
            ".word (0b0000010 << 25) | \
                (  0b00000 << 20) | \
                (     (14) << 15) | \
                (    0b000 << 12) | \
                (     (10) <<  7) | \
                (0b0101011 <<  0)   \n"
            : "=r"(reg_txid)
            : "r"(reg_size));

        return reg_txid;
    } else {
        return -1;
    }
}

/// Initiate an asynchronous 1D DMA transfer. (for local-chip transfers)
inline snrt_dma_txid_t snrt_dma_start_1d(void *dst, const void *src,
                                         size_t size) {
    uint64_t dst_wideptr = (uint64_t)dst;
    dst_wideptr += (uint64_t)snrt_cluster_base_addrh() << 32;
    uint64_t src_wideptr = (uint64_t)src;
    src_wideptr += (uint64_t)snrt_cluster_base_addrh() << 32;
    return snrt_dma_start_1d_wideptr(dst_wideptr, src_wideptr, size);
}

/// Initiate an asynchronous 2D DMA transfer with wide 64-bit pointers.
inline snrt_dma_txid_t snrt_dma_start_2d_wideptr(uint64_t dst, uint64_t src,
                                                 size_t size, size_t dst_stride,
                                                 size_t src_stride,
                                                 size_t repeat) {
    // Current DMA does not allow transfers with size == 0 (blocks)
    // TODO(colluca) remove this check once new DMA is integrated
    if (size > 0) {
        register uint32_t reg_dst_low asm("a0") = dst >> 0;       // 10
        register uint32_t reg_dst_high asm("a1") = dst >> 32;     // 11
        register uint32_t reg_src_low asm("a2") = src >> 0;       // 12
        register uint32_t reg_src_high asm("a3") = src >> 32;     // 13
        register uint32_t reg_size asm("a4") = size;              // 14
        register uint32_t reg_dst_stride asm("a5") = dst_stride;  // 15
        register uint32_t reg_src_stride asm("a6") = src_stride;  // 16
        register uint32_t reg_repeat asm("a7") = repeat;          // 17

        // dmsrc a0, a1
        asm volatile(
            ".word (0b0000000 << 25) | \
                (     (13) << 20) | \
                (     (12) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n" ::"r"(reg_src_high),
            "r"(reg_src_low));

        // dmdst a0, a1
        asm volatile(
            ".word (0b0000001 << 25) | \
                (     (11) << 20) | \
                (     (10) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n" ::"r"(reg_dst_high),
            "r"(reg_dst_low));

        // dmstr a5, a6
        asm volatile(
            ".word (0b0000110 << 25) | \
                (     (15) << 20) | \
                (     (16) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n"
            :
            : "r"(reg_dst_stride), "r"(reg_src_stride));

        // dmrep a7
        asm volatile(
            ".word (0b0000111 << 25) | \
                (     (17) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n"
            :
            : "r"(reg_repeat));

        // dmcpyi a0, a4, 0b10
        // config immediate [24:20] = 0b00010: chan=0, twod=1 (2D),
        // decouple_aw=0 (R-AW coupled; see note above)
        register uint32_t reg_txid asm("a0");  // 10
        asm volatile(
            ".word (0b0000010 << 25) | \
                (  0b00010 << 20) | \
                (     (14) << 15) | \
                (    0b000 << 12) | \
                (     (10) <<  7) | \
                (0b0101011 <<  0)   \n"
            : "=r"(reg_txid)
            : "r"(reg_size));

        return reg_txid;
    } else {
        return -1;
    }
}

/// Initiate an asynchronous 2D DMA transfer. (for local-chip transfers)
inline snrt_dma_txid_t snrt_dma_start_2d(void *dst, const void *src,
                                         size_t size, size_t dst_stride,
                                         size_t src_stride, size_t repeat) {
    uint64_t dst_wideptr = (uint64_t)dst;
    dst_wideptr += (uint64_t)snrt_cluster_base_addrh() << 32;
    uint64_t src_wideptr = (uint64_t)src;
    src_wideptr += (uint64_t)snrt_cluster_base_addrh() << 32;
    return snrt_dma_start_2d_wideptr(dst_wideptr, src_wideptr, size, dst_stride,
                                     src_stride, repeat);
}

/// Block until a transfer finishes.
inline void snrt_dma_wait(snrt_dma_txid_t tid) {
    // dmstati t0, 0  # 2=status.completed_id
    asm volatile(
        "1: \n"
        ".word (0b0000100 << 25) | \
               (  0b00000 << 20) | \
               (    0b000 << 12) | \
               (      (5) <<  7) | \
               (0b0101011 <<  0)   \n"
        "sub t0, t0, %0 \n"
        "blez t0, 1b \n" ::"r"(tid)
        : "t0");
}

/// Block until all operation on the DMA ceases.
inline void snrt_dma_wait_all() {
    // dmstati t0, 2  # 2=status.busy
    asm volatile(
        "1: \n"
        ".word (0b0000100 << 25) | \
               (  0b00010 << 20) | \
               (    0b000 << 12) | \
               (      (5) <<  7) | \
               (0b0101011 <<  0)   \n"
        "bne t0, zero, 1b \n" ::
            : "t0");
}

/**
 * @brief start tracking of dma performance region. Does not have any
 * implications on the HW. Only injects a marker in the DMA traces that can be
 * analyzed
 *
 */
inline void snrt_dma_start_tracking() { asm volatile("dmstati zero, 1"); }

/**
 * @brief stop tracking of dma performance region. Does not have any
 * implications on the HW. Only injects a marker in the DMA traces that can be
 * analyzed
 *
 */
inline void snrt_dma_stop_tracking() { asm volatile("dmstati zero, 3"); }

/**
 * @brief fast memset function performed by DMA
 *
 * @param ptr pointer to the start of the region
 * @param value value to set
 * @param len number of bytes, must be multiple of DMA bus-width
 */
inline void snrt_dma_memset(void *ptr, uint8_t value, uint32_t len) {
    // set first 64bytes to value
    // memset(ptr, value, 64);
    uint8_t *p = ptr;
    uint32_t nbytes = 64;
    while (nbytes--) {
        *p++ = value;
    }

    // DMA copy the the rest
    snrt_dma_txid_t memset_txid =
        snrt_dma_start_2d(ptr, ptr, 64, 64, 0, len / 64);
    snrt_dma_wait_all();
}
