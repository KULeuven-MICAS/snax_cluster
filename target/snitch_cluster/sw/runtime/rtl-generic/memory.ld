/* Copyright 2020 ETH Zurich and University of Bologna. */
/* Solderpad Hardware License, Version 0.51, see LICENSE for details. */
/* SPDX-License-Identifier: SHL-0.51 */

/* L1 is the cluster TCDM. Everything else -- .text, .rodata, .data, .bss --
   lands in L3, i.e. DRAM, so a plain `static` in a kernel is a DRAM object and
   reading it in a hot loop costs an L3 round trip per access. Data that a
   kernel touches per beat belongs in the .l1 section (SNRT_L1_DATA), which is
   what this region carries.

   ORIGIN is cluster_base_addr, which is 0x1000_0000 in every cfg on this
   target. LENGTH is deliberately the SMALLEST tcdm.size any of them uses
   (128 KiB); configs with 256 KiB simply have more room than declared here.
   Under-declaring is the safe direction -- it can only refuse a .l1 that is too
   big, never place one outside real memory. Generating this file from the cfg
   the way snitch_cluster_addrmap.h is generated would remove the duplication. */
MEMORY
{
    L1 (rw)   : ORIGIN = 0x10000000, LENGTH = 0x20000
    L3 (rwxa) : ORIGIN = 0x80000000, LENGTH = 0x80000000
}
