/* Copyright 2020 ETH Zurich and University of Bologna. */
/* Solderpad Hardware License, Version 0.51, see LICENSE for details. */
/* SPDX-License-Identifier: SHL-0.51 */

OUTPUT_ARCH( "riscv" )
ENTRY(_start)

/* Memory section should be provided in a separate, platform-specific */
/* file. It should define at least the L1 and L3 memory blocks. */
INCLUDE memory.ld

SECTIONS
{

  /* Program code goes into L3 */
  .text :
  {
    . = ALIGN(4);
    *(.init)
    *(.text.init)
    *(.text.startup)
    *(.text)
    *(.text*)
    *(.text)
    . = ALIGN(4);
    _etext = .;
  } >L3

  /* By default, constant data goes into L3, right after code section */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >L3

  /* HTIF section for FESVR */
  .htif         : { } >L3

  /* Thread Local Storage sections */
  .tdata    :
  {
    __tdata_start = .;
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    __tdata_end = .;
  } >L3
  .tbss :
  {
    __tbss_start = .;
    *(.tbss .tbss.* .gnu.linkonce.tb.*)
    *(.tcommon)
    __tbss_end = .;
  } >L3
  /* add a section after .tbss to put the __tbss_end symbol into for
     the LLD linker */
  .tbssend : { __tbss_end2 = .; }

  /* Cluster Local Storage sections */
  .cdata    :
  {
    __cdata_start = .;
    *(.cdata .cdata.*)
    __cdata_end = .;
  } >L3
  .cbss :
  {
    __cbss_start = .;
    *(.cbss .cbss.*)
    __cbss_end = .;
  } >L3

  /* used by the startup to initialize data */
  _sidata = LOADADDR(.data);

  /* small data section that can be addressed through the global pointer */
  .sdata          :
  {
    __SDATA_BEGIN__ = .;
    __global_pointer$ = . + 0x7f0;
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  } >L3

  /* Initialized data sections goes into L3 */
  .data :
  {
    __DATA_BEGIN__ = .;
    *(.data .data.* .gnu.linkonce.d.*)
    SORT(CONSTRUCTORS)
  } >L3
  _edata = .; PROVIDE (edata = .);

  /* small bss section */
  . = .;
  __bss_start = .;
  .sbss           :
  {
    *(.dynsbss)
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.scommon)
  } >L3

  /* Uninitialized data section */
  .bss            :
  {
   *(.dynbss)
   *(.bss .bss.* .gnu.linkonce.b.*)
   *(COMMON)
   /* Align here to ensure that the .bss section occupies space up to
      _end.  Align after .bss to ensure correct alignment even if the
      .bss section disappears because there are no input sections. */
   . = ALIGN(. != 0 ? 32 / 8 : 1);
  } >L3
  . = ALIGN(32 / 8);
  . = SEGMENT_START("ldata-segment", .);
  . = ALIGN(32 / 8);
  __BSS_END__ = .;
  __bss_end = .;
  _end = .; PROVIDE (end = .);

  /* Hot data, in the cluster TCDM.
     Everything above lands in L3 (DRAM). That is the right default for code and
     for anything touched once, but it is a trap for data a kernel reads per
     beat: a `static` array of accelerator descriptors parked in .bss cost 4500
     cycles a run in snax-flashattn, purely in L3 round trips, until it was
     moved here. Declare such objects with SNRT_L1_DATA.

     NOLOAD: this is uninitialised scratch and TCDM is not part of the loadable
     image, so anything placed here must be written before it is read. The
     64-byte tail alignment matters -- snrt_l1_start_addr() returns __l1_end, so
     it is what every heap allocation and every `snrt_l1_next() + offset` in a
     kernel inherits, and the accelerators require beat-aligned buffers. */
  .l1 (NOLOAD) :
  {
    . = ALIGN(8);
    __l1_start = .;
    *(.l1 .l1.*)
    . = ALIGN(64);
    __l1_end = .;
  } >L1

  /* Uninitialized data section in L3 */
  .dram :
  {
    *(.dram)
    _edram = .;
  } >L3
}
