# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# One cluster's share of HeMAiA's MoE prefill (src/snax-dsv2-moe-prefill.c): the experts' gate|up
# phase, chunk by chunk -- ring load, GEMM on array shape 3, dequantisation and SwiGLU -- checked
# bit for bit. The workload (data/params.hjson, data/datagen.py):
#
#   MP_SET     unit (default)  two slots of 16 and 8 rows, eight blocks each: the unit test
#              full            a cluster's share of the layer: four slots of 24, 16, 16, 8 rows,
#                              44 blocks each (60 chunks)
#   MP_ROWS    the slots' rows, e.g. 24,16,16,8 (overrides the set's)
#   MP_BLOCKS  a slot's column blocks (overrides the set's)
#   MP_BOFFS   B's offsets into its slab, e.g. b=64,paired=0 (overrides params.hjson's)
#
# The levers of the MoE prefill tracker's snax level (build knobs; the ELF rebuilds when one
# changes):
#
#   MP_LAYOUT  parity (default; HeMAiA today) | paired | b           the INT4 weights' layout
#   MP_ALAYOUT row (default; the gathered X rows) | blocked | ilv       A's layout
#   MP_LAUNCH  generic (default; HeMAiA's gemm_start) | lean | delta  how a GEMM is launched
#   MP_ASYNC   1 (default) | 0      the next GEMM is launched while the previous runs
#   MP_SCALE   slab (default) | sep    the scale tile in the weight slab, or in its own buffer
#   MP_SLABS   4 (default)             weight slabs
#   MP_PACE    0 (default)             cycles between chunk arrivals; 0 is the DRAM stream's own
#   MP_TIMELINE 1 (default)            print a line a chunk
#   MP_CENSUS  0 (default) | 1         (MP_ASYNC=0) the array's stall census of every GEMM, summed
#   MP_PERF    0 (default) | 1         the cluster's TCDM contention census over the window
#   MP_SCALE_SKEW 0 (default)          MP_SCALE=sep: bytes past a 256-byte boundary the scale slots start
#
#   make -C sw/apps/dsv2/snax-dsv2-moe-prefill MP_SET=full MP_LAUNCH=delta

APP     = snax-dsv2-moe-prefill

MP_SET ?= unit
MP_ROWS ?=
MP_BLOCKS ?=
MP_LAYOUT ?= parity
MP_LAUNCH ?= generic
MP_ASYNC ?= 1
MP_SCALE ?= slab
MP_SLABS ?= 4
MP_PACE ?= 0
MP_TIMELINE ?= 1
MP_CENSUS ?= 0
MP_PERF ?= 0
MP_SCALE_SKEW ?= 0
MP_ALAYOUT ?= row
MP_BOFFS ?=

MP_LAYOUT_N = $(if $(filter b,$(MP_LAYOUT)),1,$(if $(filter paired,$(MP_LAYOUT)),2,3))
MP_ALAYOUT_N = $(if $(filter blocked,$(MP_ALAYOUT)),1,$(if $(filter ilv,$(MP_ALAYOUT)),2,0))
MP_LAUNCH_N = $(if $(filter generic,$(MP_LAUNCH)),0,$(if $(filter lean,$(MP_LAUNCH)),1,2))
MP_SCALE_N = $(if $(filter sep,$(MP_SCALE)),1,0)

INCDIRS  = data
INCDIRS += ../include
INCDIRS += ../../include
INCDIRS += ../../../snax/versacore-to/include
INCDIRS += ../../../snax/simd/include
INCDIRS += ../../../snax/xdma/include

SNAX_LIBS  = ../../../snax/versacore-to/build/snax-versacore-to-lib.o
SNAX_LIBS += ../../../snax/simd/build/snax-simd-lib.o
SNAX_LIBS += ../../../snax/xdma/build/snax-xdma-lib.o
RISCV_LDFLAGS += $(SNAX_LIBS)

SRCS    = src/snax-dsv2-moe-prefill.c

# Operand blobs: disassemble the code only.
OBJDUMP_FLAGS = -d

include ../../common.mk
include ../dsv2.mk

RISCV_CFLAGS += -DMP_LAYOUT=$(MP_LAYOUT_N) -DMP_LAUNCH=$(MP_LAUNCH_N) -DMP_ASYNC=$(MP_ASYNC)
RISCV_CFLAGS += -DMP_SCALE=$(MP_SCALE_N) -DMP_SLABS=$(MP_SLABS) -DMP_PACE=$(MP_PACE)
RISCV_CFLAGS += -DMP_TIMELINE=$(MP_TIMELINE) -DMP_CENSUS=$(MP_CENSUS) -DMP_PERF=$(MP_PERF) -DMP_SCALE_SKEW=$(MP_SCALE_SKEW) -DMP_ALAYOUT=$(MP_ALAYOUT_N)

# The ELF rebuilds when a knob changes: it depends on a stamp named after the values.
MP_CFG_STAMP = $(BUILDDIR)/mp-cfg-$(MP_LAYOUT)-$(MP_LAUNCH)-$(MP_ASYNC)-$(MP_SCALE)-$(MP_SLABS)-$(MP_PACE)-$(MP_TIMELINE)-$(MP_CENSUS)-$(MP_PERF)-$(MP_SCALE_SKEW)-$(MP_ALAYOUT)

$(MP_CFG_STAMP): | $(BUILDDIR)
	rm -f $(BUILDDIR)/mp-cfg-*
	touch $@

$(DEP) $(ELF): $(MP_CFG_STAMP)

# The data's own stamp: data/Makefile regenerates data.h when the workload changes.
MP_DATA_ARGS = $(if $(MP_ROWS),--rows $(MP_ROWS)) $(if $(MP_BLOCKS),--blocks $(MP_BLOCKS)) $(if $(MP_BOFFS),--boff $(MP_BOFFS))
MP_SET_STAMP = $(BUILDDIR)/mp-set-$(MP_SET)-$(MP_ROWS)-$(MP_BLOCKS)-$(MP_BOFFS)

$(MP_SET_STAMP): | $(BUILDDIR)
	rm -f $(BUILDDIR)/mp-set-*
	touch $@

include ./data/Makefile

# The operand blobs link in through an assembly file of .incbin directives.
SRCS += $(DATA_S)

# data.h must exist before the dependency scan.
$(DEP): $(DATA_H)
$(ELF): $(DATA_S)
