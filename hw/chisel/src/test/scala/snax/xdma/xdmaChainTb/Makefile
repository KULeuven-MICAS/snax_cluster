# Copyright 2026 KU Leuven.
# SPDX-License-Identifier: SHL-0.51
#
# Standalone QuestaSim flow for the N-endpoint ChainGather testbenches.
#
# Deliberately NOT the target/snitch_cluster flow: that builds the whole cluster with cores and
# needs a binary to run. Here the DUT is `snax_xdma_cluster_xdma_wrapper` instantiated N times
# behind two AXI crossbars, and the stimulus is the CSR port.
#
#   make sim-p3          # 3 endpoints, one middle hop -- the smallest reproducer
#   make sim-p2          # the control (no middle hop)
#   make sim-p4          # two middle hops, the HeMAiA failing case
#   make sim-p16         # 16 endpoints, one width
#   make sim-sweep       # THE SWEEP: P in {2,4,8,16} x {lin,mom}, with a latency table
#   make sim-tree        # tree vs chain, and whether stage 2 needs a barrier
#   make sim-all         # p2, p3, p4 in order
#   make gui-p3          # same, with the GUI
#
# `compile.tcl` depends on the actual sources, so an edit cannot leave a stale compiled design
# in place -- which is the trap that has produced false results on the HeMAiA side twice.

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := $(abspath $(MKFILE_DIR)/../../../../../../../..)

# The MICAS site rc sets MGC_QUESTA_HOME and, crucially, MGLS_LICENSE_FILE. Sourcing it here
# rather than requiring the caller to have done so keeps `make sim-p3` a one-liner; without the
# licence variable vsim exits with "Invalid license environment" before compiling anything.
QUESTA_RC   ?= /esat/micas-data/data/design/scripts/questasim_2022.4.rc
QUESTA_HOME ?= $(shell . $(QUESTA_RC) >/dev/null 2>&1 && echo $$MGC_QUESTA_HOME)
QUESTA_ENV  := . $(QUESTA_RC) >/dev/null 2>&1;
VLOG        ?= $(QUESTA_HOME)/bin/vlog
VOPT        ?= $(QUESTA_HOME)/bin/vopt
VSIM        ?= $(QUESTA_HOME)/bin/vsim
VLIB        ?= $(QUESTA_HOME)/bin/vlib
BENDER      ?= bender

BUILDDIR    := $(MKFILE_DIR)work-vsim
LIB         := $(BUILDDIR)

BENDER_TARGETS := -t rtl -t simulation -t QUESTA_SIM \
                  -t snax_xdma_cluster_xdma -t xdma_chain_tb

VLOG_FLAGS  := -64 -svinputport=compat -override_timescale 1ns/1ps \
               -suppress 2583 -suppress 13314 -suppress 2286 -suppress 8386
VOPT_FLAGS  := -64 +acc
VSIM_FLAGS  := -64 -c

TB_SRCS := $(wildcard $(MKFILE_DIR)*.sv)
DUT_SRCS := \
  $(ROOT)/target/snitch_cluster/generated/snax_xdma_cluster_xdma/snax_xdma_cluster_xdma.sv \
  $(ROOT)/target/snitch_cluster/generated/snax_xdma_cluster_xdma/snax_xdma_cluster_xdma_wrapper.sv

.PHONY: all clean sim-all
all: compile

$(BUILDDIR):
	mkdir -p $(BUILDDIR)
	cd $(ROOT) && $(QUESTA_ENV) $(VLIB) $(BUILDDIR)

$(BUILDDIR)/compile.tcl: $(ROOT)/Bender.yml $(ROOT)/Bender.lock $(TB_SRCS) $(DUT_SRCS) | $(BUILDDIR)
	cd $(ROOT) && $(BENDER) script vsim $(BENDER_TARGETS) \
	  --vlog-arg="$(VLOG_FLAGS) -work $(LIB)" > $@
	echo "return 0" >> $@

.PHONY: compile
compile: $(BUILDDIR)/compile.tcl
	cd $(ROOT) && $(QUESTA_ENV) $(VSIM) -64 -c -do "source $(BUILDDIR)/compile.tcl; quit" \
	  | tee $(BUILDDIR)/compile.log
	@! grep -q "^\*\* Error" $(BUILDDIR)/compile.log

# $(1) = testbench suffix (p2, p3, ...)
define RUN_TB
.PHONY: sim-$(1)
sim-$(1): compile
	cd $(ROOT) && $(QUESTA_ENV) $(VOPT) $(VOPT_FLAGS) -work $(LIB) tb_xdma_chaingather_$(1) \
	  -o tb_xdma_chaingather_$(1)_opt | tee $(BUILDDIR)/vopt_$(1).log
	cd $(ROOT) && $(QUESTA_ENV) $(VSIM) $(VSIM_FLAGS) -work $(LIB) tb_xdma_chaingather_$(1)_opt \
	  -do "run -all; quit" | tee $(BUILDDIR)/sim_$(1).log
	@grep -q "RESULT: PASS" $(BUILDDIR)/sim_$(1).log || \
	  (echo ""; echo "  *** $(1) FAILED -- see $(BUILDDIR)/sim_$(1).log"; exit 1)

.PHONY: gui-$(1)
gui-$(1): compile
	cd $(ROOT) && $(QUESTA_ENV) $(VOPT) $(VOPT_FLAGS) -work $(LIB) tb_xdma_chaingather_$(1) \
	  -o tb_xdma_chaingather_$(1)_opt
	cd $(ROOT) && $(QUESTA_ENV) $(VSIM) -64 -work $(LIB) tb_xdma_chaingather_$(1)_opt
endef

$(foreach tb,p2 p3 p4 p16 sweep tree tree4 role,$(eval $(call RUN_TB,$(tb))))

sim-all: sim-p2 sim-p3 sim-p4

clean:
	rm -rf $(BUILDDIR)
