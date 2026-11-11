# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

LOGS_DIR ?= logs
TB_DIR   ?= $(SNITCH_ROOT)/target/common/test
UTIL_DIR ?= $(SNITCH_ROOT)/util

# SEPP packages
QUESTA_SEPP    ?=
VCS_SEPP       ?=
VERILATOR_SEPP ?=

# External executables
BENDER       ?= bender
DASM         ?= spike-dasm
VLT          ?= $(VERILATOR_SEPP) verilator
VCS          ?= $(VCS_SEPP) vcs
VERIBLE_FMT  ?= verible-verilog-format
CLANG_FORMAT ?= clang-format
VSIM         ?= $(QUESTA_SEPP) vsim
VOPT         ?= $(QUESTA_SEPP) vopt
VLOG         ?= $(QUESTA_SEPP) vlog
VLIB         ?= $(QUESTA_SEPP) vlib

# Internal executables
GENTRACE_PY      ?= $(UTIL_DIR)/trace/gen_trace.py
ANNOTATE_PY      ?= $(UTIL_DIR)/trace/annotate.py
EVENTS_PY        ?= $(UTIL_DIR)/trace/events.py
PERF_CSV_PY      ?= $(UTIL_DIR)/trace/perf_csv.py
LAYOUT_EVENTS_PY ?= $(UTIL_DIR)/trace/layout_events.py
EVENTVIS_PY      ?= $(UTIL_DIR)/trace/eventvis.py

VERILATOR_ROOT ?= $(dir $(shell $(VERILATOR_SEPP) which verilator))..
VLT_ROOT       ?= ${VERILATOR_ROOT}
VERILATOR_VERSION ?= $(shell $(VLT) --version | grep -oP 'Verilator \K\d+')

MATCH_END := '/+incdir+/ s/$$/\/*\/*/'
MATCH_BGN := 's/+incdir+//g'
SED_SRCS  := sed -e ${MATCH_END} -e ${MATCH_BGN}

VSIM_BENDER   += -t test -t rtl -t simulation -t vsim
VSIM_SOURCES   = $(shell ${BENDER} script flist ${VSIM_BENDER} | ${SED_SRCS})
VSIM_BUILDDIR ?= work-vsim
VOPT_FLAGS     = +acc

# VCS_BUILDDIR should to be the same as the `DEFAULT : ./work-vcs`
# in target/snitch_cluster/synopsys_sim.setup
VCS_BENDER   += -t test -t rtl -t simulation -t vcs
VCS_SOURCES   = $(shell ${BENDER} script flist ${VCS_BENDER} | ${SED_SRCS})
VCS_BUILDDIR := work-vcs

# For synthesis with DC compiler
SYN_FLIST ?= syn_flist.tcl
# rtl, as vsim and vcs: iDMA's sources (idma_inst64_top, its midends and backends) are under bender target rtl
SYN_BENDER += -t synthesis -t rtl

# The cluster's memories (SRAM_IMPL, SRAM_MODELS, MEM_TYPE: target/snitch_cluster/Makefile's
# header). With SRAM_IMPL, vsim and synthesis leave out tech_cells_generic's tc_sram.sv and
# tc_sram_impl.sv; the Makefile reads SRAM_IMPL's file instead.
SRAM_IMPL_16NM = /users/micas/shares/project_snax/tapeout_modules/snax_cluster/tc_sram_impl.sv
ifeq ($(strip $(MEM_TYPE)),exclude_tcsram)
  ifneq ($(strip $(SRAM_IMPL)),)
    $(error MEM_TYPE=exclude_tcsram is SRAM_IMPL=$(SRAM_IMPL_16NM): give one of MEM_TYPE and SRAM_IMPL, not both)
  endif
  override SRAM_IMPL := $(SRAM_IMPL_16NM)
else ifneq ($(strip $(MEM_TYPE)),)
  $(error MEM_TYPE=$(MEM_TYPE) is not supported: give the cluster's tc_sram_impl as SRAM_IMPL=<file> (MEM_TYPE=exclude_tcsram is SRAM_IMPL=$(SRAM_IMPL_16NM)))
endif
ifneq ($(strip $(SRAM_IMPL)),)
  override SRAM_IMPL := $(abspath $(strip $(SRAM_IMPL)))
  ifeq ($(wildcard $(SRAM_IMPL)),)
    $(error SRAM_IMPL=$(SRAM_IMPL): no such file)
  endif
  VSIM_BENDER += -t tech_cells_generic_exclude_tc_sram
  SYN_BENDER  += -t tech_cells_generic_exclude_tc_sram
endif
ifneq ($(strip $(SRAM_MODELS)),)
  ifeq ($(strip $(SRAM_IMPL)),)
    $(error SRAM_MODELS=$(SRAM_MODELS) needs SRAM_IMPL: the models are of the macros SRAM_IMPL's tc_sram_impl instantiates)
  endif
  override SRAM_MODELS := $(abspath $(strip $(SRAM_MODELS)))
  ifeq ($(wildcard $(SRAM_MODELS)),)
    $(error SRAM_MODELS=$(SRAM_MODELS): no such file)
  endif
endif

# SIM_TYPE=gate_level_sim: vsim of the cluster's netlist (target/snitch_cluster/Makefile's header).
# The files are read only by vlog, outside the image, so they are not checked here.
GLS =
ifeq ($(strip $(SIM_TYPE)),gate_level_sim)
  ifeq ($(strip $(GLS_NETLIST)),)
    $(error SIM_TYPE=gate_level_sim needs GLS_NETLIST=<the netlist .v>)
  endif
  ifeq ($(strip $(GLS_CELLS)),)
    $(error SIM_TYPE=gate_level_sim needs GLS_CELLS=<the cell library's Verilog models>)
  endif
  ifeq ($(strip $(SRAM_MODELS)),)
    $(error SIM_TYPE=gate_level_sim needs SRAM_IMPL and SRAM_MODELS: the netlist instantiates the memory macros)
  endif
  GLS = yes
  override GLS_NETLIST := $(abspath $(strip $(GLS_NETLIST)))
  override GLS_CELLS   := $(abspath $(strip $(GLS_CELLS)))
  # the netlist's library first, so snax_forge_cluster_wrapper and all below it are the netlist's;
  # registers and memories start at 0 (the netlist has no initial blocks)
  VOPT_FLAGS = +acc=n +initreg+0 -Lf $(VSIM_BUILDDIR)/gls
else ifneq ($(strip $(GLS_NETLIST)$(GLS_CELLS)),)
  $(error GLS_NETLIST and GLS_CELLS need SIM_TYPE=gate_level_sim)
endif
SYN_SOURCES = $(shell ${BENDER} script synopsys ${SYN_BENDER})
SYN_BUILDDIR := work-syn

# fesvr is being installed here
FESVR         ?= ${MKFILE_DIR}work
FESVR_VERSION ?= 98d2c29e431f3b14feefbda48c5f70c2f451acf2

VLT_BENDER   += -t rtl
VLT_SOURCES   = $(shell ${BENDER} script flist ${VLT_BENDER} | ${SED_SRCS})
VLT_BUILDDIR := work-vlt
VLT_FESVR     = $(VLT_BUILDDIR)/riscv-isa-sim
ifeq ($(VERILATOR_VERSION), 5)
	VLT_FLAGS += --timing
endif
VLT_FLAGS    += -Wno-BLKANDNBLK
VLT_FLAGS    += -Wno-LITENDIAN
VLT_FLAGS    += -Wno-CASEINCOMPLETE
VLT_FLAGS    += -Wno-CMPCONST
VLT_FLAGS    += -Wno-WIDTH
VLT_FLAGS    += -Wno-WIDTHCONCAT
VLT_FLAGS    += -Wno-UNSIGNED
VLT_FLAGS    += -Wno-UNOPTFLAT
VLT_FLAGS    += -Wno-fatal
VLT_FLAGS    += +define+SYNTHESIS
VLT_FLAGS    += --unroll-count 1024
ifeq ($(VERILATOR_VERSION), 5)
	VLT_CXXSTD_FLAGS += -std=c++20 -pthread -latomic
else 
	VLT_CXXSTD_FLAGS += -std=c++17 -pthread
endif
VLT_CFLAGS   += ${VLT_CXXSTD_FLAGS} -I ${VLT_BUILDDIR} -I $(VLT_ROOT)/include -I $(VLT_ROOT)/include/vltstd -I $(VLT_FESVR)/include -I $(TB_DIR) -I ${MKFILE_DIR}/test

ANNOTATE_FLAGS ?= -q --keep-time

# We need a recent LLVM installation (>11) to compile Verilator.
# We also need to link the binaries with LLVM's libc++.
# Define CLANG_PATH to be the path of your Clang installation.

ifneq (${CLANG_PATH},)
    CLANG_CC       := $(CLANG_PATH)/bin/clang
    CLANG_CXX      := $(CLANG_PATH)/bin/clang++
    CLANG_CXXFLAGS := -nostdinc++ -isystem $(CLANG_PATH)/include/c++/v1
    CLANG_LDFLAGS  := -nostdlib++ -fuse-ld=lld -L ${CLANG_PATH}/lib -Wl,-rpath,${CLANG_PATH}/lib -lc++
else
    CLANG_CC       ?= clang
    CLANG_CXX      ?= clang++
    CLANG_CXXFLAGS := ""
    CLANG_LDFLAGS  := ""
endif

# If requested, build verilator with LLVM and add llvm c/ld flags
ifeq ($(VLT_USE_LLVM),ON)
    CC         = $(CLANG_CC)
    CXX        = $(CLANG_CXX)
    CFLAGS     = $(CLANG_CXXFLAGS)
    CXXFLAGS   = $(CLANG_CXXFLAGS)
    LDFLAGS    = $(CLANG_LDFLAGS)
    VLT_FLAGS += --compiler clang
    VLT_FLAGS += -CFLAGS "${CLANG_CXXFLAGS}"
    VLT_FLAGS += -LDFLAGS "${CLANG_LDFLAGS}"
endif

VLOGAN_FLAGS := -assert svaext
VLOGAN_FLAGS += -assert disable_cover
VLOGAN_FLAGS += -full64
VLOGAN_FLAGS += -kdb
VHDLAN_FLAGS := -full64
VHDLAN_FLAGS += -kdb

# default on target `all`
all:

#################
# Prerequisites #
#################
# Eventually it could be an option to package this statically using musl libc.
work/${FESVR_VERSION}_unzip:
	mkdir -p $(dir $@)
	wget -O $(dir $@)/${FESVR_VERSION} https://github.com/riscv/riscv-isa-sim/tarball/${FESVR_VERSION}
	tar xfm $(dir $@)${FESVR_VERSION} --strip-components=1 -C $(dir $@)
	touch $@

work/lib/libfesvr.a: work/${FESVR_VERSION}_unzip
	cd $(dir $<)/ && ./configure --prefix `pwd` --with-boost=$(CONDA_PREFIX) --with-boost-libdir=$(CONDA_PREFIX)/lib
	$(MAKE) -C $(dir $<) install-config-hdrs install-hdrs libfesvr.a
	mkdir -p $(dir $@)
	cp $(dir $<)libfesvr.a $@
	strip --strip-debug $@

# Build fesvr seperately for verilator since this might use different compilers
# and libraries than modelsim/vcs and
$(VLT_FESVR)/${FESVR_VERSION}_unzip:
	mkdir -p $(dir $@)
	wget -O $(dir $@)/${FESVR_VERSION} https://github.com/riscv/riscv-isa-sim/tarball/${FESVR_VERSION}
	tar xfm $(dir $@)${FESVR_VERSION} --strip-components=1 -C $(dir $@)
	touch $@

$(VLT_BUILDDIR)/lib/libfesvr.a: $(VLT_FESVR)/${FESVR_VERSION}_unzip
	cd $(dir $<)/ && ./configure --prefix `pwd` --with-boost=$(CONDA_PREFIX)  \
        CC=${CC} CXX=${CXX} CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" LDFLAGS="${LDFLAGS}"
	$(MAKE) -C $(dir $<) install-config-hdrs install-hdrs libfesvr.a
	mkdir -p $(dir $@)
	cp $(dir $<)libfesvr.a $@

#############
# Verilator #
#############
# Takes the top module name as an argument.
# With VLT_HIER_CFG (hierarchical Verilation, see target/snitch_cluster/Makefile):
# - Verilator 5.034 passes the hier-block runs the source files from -f, but not
#   its +incdir+/+define+ lines, so these also go on the command line, read from
#   the bender file list once it is written (Verilator treats them as global).
# - Verilator runs without --build: its generated $(1)_hier.mk then only does
#   hier_verilation, which runs each hier block once. With --build, hier_build
#   names a block's wrapper .sv and .mk as two targets of one recipe, not a
#   grouped target, so make -j ran that block's Verilator twice at once in the
#   same folder. hier_build runs after, here, with all Verilator output up to
#   date, and only builds the C++.
define VERILATE
	mkdir -p $(dir $@)
	$(BENDER) script verilator ${VLT_BENDER} > $(dir $@)files
	+$(VLT) \
		--Mdir $(dir $@) -f $(dir $@)files $(if $(SNAX_FORGE_RTL),-f $(SNAX_FORGE_F)) $(VLT_FLAGS) \
		$(if $(VLT_HIER_CFG),$$(grep -E '^\+(incdir|define)\+' $(dir $@)files | awk '!seen[$$0]++')) \
		-j $(shell nproc) --cc $(if $(VLT_HIER_CFG),,--build) --top-module $(1)
	+$(if $(VLT_HIER_CFG),$(MAKE) -C $(dir $@) -f V$(1)_hier.mk hier_build \
		$(if $(findstring jobserver,$(MAKEFLAGS)),,-j $(shell nproc)) $(VLT_MAKE_VARS))
	touch $@
endef

############
# Modelsim #
############

$(VSIM_BUILDDIR):
	mkdir -p $@

# Expects vlog/vcom script in $< (e.g. as output by bender)
# Expects the top module name in $1
# Produces a binary used to run the simulation at the path specified by $@
define QUESTASIM
	${VSIM} -c -do "source $<; quit" | tee $(dir $<)vlog.log
	@! grep -P "Errors: [1-9]*," $(dir $<)vlog.log
	$(VOPT) $(VOPT_FLAGS) -work $(VSIM_BUILDDIR) $1 -o $(1)_opt | tee $(dir $<)vopt.log
	@! grep -P "Errors: [1-9]*," $(dir $<)vopt.log
	@mkdir -p $(dir $@)
	@echo "#!/bin/bash" > $@
	@echo 'binary=$$(realpath $$1)' >> $@
	@echo 'mkdir -p $(LOGS_DIR)' >> $@
	@echo 'echo $$binary > $(LOGS_DIR)/.rtlbinary' >> $@
	@echo '${VSIM} +permissive ${VSIM_FLAGS} $$3 -work ${MKFILE_DIR}/${VSIM_BUILDDIR} -c \
				-ldflags "-Wl,-rpath,${FESVR}/lib -L${FESVR}/lib -lfesvr -lutil" \
				$(1)_opt +permissive-off ++$$binary ++$$2' >> $@
	@chmod +x $@
	@echo "#!/bin/bash" > $@.gui
	@echo 'binary=$$(realpath $$1)' >> $@.gui
	@echo 'mkdir -p $(LOGS_DIR)' >> $@.gui
	@echo 'echo $$binary > $(LOGS_DIR)/.rtlbinary' >> $@.gui
	@echo '${VSIM} +permissive ${VSIM_FLAGS} -work ${MKFILE_DIR}/${VSIM_BUILDDIR} \
				-ldflags "-Wl,-rpath,${FESVR}/lib -L${FESVR}/lib -lfesvr -lutil" \
				$(1)_opt +permissive-off ++$$binary ++$$2' >> $@.gui
	@chmod +x $@.gui
endef

#######
# VCS #
#######
# snax_forge.f (a snax-forge bundle's RTL) is not read by VCS
$(VCS_BUILDDIR)/compile.sh:
	mkdir -p $(VCS_BUILDDIR)
	${BENDER} script vcs ${VCS_BENDER} --vlog-arg="${VLOGAN_FLAGS}" --vcom-arg="${VHDLAN_FLAGS}" > $@
	chmod +x $@
	$(VCS_SEPP) $@ > $(VCS_BUILDDIR)/compile.log

########
# Util #
########

# Common rule to generate C header with REGGEN
# $1: target name, $2: prerequisite (hjson description file)
define reggen_generate_header
	@echo "[REGGEN] Generating $1"
	@$(REGGEN) -D -o $1 $2
	@$(CLANG_FORMAT) -i $1
endef

# Arg 1: binary
# Arg 2: max size in bytes
define BINARY_SIZE_CHECK
  echo "Binary size: $$(stat -c %s $(1))B"
  @[ "$$(stat -c %s $(1))" -lt "$(2)" ] || (echo "Binary exceeds specified size of $(2)B"; exit 1)
endef

##########
# Traces #
##########

DASM_TRACES      = $(shell (ls $(LOGS_DIR)/trace_chip_??_hart_*.dasm 2>/dev/null))
TXT_TRACES       = $(shell (echo $(DASM_TRACES) | sed 's/\.dasm/\.txt/g'))
PERF_TRACES      = $(shell (echo $(DASM_TRACES) | sed 's/.dasm/_perf.json/g'))
ANNOTATED_TRACES = $(shell (echo $(DASM_TRACES) | sed 's/\.dasm/\.s/g'))
DIFF_TRACES      = $(shell (echo $(DASM_TRACES) | sed 's/\.dasm/\.diff/g'))

GENTRACE_OUTPUTS = $(TXT_TRACES) $(PERF_TRACES)
ANNOTATE_OUTPUTS = $(ANNOTATED_TRACES)
PERF_CSV         = $(LOGS_DIR)/perf.csv
EVENT_CSV        = $(LOGS_DIR)/event.csv
TRACE_CSV        = $(LOGS_DIR)/trace.csv
TRACE_JSON       = $(LOGS_DIR)/trace.json

.PHONY: traces annotate perf-csv event-csv layout
traces: $(GENTRACE_OUTPUTS)
annotate: $(ANNOTATE_OUTPUTS)
perf-csv: $(PERF_CSV)
event-csv: $(EVENT_CSV)
layout: $(TRACE_CSV) $(TRACE_JSON)

$(LOGS_DIR)/%.txt $(LOGS_DIR)/%_perf.json: $(LOGS_DIR)/%.dasm $(GENTRACE_PY)
	@CHIP=$(word 3,$(subst _, ,$*)) && \
	HART=$(word 5,$(subst _, ,$*)) && \
	echo "Processing Chip $$CHIP Hart $$HART" && \
	$(DASM) < $< | $(PYTHON) $(GENTRACE_PY) --permissive -d $(LOGS_DIR)/chip_$$CHIP\_hart_$$HART\_perf.json > $(LOGS_DIR)/trace_chip_$$CHIP\_hart_$$HART.txt
# Generate source-code interleaved traces for all harts. Reads the binary from
# the logs/.rtlbinary file that is written at start of simulation in the vsim script
BINARY ?= $(shell cat $(LOGS_DIR)/.rtlbinary)

$(LOGS_DIR)/%.s: $(LOGS_DIR)/%.txt $(ANNOTATE_PY)
	$(PYTHON) $(ANNOTATE_PY) $(ANNOTATE_FLAGS) -o $@ $(BINARY) $<
$(LOGS_DIR)/%.diff: $(LOGS_DIR)/%.txt $(ANNOTATE_PY)
	$(PYTHON) $(ANNOTATE_PY) $(ANNOTATE_FLAGS) -o $@ $(BINARY) $< -d

$(PERF_CSV): $(PERF_TRACES) $(PERF_CSV_PY)
	$(PYTHON) $(PERF_CSV_PY) -o $@ -i $(PERF_TRACES)

$(EVENT_CSV): $(PERF_TRACES) $(PERF_CSV_PY)
	$(PYTHON) $(PERF_CSV_PY) -o $@ -i $(PERF_TRACES) --filter tstart tend

$(TRACE_CSV): $(EVENT_CSV) $(LAYOUT_FILE) $(LAYOUT_EVENTS_PY)
	$(PYTHON) $(LAYOUT_EVENTS_PY) $(LAYOUT_EVENTS_FLAGS) $(EVENT_CSV) $(LAYOUT_FILE) -o $@

$(TRACE_JSON): $(TRACE_CSV) $(EVENTVIS_PY)
	$(PYTHON) $(EVENTVIS_PY) -o $@ $(TRACE_CSV)
