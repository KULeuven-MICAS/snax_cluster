# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := ${MKFILE_DIR}../../..
CVA6_SDK    ?= ${ROOT}/../cva6-sdk
DEBUG       ?= 1
EXT_JTAG    ?= 1

VIVADO ?= vivado

BENDER_TARGETS = $(shell cat $(MKFILE_DIR)../../rtl/src/bender_targets.tmp)
BENDER_TARGETS += -t hemaia
TARGET_PLATFORM?= vcu128

default: all
all: hemaia_system_vcu128

include $(ROOT)/util/Makefrag

../hemaia_chip/hemaia_chip:
	${MAKE} -C ../hemaia_chip hemaia_chip

.PHONY: hemaia_system_vcu128

hemaia_system: ../hemaia_chip/hemaia_chip define_defines_includes_no_simset.tcl
	${VIVADO} -mode batch -source hemaia_system_${TARGET_PLATFORM}.tcl -tclargs $(DEBUG) $(EXT_JTAG)

define_defines_includes_no_simset.tcl:
	${BENDER} script vivado $(BENDER_TARGETS) --only-defines --only-includes --no-simset > $@

clean:
	rm -rf .Xil hemaia_system *.jou *.log *.str define_defines_includes_no_simset.tcl

.PHONY: program flash clean
