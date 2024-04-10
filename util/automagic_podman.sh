# Copyright 2024 KU Leuven, Belgium
#                INESC-ID, Portugal
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Guilherme Paim <gppaim@ieee.org>
# Abdelrahman Ayman Fathallah <abdelrahman.ayman@esat.kuleuven.be>

#######################
# Automagic Bash      #
#######################

#[[TODO]] - turn it a makefile

# remove the flist and generated
rm -rf syn_flist.tcl vsim_flist.tcl
rm -rf .bender
rm -rf target/$PROJECT/generated/*

# bender checkout and bender flist
mkdir -p target/$PROJECT/generated
bender checkout

# Future Chisel Architectures
# SIMD=`bender path snax-streamer-simd-dev`
# RESH=`bender path snax-data-reshuffler-dev`
# cd $SIMD && make `pwd`/rtl/streamer-simd/streamer_simd_wrapper.sv
# cd $RESH && make `pwd`/tests/tb/tb_stream_dev_reshuffler.sv

GEMM=`bender path snax-streamer-gemm-dev`
cd $GEMM && make `pwd`/rtl/streamer-gemm/streamer_gemm_wrapper.sv

cd /repo
bender script synopsys -t synthesis -t $PROJECT -t $CFG > syn_flist.tcl
bender script vsim -t snitch_cluster -t snax_gemm -t test -t gate -t simulation  >vsim_flist.tcl

python3 util/clustergen.py -c target/$PROJECT/cfg/$CFG.hjson -o target/$PROJECT/generated --wrapper --mem

#check the generation
echo " "
echo "--- Check the files in the local server: ---"
echo " "
ls -la snps_flist.tcl
ls -la target/$PROJECT/generated/*
echo "---"
echo " "


