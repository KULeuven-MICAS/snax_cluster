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


cd /repo/target/snitch_cluster

pkill verible-v
pkill verible

make clean
make CFG_OVERRIDE=cfg/snax-streamer-gemm-conv.hjson bin/snitch_cluster.vlt
make CFG_OVERRIDE=cfg/snax-streamer-gemm-conv.hjson /repo/target/snitch_cluster/generated/syn_flist.tcl

#check the generation


echo " "
echo " "
echo " "
echo " "
echo "--- Check the files in the local server: ---"
echo " "
ls -la /repo/target/snitch_cluster/generated/syn_flist.tcl
ls -la generated/*
echo "---"
echo " "

exit

#bender script vsim -t snitch_cluster -t snax_gemm -t test -t gate -t simulation  >vsim_flist.tcl






