#!/bin/bash
# SimbaCore Workflow Script

set -e

#####################
# Step 1: Prepare RTL
#####################

# Generate SimbaCore.sv in `chisel-ssm`
# (Must be completed manually)

# Parse widths from SimbaCore.sv and update shell_wrapper accordingly
python ./simbacore-work/update_simbacore_params.py

# Make sure the Streamer widths in config.hjson are still correct
# (Manual verification required)


#######################################################
# Step 2: Generate RTL and wrappers in SNAX environment
#######################################################

# Start the container:
podman run -it -v `pwd`:`pwd` -w `pwd` ghcr.io/kuleuven-micas/snax:main
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson rtl-gen


########################
# Step 3: Run QuestaSim
########################

# [bash] Generate raw test data externally
cd ../chisel-ssm && sbt "test:runMain snax.MatmulDataGenerator" && cd ../snax_cluster

# [snax] Build the software
make  CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson vsim_preparation
# make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson sw -j
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson sw -j 

# [bash] Create QuestaSim binary and run programs
cd target/snitch_cluster
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson bin/snitch_cluster.vsim 

bin/snitch_cluster.vsim sw/apps/nop/build/nop.elf > vsim.log
bin/snitch_cluster.vsim sw/apps/snax-simbacore/build/snax-simbacore-main.elf > vsim.log
# Alternatively, open the GUI with VNC
bin/snitch_cluster.vsim.gui sw/apps/snax-simbacore/build/snax-simbacore-main.elf

# [snax/pixi] make traces
cd target/snitch_cluster
make traces

###########################
# Step 4 Elaborate design
###########################

# Generate flist from bender
bender script synopsys -t synthesis -t snax_simbacore -t snax_simbacore_cluster > simbacore-work/flist.tcl

#  Run Design Compiler (e.g., check with elaborate)
cd simbacore-work && dcnxt_shell -64bit -f dc.tcl > dc.log
