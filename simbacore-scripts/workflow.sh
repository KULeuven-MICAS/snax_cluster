#!/bin/bash
# SimbaCore Workflow Script

set -e

# Step 0: initialize submodules
git submodule update --init --recursive 

#####################
# Step 1: Prepare RTL
#####################

# [bash] Generate SimbaCore.sv in chisel-ssm
cd ../chisel-ssm && sbt "runMain simbacore.SimbaCoreEmitter" && cd ../snax_cluster

# Parse widths from SimbaCore.sv and update shell_wrapper accordingly
python ./simbacore-scripts/update_simbacore_params.py

# Make sure the Streamer widths in config.hjson are still correct
# (Manual verification required)


###################################
# Step 2: Generate RTL and wrappers
###################################

# Start the container (denoted as [snax] in the script):
podman run -it -v `pwd`:`pwd` -w `pwd` ghcr.io/kuleuven-micas/snax:main
# [...|target] denotes running from this directory
cd target/snitch_cluster

# [snax|target/snitch_cluster] Generate RTL and wrappers 
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson rtl-gen


########################
# Step 3: Run QuestaSim
########################

# Make sure the config in target/snitch_cluster/sw/apps/snax-simbacore/data/params.hjson is valid
# (Manual verification required)

# [bash|root] Generate raw test data via scala data generator
cd ../chisel-ssm && sbt "test:runMain simbacore.DataGenerator seqLen=64 dModel=36 dtRank=24" && cd ../snax_cluster
cd ../chisel-ssm && sbt "test:runMain simbacore.DataGenerator seqLen=128 dModel=240 dtRank=24" && cd ../snax_cluster

# When adding extra external sv sources, add to bender.yaml and run `bender update`
# Then, remove the vsim work folder and run this
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson vsim_preparation

# [snax|target] Build the software
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson sw -j 

# [bash|target] Create QuestaSim binary and run programs
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson bin/snitch_cluster.vsim 
bin/snitch_cluster.vsim sw/apps/nop/build/nop.elf | tee vsim.log # No-op test program
bin/snitch_cluster.vsim sw/apps/snax-simbacore/build/snax-simbacore-main.elf | tee vsim.log
bin/snitch_cluster.vsim.gui sw/apps/snax-simbacore/build/snax-simbacore-main.elf # GUI (with VNC)

# [snax|target] Make traces (from .dasm to .txt)
make traces

###########################
# Step 4 Elaborate design
###########################

# [bash|root] Generate flist from bender
bender script synopsys -t synthesis -t snax_simbacore -t snax_simbacore_cluster  > simbacore-work/flist.tcl

# [bash|root] Run Design Compiler (e.g., check with elaborate)
cd simbacore-work && dcnxt_shell -64bit -f dc.tcl > dc.log


###############
# Documentation
###############

# Current flaws in this flow:
# 1) Raw test data generation via scala generator is part of the sw makefile, but cannot be executed in the snax podman shell

