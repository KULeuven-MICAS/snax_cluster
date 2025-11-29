#!/bin/bash
# SimbaCore Workflow Script

set -e


#####################
# Step 0: Prepare RTL
#####################

# This is entirely managed by Robin. 

# Generate SimbaCore.sv in chisel-ssm and push to git
# When architecture changes: parse widths from SimbaCore.sv and update shell_wrapper accordingly
python ./simbacore-scripts/update_simbacore_params.py
# Manually check the Streamer widths in config.hjson are still correct 

###################################
# Step 1: Generate RTL and wrappers
###################################

# [bash] Make sure bender is up-to-date
bender update

# Start the container (denoted as [snax] in the script):
podman run -it -v `pwd`:`pwd` -w `pwd` ghcr.io/kuleuven-micas/snax:main
# [...|target] denotes running from this directory
cd target/snitch_cluster

# [snax|target] Generate RTL and wrappers 
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson rtl-gen


########################
# Step 2: Run QuestaSim
########################

# When adding extra external sv sources, add to bender.yaml and run `bender update`
# Then, remove the vsim work folder and run this
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson vsim_preparation

# [snax|target] Build the software
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson sw -j 

# [bash|target] Create QuestaSim binary and run programs
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson bin/snitch_cluster.vsim 
# [bash|target] Run no-op test program just to see if snitch core is alive
bin/snitch_cluster.vsim sw/apps/nop/build/nop.elf | tee vsim.log 
# [bash|target] Run SimbaCore test program
bin/snitch_cluster.vsim sw/apps/snax-simbacore/build/snax-simbacore-main.elf | tee vsim.log
# [bash|target] Run SimbaCore test program in GUI (with VNC)
bin/snitch_cluster.vsim.gui sw/apps/snax-simbacore/build/snax-simbacore-main.elf 

# [snax|target] Make traces (from .dasm to .txt)
make traces

###########################
# Step 3 Elaborate design
###########################

# [bash|root] Generate flist from bender
bender script synopsys -t synthesis -t snax_simbacore -t snax_simbacore_cluster  > simbacore-work/flist.tcl

# [bash|root] Run Design Compiler (e.g., check with elaborate)
cd simbacore-work && dcnxt_shell -64bit -f dc.tcl > dc.log


