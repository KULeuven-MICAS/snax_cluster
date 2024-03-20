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

TARGET_ROOT=$HOME
PROJECT=snitch_cluster
CFG=default

# remove the flist and generated
rm -rf snps_flist.tcl
rm -rf .bender
rm -rf target/$PROJECT/generated/*

# bender checkout and bender flist
mkdir -p target/$PROJECT/generated
bender checkout && bender script synopsys -t synthesis -t $PROJECT > snps_flist.tcl
python3 util/clustergen.py -c target/$PROJECT/cfg/$CFG.hjson -o target/$PROJECT/generated --wrapper --mem

#check the generation
echo " "
echo "--- Check the files in the local server: ---"
echo " "
ls -la snps_flist.tcl
ls -la target/$PROJECT/generated/*
echo "---"
echo " "

echo "--- Enter with your password my Lord: ---"
echo " "

#transfer

ssh $USER@cygni-gw "mkdir -p $TARGET_ROOT/$PROJECT" && scp -r .bender/ hw/ target/$PROJECT/generated/ snps_flist.tcl $USER@cygni-gw:$TARGET_ROOT/$PROJECT && clear && echo " " && echo "Files Benderized, Generated and Transfered -- Check the Synthesis Server" && echo " " 

