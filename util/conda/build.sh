#!/bin/bash
# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
 

clean_and_fetch_git() {
    cd ..
    rm -rf snax_cluster
    git clone --recursive https://github.com/kuleuven-micas/snax_cluster
    cd snax_cluster
    git checkout 86e2a519805b255377ed379d08686248379a24d5
    return 0
}

# Function to build multiple snitch_cluster.vlt instances
build_snax_verilator() {
    local config_file="$1"
    local target_dir="$2"
    
    if [ -z "$config_file" ]; then
        echo "Error: Configuration file not provided"
        echo "Usage: build_snitch_cluster <config_file>"
        return 1
    fi

    if [ ! -f "target/snitch_cluster/$config_file" ]; then
        echo "Error: Configuration file '$config_file' not found"
        return 1
    fi 
    
    # Generate RTL
    make -C target/snitch_cluster rtl-gen \
        CFG_OVERRIDE="$config_file"
    
    # Build software
    make DEBUG=ON sw -j$(nproc) \
        -C target/snitch_cluster \
        SELECT_TOOLCHAIN=llvm-generic \
        SELECT_RUNTIME=rtl-generic \
        CFG_OVERRIDE="$config_file"
    
    # Generate Verilator files
    make -C target/snitch_cluster bin/snitch_cluster.vlt \
        CFG_OVERRIDE="$config_file" -j$(nproc)
    
    echo "Build completed for '$config_file' successfully"

    cp --parents target/snitch_cluster/bin/snitch_cluster.vlt ${target_dir}/bin/snitch_cluster.vlt && \
    cp --parents sw/snRuntime ${target_dir}/sw/snRuntime && \
    cp --parents target/snitch_cluster/sw/runtime/rtl ${target_dir}/target/snitch_cluster/sw/runtime/rtl && \
    cp --parents target/snitch_cluster/sw/runtime/rtl-generic ${target_dir}/target/snitch_cluster/sw/runtime/rtl-generic && \
    cp --parents target/snitch_cluster/sw/runtime/common ${target_dir}/target/snitch_cluster/sw/runtime/common && \
    cp --parents target/snitch_cluster/sw/snax/ ${target_dir}/target/snitch_cluster/sw/snax && \
    cp --parents sw/math/ ${target_dir}/sw/math/ && \
    cp --parents sw/deps/riscv-opcodes ${target_dir}/sw/deps/riscv-opcodes && \
    cp --parents sw/deps/printf ${target_dir}/sw/deps/printf

    echo "Successfully installed in '$target_dir'"

    make -C target/snitch_cluster clean \
        CFG_OVERRIDE="$config_file" -j$(nproc)

    return 0
}

export VLT_ROOT="${BUILD_PREFIX}/share/verilator"
PIP_NO_INDEX= pip install hjson #Unset PIP_NO_INDEX to allow pypi installation
#clean_and_fetch_git
build_snax_verilator cfg/snax_mac_cluster.hjson ${PREFIX}/snax-utils/snax-mac
#clean_and_fetch_git
build_snax_verilator cfg/snax_alu_cluster.hjson ${PREFIX}/snax-utils/snax-alu
#clean_and_fetch_git
build_snax_verilator cfg/snax_streamer_gemm_add_c_cluster.hjson ${PREFIX}/snax-utils/snax-streamer-gemm-add-c
#clean_and_fetch_git
build_snax_verilator cfg/snax_KUL_cluster.hjson ${PREFIX}/snax-utils/snax-kul-cluster-mixed-narrow-wide
cp /bin/spike-dasm ${PREFIX}/bin/spike-dasm
