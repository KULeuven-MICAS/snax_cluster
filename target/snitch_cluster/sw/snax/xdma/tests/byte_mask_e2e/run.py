#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Run the C driver against complete 512-bit-bank XDMA RTL and SRAM in Verilator."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

TEST = Path(__file__).resolve().parent
REPO = TEST.parents[6]


def execute(command, cwd, timeout=900):
    print("+ " + " ".join(str(arg) for arg in command), flush=True)
    subprocess.run([str(arg) for arg in command], cwd=cwd, check=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO)
    parser.add_argument("--build-dir", type=Path, default=TEST / "build")
    args = parser.parse_args()
    source = args.source_root.resolve()
    build = args.build_dir.resolve()
    for tool in ("sbt", "verilator", "clang"):
        if not shutil.which(tool):
            parser.error(tool + " is missing; run this test inside snax-shell")
    rtl = build / "rtl"
    include = build / "include"
    objects = build / "obj"
    for directory in (rtl, include, objects):
        directory.mkdir(parents=True, exist_ok=True)
    # Keep the existing full datapath's two destination slots; only the first is used.
    cfg = {
        "cfg_io_width": 32, "max_multicast": 2, "max_dimension": 2,
        "max_mem_size_kiB": 4096,
        "reader_agu_temporal_dimension": 2, "writer_agu_temporal_dimension": 2,
        "reader_buffer": 2, "writer_buffer": 2,
        "reader_extensions": {}, "writer_extensions": {},
    }
    command = (
        "runMain snax.xdma.xdmaTop.XDMATopGen --clusterName byte_mask_e2e"
        " --xdmaCfg " + json.dumps(json.dumps(cfg, separators=(",", ":"))) +
        " --axiAddrWidth 48 --axiDataWidth 512 --tcdmDataWidth 512 --tcdmSize 128"
        " --hw-target-dir " + str(rtl) +
        " --sw-target-dir " + str(include / "snax-xdma-addr.h")
    )
    execute(["sbt", "-batch", command], source / "hw/chisel")
    driver = source / "target/snitch_cluster/sw/snax/xdma"
    # The driver and generated address header come from the source being tested.
    # Only the SNRT accessors are replaced with transactions on the real RTL bus.
    shutil.copyfile(driver / "include/snax-xdma-lib.h", include / "snax-xdma-lib.h")
    shutil.copyfile(TEST / "snrt.h", include / "snrt.h")
    for name, path in (
        ("test", TEST / "byte_mask_e2e.c"),
        ("driver", driver / "src/snax-xdma-lib.c"),
    ):
        execute([
            "clang", "-std=c11", "-O2", "-g", "-funwind-tables",
            "-I" + str(include), "-c", path, "-o", build / (name + ".o"),
        ], TEST)
    # This is also the repository CI workaround for Verilator's precompiled headers.
    os.environ["MAKEFLAGS"] = (
        os.environ.get("MAKEFLAGS", "") + " VK_PCH_I_FAST= VK_PCH_I_SLOW="
    ).strip()
    execute([
        "verilator", "--cc", "--exe", "--build", "--assert", "-j", "4",
        "-Wno-fatal", "--top-module", "byte_mask_e2e_top",
        "--Mdir", objects, "-CFLAGS", "-std=c++17 -I" + str(include),
        "-LDFLAGS", str(build / "test.o") + " " + str(build / "driver.o"),
        TEST / "top.sv", rtl / "byte_mask_e2e_xdma.sv", TEST / "sim.cc",
    ], TEST)
    execute([objects / "Vbyte_mask_e2e_top"], TEST, timeout=30)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
    except subprocess.TimeoutExpired as error:
        print("FAIL: timed out running " + str(error.cmd), flush=True)
        raise SystemExit(2)
