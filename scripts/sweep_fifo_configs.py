#!/usr/bin/env python3
"""
Minimal FIFO sweep.

Edit the hard-coded CONFIGS list below, then run:

  python3 scripts/sweep_fifo_configs.py
"""

import datetime as dt
import os
import re
import shlex
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG = os.path.join(ROOT, "target", "snitch_cluster", "cfg", "snax_simbacore_cluster.hjson")
VSIM_DIR = os.path.join(ROOT, "target", "snitch_cluster")
OUT_DIR = os.path.join(ROOT, "regression_test_out", "fifo_sweep")


# Edit this list to define which FIFO configurations you want to test.
# Each entry is a tuple: (reader_fifo_depth_list, writer_fifo_depth_list).
CONFIGS = [
    # 0    1    2    3    4    5    6    7    8    9    10   11   12   13 # 0 1 2 3
    ([9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9]),
]


def patch_fifo_depths(reader, writer):
    txt = open(CFG).read()

    def repl_occurrence(values, which):
        pat = re.compile(r"(fifo_depth:\s*)\[[^\]]*\]")
        seen = 0

        def repl(m):
            nonlocal seen
            seen += 1
            if seen == which:
                vals = ", ".join(str(v) for v in values)
                return f"{m.group(1)}[ {vals} ]"
            return m.group(0)

        return pat.sub(repl, txt if which == 1 else repl_occurrence(values, 1)[0], count=0)

    # First occurrence = reader, second = writer
    def apply(values, which, text):
        pat = re.compile(r"(fifo_depth:\s*)\[[^\]]*\]")
        seen = 0

        def repl(m):
            nonlocal seen
            seen += 1
            if seen == which:
                vals = ", ".join(str(v) for v in values)
                return f"{m.group(1)}[ {vals} ]"
            return m.group(0)

        return pat.sub(repl, text)

    txt = apply(reader, 1, txt)
    txt = apply(writer, 2, txt)
    open(CFG, "w").write(txt)


def run_build_and_sim(log_path):
    subprocess.check_call(["bash", "scripts/build_sim.sh"], cwd=ROOT)
    # Stream to console and write directly to the per-config log file.
    quoted = shlex.quote(log_path)
    cmd = f"bin/snitch_cluster.vsim sw/apps/snax-simbacore-simd/build/snax-simbacore-simd.elf | tee {quoted}"
    subprocess.check_call(cmd, cwd=VSIM_DIR, shell=True)


def parse_cycles(log_path):
    s = open(log_path).read()
    nums = [int(m.group(1)) for m in re.finditer(r"SimbaCore took\s+(\d+)\s+cycles", s)]
    if len(nums) < 2:
        raise RuntimeError("Expected at least two 'SimbaCore took ... cycles' lines")
    return nums[-2], nums[-1]


def main():
    cfgs = CONFIGS

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(OUT_DIR, f"fifo_sweep_summary-{ts}.txt")

    orig = open(CFG).read()
    with open(summary_path, "w") as out:
        out.write("idx\treader_fifo_depth\twriter_fifo_depth\tphase1\tphase2\n")
        for idx, (r, w) in enumerate(cfgs):
            name = f"cfg{idx}"
            log_path = os.path.join(OUT_DIR, f"vsim_{name}.log")
            print(f"== {name} ==")
            patch_fifo_depths(r, w)
            run_build_and_sim(log_path)
            p1, p2 = parse_cycles(log_path)
            out.write(f"{idx}\t{' '.join(map(str, r))}\t{' '.join(map(str, w))}\t{p1}\t{p2}\n")
    open(CFG, "w").write(orig)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
