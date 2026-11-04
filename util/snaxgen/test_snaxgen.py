# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Tests of snaxgen's streamer sets: the CSR range table, the refusals, the per-set names,
# snax_forge.f and clustergen's per-set count, and (with sbt) the headers the sw_only path
# writes. Run from the repository: python -m pytest util/snaxgen

import copy
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "util" / "clustergen"))

import snaxgen  # noqa: E402
import cluster  # noqa: E402

CLUSTER_FILE = HERE / "test_streamer_sets.hjson"
TAG = "snaxgen_sets_test"
HEADER_DIR = REPO / "target" / "snitch_cluster" / "sw" / "snax" / TAG / "include"

EXPECTED = [
    # name, streamer CSRs, own CSRs, shell offsets
    ("vmul", (960, 977), (978, 981), (0, 3)),
    ("vsum", (982, 994), (995, 999), (4, 8)),
]


# The cluster file's entry as plain dicts and lists
def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


@pytest.fixture
def entry():
    cfg = plain(snaxgen.get_config(str(CLUSTER_FILE)))
    return cfg["cluster"]["hives"][0]["cores"][0]["snax_acc_cfg"][0]


def test_range_table(entry):
    table = snaxgen.csr_range_table(entry)
    assert [row["name"] for row in table] == ["vmul", "vsum"]
    for row, (name, streamer, own, shell) in zip(table, EXPECTED):
        assert (row["streamer_base"], row["streamer_base"] + row["streamer_csr_num"] - 1) \
            == streamer, name
        assert row["csr_base"] == row["streamer_base"] + row["streamer_csr_num"], name
        assert (row["csr_base"], row["csr_base"] + row["csr_num"] - 1) == own, name
        assert (row["shell_offset"], row["shell_offset"] + row["csr_num"] - 1) == shell, name
    assert [(r["num_rw_csr"], r["num_ro_csr"]) for r in table] == [(2, 2), (3, 2)]


def test_single_set_entry_unchanged(entry):
    # An entry with snax_streamer_cfg is one set named after it, from 960, no extra args
    single = {
        "snax_acc_name": "single",
        "tag_name": "single",
        "snax_num_rw_csr": 2,
        "snax_num_ro_csr": 2,
        "snax_streamer_cfg": entry["snax_streamer_sets"][0]["snax_streamer_cfg"],
    }
    assert [s["name"] for s in snaxgen.streamer_sets(single)] == ["single"]
    table = snaxgen.csr_range_table(single)
    assert len(table) == 1
    assert table[0]["streamer_base"] == 960
    assert table[0]["streamer_csr_num"] == snaxgen.streamer_csr_num(single)
    single["csr_range_table"] = table
    [(unit_cfg, extra_args, header_dir)] = snaxgen.streamer_units(single)
    assert unit_cfg is single
    assert extra_args == ""
    assert header_dir == "../../target/snitch_cluster/sw/snax/single/include"
    # An entry with neither key has no sets
    assert snaxgen.streamer_sets({"snax_acc_name": "none"}) == []


def test_set_units(entry):
    entry["tag_name"] = entry["snax_acc_name"]
    entry["csr_range_table"] = snaxgen.csr_range_table(entry)
    units = snaxgen.streamer_units(entry)
    assert [u[0]["tag_name"] for u in units] == [f"{TAG}_vmul", f"{TAG}_vsum"]
    assert units[0][1] == (
        " --csr-base 960 --macro-prefix VMUL_ --header-name streamer_vmul_csr_addr_map.h"
    )
    assert units[1][1] == (
        " --csr-base 982 --macro-prefix VSUM_ --header-name streamer_vsum_csr_addr_map.h"
    )
    assert {u[2] for u in units} == {f"../../target/snitch_cluster/sw/snax/{TAG}/include"}
    assert [u[0]["streamer_csr_num"] for u in units] == [18, 13]
    # The entry itself is left as it was
    assert entry["tag_name"] == TAG and "snax_streamer_cfg" not in entry


def refusal(acc):
    with pytest.raises(SystemExit) as error:
        snaxgen.csr_range_table(acc)
    message = str(error.value)
    assert f"accelerator entry '{TAG}'" in message
    return message


def test_refuses_both_keys(entry):
    entry["snax_streamer_cfg"] = entry["snax_streamer_sets"][0]["snax_streamer_cfg"]
    assert "both snax_streamer_cfg and snax_streamer_sets" in refusal(entry)


def test_refuses_sets_without_shell_csrs(entry):
    entry.pop("snax_disable_csr_manager")
    assert "streamer sets need the shell's own CSRs" in refusal(entry)
    entry["snax_disable_csr_manager"] = False
    assert "streamer sets need the shell's own CSRs" in refusal(entry)


def test_refuses_duplicate_names(entry):
    entry["snax_streamer_sets"][1]["name"] = "vmul"
    assert "duplicate streamer set names ['vmul']" in refusal(entry)


def test_refuses_layout_past_1533(entry):
    # vsum's own CSRs end at 999; 534 more R/W CSRs end exactly at 1533
    entry["snax_streamer_sets"][1]["snax_num_rw_csr"] += 1533 - 999
    assert snaxgen.csr_range_table(entry)[-1]["csr_base"] + 539 - 1 == 1533
    entry["snax_streamer_sets"][1]["snax_num_rw_csr"] += 1
    assert "its CSRs end at 1534, past 1533" in refusal(entry)


def test_snax_forge_f_lists_each_set(entry, tmp_path):
    (tmp_path / "rtl").mkdir()
    (tmp_path / "rtl" / "acc.sv").write_text("")
    (tmp_path / "rtl" / "files.f").write_text("acc.sv\n")
    entry["snax_forge_rtl"] = "rtl/files.f"
    gen = tmp_path / "generated"
    names = [f"{TAG}_vmul_Streamer.sv", f"{TAG}_vmul_streamer_wrapper.sv",
             f"{TAG}_vsum_Streamer.sv", f"{TAG}_vsum_streamer_wrapper.sv",
             f"{TAG}_wrapper.sv"]
    (gen / TAG).mkdir(parents=True)
    for name in names:
        (gen / TAG / name).write_text("")
    snaxgen.write_snax_forge_f([{"snax_acc_cfg": [entry]}], str(tmp_path / "c.hjson"), str(gen))
    listed = (gen / "snax_forge.f").read_text().splitlines()
    assert listed == [str(tmp_path / "rtl" / "acc.sv")] + [str(gen / TAG / n) for n in names]


def test_clustergen_counts_per_set(entry):
    single = copy.deepcopy(entry["snax_streamer_sets"][0])
    single["snax_acc_name"] = "single"
    stub = type("Stub", (), {})()
    stub.cfg = {"cores": [{"snax_acc_cfg": [entry, single]}, {"snax_acc_cfg": [{}]}]}
    cluster.SnitchCluster.parse_streamer_csr(stub)
    assert stub.cfg["streamer_csr_num_list"] == [[18, 13, 18], []]


def defines(path):
    return {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"^#define (\w+) (\d+)$", path.read_text(), re.M)
    }


@pytest.mark.skipif(shutil.which("sbt") is None, reason="needs sbt (StreamerSwHeaderGen)")
def test_sw_only_headers_agree(entry, tmp_path):
    shutil.rmtree(HEADER_DIR.parent, ignore_errors=True)
    try:
        subprocess.run(
            [sys.executable, str(HERE / "snaxgen.py"),
             f"--cfg_path={CLUSTER_FILE}",
             f"--tpl_path={REPO}/hw/templates/",
             f"--chisel_path={REPO}/hw/chisel/",
             "--sw_only=true",
             f"--gen_path={tmp_path}/"],
            check=True, cwd=REPO / "target" / "snitch_cluster",
        )
        assert sorted(os.listdir(HEADER_DIR)) == [
            "streamer_vmul_csr_addr_map.h", "streamer_vsum_csr_addr_map.h"]
        for row in snaxgen.csr_range_table(entry):
            prefix = row["name"].upper() + "_"
            header = defines(HEADER_DIR / f"streamer_{row['name']}_csr_addr_map.h")
            assert header, row["name"]
            assert all(name.startswith(prefix) for name in header), row["name"]
            first = row["streamer_base"]
            last = first + row["streamer_csr_num"] - 1
            assert header[prefix + "BASE_PTR_READER_0_LOW"] == first
            assert header[prefix + "STREAMER_START_CSR"] == last - 2
            assert header[prefix + "STREAMER_BUSY_CSR"] == last - 1
            assert header[prefix + "STREAMER_PERFORMANCE_COUNTER_CSR"] == last
            # Every CSR number in the header (not the counts, *_NUM_* and *_CSR_NUM) lies in
            # the set's streamer range, and the movers take 5 CSRs each
            numbers = {v for k, v in header.items() if "_NUM" not in k}
            assert min(numbers) == first and max(numbers) == last, row["name"]
            movers = [k for k in header if re.fullmatch(prefix + r"BASE_PTR_\w+_LOW", k)]
            assert len(movers) * 5 + 3 == row["streamer_csr_num"], row["name"]
        assert defines(HEADER_DIR / "streamer_vmul_csr_addr_map.h")[
            "VMUL_BASE_PTR_READER_0_LOW"] == 960
        assert defines(HEADER_DIR / "streamer_vsum_csr_addr_map.h")[
            "VSUM_BASE_PTR_READER_0_LOW"] == 982
    finally:
        shutil.rmtree(HEADER_DIR.parent, ignore_errors=True)
