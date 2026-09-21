#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Derive the TCDM interconnect port map of a cluster cfg and emit it as C.

The cluster's narrow TCDM interconnect concatenates its inputs as
`{axi_soc_req, tcdm_req, snax_tcdm_req_i}`, so a port index means something
only against the cfg that produced it: which engine owns port 17 depends on how
many channels the engines before it asked for. The performance counters address
ports by that index (`PORT_INDEX`) and aggregate them by owning core
(`PORT_GROUP`), which makes the map software's problem.

Nothing emitted it, so the alternative was for every kernel that reads a
contention counter to hardcode the boundaries of the one cfg it was written
against. That fails silently and in the worst possible way: the counter still
counts, the numbers still look plausible, and they describe a different
engine's channels.

This module derives the map from the same rules the wrapper template and the
Chisel streamer use, and emits it:

  * SNAX ports are packed in core order, each core taking the port count its
    accelerator block asks for (`snitch_cluster_wrapper.sv.tpl`).
  * Within a streamer, ports run readers, then writers, then reader-writers,
    and a reader-writer pair SHARES one set of ports -- only the even-indexed
    entry of `data_reader_writer_params` contributes (`DesignParams.scala`,
    `readerWriterTcdmPorts`).
  * An xDMA or SIMD engine takes `num_channel` read ports followed by
    `num_channel` write ports.
  * Then come the Snitch cores' own data ports, one per core unless the core
    has SSRs, and finally the single narrow SoC port.

Run standalone:

    util/snaxgen/tcdm_ports.py --cfg_path <cluster.hjson> --out <header.h>
"""

import argparse
import os

import hjson
from jsonref import JsonRef

from core_roles import (  # noqa: I100  -- same directory, not a package
    RoleError,
    cluster_cores,
    derive_roles,
    roles_of,
    _source_label,
)


class PortMapError(ValueError):
    """A cfg whose TCDM port map cannot be derived."""


def get_config(cfg_path):
    """Parse a cluster hjson with its $refs resolved, as snaxgen does."""
    with open(cfg_path, "r") as f:
        return JsonRef.replace_refs(hjson.loads(f.read(), use_decimal=True))


def _streamer_channel_groups(acc):
    """Channel groups of one accelerator's streamer, in TCDM port order.

    Returns `[(label, count)]`. The labels name the streamer's data movers the
    way the cfg spells them, because that is the vocabulary the kernel author
    is holding: `data_reader_params[1]` is the B operand of a matmul only by
    convention, and encoding that convention here would make the header lie the
    first time somebody wires the operands the other way round.
    """
    cfg = acc.get("snax_streamer_cfg") or {}
    groups = []
    for i, n in enumerate((cfg.get("data_reader_params") or {}).get("num_channel", [])):
        groups.append(("rd{}".format(i), int(n)))
    for i, n in enumerate((cfg.get("data_writer_params") or {}).get("num_channel", [])):
        groups.append(("wr{}".format(i), int(n)))
    # A reader-writer pair shares one set of TCDM ports; only the reader half
    # (even index) contributes ports, and both halves drive them.
    rw = (cfg.get("data_reader_writer_params") or {}).get("num_channel", [])
    for i, n in enumerate(rw):
        if i % 2 == 0:
            groups.append(("rw{}".format(i // 2), int(n)))
    return groups


def _core_channel_groups(core, cluster):
    """Channel groups of one core's SNAX block, in TCDM port order.

    The branch order mirrors the wrapper template exactly -- accelerator, else
    xDMA, else SIMD, else nothing -- because a core carrying two of them gets
    ports for the first one only, and the map has to agree with the wiring
    rather than with what the cfg looks like it asks for.
    """
    accs = core.get("snax_acc_cfg") or []
    if accs and accs[0]:
        groups = []
        for acc in accs:
            if "snax_tcdm_ports" not in acc:
                raise PortMapError(
                    "accelerator {!r} has no snax_tcdm_ports".format(
                        acc.get("snax_acc_name")
                    )
                )
            declared = int(acc["snax_tcdm_ports"])
            derived = _streamer_channel_groups(acc)
            total = sum(n for _, n in derived)
            if derived and total != declared:
                raise PortMapError(
                    "accelerator {!r} declares snax_tcdm_ports {} but its streamer "
                    "channels add up to {}. One of the two is stale, and the port "
                    "map would be wrong either way.".format(
                        acc.get("snax_acc_name"), declared, total
                    )
                )
            name = str(acc.get("snax_acc_name", "acc")).replace("snax_", "")
            if derived:
                groups += [("{}.{}".format(name, lbl), n) for lbl, n in derived]
            else:
                # An accelerator with no streamer description still owns ports;
                # name them as one block rather than inventing a breakdown.
                groups.append((name, declared))
        return groups

    narrow = round(int(cluster["dma_data_width"]) / int(cluster["data_width"]))
    if "snax_xdma_cfg" in core:
        return [("xdma.rd", narrow), ("xdma.wr", narrow)]
    if "snax_simd_cfg" in core:
        simd = core["snax_simd_cfg"]
        n = int(simd["num_channel"]) if "num_channel" in simd else narrow
        return [("simd.rd", n), ("simd.wr", n)]
    return []


def _snitch_data_ports(core):
    """TCDM data ports of the Snitch core itself: one, or one per SSR.

    This mirrors `get_tcdm_ports` in snitch_cluster.sv, which reads the
    `NumSsrs` the wrapper emits. Clustergen writes that as the core's
    `num_ssrs`; a raw cfg that has not been through clustergen only carries the
    `ssrs` list, so both are accepted and `num_ssrs` wins.
    """
    if "num_ssrs" in core:
        n = int(core["num_ssrs"])
    else:
        n = len(core.get("ssrs") or [])
    return n if n > 1 else 1


def derive_port_map(cfg, cfg_path):
    """Derive the whole interconnect input map. Returns a dict for emission."""
    if "cluster" not in cfg:
        raise PortMapError(
            "no `cluster` key: this is not a cluster cfg. cfg/lru.hjson is a "
            "SYMLINK to the active cfg -- point it at a real one with CFG_OVERRIDE."
        )
    cluster = cfg["cluster"]
    cores = cluster_cores(cfg)
    roles = derive_roles(cores)

    channels = []  # (label, base, count)
    per_core = []  # (base, count)
    base = 0
    for i, core in enumerate(cores):
        held = roles_of(cores, roles, i)
        prefix = held[0] if held else "hart{}".format(i)
        core_base, core_count = base, 0
        for label, count in _core_channel_groups(core, cluster):
            # The role name is the one a kernel author says out loud. Where the
            # cfg already names the block (`xdma.rd`), keep its own name.
            if not label.startswith(("xdma.", "simd.")):
                label = "{}.{}".format(prefix, label.split(".", 1)[-1])
            channels.append((label, base, count))
            base += count
            core_count += count
        per_core.append((core_base, core_count))

    snax_ports = base
    snitch_base = snax_ports
    snitch_ports = sum(_snitch_data_ports(c) for c in cores)
    soc_port = snitch_base + snitch_ports
    return {
        "source": _source_label(cfg_path),
        "num_cores": len(cores),
        "channels": channels,
        "per_core": per_core,
        "snax_ports": snax_ports,
        "snitch_base": snitch_base,
        "snitch_ports": snitch_ports,
        "soc_port": soc_port,
        "num_ports": soc_port + 1,
        "num_banks": int(cluster["tcdm"]["banks"]),
        "narrow_bytes": int(cluster["data_width"]) // 8,
        "wide_bytes": int(cluster["dma_data_width"]) // 8,
        "roles": roles,
        "cores": cores,
    }


def emit_header(m):
    """Render the generated C header as a string."""
    lines = [
        "// Copyright 2026 KU Leuven.",
        "// Licensed under the Apache License, Version 2.0, see LICENSE for details.",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        "// GENERATED by util/snaxgen/tcdm_ports.py -- do not edit.",
        "// Source: {}".format(m["source"]),
        "//",
        "// The narrow TCDM interconnect's input map, which is what the cluster",
        "// performance counters address: PORT_INDEX picks one input, PORT_GROUP",
        "// aggregates the inputs of one core. Port indices mean nothing across",
        "// cfgs, so read them from here rather than writing them down.",
        "//",
        "//   index      owner",
    ]
    for label, base, count in m["channels"]:
        lines.append(
            "//   {:>3}..{:<3}  {}".format(base, base + count - 1, label)
            if count > 1
            else "//   {:>3}       {}".format(base, label)
        )
    lines.append(
        "//   {:>3}..{:<3}  snitch core data ports".format(
            m["snitch_base"], m["snitch_base"] + m["snitch_ports"] - 1
        )
    )
    lines.append("//   {:>3}       narrow SoC port".format(m["soc_port"]))
    lines += [
        "",
        "#pragma once",
        "",
        "#define SNAX_TCDM_NUM_PORTS {}".format(m["num_ports"]),
        "#define SNAX_TCDM_SNAX_PORTS {}".format(m["snax_ports"]),
        "#define SNAX_TCDM_SNITCH_BASE {}".format(m["snitch_base"]),
        "#define SNAX_TCDM_SNITCH_PORTS {}".format(m["snitch_ports"]),
        "#define SNAX_TCDM_SOC_PORT {}".format(m["soc_port"]),
        "#define SNAX_TCDM_NUM_BANKS {}".format(m["num_banks"]),
        "// Bytes one narrow port moves per granted request, and one wide DMA beat.",
        "#define SNAX_TCDM_NARROW_BYTES {}".format(m["narrow_bytes"]),
        "#define SNAX_TCDM_WIDE_BYTES {}".format(m["wide_bytes"]),
        "",
        "// PORT_GROUP selectors. Group 0 is every input; then one group per core,",
        "// holding that core's SNAX ports; then all Snitch data ports; then the SoC.",
        "#define SNAX_TCDM_GRP_ALL 0",
    ]
    for i in range(m["num_cores"]):
        lines.append("#define SNAX_TCDM_GRP_HART{} {}".format(i, i + 1))
    for role, idx in sorted(m["roles"].items(), key=lambda kv: kv[1]):
        lines.append(
            "#define SNAX_TCDM_GRP_{} SNAX_TCDM_GRP_HART{}".format(role.upper(), idx)
        )
    lines += [
        "#define SNAX_TCDM_GRP_SNITCH {}".format(m["num_cores"] + 1),
        "#define SNAX_TCDM_GRP_SOC {}".format(m["num_cores"] + 2),
        "#define SNAX_TCDM_NUM_GROUPS {}".format(m["num_cores"] + 3),
        "",
        "// Per-core SNAX port ranges, for sweeping a group port by port.",
    ]
    for i, (base, count) in enumerate(m["per_core"]):
        lines.append("#define SNAX_TCDM_HART{}_BASE {}".format(i, base))
        lines.append("#define SNAX_TCDM_HART{}_PORTS {}".format(i, count))
    lines += [
        "",
        "// Every channel group of every engine, in port order. A kernel can walk this",
        "// and label a per-port census without knowing which cluster it is running on.",
        "typedef struct {",
        "    const char *name;",
        "    unsigned short base;",
        "    unsigned short count;",
        "} snax_tcdm_channel_group_t;",
        "",
        "#define SNAX_TCDM_NUM_CHANNEL_GROUPS {}".format(len(m["channels"])),
        "#define SNAX_TCDM_CHANNEL_GROUPS_INIT \\",
    ]
    body = ", \\\n".join(
        '        {{"{}", {}, {}}}'.format(label, base, count)
        for label, base, count in m["channels"]
    )
    lines.append("    { \\")
    lines.append(body + " \\")
    lines.append("    }")
    return "\n".join(lines).rstrip("\n") + "\n"


def generate(cfg, cfg_path, out_path):
    """Derive and write the port-map header. Returns the derived map."""
    m = derive_port_map(cfg, cfg_path)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w") as f:
        f.write(emit_header(m))
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cfg_path", required=True, help="Path to the cluster hjson cfg"
    )
    parser.add_argument(
        "--out", required=True, help="Path of the generated C header to write"
    )
    args = parser.parse_args()

    cfg = get_config(args.cfg_path)
    try:
        m = generate(cfg, args.cfg_path, args.out)
    except (PortMapError, RoleError) as e:
        raise SystemExit("[tcdm_ports] {}: {}".format(args.cfg_path, e))
    print(
        "[tcdm_ports] {} -> {} interconnect inputs ({} snax, {} snitch, 1 soc)".format(
            os.path.basename(args.cfg_path),
            m["num_ports"],
            m["snax_ports"],
            m["snitch_ports"],
        )
    )


if __name__ == "__main__":
    main()
