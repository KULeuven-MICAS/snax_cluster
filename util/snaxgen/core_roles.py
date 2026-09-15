#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Derive the per-hart role map of a cluster cfg and emit it as C #defines.

The cluster hjson already IS the ground truth for which hart carries which
engine: `snax_acc_cfg` is the accelerator, `snax_simd_cfg` the SIMD operator
bank, `snax_xdma_cfg` the SNAX transfer engine, and the `xdma` boolean the
Snitch DMA ISA extension. Nothing emitted that, so the same fact was written
out by hand in `snax-core-roles.h`, in `sw/apps/common.mk` -- as a match on the
cfg FILE NAME -- and again in every downstream integrator. None of those
failures is loud: a kernel placed on the wrong hart programs THAT hart's
accelerator at the same CSR offsets and reports success.

This module derives the map structurally and emits it, so software reads the
placement from a generated header instead of assuming it.

The optional `roles:` list on a core is a LABEL, cross-checked against the
derivation -- never a second source of truth. A label that disagrees with what
the cfg structurally describes fails generation, so a stale label is a build
error rather than a quiet second opinion.

Run standalone:

    util/snaxgen/core_roles.py --cfg_path <cluster.hjson> --out <header.h>
"""

import argparse
import os

import hjson
from jsonref import JsonRef

# The roles a cluster cfg may declare, in the order they are emitted.
#
# `host` is deliberately absent and should stay absent: on a multi-chiplet
# integration the host is the chiplet's CVA6 sitting at index
# N_CORES_PER_CLUSTER, one past the last cluster core. That is an
# integrator-level concept which a cluster cfg cannot describe.
ROLES = ("gemm", "simd", "xdma", "idma")

# `gemm` is the matmul ARRAY, not "whatever accelerator hart 0 carries".
# snax_alu / snax_cgra / snax_dimc / snax_hypercorex are accelerators too, but
# no kernel places itself by asking for a matmul on them, and calling them
# `gemm` would make the generated header lie about what the hart can do.
MATMUL_ACC_NAMES = ("snax_versacore", "snax_streamer_gemmX")


class RoleError(ValueError):
    """A cfg whose role map cannot be emitted as written."""


def get_config(cfg_path):
    """Parse a cluster hjson with its $refs resolved, as snaxgen does."""
    with open(cfg_path, "r") as f:
        return JsonRef.replace_refs(hjson.loads(f.read(), use_decimal=True))


def cluster_cores(cfg):
    """Cores of the cluster, flattened in CLUSTER-LOCAL index order.

    Roles are emitted as cluster-local indices because that is what software
    compares against (`snrt_cluster_core_idx()`), while the cfg nests cores by
    hive. Every cfg in this repo has exactly one hive, so the flattening is a
    no-op today; it is written out so that adding a second hive reindexes the
    map the way software reads it rather than silently emitting hive-local
    indices.
    """
    if "cluster" not in cfg:
        raise RoleError(
            "no `cluster` key: this is not a cluster cfg. cfg/default.hjson is "
            "empty, and cfg/lru.hjson is a SYMLINK to the active cfg -- point "
            "it at a real one with CFG_OVERRIDE."
        )
    cores = []
    for hive in cfg["cluster"]["hives"]:
        cores.extend(hive["cores"])
    return cores


def _claim(roles, role, idx, why):
    """Assign `role` to hart `idx`, refusing a second claimant."""
    if role in roles and roles[role] != idx:
        raise RoleError(
            "two cores claim the role '{}': hart {} and hart {} ({}). A role "
            "names ONE hart, because software resolves it to a single core "
            "index.".format(role, roles[role], idx, why)
        )
    roles[role] = idx


def derive_roles(cores):
    """Derive `role -> cluster-local core index` from the cfg structure alone.

    The rules are independent and a core matches as many as apply, which is how
    the unsplit two-core shape derives [idma, xdma, simd] for its single data
    mover from one cfg block.
    """
    last = len(cores) - 1
    roles = {}

    # gemm -- a matmul accelerator block.
    for i, core in enumerate(cores):
        for acc in core.get("snax_acc_cfg") or []:
            if acc.get("snax_acc_name") in MATMUL_ACC_NAMES:
                _claim(roles, "gemm", i, "both carry a matmul snax_acc_cfg")
                break

    # xdma -- the SNAX transfer engine. NOT the `xdma:` boolean below.
    for i, core in enumerate(cores):
        if "snax_xdma_cfg" in core:
            _claim(roles, "xdma", i, "both carry snax_xdma_cfg")

    # simd -- the operator bank. A cluster that splits it onto its own hart says
    # so with snax_simd_cfg. A cluster that does not has the same extensions
    # sitting on the xDMA's reader socket, and that core is then the SIMD core --
    # which is exactly what the hand-written header's
    # `snax_is_simd_core() { return snrt_is_dm_core(); }` used to mean.
    #
    # The reader_extensions fallback is consulted ONLY when no core has
    # snax_simd_cfg. A split cluster's xDMA also declares reader_extensions -- a
    # Transposer, which is a wire permutation rather than compute and so belongs
    # with data movement -- and reading that as a SIMD claim would hand the role
    # to two harts at once.
    simd = [i for i, c in enumerate(cores) if "snax_simd_cfg" in c]
    if not simd:
        simd = [
            i
            for i, c in enumerate(cores)
            if "reader_extensions" in (c.get("snax_xdma_cfg") or {})
        ]
    for i in simd:
        _claim(roles, "simd", i, "both carry a SIMD operator bank")

    # idma -- the classic Snitch DMA, spelled `xdma: true`.
    #
    # VOCABULARY WARNING: that boolean is the Snitch DMA ISA extension and has
    # nothing to do with snax_xdma_cfg, the SNAX transfer engine that gives the
    # `xdma` role above. The two keys are confusingly named and sit on DIFFERENT
    # harts in a split cluster -- there, hart 3 has `xdma: true` and no snax
    # block, while hart 2 has snax_xdma_cfg with `xdma: false`.
    for i, core in enumerate(cores):
        has_idma = core.get("xdma", False)
        # Insist on a real boolean. hjson would hand back the STRING "false" for
        # a quoted value, which is truthy -- and would hand the iDMA role to a
        # core that does not have it, silently. No cfg quotes it today; the
        # check is here so none ever can.
        if not isinstance(has_idma, bool):
            raise RoleError(
                "hart {}: `xdma` must be a boolean, got {!r}. A quoted "
                "\"false\" is truthy and would claim the idma role.".format(
                    i, has_idma
                )
            )
        if has_idma:
            if i != last:
                raise RoleError(
                    "hart {} sets `xdma: true` (the Snitch DMA ISA) but is not "
                    "the last core (hart {}). snRuntime picks its data-mover "
                    "core STRUCTURALLY as the last one -- snrt_is_dm_core() is "
                    "!(core_idx < core_num - SNRT_CLUSTER_DM_CORE_NUM) in "
                    "sw/snRuntime/src/team.h -- and runs every singleton task "
                    "there: the L1/L3 allocator init in alloc.h and "
                    "global-barrier participation in sync.h. A cfg that puts "
                    "the iDMA anywhere else describes a machine snRuntime "
                    "cannot drive.".format(i, last)
                )
            _claim(roles, "idma", i, "both set `xdma: true`")

    return roles


def check_labels(cores, derived):
    """Fail if any `roles:` label disagrees with the derivation.

    The derivation is the source of truth. The label exists so the cfg SAYS
    what it is, and so that changing the structure without the label -- or the
    label without the structure -- is a generation-time error instead of a
    kernel silently landing on the wrong hart.
    """
    labelled = {}
    for i, core in enumerate(cores):
        if "roles" not in core:
            continue
        declared = core["roles"]
        if not isinstance(declared, list):
            raise RoleError(
                "hart {}: `roles` must be a list of role names, got {!r}. Use a "
                "list even for a single role -- one core commonly holds several "
                "(the unsplit data mover is idma, xdma and simd at once).".format(
                    i, declared
                )
            )
        for role in declared:
            if role not in ROLES:
                raise RoleError(
                    "hart {}: unknown role '{}'. Known roles: {}.".format(
                        i, role, ", ".join(ROLES)
                    )
                )
            _claim(labelled, role, i, "both are labelled with it")

        want = sorted(r for r, idx in derived.items() if idx == i)
        got = sorted(set(declared))
        if got != want:
            raise RoleError(
                "hart {}: `roles: [{}]` disagrees with what the cfg "
                "structurally describes, [{}]. The label does not decide "
                "placement -- the accelerator blocks on the core do -- so fix "
                "whichever of the two is wrong.".format(
                    i, ", ".join(got), ", ".join(want)
                )
            )
    return labelled


def roles_of(cores, derived, idx):
    """The roles held by hart `idx`, in emission order."""
    return [r for r in ROLES if derived.get(r) == idx]


# <repo>/util/snaxgen/core_roles.py -> <repo>
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _source_label(cfg_path):
    """A repo-relative name for the cfg, independent of the caller's cwd.

    The emitted header must be a pure function of the cfg: HeMAiA drives this
    through `make -C <snitch>/target/snitch_cluster` from its own root, so a
    cwd-relative path would put `../../..` chains into the file and make the
    SAME cfg generate different bytes depending on where make was run.
    """
    path = os.path.abspath(cfg_path)
    rel = os.path.relpath(path, _REPO_ROOT)
    return os.path.basename(path) if rel.startswith(os.pardir) else rel


def emit_header(cfg_path, cores, derived):
    """Render the generated C header as a string."""
    rel = _source_label(cfg_path)
    lines = [
        "// Copyright 2026 KU Leuven.",
        "// Licensed under the Apache License, Version 2.0, see LICENSE for details.",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        "// GENERATED by util/snaxgen/core_roles.py -- do not edit.",
        "// Source: {}".format(rel),
        "//",
        "// Which hart carries which engine, derived from the cluster cfg rather",
        "// than assumed. Use the helpers in snax-core-roles.h, not these macros",
        "// directly.",
        "//",
    ]
    for i in range(len(cores)):
        held = roles_of(cores, derived, i)
        lines.append(
            "//   hart {}  {}".format(i, ", ".join(held) if held else "(plain compute)")
        )
    lines += [
        "//",
        "// SNAX_HAS_<ROLE>_CORE is 0 for an engine this cluster does not have, and",
        "// SNAX_CORE_<ROLE> is then not defined at all -- so a kernel that needs",
        "// the engine fails to build instead of silently running on hart 0.",
        "",
        "#pragma once",
        "",
        "#define SNAX_CLUSTER_NUM_CORES {}".format(len(cores)),
        "",
    ]
    for role in ROLES:
        upper = role.upper()
        if role in derived:
            lines.append("#define SNAX_HAS_{}_CORE 1".format(upper))
            lines.append("#define SNAX_CORE_{} {}".format(upper, derived[role]))
        else:
            lines.append("#define SNAX_HAS_{}_CORE 0".format(upper))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def generate(cfg, cfg_path, out_path):
    """Derive, validate and write the role header. Returns the role map."""
    cores = cluster_cores(cfg)
    derived = derive_roles(cores)
    check_labels(cores, derived)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w") as f:
        f.write(emit_header(cfg_path, cores, derived))
    return derived


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
    derived = generate(cfg, args.cfg_path, args.out)
    cores = cluster_cores(cfg)
    print(
        "[core_roles] {} -> {}".format(
            os.path.basename(args.cfg_path),
            ", ".join(
                "hart {}: {}".format(i, "+".join(roles_of(cores, derived, i)))
                for i in range(len(cores))
                if roles_of(cores, derived, i)
            )
            or "no roles",
        )
    )


if __name__ == "__main__":
    main()
