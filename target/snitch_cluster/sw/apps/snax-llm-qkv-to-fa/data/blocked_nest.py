# MIRROR of HeMAiA target/sw/host/runtime/libbingo/mini_compiler/kernels/blocked_nest.py,
# vendored so this app can test the descriptors that module derives. Here the beat geometry
# (lanes per beat, bytes per lane) is a parameter taken from the cluster cfg.
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Deriving the SIMD BLOCKED pass for any GEMM mesh -- and checking it.

The blocked pass writes a GEMM operand layout straight from a plain fp16 tile, by reading
the tile in the operand's own order (see simd_pass_blocked in the device's simd.h). Which
order works depends on the operand block the mesh defines, so the order is DERIVED here,
per (tile, mesh, output precision), and handed to the device as a descriptor it executes
without interpreting. Nothing the derivation produces is trusted: `blocked_nest` walks the
final descriptor the way the hardware will and compares every element against the target
layout's index map, the same discipline comm/nest.py applies to the xDMA.

THE PROBLEM, stated once. The source S [R, C] is fp16 with rows `pitch` bytes apart. The
target is the A-layout of S with Mb x Kb blocks -- (m, k, r, s), element (i, j) at

    ((i // Mb * (C // Kb) + j // Kb) * Mb + i % Mb) * Kb + j % Kb

which serves both operands: A of y [T, d] is this with S = y, (Mb, Kb) = (meshRow,
tileSize); B of y is this with S = y^T and (Mb, Kb) = (meshCol, tileSize), because B's
atom runs along the contraction axis exactly as A's does.

THE HARDWARE it has to fit, and the three facts that bound what is expressible. A LANE
here is the AGU's spatial unit, one TCDM channel of the beat (the shape API's lane_stride
and lane_mask), not an FP16 operator lane: a beat is `lanes` lanes of `lane_bytes` each,
both read from the cluster cfg (simd_beat_bytes / data_width).

  1. A LANE IS 8 B. A reader lane is 4 contiguous fp16 of S; a writer lane is 8 B of the
     target, 4 fp16 or 8 int8, and must be contiguous there. So an fp16 output needs
     tileSize % 4 == 0, and no engine on this cluster can write a tileSize = 2 operand
     from fp16 -- the xDMA's atom is the same 8 B.
  2. ONE LANE STRIDE PER SIDE. Lane c sits at base + c*stride, so the lanes of a beat
     have to be evenly spaced runs on both sides. This is what refuses an int8 operand
     whose atom sits inside a taller block and spans more than one lane: the lanes fill
     an atom and the next atom is a block away, which no single stride reaches.
  3. THREE LOOPS PER SIDE, one of them spent on the operand interleave when the pass also
     multiplies. A nest that needs more is split into repeated tasks, the outermost loop
     peeled into the core's own loop -- still one kernel, one extra task per iteration.

Fp16ToInt8 turns two consecutive fp16 beats into one int8 beat, in order, so int8 output
lane c is lanes 2c and 2c+1 of the pair (the first beat's lanes, then the second's). That
is why the int8 and fp16 nests of one mesh can differ: the pairing has to land on 8
contiguous target bytes.
"""

import itertools
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np



@dataclass(frozen=True)
class BlockedNest:
    """What the device executes. Loops are innermost first, as (bound, byte stride).

    `rd` excludes the operand interleave: when the pass multiplies, the device puts that
    loop innermost itself, because only it knows where its scratch operand lives.
    `reps` repeats the whole task with both bases advanced by the rep strides.
    `lanes` x `lane_bytes` is the beat it was derived for.
    """
    lanes: int
    lane_bytes: int
    pitch: int
    rd_lane: int
    rd: Tuple[Tuple[int, int], ...]
    wr_lane: int
    wr: Tuple[Tuple[int, int], ...]
    reps: int
    rep_rd: int
    rep_wr: int
    lane_dir: str

    @property
    def tasks(self) -> int:
        return self.reps


def target_index(rows: int, cols: int, mb: int, kb: int) -> np.ndarray:
    i = np.arange(rows)[:, None]
    j = np.arange(cols)[None, :]
    return ((i // mb * (cols // kb) + j // kb) * mb + i % mb) * kb + j % kb


def _pow2_splits(bound: int) -> List[int]:
    return [d for d in range(2, bound) if bound % d == 0 and d & (d - 1) == 0]


def _side_dims(loops):
    """The AGU loops one side needs: drop bound-1 loops, merge adjacent ones that compose."""
    out = []
    for b, s in loops:
        if b == 1:
            continue
        if out and s == out[-1][1] * out[-1][0]:
            out[-1] = (out[-1][0] * b, out[-1][1])
        else:
            out.append((b, s))
    return out


def _stream(loops, lane_step, lanes):
    """(i, j) of every lane of every beat, in stream order: shape (beats, lanes, 2)."""
    bounds = [lp[0] for lp in loops]
    grids = np.indices(bounds[::-1]).reshape(len(bounds), -1)[::-1]   # innermost first
    i = sum(g * lp[2][0] for g, lp in zip(grids, loops))
    j = sum(g * lp[2][1] for g, lp in zip(grids, loops))
    c = np.arange(lanes)
    return np.stack([i[:, None] + c * lane_step[0], j[:, None] + c * lane_step[1]], -1)


def _lane_targets(beats, tmap, out_bytes, fp16_per_lane):
    """Target byte address of each OUTPUT lane, or None if a lane is not contiguous."""
    lanes = beats.shape[1]
    run = np.arange(fp16_per_lane)
    t = tmap[beats[..., 0][..., None], beats[..., 1][..., None] + run]  # (beats, lanes, run)
    if out_bytes == 2:
        if np.any(np.diff(t, axis=-1) != 1):
            return None
        return t[..., 0] * 2
    if len(t) % 2:
        return None
    pairs = t.reshape(len(t) // 2, lanes, 2, fp16_per_lane).reshape(-1, lanes, 2 * fp16_per_lane)
    if np.any(np.diff(pairs, axis=-1) != 1):
        return None
    return pairs[..., 0]


def _fit_writer(bounds, addrs):
    """Writer lane stride and per-loop strides that reproduce `addrs`, or None."""
    lw = int(addrs[0, 1] - addrs[0, 0])
    if np.any(addrs - addrs[:, :1] != np.arange(addrs.shape[1]) * lw):
        return None
    bases = addrs[:, 0]
    strides, n = [], 1
    for b in bounds:
        strides.append(int(bases[n] - bases[0]) if b > 1 else 0)
        n *= b
    grids = np.indices(bounds[::-1]).reshape(len(bounds), -1)[::-1]
    pred = bases[0] + sum(g * s for g, s in zip(grids, strides))
    if np.any(pred != bases):
        return None
    return lw, list(zip(bounds, strides))


def _candidates(rows, cols, pitch, lanes, lane_bytes):
    """(lane_dir, reader lane stride, lane step, base loops) for each lane direction."""
    per = lane_bytes // 2
    if rows % lanes == 0:        # lane c = row i0+c, `per` fp16 at j
        yield ("row", pitch, (1, 0),
               [(rows // lanes, lanes * pitch, (lanes, 0)),
                (cols // per, lane_bytes, (0, per))])
    if cols % (lanes * per) == 0:     # lane c = `per` fp16 at j0 + per*c of one row
        yield ("col", lane_bytes, (0, per),
               [(rows, pitch, (1, 0)),
                (cols // (lanes * per), lanes * lane_bytes, (0, lanes * per))])


def _splits(loop):
    b, s, (di, dj) = loop
    yield [loop]
    for d in _pow2_splits(b):
        yield [(d, s, (di, dj)), (b // d, s * d, (di * d, dj * d))]


def _search(rows, cols, pitch, mb, kb, out_bytes, fused_mul, agu_dims, lanes, lane_bytes):
    tmap = target_index(rows, cols, mb, kb)
    budget = agu_dims - (1 if fused_mul else 0)
    best = None
    for lane_dir, rd_lane, lane_step, base in _candidates(rows, cols, pitch, lanes,
                                                          lane_bytes):
        for sa in _splits(base[0]):
            for sb in _splits(base[1]):
                for perm in itertools.permutations(sa + sb):
                    loops = list(perm)
                    addrs = _lane_targets(_stream(loops, lane_step, lanes), tmap, out_bytes,
                                          lane_bytes // 2)
                    if addrs is None:
                        continue
                    rd = [(lp[0], lp[1]) for lp in loops]
                    wb = [lp[0] for lp in loops]
                    if out_bytes == 1:
                        if wb[0] % 2:
                            continue
                        wb[0] //= 2
                    fw = _fit_writer(wb, addrs)
                    if fw is None:
                        continue
                    wr_lane, wr = fw
                    if out_bytes == 1:
                        # The pair loop exists on the reader only; give the writer a
                        # bound-1 placeholder so the two lists stay index-aligned.
                        (b0, s0) = rd[0]
                        rd = [(2, s0), (b0 // 2, 2 * s0)] + rd[1:]
                        wr = [(1, 0)] + wr
                    # Each side merges on its own strides; a repeat peels the SAME outer
                    # loop off both.
                    reps, rep_rd, rep_wr = 1, 0, 0
                    while True:
                        rdims, wdims = _side_dims(rd), _side_dims(wr)
                        if len(rdims) <= budget and len(wdims) <= agu_dims:
                            break
                        if reps > 1 or len(rd) < 2:
                            rdims = None       # one repeat level is all the device runs
                            break
                        (reps, rep_rd), (_, rep_wr) = rd[-1], wr[-1]
                        rd, wr = rd[:-1], wr[:-1]
                    if rdims is None:
                        continue
                    cand = BlockedNest(lanes, lane_bytes, pitch, rd_lane, tuple(rdims),
                                       wr_lane, tuple(wdims), reps, rep_rd, rep_wr, lane_dir)
                    if best is None or cand.tasks < best.tasks:
                        best = cand
                        if best.tasks == 1:
                            return best
    return best


def verify(nest: BlockedNest, rows, cols, mb, kb, out_bytes):
    """Walk the descriptor exactly as the device runs it; compare with the index map.

    Independent of the search: it knows nothing of loops, splits or merges, only the
    descriptor. Every source element must land at the byte its target index says, and
    every target byte must be written exactly once.
    """
    lanes, per = nest.lanes, nest.lane_bytes // 2

    def walk(lane, loops, rep_stride):
        bounds = [b for b, _ in loops] or [1]
        strides = [s for _, s in loops] or [0]
        grids = np.indices(bounds[::-1]).reshape(len(bounds), -1)[::-1]
        base = sum(g * s for g, s in zip(grids, strides))
        base = np.concatenate([base + r * rep_stride for r in range(nest.reps)])
        return base[:, None] + np.arange(lanes) * lane               # (beats, lanes) bytes

    src = walk(nest.rd_lane, nest.rd, nest.rep_rd)                     # fp16 source bytes
    dst = walk(nest.wr_lane, nest.wr, nest.rep_wr)                     # target bytes
    # source byte -> (i, j) -> the 4 elements the lane carries
    i, jb = np.divmod(src, nest.pitch)
    if np.any(jb % 2) or np.any(jb // 2 + per > cols) or np.any(i >= rows):
        raise ValueError("blocked nest: the reader addresses outside the tile")
    elems = (i * cols + jb // 2)[..., None] + np.arange(per)  # (beats, lanes, per)
    if out_bytes == 1:
        if len(elems) != 2 * len(dst):
            raise ValueError("blocked nest: reader and writer beat counts disagree")
        elems = elems.reshape(len(dst), lanes, 2 * per)
    elif len(elems) != len(dst):
        raise ValueError("blocked nest: reader and writer beat counts disagree")
    per_lane = elems.shape[-1]
    dst_elem = (dst[..., None] // out_bytes) + np.arange(per_lane)
    want = target_index(rows, cols, mb, kb).reshape(-1)
    got = np.full(rows * cols, -1, dtype=np.int64)
    flat_src, flat_dst = elems.reshape(-1), dst_elem.reshape(-1)
    if np.any(flat_dst >= rows * cols) or np.any(flat_dst < 0):
        raise ValueError("blocked nest: the writer addresses outside the operand")
    got[flat_dst] = flat_src
    if len(np.unique(flat_dst)) != rows * cols or np.any(want[got] != np.arange(rows * cols)):
        bad = int(np.count_nonzero(want[got] != np.arange(rows * cols)))
        raise ValueError(f"blocked nest: {bad} of {rows * cols} elements misplaced")


def blocked_nest(rows: int, cols: int, pitch: int, mb: int, kb: int, out_bytes: int,
                 fused_mul: bool, *, lanes: int, lane_bytes: int = 8,
                 agu_dims: int = 3) -> BlockedNest:
    """The blocked pass for S [rows, cols] fp16 at `pitch` -> A-layout of S with (mb, kb)
    blocks, fp16 (out_bytes 2) or int8 (1), on a beat of `lanes` lanes of `lane_bytes`.
    Verified. Raises with the reason otherwise."""
    per = lane_bytes // 2
    if rows % mb or cols % kb:
        raise ValueError(
            f"a [{rows}, {cols}] tile does not tile {mb}x{kb} blocks; a partial block "
            f"leaves the operand's tail unwritten.")
    if kb % per and out_bytes == 2:
        raise ValueError(
            f"tileSize {kb}: an fp16 lane carries {per} consecutive elements, "
            f"so a {kb}-element atom splits it across two atoms. No {lane_bytes} B engine on "
            f"this cluster writes that layout from fp16.")
    if kb % per and kb * out_bytes < lane_bytes:
        raise ValueError(
            f"tileSize {kb}: an int8 atom of {kb} B is narrower than the 4-element fp16 "
            f"lane it is quantised from, so one input lane would straddle two atoms.")
    nest = _search(rows, cols, pitch, mb, kb, out_bytes, fused_mul, agu_dims, lanes, lane_bytes)
    if nest is None:
        raise ValueError(
            f"no blocked nest writes {'int8' if out_bytes == 1 else 'fp16'} blocks of "
            f"{mb}x{kb} from a [{rows}, {cols}] tile: every order either splits a lane "
            f"across blocks at addresses one lane stride cannot reach, or needs more than "
            f"one repeat level. A 2-D lane grid on the SIMD writer would lift the first.")
    verify(nest, rows, cols, mb, kb, out_bytes)
    return nest
