#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""crest_codec.py: reference codec of CREST (Centre-Relative Encoding with Split Tiers), the
multi-format weight format that CrestDecompressor decodes (see crest_decompressor.md). Lossless:
decode(encode(x)) == x for any payload of the supported modes.

The payload is the weight tensor as stored, in 64-byte beats. Each beat holds L lanes:
  mode 0 nib   128 lanes of 4 bits (INT4 / MXFP4 / NVFP4 nibbles)          symbol = the nibble
  mode 1 byte   64 lanes of 8 bits (INT8 / FP8)                            symbol = the byte
  mode 2 bf16   32 lanes of 16 bits (BF16)   symbol = the exponent byte; sign + mantissa travel raw

Symbols are ranked by a KEY, small keys first, with no table:
  keymap 0 int  v = signed (s - centre) mod 2^w;  key = zigzag(v) = 0, -1, +1, -2, ...
  keymap 1 sm   sign-magnitude: key = 2 * zigzag(|m| - centre) + sign
With K0 = 2^p - 2, a lane's p-bit PLANE code is its key if key < K0, K0 if the symbol is a tier-1
escape (key in [K0, K0 + 2^e1), entry = key - K0 in e1 bits), K0 + 1 if it is a tier-2 escape (the
raw symbol, w bits). Both escapes are flagged by the plane alone, so the decoder finds every entry
with two prefix popcounts and never parses variable-length codes.

Stream: word 0 = stream header ([1:0] mode, [2] keymap, [10:3] centre, [13:11] p, [16:14] e1),
then one record per group of G output beats (the last group may be shorter):
  coded  E escape words: bits [14:0] E, [15] 0, [31:16] n1 = tier-1 entries; from bit 32 the n1
         tier-1 entries, then the tier-2 entries, both in lane order, zero padded. Then the plane:
         beat j, lane i at plane bit j * L * lb + i * lb, lb = p (+ 8 raw bits in mode bf16, after
         the code), ceil(gb * L * lb / 512) words.
  raw    one header word with only bit 15 set, then the gb beats unchanged.
A group is raw when it would need more than EMAX escape words or would not be shorter.

    python3 crest_codec.py selftest                        synthetic cases + cached real tensors
    python3 crest_codec.py vectors --case NAME --out DIR   in.hex / out.hex / csr.txt / meta.json
    python3 crest_codec.py cases                           list the cases

The real_* cases read published-checkpoint tensors from the directory that the environment
variable CREST_WEIGHT_CACHE names (the format study's cache); without it only the synthetic
cases exist.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

W = 512
G = 64            # output beats per group
EMAX = 12         # escape words a group may use (the decoder's escape buffer)
HB = 32           # group header bits
MODES = {"nib": 0, "byte": 1, "bf16": 2}
LANES = {0: 128, 1: 64, 2: 32}
SB = {0: 4, 1: 8, 2: 8}
EXTRA = {0: 0, 1: 0, 2: 8}
P_RANGE = {0: (2, 3), 1: (2, 7), 2: (2, 7)}


# ---- symbols and keys ---------------------------------------------------------------------------
def to_symbols(payload, mode):
    """(symbols int64 [beats * L], raw bytes or None) of a payload (uint8, whole beats)."""
    b = np.frombuffer(bytes(payload), np.uint8)
    if mode == 0:
        s = np.empty(b.size * 2, np.int64)
        s[0::2], s[1::2] = b & 15, b >> 4
        return s, None
    if mode == 1:
        return b.astype(np.int64), None
    u = b.view("<u2").astype(np.int64)
    return (u >> 7) & 0xFF, ((u >> 8) & 0x80) | (u & 0x7F)


def from_symbols(s, raw, mode):
    s = np.asarray(s, np.int64)
    if mode == 0:
        return (s[0::2] | (s[1::2] << 4)).astype(np.uint8)
    if mode == 1:
        return s.astype(np.uint8)
    u = ((raw & 0x80) << 8) | (s << 7) | (raw & 0x7F)
    return u.astype("<u2").view(np.uint8)


def zig(v):
    return np.where(v >= 0, 2 * v, -2 * v - 1)


def unzig(k):
    return np.where(k & 1, -((k + 1) >> 1), k >> 1)


def keys(s, keymap, centre, w):
    if keymap == 0:
        half = 1 << (w - 1)
        return zig(((s - centre + half) % (1 << w)) - half)
    half = 1 << (w - 1)
    return 2 * zig((s % half) - centre) + (s >= half)


def symbol_of_key(k, keymap, centre, w):
    if keymap == 0:
        return (centre + unzig(k)) % (1 << w)
    half = 1 << (w - 1)
    return (k & 1) * half + centre + unzig(k >> 1)


# ---- bit packing --------------------------------------------------------------------------------
def pack_fields(values, widths):
    """Little-endian bit stream of fields (value i in widths[i] bits, LSB first)."""
    values = np.asarray(values, np.int64)
    widths = np.asarray(widths, np.int64)
    pos = np.concatenate([[0], np.cumsum(widths)[:-1]]) if widths.size else np.zeros(0, np.int64)
    total = int(widths.sum())
    bits = np.zeros(total, np.uint8)
    for b in range(int(widths.max()) if widths.size else 0):
        m = widths > b
        bits[pos[m] + b] = (values[m] >> b) & 1
    return bits


def words_of(bits):
    bits = np.concatenate([bits, np.zeros((-bits.size) % W, np.uint8)])
    return np.packbits(bits.reshape(-1, 8), axis=1, bitorder="little").reshape(-1)


def bits_of(words):
    return np.unpackbits(np.asarray(words, np.uint8), bitorder="little")


def field(bits, pos, width):
    return int(sum(int(bits[pos + i]) << i for i in range(width)))


# ---- parameter search ---------------------------------------------------------------------------
def _cost(h, gb, kord, sb, extra, L, p, e1, emax):
    """Words of a candidate, from per-group symbol histograms h [ng, 2^sb] taken in key order."""
    hk = h[:, kord]
    cum = np.concatenate([np.zeros((h.shape[0], 1), np.int64), np.cumsum(hk, 1)], 1)
    space = hk.shape[1]
    k0 = 2 ** p - 2
    k1 = min(k0 + 2 ** e1, space)
    esc = e1 * (cum[:, k1] - cum[:, k0]) + sb * (cum[:, space] - cum[:, k1])
    ne = -(-(HB + esc) // W)
    npl = -(-gb * L * (p + extra) // W)
    coded = (ne <= emax) & (ne + npl <= gb)
    return int(np.where(coded, ne + npl, 1 + gb).sum())


def choose(s, mode, g=G, emax=EMAX):
    """(keymap, centre, p, e1) with the fewest words for this payload."""
    sb, extra, L = SB[mode], EXTRA[mode], LANES[mode]
    beats = s.size // L
    ng = -(-beats // g)
    gid = np.repeat(np.arange(ng), g * L)[:s.size]
    h = np.bincount(gid * (1 << sb) + s, minlength=ng * (1 << sb)).reshape(ng, 1 << sb)
    gb = np.full(ng, g)
    gb[-1] = beats - (ng - 1) * g
    allsym = np.arange(1 << sb)
    best = None
    maps = [(0, c) for c in range(1 << sb)]
    if mode != 2:
        maps += [(1, c) for c in range(1 << (sb - 1))]
    for km, c in maps:
        kord = np.argsort(keys(allsym, km, c, sb), kind="stable")
        for p in range(P_RANGE[mode][0], P_RANGE[mode][1] + 1):
            for e1 in range(1, (3 if mode == 0 else 7) + 1):
                n = _cost(h, gb, kord, sb, extra, L, p, e1, emax)
                if best is None or n < best[0]:
                    best = (n, km, c, p, e1)
    return best[1:]


# ---- encoder ------------------------------------------------------------------------------------
def encode(payload, mode, params=None, g=G, emax=EMAX):
    """payload (uint8, whole 64-byte beats) -> (words uint8 [64 * n], info)."""
    mode = MODES.get(mode, mode)
    payload = np.frombuffer(bytes(payload), np.uint8)
    if payload.size % 64:
        raise ValueError("the payload must be whole 64-byte beats")
    s, raw = to_symbols(payload, mode)
    sb, extra, L = SB[mode], EXTRA[mode], LANES[mode]
    km, c, p, e1 = params if params is not None else choose(s, mode, g, emax)
    lo, hi = P_RANGE[mode]
    if not (lo <= p <= hi and 1 <= e1 <= (3 if mode == 0 else 7)):
        raise ValueError(f"p = {p}, e1 = {e1} out of range for mode {mode}")
    k = keys(s, km, c, sb)
    k0 = 2 ** p - 2
    t1 = (k >= k0) & (k < k0 + 2 ** e1)
    t2 = k >= k0 + 2 ** e1
    code = np.where(t2, k0 + 1, np.where(t1, k0, k))
    beats = payload.size // 64
    hdr = pack_fields([mode | km << 2 | c << 3 | p << 11 | e1 << 14], [32])
    out = [words_of(hdr)]
    info = dict(beats=beats, groups=0, raw_groups=0, t1=0, t2=0, params=(km, c, p, e1), words=1)
    lb = p + extra
    for g0 in range(0, beats, g):
        gb = min(g, beats - g0)
        sl = slice(g0 * L, (g0 + gb) * L)
        n1, n2 = int(t1[sl].sum()), int(t2[sl].sum())
        esc_bits = HB + n1 * e1 + n2 * sb
        ne = -(-esc_bits // W)
        npl = -(-gb * L * lb // W)
        info["groups"] += 1
        if ne > emax or ne + npl > gb:
            out += [words_of(pack_fields([1 << 15], [32])), payload[g0 * 64:(g0 + gb) * 64]]
            info["raw_groups"] += 1
            info["words"] += 1 + gb
            continue
        ev = np.concatenate([[ne | n1 << 16], (k[sl][t1[sl]] - k0), s[sl][t2[sl]]])
        ew = np.concatenate([[HB], np.full(n1, e1), np.full(n2, sb)])
        out.append(words_of(pack_fields(ev, ew)))
        if extra:
            pv = np.stack([code[sl], raw[sl]], 1).reshape(-1)
            pw = np.tile([p, extra], gb * L)
        else:
            pv, pw = code[sl], np.full(gb * L, p)
        out.append(words_of(pack_fields(pv, pw)))
        info["t1"] += n1
        info["t2"] += n2
        info["words"] += ne + npl
    words = np.concatenate(out)
    info["ratio"] = beats / info["words"]
    return words, info


# ---- decoder: what the hardware does, beat by beat ----------------------------------------------
def decode(words, beats, g=G):
    words = np.asarray(words, np.uint8).reshape(-1, 64)
    sh = field(bits_of(words[0]), 0, 32)
    mode, km, c, p, e1 = sh & 3, (sh >> 2) & 1, (sh >> 3) & 0xFF, (sh >> 11) & 7, (sh >> 14) & 7
    sb, extra, L = SB[mode], EXTRA[mode], LANES[mode]
    k0, lb = 2 ** p - 2, p + extra
    wi, res = 1, []
    for g0 in range(0, beats, g):
        gb = min(g, beats - g0)
        hdr = field(bits_of(words[wi]), 0, 32)
        if hdr >> 15 & 1:
            res.append(words[wi + 1:wi + 1 + gb].reshape(-1))
            wi += 1 + gb
            continue
        ne, n1 = hdr & 0x7FFF, hdr >> 16
        eb = bits_of(words[wi:wi + ne])
        wi += ne
        npl = -(-gb * L * lb // W)
        pb = bits_of(words[wi:wi + npl])
        wi += npl
        q1, q2 = HB, HB + n1 * e1                 # tier-1 and tier-2 entry pointers (bits)
        for j in range(gb):
            base = j * L * lb
            fld = np.array([field(pb, base + i * lb, lb) for i in range(L)], np.int64)
            cd, rw = fld & ((1 << p) - 1), fld >> p
            f1, f2 = cd == k0, cd == k0 + 1
            i1 = np.cumsum(f1) - f1                # prefix popcounts: each lane's entry number
            i2 = np.cumsum(f2) - f2
            key = cd.copy()
            for lane in np.nonzero(f1)[0]:
                key[lane] = k0 + field(eb, q1 + e1 * i1[lane], e1)
            sym = symbol_of_key(key, km, c, sb)
            for lane in np.nonzero(f2)[0]:
                sym[lane] = field(eb, q2 + sb * i2[lane], sb)
            q1 += e1 * int(f1.sum())
            q2 += sb * int(f2.sum())
            res.append(from_symbols(sym, rw if extra else None, mode))
    return np.concatenate(res).astype(np.uint8)


# ---- cases --------------------------------------------------------------------------------------
REAL = os.environ.get("CREST_WEIGHT_CACHE", "")   # directory of cached checkpoint tensors, or unset
REAL_CASES = {   # case: (mode, cached tensor, beats)
    "real_int4_gptq": ("nib", "Qwen__Qwen2.5-7B-Instruct-GPTQ-Int4/"
                       "model.layers.14.mlp.down_proj.qweight.bin", 4096),
    "real_int4_awq": ("nib", "Qwen__Qwen2.5-7B-Instruct-AWQ/"
                      "model.layers.14.mlp.down_proj.qweight.bin", 1024),
    "real_mxfp4": ("nib", "openai__gpt-oss-20b/"
                   "model.layers.12.mlp.experts.down_proj_blocks.bin", 1024),
    "real_fp8_qwen3": ("byte", "Qwen__Qwen3-8B-FP8/"
                       "model.layers.18.mlp.down_proj.weight.bin", 4096),
    "real_fp8_dsv3": ("byte", "deepseek-ai__DeepSeek-V3/"
                      "model.layers.30.mlp.experts.0.down_proj.weight.bin", 4096),
    "real_int8": ("byte", "RedHatAI__Qwen2.5-7B-Instruct-quantized.w8a8/"
                  "model.layers.14.mlp.down_proj.weight.bin", 4096),
    "real_bf16": ("bf16", "Qwen__Qwen2.5-7B-Instruct/"
                  "model.layers.14.mlp.down_proj.weight.bin", 4096),
}


def synth(name, rng):
    """(payload, mode, params or None) of a synthetic edge case."""
    def gauss_int(n, w, sigma, centre=0):
        v = np.clip(np.rint(rng.normal(0, sigma, n)), -(1 << (w - 1)) + 1, (1 << (w - 1)) - 1)
        return ((v + centre) % (1 << w)).astype(np.int64)
    if name == "nib_small":                       # INT4 around 0, one partial group
        s = gauss_int(128 * 70, 4, 1.4)
        return from_symbols(s, None, 0), "nib", (0, 0, 3, 3)
    if name == "nib_zp8_dense":         # zero point 8, one beat all tier-1, one all tier-2
        s = gauss_int(128 * 64, 4, 1.6, 8)
        s[128 * 5:128 * 6] = 8 + 3                 # key 6 = tier-1 at p = 3
        s[128 * 9:128 * 10] = 0                    # far from 8: tier 2
        return from_symbols(s, None, 0), "nib", (0, 8, 3, 2)
    if name == "byte_int8":
        s = gauss_int(64 * 200, 8, 9.0)
        return from_symbols(s, None, 1), "byte", None
    if name == "byte_sm_straddle":      # sign-magnitude: both tiers, 8-bit entries across
        n = 64 * 64                      # words, then a uniform-noise group that goes raw
        mag = np.clip(np.rint(rng.normal(90, 8, n)), 0, 127).astype(np.int64)
        s = np.concatenate([(rng.integers(0, 2, n) << 7) | mag, rng.integers(0, 256, n)])
        return from_symbols(s, None, 1), "byte", (1, 90, 6, 4)
    if name == "nib_sm":                           # E2M1-like sign-magnitude nibbles around |m| = 2
        mag = np.clip(np.rint(np.abs(rng.normal(2, 1.3, 128 * 64))), 0, 7).astype(np.int64)
        s = (rng.integers(0, 2, mag.size) << 3) | mag
        return from_symbols(s, None, 0), "nib", (1, 2, 3, 2)
    if name == "nib_esc_full":                     # an escape region of ~11 words (EMAX is 12)
        s = gauss_int(128 * 64, 4, 1.25)
        idx = rng.choice(s.size, 1350, replace=False)
        s[idx] = 8                                 # -8: key 15, tier 2 at p = 3, e1 = 3
        return from_symbols(s, None, 0), "nib", (0, 0, 3, 3)
    if name == "bf16_dense":                       # one beat with every lane tier 2, one all tier 1
        x = rng.normal(0, 0.02, 32 * 64).astype(np.float32)
        x[32 * 3:32 * 4] = 1e30
        x[32 * 7:32 * 8] = 0.02 * 2 ** -3
        u = (x.view(np.uint32) >> 16).astype("<u2")
        return u.view(np.uint8), "bf16", (0, 121, 3, 2)
    if name == "bf16_gauss":
        x = rng.normal(0, 0.02, 32 * 160).astype(np.float32)
        u = (x.view(np.uint32) >> 16).astype("<u2")
        return u.view(np.uint8), "bf16", None
    if name == "raw_mix":                          # coded, raw (uniform noise), coded
        s = np.concatenate([gauss_int(128 * 64, 4, 1.2), rng.integers(0, 16, 128 * 64),
                            gauss_int(128 * 64, 4, 1.2)])
        return from_symbols(s, None, 0), "nib", (0, 0, 3, 3)
    raise KeyError(name)


SYNTH = ["nib_small", "nib_zp8_dense", "nib_sm", "nib_esc_full", "byte_int8", "byte_sm_straddle",
         "bf16_gauss", "bf16_dense", "raw_mix"]


def case(name, rng):
    if name in REAL_CASES:
        mode, f, beats = REAL_CASES[name]
        if not REAL:
            raise FileNotFoundError(f"{name}: CREST_WEIGHT_CACHE is not set")
        raw = np.fromfile(os.path.join(REAL, f), np.uint8)
        if raw.size < beats * 64:
            raise FileNotFoundError(f)
        return raw[:beats * 64], mode, None
    return synth(name, rng)


def available():
    if not REAL:
        return list(SYNTH)
    return SYNTH + [n for n, (_, f, _) in REAL_CASES.items() if glob.glob(os.path.join(REAL, f))]


def _hex(words):
    return [bytes(w[::-1]).hex() for w in np.asarray(words, np.uint8).reshape(-1, 64)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "vectors", "cases"])
    ap.add_argument("--case", default="all")
    ap.add_argument("--out", default="crest_vectors")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = available() if a.case == "all" else [a.case]
    if a.cmd == "cases":
        print("\n".join(names))
        return
    for n in names:
        rng = np.random.default_rng(a.seed)
        payload, mode, params = case(n, rng)
        words, info = encode(payload, mode, params)
        beats = len(payload) // 64
        ok = np.array_equal(decode(words, beats), np.frombuffer(bytes(payload), np.uint8))
        if a.cmd == "selftest":
            status = "ok  " if ok else "FAIL"
            print(f"{n:18s} {status} {beats:5d} beats <- {words.size // 64:5d} words "
                  f"({info['ratio']:.3f}x) params {info['params']}, {info['groups']} groups, "
                  f"{info['raw_groups']} raw, tier1 {info['t1']}, tier2 {info['t2']}")
            if not ok:
                sys.exit(1)
        else:
            if not ok:
                raise AssertionError(f"{n}: the reference does not round-trip")
            out = a.out if a.case != "all" else os.path.join(a.out, n)
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, "in.hex"), "w").write("\n".join(_hex(words)) + "\n")
            open(os.path.join(out, "out.hex"), "w").write("\n".join(_hex(payload)) + "\n")
            open(os.path.join(out, "csr.txt"), "w").write(f"{beats}\n")
            json.dump(dict(info, case=n, in_words=words.size // 64, out_beats=beats),
                      open(os.path.join(out, "meta.json"), "w"), indent=1, default=int)
            print(f"{n}: {words.size // 64} words in -> {beats} beats out")


if __name__ == "__main__":
    main()
