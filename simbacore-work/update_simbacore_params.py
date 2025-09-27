"""
Parses SimbaCore.sv to extract port widths and updates snax_simbacore_shell_wrapper.sv parameter widths accordingly.
AI generated.
"""

from __future__ import annotations
import os
import re
import math
from typing import Dict, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
simbacore_path = os.path.abspath(os.path.join(script_dir, "../../chisel-ssm/generated/SimbaCore/SimbaCore.sv"))
wrapper_path = os.path.abspath(os.path.join(script_dir, "../hw/snax_simbacore/snax_simbacore_shell_wrapper.sv"))

PORT_MAPPING: Dict[str, str] = {
    "OSCoreInAWidth": "io_osCore_in_a_bits",
    "OSCoreInBWidth": "io_osCore_in_b_bits",
    "OSCoreOutDWidth": "io_osCore_out_d_bits",
    "SwitchCoreInMatmulWidth": "io_switchCore_in_matmul_bits",
    "SwitchCoreInWeightWidth": "io_switchCore_in_weight_bits",
    "SwitchCoreInBiasWidth": "io_switchCore_in_bias_bits",
    "SwitchCoreInMatmulWeightWidth": "io_switchCore_in_matmulWeight_bits",
    "SUCoreInAWidth": "io_suCore_in_A_bits",
    "SUCoreInBCWidth": "io_suCore_in_BC_bits",
    "SUCoreInDWidth": "io_suCore_in_D_bits",
    "SUCoreInXWidth": "io_suCore_in_x_bits",
    "SUCoreInZWidth": "io_suCore_in_z_bits",
    "SUCoreOutYWidth": "io_suCore_out_y_bits",
    "ISCoreInAWidth": "io_isCore_in_a_bits",
    "ISCoreInBWidth": "io_isCore_in_b_bits",
    "ISCoreInCWidth": "io_isCore_in_c_bits",
    "ISCoreOutDWidth": "io_isCore_out_d_bits",
}
BANK_WIDTH = 64


def _strip_comment(line: str) -> str:
    return line.split("//", 1)[0]


def parse_simbacore(simbacore_path: str):
    """Parse SimbaCore.sv and return a mapping port_name -> width."""
    MODULE_RE = re.compile(r"module\s+SimbaCore\s*\((?P<ports>.*?)\);", re.DOTALL)
    WIDTH_TOKEN_RE = re.compile(r"\[(\d+)\s*:\s*0\]")

    with open(simbacore_path, "r") as f:
        content = f.read()

    m = MODULE_RE.search(content)
    if not m:
        raise ValueError("SimbaCore module not found in file")

    ports_region = m.group("ports")
    body_region = content[m.end() :]  # Module body after port list

    port_widths = {}
    last_width = None
    last_direction = None

    for raw_line in ports_region.split("\n"):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue

        if line.endswith(","):
            line = line[:-1]

        tokens = line.split()
        if not tokens:
            continue

        # Direction
        if tokens[0] in ("input", "output"):
            last_direction = tokens[0]
            tokens = tokens[1:]
        if last_direction is None:
            # Line without direction before any direction established; skip
            continue

        # Optional 'logic'
        if tokens and tokens[0] == "logic":
            tokens = tokens[1:]

        # Optional width token
        width = last_width
        if tokens and WIDTH_TOKEN_RE.match(tokens[0]):
            wmatch = WIDTH_TOKEN_RE.match(tokens[0])
            if wmatch:
                width = int(wmatch.group(1)) + 1
                last_width = width
                tokens = tokens[1:]

        # Remaining tokens combined may include multiple comma separated names
        names_part = " ".join(tokens)
        for name in [n.strip() for n in names_part.split(",") if n.strip()]:
            name = name.rstrip(",")
            if name.endswith("_bits"):
                port_widths[name] = width if width is not None else None

    # Infer widths for any unresolved (None) or width==None ports by body scan
    for pname, pwidth in list(port_widths.items()):
        if pwidth is not None and pwidth != 1:
            continue  # Already have a non-trivial width
        # Search slices name[NN:0]
        slice_pat = re.compile(rf"{re.escape(pname)}\[(\d+)\s*:\s*0\]")
        candidates = [int(mm.group(1)) + 1 for mm in slice_pat.finditer(body_region)]
        if not candidates:
            # Search individual indices name[NN]
            idx_pat = re.compile(rf"{re.escape(pname)}\[(\d+)\]")
            candidates = [int(mm.group(1)) + 1 for mm in idx_pat.finditer(body_region)]
        inferred = max(candidates) if candidates else (pwidth if pwidth is not None else 1)
        port_widths[pname] = inferred

    return port_widths


def update_wrapper_file(wrapper_path: str, port_widths: Dict[str, int], mapping: Dict[str, str]) -> None:

    with open(wrapper_path, "r") as f:
        content = f.read()

    for param, port in mapping.items():
        if port not in port_widths:
            print(f"Warning: Port '{port}' not found in SimbaCore.sv for parameter '{param}'")
            continue

        width = port_widths[port]
        streamer_width = math.ceil(width / BANK_WIDTH)
        pattern = re.compile(rf"^(\s*)parameter\s+int\s+unsigned\s+{re.escape(param)}\s*=\s*\d+.*$", re.MULTILINE)

        def _repl(match):
            indent = match.group(1)  # Preserve original indentation
            comma = "," if "," in match.group(0) else ""
            return f"{indent}parameter int unsigned {param} = {width}{comma} // {streamer_width}"

        new_content, nsubs = pattern.subn(_repl, content)
        if nsubs == 0:
            print(f"Note: Parameter '{param}' not found for substitution in wrapper")
        content = new_content

    with open(wrapper_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    port_widths = parse_simbacore(simbacore_path)
    if not port_widths:
        print("No _bits ports found.")
    else:
        update_wrapper_file(wrapper_path, port_widths, PORT_MAPPING)
        print("Wrapper update complete.")
