"""
Parses SimbaCore.sv to extract port widths and updates snax_simbacore_shell_wrapper.sv parameter widths accordingly.
"""

from __future__ import annotations
import os
import re
import math
from pathlib import Path
from typing import Dict

# Use pathlib for cleaner path handling
SCRIPT_DIR = Path(__file__).parent
SIMBACORE_PATH = SCRIPT_DIR / "../../chisel-ssm/generated/SimbaCore/SimbaCore.sv"
WRAPPER_PATH = SCRIPT_DIR / "../hw/snax_simbacore/snax_simbacore_shell_wrapper.sv"

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
    "SwitchCoreOutWidth": "io_switchCore_out_x_bits",
    "SUCoreInXWidth": "io_suCore_in_x_bits",
    "SUCoreInZWidth": "io_suCore_in_z_bits",
    "SUCoreOutYWidth": "io_suCore_out_y_bits",
    "ISCoreInAWidth": "io_isCore_in_a_bits",
    "ISCoreInBWidth": "io_isCore_in_b_bits",
    "ISCoreInCWidth": "io_isCore_in_c_bits",
    "ISCoreOutDWidth": "io_isCore_out_d_bits",
    "ModeWidth": "io_config_bits_mode",
}

BANK_WIDTH = 64


def _strip_comment(line: str) -> str:
    """Remove comment from line, returning everything before '//'."""
    return line.split("//", 1)[0]


def _extract_width_from_slice(token: str) -> int | None:
    """Extract width from SystemVerilog slice notation [N:0]."""
    match = re.match(r"\[(\d+)\s*:\s*0\]", token)
    return int(match.group(1)) + 1 if match else None


def parse_simbacore(simbacore_path: Path) -> Dict[str, int]:
    """Parse SimbaCore.sv and return a mapping port_name -> width."""
    MODULE_RE = re.compile(r"module\s+SimbaCore\s*\((?P<ports>.*?)\);", re.DOTALL)

    try:
        content = simbacore_path.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f"SimbaCore file not found: {simbacore_path}")

    module_match = MODULE_RE.search(content)
    if not module_match:
        raise ValueError("SimbaCore module not found in file")

    ports_region = module_match.group("ports")
    body_region = content[module_match.end() :]  # Module body after port list

    port_widths = {}
    last_width = None
    last_direction = None

    for raw_line in ports_region.split("\n"):
        line = _strip_comment(raw_line).strip().rstrip(",")
        if not line:
            continue

        tokens = line.split()
        if not tokens:
            continue

        # Handle direction (input/output)
        if tokens[0] in ("input", "output"):
            last_direction = tokens[0]
            tokens = tokens[1:]

        if last_direction is None:
            continue  # Skip lines without direction

        # Skip 'logic' keyword
        if tokens and tokens[0] == "logic":
            tokens = tokens[1:]

        # Extract width from slice notation
        width = last_width
        if tokens:
            extracted_width = _extract_width_from_slice(tokens[0])
            if extracted_width:
                width = extracted_width
                last_width = width
                tokens = tokens[1:]

        # Process port names (handle comma-separated names)
        names_part = " ".join(tokens)
        for name in (n.strip().rstrip(",") for n in names_part.split(",") if n.strip()):
            if "_bits" in name:
                port_widths[name] = width

    # Infer widths from body for unresolved ports
    for port_name, port_width in list(port_widths.items()):
        if port_width is not None and port_width != 1:
            continue  # Already have a non-trivial width

        inferred_width = _infer_width_from_body(port_name, body_region, port_width)
        port_widths[port_name] = inferred_width

    return port_widths


def _infer_width_from_body(port_name: str, body: str, default_width: int | None) -> int:
    """Infer port width by scanning the module body for slice patterns."""
    escaped_name = re.escape(port_name)

    # Look for slice patterns [N:0]
    slice_pattern = rf"{escaped_name}\[(\d+)\s*:\s*0\]"
    slice_matches = [int(m.group(1)) + 1 for m in re.finditer(slice_pattern, body)]

    if slice_matches:
        return max(slice_matches)

    # Look for individual index patterns [N]
    index_pattern = rf"{escaped_name}\[(\d+)\]"
    index_matches = [int(m.group(1)) + 1 for m in re.finditer(index_pattern, body)]

    if index_matches:
        return max(index_matches)

    return default_width if default_width is not None else 1


def update_wrapper_file(wrapper_path: Path, port_widths: Dict[str, int], mapping: Dict[str, str]) -> None:
    """Update wrapper file parameters based on extracted port widths."""
    try:
        content = wrapper_path.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f"Wrapper file not found: {wrapper_path}")

    for param, port in mapping.items():
        if port not in port_widths:
            print(f"Warning: Port '{port}' not found in SimbaCore.sv for parameter '{param}'")
            continue

        width = port_widths[port]
        streamer_width = math.ceil(width / BANK_WIDTH)

        # Create pattern to match parameter declaration
        pattern = re.compile(rf"^(\s*)parameter\s+int\s+unsigned\s+{re.escape(param)}\s*=\s*\d+.*$", re.MULTILINE)

        def _replacement(match):
            indent = match.group(1)  # Preserve original indentation
            comma = "," if "," in match.group(0) else ""
            return f"{indent}parameter int unsigned {param} = {width}{comma} // {streamer_width}"

        new_content, num_substitutions = pattern.subn(_replacement, content)
        if num_substitutions == 0:
            print(f"Note: Parameter '{param}' not found for substitution in wrapper")
        content = new_content

    wrapper_path.write_text(content)


if __name__ == "__main__":
    try:
        port_widths = parse_simbacore(SIMBACORE_PATH)
        if not port_widths:
            print("No _bits ports found.")

        update_wrapper_file(WRAPPER_PATH, port_widths, PORT_MAPPING)
        print("Wrapper update complete.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
