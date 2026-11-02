# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Check default compatibility, invalid bank groups and the real XDMA wrapper."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "snaxgen", ROOT / "util/snaxgen/snaxgen.py")
snaxgen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snaxgen)


class WideTcdmConfigTest(unittest.TestCase):
    def config(self, wide=None):
        cfg = {
            "name": "wide_config_test", "data_width": 64,
            "dma_data_width": 512, "addr_width": 48,
            "tcdm": {"size": 128, "banks": 32},
            "max_mem_size_kiB": 4096,
        }
        if wide is not None:
            cfg["xdma_use_wide_tcdm"] = wide
        return cfg

    def test_default_and_explicit_false_keep_narrow_ports(self):
        for flag in (None, False):
            with self.subTest(flag=flag):
                self.assertEqual(
                    snaxgen.get_xdma_tcdm_data_width(self.config(flag)), 64)

    def test_wide_word_spans_existing_banks(self):
        cfg = self.config(True)
        self.assertEqual(snaxgen.get_xdma_tcdm_data_width(cfg), 512)
        self.assertEqual(cfg["data_width"], 64)

    def test_reject_incomplete_or_non_power_of_two_groups(self):
        for width, banks in ((384, 32), (512, 33), (32, 32), (520, 32)):
            cfg = self.config(True)
            cfg["dma_data_width"] = width
            cfg["tcdm"]["banks"] = banks
            with self.subTest(width=width, banks=banks):
                with self.assertRaises(ValueError):
                    snaxgen.get_xdma_tcdm_data_width(cfg)

    def test_wrapper_matches_local_and_metadata_widths(self):
        template = snaxgen.get_template(
            str(ROOT / "hw/templates/snax_xdma_wrapper.sv.tpl"))
        for wide, width, ports in ((False, 64, 16), (True, 512, 2)):
            with self.subTest(wide=wide):
                rtl = template.render_unicode(cfg=self.config(wide))
                self.assertRegex(rtl, rf"TCDMDataWidth\s*=\s*{width}")
                self.assertRegex(rtl, rf"TCDMNumPorts\s*=\s*{ports}")
                self.assertRegex(rtl, r"WordlineWidth\s*=\s*" + str(width) + ";")
                # Remote AXI control and the CPU/bank width remain narrow.
                self.assertRegex(rtl, r"AxiNarrowDataWidth\s*=\s*32'd64;")
                self.assertEqual(
                    ".q.user.tcdm_priority" in rtl, not wide)
                self.assertIn(".WordlineWidth            (WordlineWidth)", rtl)


if __name__ == "__main__":
    unittest.main()
