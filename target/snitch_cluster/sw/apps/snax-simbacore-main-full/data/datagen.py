#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

import pathlib
import sys
import os
import importlib.util

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../snax-simbacore-main/data"))

# Dynamically load DataGenerator from the other datagen.py to avoid name clash
_this_dir = os.path.dirname(__file__)
_other_datagen_path = os.path.abspath(os.path.join(_this_dir, "../../snax-simbacore-main/data/datagen.py"))
_spec = importlib.util.spec_from_file_location("snax_simbacore_main_datagen", _other_datagen_path)
_snax_simbacore_main_datagen = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_snax_simbacore_main_datagen)

DataGenerator = _snax_simbacore_main_datagen.DataGenerator

from datagen_cli import main as datagen_cli_main  # type: ignore[import]


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
