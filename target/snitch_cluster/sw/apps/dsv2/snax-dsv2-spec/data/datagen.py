#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-spec: snax-dsv2-layer's generator, run on this app's params.hjson (four
# tokens per pass).

import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "../../snax-dsv2-layer/data/datagen.py"), run_name="__main__")
