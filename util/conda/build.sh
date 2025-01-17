#!/bin/bash
# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

PIP_NO_INDEX= pip install hjson #Unset PIP_NO_INDEX to allow pypi installation
make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson
make -C target/snitch_cluster bin/snitch_cluster.vlt CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson -j
make -C target/snitch_cluster sw CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson -j
