#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>


import re
import sys
import os
import inspect
import random
from abc import ABC

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from snax_utils import align_wide_addr  # noqa E402

DATA_OUT_DIR = os.path.join(os.path.dirname(__file__), "generated")
NUM_LOOPS = 4  # NOTE this must match the hjson config
BANKWIDTH = 64
BANK_BYTES = BANKWIDTH // 8
NB_TEST_SAMPLES = 25


class DataGeneratorBase(ABC):
    """Abstract base class to centralize data generation logic."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data: list[str] = []

    def run(self):
        pass

    def emit_header_file(self):
        self.data.append("#include <stdint.h>\n")
        self.format("uint32_t", "nb_test_samples", NB_TEST_SAMPLES)
        self.run()
        return "\n".join(self.data)

    def format(self, type: str, var_name: str, value: int):
        self.data.append(format_scalar_definition(type, var_name, value))

    def format_int(self, value: int):
        """Format `value` as integer. The name will be the variable name in the caller's scope."""
        callers_local_vars = (
            inspect.currentframe().f_back.f_locals.items()  # pyright: ignore[reportOptionalMemberAccess]
        )
        variable_name = next((name for name, val in callers_local_vars if val is value), None)
        assert variable_name is not None, "Variable name not found"
        self.format("uint32_t", variable_name, value)

    def format_params(self):
        """Takes all parameters from the kwargs and formats them as integers."""
        for key, value in self.kwargs.items():
            self.format("uint32_t", key, value)

    def format_vector(self, type: str, var_name: str, value: list[int]):
        self.data.append(format_vector_definition(type, var_name, value))

    def read_and_format_vector(self, type: str, tensor_name: str):
        try:
            tensor_data = self._read_data_int(tensor_name)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Error loading test data for tensor {tensor_name}: {e}. Did you run the scala data generator and is the data directory correct?"
            )
        self.format_vector(type, tensor_name, tensor_data)

    def format_temporal_bounds_strides(self, streamer_name: str, mode: int, bounds: list[int], strides: list[int]):
        """Format temporal bounds and strides for a streamer by automatically naming the variables and adding defaults.
        bounds are from inner to outer loop.

        The naming is, e.g., M0_R3_tb or M2_W1_ts. tb = temporal bounds, ts = temporal strides.
        """
        # Extend with defaults
        bounds = bounds + [1] * (NUM_LOOPS - len(bounds))
        strides = strides + [0] * (NUM_LOOPS - len(strides))
        # Save each bound/stride as a separate variable
        # for i in range(NUM_LOOPS):
        #     self.format("uint32_t", f"M{mode}_{streamer_name}_tb{i}", bounds[i])
        #     self.format("uint32_t", f"M{mode}_{streamer_name}_ts{i}", strides[i])
        # Group values into array
        self.data += [f"int32_t M{mode}_{streamer_name}_tb[] = {{{', '.join(map(str, bounds))}}};"]
        self.data += [f"int32_t M{mode}_{streamer_name}_ts[] = {{{', '.join(map(str, strides))}}};"]

    def format_spatial_stride(self, streamer_name: str, mode: int, stride: int):
        # self.format("uint32_t", f"M{mode}_{streamer_name}_ss0", stride)
        self.data += [f"int32_t M{mode}_{streamer_name}_ss[] = {{{stride}}};"]

    def _read_data_int(self, tensor_name: str):
        """Read a vec from a file."""
        with open(os.path.join(DATA_OUT_DIR, tensor_name + ".bin"), "r") as f:
            lines = f.readlines()
        data_lines = [line.strip() for line in lines if not line.startswith("#")]
        return [int(x) for x in data_lines]

    def format_test_samples(self, tensor_name: str, tensor_size: int, nb_test_samples: int):
        """Format variables used to test only a subset of the output."""
        self.format_vector(
            "int32_t",
            f"test_samples_{tensor_name}",
            [random.randint(0, tensor_size - 1) for _ in range(nb_test_samples)],
        )

    def enable_channel(self, streamer_name: str, mode: int):
        self.format("uint32_t", f"M{mode}_{streamer_name}_en", 1)

    def disable_channel(self, streamer_name: str, mode: int):
        self.format("uint32_t", f"M{mode}_{streamer_name}_en", 0)

    def build_mode(
        self,
        mode: int,
        streamers: dict[str, tuple[list[int], list[int]]],
        scalars: dict[str, int],
        test_data: dict[str, str],
        tests: dict[str, int],
    ):
        """Process all settings of a single mode and convert them to C code.

        Args:
        - mode: mode bit
        - streamers: dict[streamer name, (temporal bounds, temporal strides)]
        - scalars: dict[scalar name, value]
        - test_data: dict[tensor name, dtype]
        - tests: dict[test name, tensor size]
        """
        # Iterate over all streamer names
        assert all(re.match(r"^(R([0-9]|1[0-3])|W([0-9]|1[0-3]))$", key) for key in streamers.keys())
        for name in [f"R{i}" for i in range(14)] + [f"W{i}" for i in range(4)]:
            if name in streamers:
                (bounds, strides) = streamers[name]
                self.format_temporal_bounds_strides(name, mode, bounds, strides)
                self.format_spatial_stride(name, mode, BANK_BYTES)
                self.enable_channel(name, mode)
            else:
                self.disable_channel(name, mode)

        # Format scalar values # TODO prefix with mode bit
        for key, value in scalars.items():
            self.format("uint32_t", key, int(value))

        for tensor, size in tests.items():
            self.format_test_samples(tensor, tensor_size=size, nb_test_samples=NB_TEST_SAMPLES)

        # Read and format test data
        for tensor, dtype in test_data.items():
            self.read_and_format_vector(dtype, tensor)
