// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Reuse same implementation as regular rtl runtime
// This is the putchar implementation for the testbench with fesvr
// #include "../../rtl/src/putchar.c"

#include "uart_snitch.h"

static uint8_t uart_initialized = 0;

void _putchar(char character) {
    if (!uart_initialized) {
        init_uart(32, 1);
        uart_initialized = 1;
    }
    write_serial(character);
}
