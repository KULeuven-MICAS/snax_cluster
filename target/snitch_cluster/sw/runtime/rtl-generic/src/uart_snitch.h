// Copyright 2021 ETH Zurich, University of Bologna; Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// extern uint32_t __base_uart_sym;

uint32_t __base_uart_sym = 0x02002000;

#define UART_BASE_ADDR (uintptr_t) & __base_uart_sym

#define UART_RBR UART_BASE_ADDR + 0
#define UART_THR UART_BASE_ADDR + 0
#define UART_INTERRUPT_ENABLE UART_BASE_ADDR + 4
#define UART_INTERRUPT_IDENT UART_BASE_ADDR + 8
#define UART_FIFO_CONTROL UART_BASE_ADDR + 8
#define UART_LINE_CONTROL UART_BASE_ADDR + 12
#define UART_MODEM_CONTROL UART_BASE_ADDR + 16
#define UART_LINE_STATUS UART_BASE_ADDR + 20
#define UART_MODEM_STATUS UART_BASE_ADDR + 24
#define UART_DLAB_LSB UART_BASE_ADDR + 0
#define UART_DLAB_MSB UART_BASE_ADDR + 4

/*
    UART_LINE_CONTROL[1:0]: iLCR_WLS Word Length Select, 2'b11 for two bits mode
    UART_LINE_CONTROL[2]: iLCE_STB, 1'b0 for one bit stop bit
    UART_LINE_CONTROL[3]: iLCR_PEN, Parity Enable, disable as there is higher level parity check algorithm
    UART_LINE_CONTROL[4]: iLCR_PES, also related to parity check
    UART_LINE_CONTROL[5]: iLCR_SP, also related to parity check
    UART_LINE_CONTROL[6]: iLCR_BC, signaling a break control
    UART_LINE_CONTROL[7]: iLCR_DLAB, don't know what it is

*/
/*
    UART_MODEM_CONTROL[0]: iMCR_DTR (DTR output, not used)
    UART_MODEM_CONTROL[1]: iMCR_RTS (RTS output, set 1 to inform the device is ready to receive the data)
    UART_MODEM_CONTROL[2]: iMCR_OUT1 (General Purpose Output 1, not used)
    UART_MODEM_CONTROL[3]: iMCR_OUT2 (General Purpose Output 2, not used)
    UART_MODEM_CONTROL[4]: iMCR_LOOP (Internal Loopback, should set to 0)
    UART_MODEM_CONTROL[5]: iMCR_AFE (Automatic Flow Control, set to 1 to automatically manage DTR and RTS)
*/

inline static void write_reg_u8(uintptr_t addr, uint8_t value)
{
    volatile uint8_t *loc_addr = (volatile uint8_t *)addr;
    *loc_addr = value;
}

inline static uint8_t read_reg_u8(uintptr_t addr)
{
    return *(volatile uint8_t *)addr;
}

inline static int is_data_ready()
{
    return read_reg_u8(UART_LINE_STATUS) & 0x01;
}

inline static int is_data_overrun()
{
    return read_reg_u8(UART_LINE_STATUS) & 0x02;
}

inline static int is_transmit_empty()
{
    return read_reg_u8(UART_LINE_STATUS) & 0x20;
}

inline static int is_transmit_done()
{
    return read_reg_u8(UART_LINE_STATUS) & 0x40;
}

inline static void write_serial(char a)
{
    while (is_transmit_empty() == 0)
    {
    };

    write_reg_u8(UART_THR, a);
}

inline static uint8_t read_serial()
{
    while (is_data_ready() == 0)
    {
    };

    return read_reg_u8(UART_RBR);
}
inline static void init_uart(uint32_t freq, uint32_t baud)
{
    uint32_t divisor = freq / (baud << 4);

    write_reg_u8(UART_INTERRUPT_ENABLE, 0x00); // Disable all interrupts
    write_reg_u8(UART_LINE_CONTROL,
                 0x80);                                 // Enable DLAB (set baud rate divisor)
    write_reg_u8(UART_DLAB_LSB, divisor);               // divisor (lo byte)
    write_reg_u8(UART_DLAB_MSB, (divisor >> 8) & 0xFF); // divisor (hi byte)
    write_reg_u8(UART_LINE_CONTROL, 0x03);              // 8 bits, no parity, one stop bit
    write_reg_u8(UART_FIFO_CONTROL,
                 0xC7);                     // Enable FIFO, clear them, with 14-byte threshold
    write_reg_u8(UART_MODEM_CONTROL, 0x22); // Flow control enabled, auto flow control mode
}

inline static void print_uart(const char *str)
{
    const char *cur = &str[0];
    while (*cur != '\0')
    {
        write_serial((uint8_t)*cur);
        ++cur;
    }
    while (!is_transmit_done())
        ;
}

inline static void print_uart_hex(char *str, uint32_t length)
{
    uint8_t lut[16] = {'0', '1', '2', '3', '4', '5', '6', '7',
                       '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
    for (uint64_t i = (uint64_t)str; i < (uint64_t)str + length; i++)
    {
        if (i % 16 == 0)
        {
            write_serial('\r');
            write_serial('\n');
            for (int j = 28; j >= 0; j = j - 4)
                write_serial(lut[(i >> j) % 16]);
            write_serial(':');
            write_serial(' ');
        }
        char temp = *((char *)i);
        write_serial(lut[temp / 16]);
        write_serial(lut[temp % 16]);
        write_serial(' ');
    }
    while (!is_transmit_done())
        ;
}

inline static void scan_uart(char *str)
{
    char *cur = &str[0];
    while (1)
    {
        *cur = read_serial();
        if (*cur == '\r')
        {
            *cur = '\0';
            return;
        }
        else
            cur++;
    }
}
