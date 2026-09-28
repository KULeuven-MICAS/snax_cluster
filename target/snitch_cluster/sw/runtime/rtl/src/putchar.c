// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

extern uintptr_t volatile tohost, fromhost;

// Each hart hands the host whole lines. printf's characters collect in a line
// buffer in the hart's thread-local storage, which is in the TCDM; at a newline,
// or when the line buffer is full, the line is copied into the hart's buffer in
// main memory, where the host reads it, and handed over in one write syscall. A
// character therefore costs a TCDM store, not a main-memory round trip.
extern uint32_t _edram;
#define PUTC_BUFFER_LEN (1024 - sizeof(size_t))
struct putc_buffer_header {
    size_t size;
    uint64_t syscall_mem[8];
};
static volatile struct putc_buffer {
    struct putc_buffer_header hdr;
    char data[PUTC_BUFFER_LEN];
} *const putc_buffer = (void *)&_edram;

#define PUTC_LINE_LEN 128
static __thread uint32_t putc_line_len;
static __thread uint32_t putc_line[PUTC_LINE_LEN / 4];

// Provide an implementation for putchar.
void _putchar(char character) {
    ((char *)putc_line)[putc_line_len++] = character;
    if (putc_line_len == PUTC_LINE_LEN || character == '\n') {
        volatile struct putc_buffer *buf = &putc_buffer[snrt_hartid()];
        volatile uint32_t *dst = (volatile uint32_t *)buf->data;
        for (uint32_t w = 0; w < (putc_line_len + 3) / 4; w++)
            dst[w] = putc_line[w];
        buf->hdr.size = putc_line_len;
        buf->hdr.syscall_mem[0] = 64;  // sys_write
        buf->hdr.syscall_mem[1] = 1;   // file descriptor (1 = stdout)
        buf->hdr.syscall_mem[2] = (uintptr_t)&buf->data;  // buffer
        buf->hdr.syscall_mem[3] = putc_line_len;          // length

        tohost = (uintptr_t)buf->hdr.syscall_mem;
        while (fromhost == 0)
            ;
        fromhost = 0;

        putc_line_len = 0;
    }
}
