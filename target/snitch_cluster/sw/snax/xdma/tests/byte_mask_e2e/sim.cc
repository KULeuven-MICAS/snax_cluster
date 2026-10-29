// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include "Vbyte_mask_e2e_top.h"
#include "snrt.h"
#include "verilated.h"

extern "C" int byte_mask_e2e_main(void);

static VerilatedContext context;
static std::unique_ptr<Vbyte_mask_e2e_top> dut;
static uint64_t cycles;

static void tick() {
    if (++cycles > 200000) throw std::runtime_error("simulation cycle limit");
    dut->clk = 0;
    dut->eval();
    context.timeInc(1);
    dut->clk = 1;
    dut->eval();
    context.timeInc(1);
    dut->clk = 0;
    dut->eval();
    if (context.gotFinish()) throw std::runtime_error("RTL stopped simulation");
}

static void request(uint32_t address, bool write, uint32_t value) {
    if (address < 960 || address >= 1088)
        throw std::runtime_error("CSR address outside XDMA region");
    dut->csr_addr = address - 960;
    dut->csr_write = write;
    dut->csr_data = value;
    dut->csr_strb = 15;
    dut->csr_valid = 1;
    dut->eval();
    while (!dut->csr_ready) tick();
    tick();
    dut->csr_valid = 0;
    dut->eval();
}

extern "C" void csrw_ss(uint32_t address, uint32_t value) {
    request(address, true, value);
}

extern "C" uint32_t csrr_ss(uint32_t address) {
    request(address, false, 0);
    while (!dut->rsp_valid) tick();
    uint32_t value = dut->rsp_data;
    dut->rsp_ready = 1;
    tick();
    dut->rsp_ready = 0;
    dut->eval();
    return value;
}

extern "C" void e2e_mem_write(uint32_t address, uint8_t value) {
    if (address >= 128 * 1024)
        throw std::runtime_error("host memory write out of range");
    dut->host_addr = address;
    dut->host_data = value;
    dut->host_write = 1;
    tick();
    dut->host_write = 0;
    dut->eval();
}

extern "C" uint8_t e2e_mem_read(uint32_t address) {
    if (address >= 128 * 1024)
        throw std::runtime_error("host memory read out of range");
    dut->host_addr = address;
    dut->eval();
    return dut->host_result;
}

int main(int argc, char** argv) {
    try {
        context.commandArgs(argc, argv);
        dut = std::make_unique<Vbyte_mask_e2e_top>(&context);
        dut->reset = 1;
        dut->csr_valid = 0;
        dut->rsp_ready = 0;
        dut->host_write = 0;
        dut->host_addr = 0;
        dut->host_data = 0;
        tick();
        tick();
        dut->reset = 0;
        tick();
        int result = byte_mask_e2e_main();
        dut->final();
        std::printf("Simulation finished after %llu cycles\n",
                    static_cast<unsigned long long>(cycles));
        return result;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "HARNESS ERROR: %s\n", error.what());
        return 2;
    }
}
