package snax.csr_manager

import chisel3._

import chiseltest._

trait HasCsrManagerTestUtils {

  // write csr helper function without strobe
  def write_csr[T <: CsrManager](dut: T, addr: Int, data: Int) = {

    // give the data and address to the right ports
    dut.io.csr_config_in.req.bits.write.poke(1.B)
    dut.io.csr_config_in.req.bits.strb.poke("b1111".U)
    dut.io.csr_config_in.req.bits.data.poke(data.U)
    dut.io.csr_config_in.req.bits.addr.poke(addr.U)
    dut.io.csr_config_in.req.valid.poke(1.B)

    // wait for grant
    while (dut.io.csr_config_in.req.ready.peekBoolean() == false) {
      dut.clock.step(1)
    }

    dut.clock.step(1)

    dut.io.csr_config_in.req.valid.poke(0.B)

  }

  // write csr helper function with strobe
  def write_csr[T <: CsrManager](dut: T, addr: BigInt, data: BigInt, strb: Int) = {

    // give the data and address to the right ports
    dut.io.csr_config_in.req.bits.write.poke(1.B)
    dut.io.csr_config_in.req.bits.strb.poke(strb.U)
    dut.io.csr_config_in.req.bits.data.poke(data.U)
    dut.io.csr_config_in.req.bits.addr.poke(addr.U)
    dut.io.csr_config_in.req.valid.poke(1.B)

    // wait for grant
    while (dut.io.csr_config_in.req.ready.peekBoolean() == false) {
      dut.clock.step(1)
    }

    dut.clock.step(1)

    dut.io.csr_config_in.req.valid.poke(0.B)

  }

  // read csr helper function
  def read_csr[T <: CsrManager](dut: T, addr: Int) = {
    dut.clock.step(1)

    // give the data and address to the right ports
    dut.io.csr_config_in.req.bits.write.poke(0.B)
    dut.io.csr_config_in.req.bits.addr.poke(addr.U)
    dut.io.csr_config_in.req.valid.poke(1.B)

    // wait for grant
    while (dut.io.csr_config_in.req.ready.peekBoolean() == false) {
      dut.clock.step(1)
    }

    // wait for valid signal
    while (dut.io.csr_config_in.rsp.valid.peekBoolean() == false) {
      dut.clock.step(1)
    }

    // return read csr result
    val result = dut.io.csr_config_in.rsp.bits.data.peekInt()

    dut.clock.step(1)

    dut.io.csr_config_in.req.valid.poke(0.B)

    // give read out ready signal
    dut.io.csr_config_in.rsp.ready.poke(1.B)

    result
  }
}
