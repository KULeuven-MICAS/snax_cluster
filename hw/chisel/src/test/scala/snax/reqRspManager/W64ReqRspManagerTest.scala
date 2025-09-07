package snax.reqRspManager

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import snax.reqRspManager.ReqRspManager

class W64ReqRspManagerTest extends AnyFlatSpec with ChiselScalatestTester with Matchers with HasReqRspManagerTest {

  "DUT" should "pass" in {
    test(
      new ReqRspManager(
        numReadWriteReg = 7,
        numReadOnlyReg = 2,
        addrWidth = 32,
        dataWidth = 64,
        moduleTagName = "Test"
      )
    ).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // Strobe Moves step by step
      writeReg(dut, 0, BigInt("FFFFFFFFFFFFFFFF", 16), 0b00000001)
      dut.io.readWriteRegIO.bits(0).expect(0x000000FFL)
      dut.io.readWriteRegIO.bits(1).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(2).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(3).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(4).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(5).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(6).expect(0x00000000L)
      if (readReg(dut, 0) != BigInt("00000000000000FF", 16)) throw new Exception("Value not written correctly")

      writeReg(dut, 0, BigInt("FFFFFFFFFFFFFFFF", 16), 0b00000010)
      dut.io.readWriteRegIO.bits(0).expect(0x0000FFFFL)
      dut.io.readWriteRegIO.bits(1).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(2).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(3).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(4).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(5).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(6).expect(0x00000000L)
      if (readReg(dut, 0) != BigInt("000000000000FFFF", 16)) throw new Exception("Value not written correctly")

      writeReg(dut, 0, BigInt("FFFFFFFFFFFFFFFF", 16), 0b00000100)
      dut.io.readWriteRegIO.bits(0).expect(0x00FFFFFFL)
      dut.io.readWriteRegIO.bits(1).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(2).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(3).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(4).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(5).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(6).expect(0x00000000L)
      if (readReg(dut, 0) != BigInt("0000000000FFFFFF", 16)) throw new Exception("Value not written correctly")

      writeReg(dut, 0, BigInt("FFFFFFFFFFFFFFFF", 16), 0b00001000)
      dut.io.readWriteRegIO.bits(0).expect(0xFFFFFFFFL)
      dut.io.readWriteRegIO.bits(1).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(2).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(3).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(4).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(5).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(6).expect(0x00000000L)
      if (readReg(dut, 0) != BigInt("00000000FFFFFFFF", 16)) throw new Exception("Value not written correctly")

      writeReg(dut, 0, BigInt("FFFFFFFFFFFFFFFF", 16), 0b00010000)
      dut.io.readWriteRegIO.bits(0).expect(0xFFFFFFFFL)
      dut.io.readWriteRegIO.bits(1).expect(0x000000FFL)
      dut.io.readWriteRegIO.bits(2).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(3).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(4).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(5).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(6).expect(0x00000000L)
      if (readReg(dut, 0) != BigInt("000000FFFFFFFFFF", 16)) throw new Exception("Value not written correctly")

      writeReg(dut, 0, BigInt("FFFFFFFFFFFFFFFF", 16), 0b00100000)
      dut.io.readWriteRegIO.bits(0).expect(0xFFFFFFFFL)
      dut.io.readWriteRegIO.bits(1).expect(0x0000FFFFL)
      dut.io.readWriteRegIO.bits(2).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(3).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(4).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(5).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(6).expect(0x00000000L)
      if (readReg(dut, 0) != BigInt("0000FFFFFFFFFFFF", 16)) throw new Exception("Value not written correctly")

      writeReg(dut, 0, BigInt("FFFFFFFFFFFFFFFF", 16), 0b01000000)
      dut.io.readWriteRegIO.bits(0).expect(0xFFFFFFFFL)
      dut.io.readWriteRegIO.bits(1).expect(0x00FFFFFFL)
      dut.io.readWriteRegIO.bits(2).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(3).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(4).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(5).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(6).expect(0x00000000L)
      if (readReg(dut, 0) != BigInt("00FFFFFFFFFFFFFF", 16)) throw new Exception("Value not written correctly")

      writeReg(dut, 0, BigInt("FFFFFFFFFFFFFFFF", 16), 0b10000000)
      dut.io.readWriteRegIO.bits(0).expect(0xFFFFFFFFL)
      dut.io.readWriteRegIO.bits(1).expect(0xFFFFFFFFL)
      dut.io.readWriteRegIO.bits(2).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(3).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(4).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(5).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(6).expect(0x00000000L)
      if (readReg(dut, 0) != BigInt("FFFFFFFFFFFFFFFF", 16)) throw new Exception("Value not written correctly")

      writeReg(dut, 0, 0, 0b00000000)
      dut.io.readWriteRegIO.bits(0).expect(0xFFFFFFFFL)
      dut.io.readWriteRegIO.bits(1).expect(0xFFFFFFFFL)
      dut.io.readWriteRegIO.bits(2).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(3).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(4).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(5).expect(0x00000000L)
      dut.io.readWriteRegIO.bits(6).expect(0x00000000L)
      if (readReg(dut, 0) != BigInt("FFFFFFFFFFFFFFFF", 16)) throw new Exception("Value not written correctly")
    }
  }

}
