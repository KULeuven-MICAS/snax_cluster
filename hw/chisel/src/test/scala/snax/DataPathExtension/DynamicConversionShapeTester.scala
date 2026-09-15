// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.DataPathExtension

import chisel3._

import chiseltest._
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import snax.streamer.StreamerGen

class DynamicConversionShapeHarness(extension: HasDataPathExtension) extends Module with RequireAsyncReset {
  val dut = extension.instantiate("shape_test")
  val io  = IO(chiselTypeOf(dut.io))
  io <> dut.io
}

class DynamicConversionShapeTester extends AnyFlatSpec with ChiselScalatestTester {
  private def pack(values: Seq[Int], width: Int): BigInt =
    values.zipWithIndex.foldLeft(BigInt(0)) { case (word, (value, lane)) =>
      word | (BigInt(value) << (lane * width))
    }

  // Test values are small positive integers, exactly representable in FP16.
  private def fp16(value: Int): Int = {
    val exponent = 31 - Integer.numberOfLeadingZeros(value)
    ((exponent + 15) << 10) | ((value - (1 << exponent)) << (10 - exponent))
  }

  private def checkPacking(
    extension:   HasDataPathExtension,
    factors:     Seq[Int],
    outputWidth: Int,
    csrs:        Int => Seq[Int],
    convert:     Int => Int
  ): Unit = {
    test(new DynamicConversionShapeHarness(extension)).withAnnotations(Seq(TreadleBackendAnnotation)) { dut =>
      val dataWidth   = extension.extensionParam.dataWidth
      val inputLanes  = dataWidth / 32
      val outputLanes = dataWidth / outputWidth
      dut.io.enable_i.poke(true.B)
      dut.io.data_i.valid.poke(false.B)
      dut.io.data_o.ready.poke(true.B)

      for ((factor, shape) <- factors.zipWithIndex) {
        dut.io.csr_i.zip(csrs(shape)).foreach { case (port, value) => port.poke(value.U) }
        dut.io.start_i.poke(true.B)
        dut.clock.step()
        dut.io.start_i.poke(false.B)

        // Two full output words exercise counter wrap and preservation of lane order.
        val values      = Seq.tabulate(outputLanes * 2)(i => i % 63 + 1)
        val activeLanes = inputLanes / factor
        val inputs      = values
          .grouped(activeLanes)
          .map { active =>
            pack(active ++ Seq.fill(inputLanes - activeLanes)(117), 32)
          }
          .toSeq
        val outputs     = values.map(convert).grouped(outputLanes).map(pack(_, outputWidth)).toSeq
        var sent        = 0
        var received    = 0
        var cycles      = 0

        while (received < outputs.length && cycles < inputs.length * 4 + 20) {
          dut.io.data_i.valid.poke((sent < inputs.length).B)
          if (sent < inputs.length) dut.io.data_i.bits.poke(inputs(sent).U)
          if (dut.io.data_i.valid.peekBoolean() && dut.io.data_i.ready.peekBoolean()) sent += 1
          if (dut.io.data_o.valid.peekBoolean()) {
            dut.io.data_o.bits.expect(outputs(received).U)
            received += 1
          }
          dut.clock.step()
          cycles += 1
        }

        assert(sent == inputs.length, s"shape $shape did not consume all input beats")
        assert(received == outputs.length, s"shape $shape did not produce both packed output words")
        dut.io.data_i.valid.poke(false.B)
        dut.io.data_o.valid.expect(false.B)
      }
    }
  }

  "Dynamic conversion extensions" should "pack rescaled lanes using the CSR-selected shape table" in {
    val factors = Seq(1, 4, 2)
    checkPacking(
      new HasRescaleDownEfficientDynamic(extra_loops_choice = factors),
      factors,
      8,
      shape => Seq(0, 2, 0, 1, shape),
      identity
    )
  }

  it should "pack FP16 lanes using the CSR-selected shape table" in {
    val factors = Seq(1, 4, 2)
    checkPacking(new HasInt32ToFp16Converter(extra_loops_choice = factors), factors, 16, shape => Seq(shape), fp16)
  }

  it should "accept named extension parameters in any JSON order with omitted defaults" in {
    val config     = """{"snax_streamer_cfg": {"data_reader_params": {"datapath_extensions": [{
      "HasRescaleDownEfficientDynamic": {"extra_loops_choice": [1, 4, 2], "dataWidth": 256},
      "HasInt32ToFp16Converter": {"extra_loops_choice": [1, 4, 2], "dataWidth": 256}
    }]}}}"""
    val (param, _) = StreamerGen.buildParam(Array("--streamercfg", config))
    val extensions = param.readerDatapathExtention.head
    assert(extensions.length == 2)
    assert(extensions.map(_.extensionParam.userCsrNum) == Seq(5, 1))
    for (extension <- extensions) {
      assert(extension.extensionParam.dataWidth == 256)
      ChiselStage.emitCHIRRTL(new DynamicConversionShapeHarness(extension))
    }
  }

  it should "reject empty, nonpositive, and fractional lane aggregation factors" in {
    for (factors <- Seq(Seq.empty[Int], Seq(0), Seq(-1), Seq(3), Seq(32))) {
      val extensions = Seq(
        new HasRescaleDownEfficientDynamic(extra_loops_choice = factors),
        new HasInt32ToFp16Converter(extra_loops_choice = factors)
      )
      for (extension <- extensions) {
        val error = intercept[IllegalArgumentException] {
          ChiselStage.emitCHIRRTL(new DynamicConversionShapeHarness(extension))
        }
        assert(error.getMessage.contains("extra_loops_choice"))
      }
    }
  }
}
