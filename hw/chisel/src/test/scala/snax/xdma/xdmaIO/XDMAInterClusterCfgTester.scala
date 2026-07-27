package snax.xdma.xdmaIO

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec
import firrtl2.options.TargetDirAnnotation

import snax.xdma.DesignParams._
import snax.readerWriter.ReaderWriterParam
import snax.DataPathExtension._

/** Round-trip test for the cross-cluster config serialize<->deserialize path, specifically that the added
  * `writerExtCfg` (the writer-side DataPathExtension config carried across the die-to-die link) survives
  * serialization exactly. Passing elaboration only proves widths line up; this proves the BITS land in the
  * right slots (the one thing the symmetric-reasoning argument cannot cover). Covers a single-frame config and
  * a multi-frame config (enough writer extensions to exceed one 507-bit frame body).
  */
class XDMAInterClusterCfgRoundTrip(param: XDMAParam) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val cfgIn  = Flipped(Decoupled(new XDMAInterClusterCfgIO(param, param)))
    val cfgOut = Decoupled(new XDMAInterClusterCfgIO(param, param))
  })
  val ser = Module(new XDMAInterClusterCfgIOSerializer(param))
  val des = Module(new XDMAInterClusterCfgIODeserializer(param))
  ser.io.cfgIn  <> io.cfgIn
  des.io.cfgIn  <> ser.io.cfgOut
  io.cfgOut     <> des.io.cfgOut
}

class XDMAInterClusterCfgTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  def mkParam(nExt: Int) = new XDMAParam(
    cfgParam          = new XDMAConfigParam(addrWidth = 32, dataWidth = 32),
    axiParam          = new XDMAAXIParam,
    crossClusterParam = new XDMACrossClusterParam,
    rwParam           = new ReaderWriterParam,
    extParam          = Seq.fill(nExt)(new HasVerilogMemset) // 1 CSR each
  )

  def roundTrip(nExt: Int, tag: String): Unit = {
    val param  = mkParam(nExt)
    val extLen = nExt + 1 // total writer-ext userCsrNum (nExt) + 1 bypass
    test(new XDMAInterClusterCfgRoundTrip(param))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags, TargetDirAnnotation(s"test_run_dir/xcfg_$tag"))) { dut =>
        val rng     = new scala.util.Random(7)
        val taskID  = BigInt(0xa)
        val rdPtr   = BigInt("123456789A", 16) & ((BigInt(1) << 48) - 1)
        val wp0     = BigInt("ABCDEF01", 16)
        val wp1     = BigInt("00002222", 16)
        val extVals = (0 until extLen).map(_ => BigInt(rng.nextLong() & 0xffffffffL))

        dut.io.cfgIn.bits.taskID.poke(taskID.U)
        dut.io.cfgIn.bits.isWriterSide.poke(true.B)
        dut.io.cfgIn.bits.readerPtr.poke(rdPtr.U)
        dut.io.cfgIn.bits.writerPtr(0).poke(wp0.U)
        dut.io.cfgIn.bits.writerPtr(1).poke(wp1.U)
        dut.io.cfgIn.bits.axiTransferBeatSize.poke(0x111.U)
        dut.io.cfgIn.bits.spatialStride.poke(0x222.U)
        for (i <- 0 until 5) dut.io.cfgIn.bits.temporalBounds(i).poke((0x30 + i).U)
        for (i <- 0 until 5) dut.io.cfgIn.bits.temporalStrides(i).poke((0x40 + i).U)
        dut.io.cfgIn.bits.enabledChannel.poke(0x5.U)
        dut.io.cfgIn.bits.enabledByte.poke(0x6.U)
        // The chain position the initiator stamped. It must survive the wire: a gather head is configured
        // remotely, so the receiver cannot re-derive its role from its own `origination`.
        dut.io.cfgIn.bits.chainRole.poke(XDMAChainRole.MIDDLE.U)
        dut.io.cfgIn.bits.collectiveMode.poke(true.B)
        for (i <- 0 until extLen) dut.io.cfgIn.bits.writerExtCfg(i).poke(extVals(i).U)

        dut.io.cfgOut.ready.poke(true.B)
        dut.io.cfgIn.valid.poke(true.B)

        var cyc = 0
        while (!dut.io.cfgOut.valid.peekBoolean() && cyc < 500) { dut.clock.step(1); cyc += 1 }
        assert(dut.io.cfgOut.valid.peekBoolean(), s"[$tag] no cfgOut after $cyc cycles")

        // field-by-field equality (regression on the existing fields + the new writerExtCfg)
        assert(dut.io.cfgOut.bits.isWriterSide.peekBoolean(), s"[$tag] isWriterSide")
        assert(dut.io.cfgOut.bits.taskID.peekInt() == taskID, s"[$tag] taskID")
        assert(dut.io.cfgOut.bits.readerPtr.peekInt() == rdPtr, s"[$tag] readerPtr")
        assert(dut.io.cfgOut.bits.writerPtr(0).peekInt() == wp0, s"[$tag] writerPtr0")
        assert(dut.io.cfgOut.bits.writerPtr(1).peekInt() == wp1, s"[$tag] writerPtr1")
        assert(dut.io.cfgOut.bits.enabledChannel.peekInt() == 5, s"[$tag] enabledChannel")
        assert(dut.io.cfgOut.bits.enabledByte.peekInt() == 6, s"[$tag] enabledByte")
        assert(dut.io.cfgOut.bits.chainRole.peekInt() == XDMAChainRole.MIDDLE, s"[$tag] chainRole")
        assert(dut.io.cfgOut.bits.collectiveMode.peekBoolean(), s"[$tag] collectiveMode")
        assert(dut.io.cfgOut.bits.temporalBounds(2).peekInt() == 0x32, s"[$tag] temporalBounds(2)")
        assert(dut.io.cfgOut.bits.temporalStrides(4).peekInt() == 0x44, s"[$tag] temporalStrides(4)")
        for (i <- 0 until extLen) {
          val got = dut.io.cfgOut.bits.writerExtCfg(i).peekInt()
          assert(got == extVals(i), f"[$tag] writerExtCfg($i) got 0x${got}%x want 0x${extVals(i)}%x")
        }
        println(f"[XCfg $tag] round-trip EXACT: isWriterSide + taskID + ptrs + bounds/strides + chainRole + $extLen writerExtCfg entries ($cyc cyc, extLen=$extLen)")
      }
  }

  "XDMAInterClusterCfg_singleframe" should "round-trip writerExtCfg (1 writer ext -> single frame)" in {
    roundTrip(nExt = 1, tag = "single")
  }
  "XDMAInterClusterCfg_multiframe" should "round-trip writerExtCfg (5 writer exts -> multi-frame)" in {
    roundTrip(nExt = 5, tag = "multi")
  }
}
