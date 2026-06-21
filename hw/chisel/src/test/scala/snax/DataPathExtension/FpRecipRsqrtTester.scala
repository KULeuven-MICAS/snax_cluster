package snax.DataPathExtension

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Tier-1 accuracy test for the NR FpRecipRsqrt core vs 1/x and 1/sqrt(x). Reports worst relative error. */
class FpRecipRsqrtTester extends AnyFlatSpec with ChiselScalatestTester {

  def fbits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)
  def asF(bits: BigInt): Float = java.lang.Float.intBitsToFloat(bits.toInt)

  "FpRecipRsqrt" should "match 1/x and 1/sqrt(x)" in {
    test(new FpRecipRsqrt)
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) {
        dut =>
          dut.clock.step(1)
          var maxRecip = 0.0
          var maxRsqrt = 0.0
          // positive inputs spanning softmax/rmsnorm sum ranges
          val xs = (1 to 4000).map(_ * 0.5f)
          for (x <- xs) {
            // RECIP
            dut.io.in.poke(fbits(x).U); dut.io.mode.poke(0.U); dut.clock.step(1)
            val r  = asF(dut.io.out.peekInt()); val gr = 1.0f / x
            maxRecip = math.max(maxRecip, math.abs((r - gr) / gr).toDouble)
            // RSQRT
            dut.io.mode.poke(1.U); dut.clock.step(1)
            val s  = asF(dut.io.out.peekInt()); val gs = (1.0 / math.sqrt(x.toDouble)).toFloat
            maxRsqrt = math.max(maxRsqrt, math.abs((s - gs) / gs).toDouble)
          }
          println(f"[FpRecipRsqrt] worst rel-err: recip=$maxRecip%.3e rsqrt=$maxRsqrt%.3e")
          // NR to ~machine precision; allow a few FP32 ULP
          assert(maxRecip <= 1e-5, s"recip rel-err too high: $maxRecip")
          assert(maxRsqrt <= 1e-5, s"rsqrt rel-err too high: $maxRsqrt")
      }
  }

  // specialized single-mode core: only the listed NR path is built; the mode input is ignored.
  def checkSingle(mode: String): Unit = {
    test(new FpRecipRsqrt(Seq(mode)))
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) {
        dut =>
          dut.clock.step(1)
          var maxErr = 0.0
          for (x <- (1 to 4000).map(_ * 0.5f)) {
            dut.io.in.poke(fbits(x).U); dut.io.mode.poke(0.U); dut.clock.step(1)
            val o = asF(dut.io.out.peekInt())
            val g = if (mode == "RECIP") 1.0f / x else (1.0 / math.sqrt(x.toDouble)).toFloat
            maxErr = math.max(maxErr, math.abs((o - g) / g).toDouble)
          }
          println(f"[FpRecipRsqrt:$mode-only] worst rel-err=$maxErr%.3e")
          assert(maxErr <= 1e-5, s"$mode-only rel-err too high: $maxErr")
      }
  }

  "FpRecipRsqrt_RSQRTonly" should "match 1/sqrt(x) with recip path dropped" in { checkSingle("RSQRT") }
  "FpRecipRsqrt_RECIPonly" should "match 1/x with rsqrt path dropped" in { checkSingle("RECIP") }
}
