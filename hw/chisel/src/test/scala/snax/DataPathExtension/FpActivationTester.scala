package snax.DataPathExtension

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** Differential Tier-1 test for the merged FpActivation: it must reproduce the standalone FpExp / FpSilu
  * (the proven golden references) BIT-FOR-BIT in every build mode (exp-only, silu-only, and both-built with
  * either func selected). Combinational (pipelined=false) so ref and act are pure functions of `in` and can
  * be compared directly across a dense input sweep. Stronger than a ULP check — proves the FP-unit sharing
  * introduced no numerical drift.
  */
class FpActivationTester extends AnyFlatSpec with ChiselScalatestTester {

  private class ActDiff(hasExp: Boolean, hasSilu: Boolean, funcSilu: Boolean, expLutN: Int, siluN: Int)
      extends Module
      with RequireAsyncReset {
    val io = IO(new Bundle {
      val in     = Input(UInt(32.W))
      val golden = Output(UInt(32.W)) // standalone FpExp/FpSilu
      val merged = Output(UInt(32.W)) // merged FpActivation
    })
    val a = Module(new FpActivation(pipelined = false, hasExp = hasExp, hasSilu = hasSilu, expLutN, siluN))
    a.io.in := io.in; a.io.func := funcSilu.B; io.merged := a.io.out
    if (funcSilu) {
      val m = Module(new FpSilu(pipelined = false, siluN)); m.io.in := io.in; io.golden := m.io.out
    } else {
      val m = Module(new FpExp(pipelined = false, expLutN)); m.io.in := io.in; io.golden := m.io.out
    }
  }

  private def bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)

  def diff(hasExp: Boolean, hasSilu: Boolean, funcSilu: Boolean, tag: String): Unit =
    test(new ActDiff(hasExp, hasSilu, funcSilu, 128, 256)) { dut =>
      // dense sweep over the full activation domain + edge/saturation/near-subnormal points
      val xs = (-1800 to 1800).map(_ * 0.05) ++
        Seq(0.0, -0.0, 1e-30, -1e-30, 5e-8, -5e-8, 16.0, -16.0, 88.0, -88.0, 100.0, -100.0, 1e9, -1e9)
      var mism  = 0
      var first = ""
      for (xd <- xs) {
        val x = xd.toFloat
        dut.io.in.poke(bits(x).U)
        val ref = dut.io.golden.peekInt()
        val act = dut.io.merged.peekInt()
        if (ref != act) { mism += 1; if (first.isEmpty) first = f"x=$x%.6g ref=0x$ref%08x act=0x$act%08x" }
      }
      assert(mism == 0, s"FpActivation($tag) differs from reference in $mism/${xs.length} cases; first: $first")
    }

  "FpActivation_exp_only"  should "match FpExp bit-exact"               in { diff(true, false, false, "exp-only") }
  "FpActivation_silu_only" should "match FpSilu bit-exact"              in { diff(false, true, true, "silu-only") }
  "FpActivation_both_exp"  should "match FpExp bit-exact (both built)"  in { diff(true, true, false, "both-exp") }
  "FpActivation_both_silu" should "match FpSilu bit-exact (both built)" in { diff(true, true, true, "both-silu") }
}
