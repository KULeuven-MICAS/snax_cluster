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
    val a = Module(new FpActivation(pipelined = false, hasExp = hasExp, hasSilu = hasSilu, hasGelu = false,
                                    expLutN = expLutN, siluN = siluN))
    a.io.in := io.in; a.io.func := funcSilu.B; a.io.gelu := false.B; io.merged := a.io.out
    if (funcSilu) {
      val m = Module(new FpSilu(pipelined = false, siluN)); m.io.in := io.in; io.golden := m.io.out
    } else {
      val m = Module(new FpExp(pipelined = false, expLutN)); m.io.in := io.in; io.golden := m.io.out
    }
  }

  // GELU-only build: the merged core with hasGelu (isExp/isGelu constant-fold), checked numerically vs the
  // true x*Phi(x) (no standalone FpGelu golden exists).
  private class GeluOnly(siluN: Int) extends Module with RequireAsyncReset {
    val io = IO(new Bundle { val in = Input(UInt(32.W)); val out = Output(UInt(32.W)) })
    val a = Module(new FpActivation(pipelined = false, hasExp = false, hasSilu = false, hasGelu = true,
                                    expLutN = 32, siluN = siluN))
    a.io.in := io.in; a.io.func := true.B; a.io.gelu := true.B; io.out := a.io.out
  }

  private def bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)
  private def f32(b: BigInt): Float = java.lang.Float.intBitsToFloat(b.toInt)
  private def erf(x: Double): Double = {
    val t = 1.0 / (1.0 + 0.3275911 * math.abs(x))
    val y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t * math.exp(-x * x)
    if (x >= 0) y else -y
  }
  private def geluRef(x: Double): Double = x * 0.5 * (1.0 + erf(x / math.sqrt(2.0)))
  private def f32ToF16bits(f: Float): Int = {
    val b = java.lang.Float.floatToIntBits(f); val s = (b >>> 16) & 0x8000
    val e = (b >>> 23) & 0xff; val m = b & 0x7fffff
    if (e == 0xff) return s | 0x7c00 | (if (m != 0) 0x200 else 0)
    val exp = e - 127 + 15
    if (exp >= 0x1f) return s | 0x7c00
    if (exp <= 0) { if (exp < -10) return s
      val mm = m | 0x800000; val sh = 14 - exp; val half = mm >>> sh
      val rem = mm & ((1 << sh) - 1); val hw = 1 << (sh - 1)
      var r = half; if (rem > hw || (rem == hw && (half & 1) == 1)) r += 1; return s | r }
    var h = s | (exp << 10) | (m >>> 13); val rem = m & 0x1fff
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1)) h += 1; h
  }
  // monotonic key: adjacent FP16 codes -> adjacent keys across zero (signed values)
  private def f16mono(h: Int): Int = { val mag = h & 0x7fff; if ((h & 0x8000) != 0) 0x8000 - mag else 0x8000 + mag }

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

  "FpActivation_gelu" should "match true x*Phi(x) to <=2 FP16 ULP" in {
    test(new GeluOnly(256)) { dut =>
      var worst = 0
      var wx    = ""
      // dense sweep; the significant FP16 range of gelu (|x| up to ~6, then Phi saturates -> gelu ~ x)
      for (xd <- (-1600 to 1600).map(_ * 0.01)) {
        val x = xd.toFloat
        dut.io.in.poke(bits(x).U)
        val g16 = f32ToF16bits(geluRef(x.toDouble).toFloat)
        if ((g16 & 0x7c00) != 0) { // skip tiny/subnormal golden (HW flushes; contributes ~0)
          val h16 = f32ToF16bits(f32(dut.io.out.peekInt()))
          val ulp = math.abs(f16mono(h16) - f16mono(g16))
          if (ulp > worst) { worst = ulp; wx = f"x=$x%.3f got=0x$h16%04x gold=0x$g16%04x" }
          assert(ulp <= 2, s"GELU exceeds 2 FP16 ULP at $wx (ulp=$ulp)")
        }
      }
      println(f"[FpActivation gelu] worst FP16 ULP=$worst over x in [-16,16]  ($wx)")
    }
  }
}
