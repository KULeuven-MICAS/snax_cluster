package snax.DataPathExtension

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** Tier-1 test for FpActivation's RSQRT function.
  *
  * Unlike the exp/silu tests next door this is not a differential check -- there is no standalone golden
  * module to diff against -- so it scores the hardware against `1/sqrt(x)` in double precision, in FP16
  * ULP, which is the grid the result is narrowed onto downstream. Two properties are asserted separately
  * because they fail for different reasons:
  *
  *   ACCURACY over the reachable domain. The consumer is rmsnorm's mean = SUM(x^2)/D, so the sweep runs
  *   over every FP16-representable decade of a positive mean and, within each, a dense sample of
  *   significands -- the table is indexed by the significand alone, so a bug in the node/frac split shows
  *   up at ANY exponent, and a bug in the exponent half shows up at ONLY some. Sweeping both separates them.
  *
  *   THE EXPONENT HALF IS EXACT, not approximate. rsqrt(4^k) must come back as EXACTLY 2^-k with a zero
  *   mantissa for every k in range: that is the whole reason the exponent is a shift and not a table, and
  *   it is the property that would silently rot if the parity fold were wrong.
  *
  * Plus totality: x <= 0, subnormal, inf and NaN must all give +0, never a wrapped exponent.
  */
class FpActivationRsqrtTester extends AnyFlatSpec with snax.utils.VerilatorTester {

  private class RsqDut(hasExp: Boolean, hasSilu: Boolean, rsqN: Int) extends Module with RequireAsyncReset {
    val io = IO(new Bundle {
      val in  = Input(UInt(32.W))
      val out = Output(UInt(32.W))
    })
    val a = Module(
      new FpActivation(pipelined = false, hasExp = hasExp, hasSilu = hasSilu, 32, 256,
                       hasRsqrt = true, rsqN = rsqN)
    )
    a.io.in := io.in; a.io.func := FpActivation.RSQRT.U; io.out := a.io.out
  }

  private def bits(f:  Float):  BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)
  private def fromBits(b: BigInt): Float = java.lang.Float.intBitsToFloat(b.toInt)

  /** FP16 ULP distance -- the same monotonic-key measure the device-side apps use. */
  private def f16ulp(a: Float, b: Float): Int = snax.utils.TestFp16.ulp(a.toDouble, b.toDouble)

  /** The domain a sum-of-squares mean actually occupies, sampled densely in the significand. */
  private def sweep: Seq[Float] = {
    val exps = -14 to 15
    for {
      e <- exps
      i <- 0 until 64
      m = 1.0 + i / 64.0
    } yield (m * math.pow(2.0, e)).toFloat
  }

  private def accuracy(hasExp: Boolean, hasSilu: Boolean, rsqN: Int, budget: Int, tag: String): Unit =
    test(new RsqDut(hasExp, hasSilu, rsqN)) { dut =>
      var worst  = 0
      var atWhat = ""
      for (x <- sweep) {
        dut.io.in.poke(bits(x).U)
        val got = fromBits(dut.io.out.peekInt())
        val ref = (1.0 / math.sqrt(x.toDouble)).toFloat
        // Only score where the true result is a normal FP16; outside that the narrow, not this unit, decides.
        if (math.abs(ref) >= 6.1e-5f && math.abs(ref) <= 65504.0f) {
          val u = f16ulp(got, ref)
          if (u > worst) { worst = u; atWhat = f"x=$x%.6g got=$got%.6g ref=$ref%.6g" }
        }
      }
      assert(worst <= budget, s"FpActivation RSQRT($tag): worst $worst FP16 ULP > $budget; at $atWhat")
      println(s"[FpActivationRsqrt] $tag: worst $worst FP16 ULP over ${sweep.length} points")
    }

  private def exactPowers(rsqN: Int): Unit =
    test(new RsqDut(false, false, rsqN)) { dut =>
      // rsqrt(2^(2k)) = 2^-k exactly: mantissa must be ZERO, not merely close.
      for (k <- -7 to 7) {
        val x = math.pow(2.0, 2 * k).toFloat
        dut.io.in.poke(bits(x).U)
        val got = dut.io.out.peekInt()
        val want = bits(math.pow(2.0, -k).toFloat)
        assert(got == want,
               f"FpActivation RSQRT: rsqrt(2^${2 * k}) must be exactly 2^${-k}; got 0x$got%08x want 0x$want%08x")
      }
    }

  private def totality(rsqN: Int): Unit =
    test(new RsqDut(true, true, rsqN)) { dut =>
      val zeros = Seq(
        bits(0.0f), bits(-0.0f), bits(-1.0f), bits(-1e30f), bits(1e-42f), bits(-1e-42f),
        BigInt(0x7f800000L), BigInt(0xff800000L), BigInt(0x7fc00000L) // +inf, -inf, NaN
      )
      for (b <- zeros) {
        dut.io.in.poke(b.U)
        val got = dut.io.out.peekInt()
        assert(got == 0, f"FpActivation RSQRT: input 0x$b%08x must flush to +0, got 0x$got%08x")
      }
    }

  "FpActivation_rsqrt_only" should "be within 1 FP16 ULP of 1/sqrt(x)" in {
    accuracy(false, false, 64, 1, "rsqrt-only, rsqN=64")
  }
  "FpActivation_rsqrt_with_exp_silu" should "be unchanged when all three are built" in {
    accuracy(true, true, 64, 1, "exp+silu+rsqrt, rsqN=64")
  }
  "FpActivation_rsqrt_exponent" should "be exact on powers of four" in { exactPowers(64) }
  "FpActivation_rsqrt_totality" should "flush non-positive and non-finite inputs to zero" in { totality(64) }
}
