package snax.simd

import scala.util.Random

import org.scalatest.flatspec.AnyFlatSpec
import chiseltest._
import snax.DataPathExtension.DataPathExtensionHarness
import snax.DataPathExtension.HasStreamMap

/** SimdStreamMap: LINEAR against an exact model (one fused a*x + b, one narrow); EXP, SILU and RSQRT bit for bit
  * against the xDMA's StreamMap, whose per-lane math is the same, built at the same 16 lanes with no time-sharing.
  */
class SimdStreamMapTester extends AnyFlatSpec with ChiselScalatestTester {
  import SimdFp._

  val dw     = 256
  val lanes  = dw / 16
  val funcs  = Seq("LINEAR_FP16", "EXP_FP16", "SILU_FP16", "RSQRT_FP16")
  val LINEAR = 0; val EXP = 1; val SILU = 2; val RSQRT = 3

  def f32(v: Float): BigInt = BigInt(bits32(v))

  val rnd = new Random(0x5a)
  def beats(n: Int, mag: Int) = Seq.fill(n)(Seq.fill(lanes)(randF16(rnd, mag)))

  // (name, func, a, b, rows, full rate)
  val linear    = Seq((2.0f, 1.0f), (-0.375f, 3.5f), (1.0f / 3.0f, -0.0f)).zipWithIndex.map { case ((a, b), i) =>
    val in  = beats(40, 100)
    val out = in.map(row => pack(row.map(x => f32ToF16(Math.fma(a, f16ToF32(x), b))), 16))
    (SimdTask(Seq(f32(a), f32(b), BigInt(LINEAR)), in.map(pack(_, 16)), in.length, s"linear$i"), out)
  }
  val nonlinear = Seq(
    (EXP, 1.0f, -8.0f, 6), // exp(x - 8) over x in [-6, 6]
    (SILU, 1.0f, 0.0f, 12),
    (RSQRT, 1.0f / 64, 0.0f, 100)
  ).map { case (fn, a, b, mag) =>
    val in = beats(64, mag).map(_.map(h => if (fn == RSQRT) h & 0x7fff else h)) // rsqrt: positive inputs
    SimdTask(Seq(f32(a), f32(b), BigInt(fn)), in.map(pack(_, 16)), in.length, s"func$fn")
  }

  behavior of "SimdStreamMap"

  it should "compute LINEAR exactly and stream one beat per cycle" in {
    test(new DataPathExtensionHarness(new HasSimdStreamMap(funcs, 16, dw)))
      .withAnnotations(SimdStreamDriver.annotations) { dut =>
        for ((gap, stall, mode) <- Seq((0.0, 0.0, "full rate"), (0.3, 0.4, "stalled"))) {
          val res = SimdStreamDriver.run(dut, linear.map(_._1), seed = 3, gap, stall)
          for (((t, expect), (got, held)) <- linear.zip(res)) {
            assert(got == expect, s"${t.name} ($mode): output differs from a*x+b")
            if (mode == "full rate") assert(held == 0, s"${t.name}: input held off $held cycles at full rate")
          }
        }
      }
  }

  // The reference run and the compared run are separate cases: chiseltest keys its build directory on the case.
  var ref: Seq[Seq[BigInt]] = Nil
  it should "run EXP, SILU and RSQRT on the xDMA StreamMap at 16 lanes (the reference)" in {
    test(new DataPathExtensionHarness(new HasStreamMap(16, funcs, 16, dw, 1)))
      .withAnnotations(SimdStreamDriver.annotations) { dut =>
        ref = SimdStreamDriver.run(dut, nonlinear, seed = 5, 0.0, 0.0).map(_._1)
      }
  }

  it should "match the xDMA StreamMap bit for bit on EXP, SILU and RSQRT" in {
    assert(ref.nonEmpty, "the reference case did not run")
    test(new DataPathExtensionHarness(new HasSimdStreamMap(funcs, 16, dw)))
      .withAnnotations(SimdStreamDriver.annotations) { dut =>
        for ((gap, stall) <- Seq((0.0, 0.0), (0.3, 0.4))) {
          val res = SimdStreamDriver.run(dut, nonlinear, seed = 9, gap, stall)
          for (((t, (got, _)), want) <- nonlinear.zip(res).zip(ref))
            assert(got == want, s"${t.name}: differs from the xDMA StreamMap")
        }
      }
  }
}
