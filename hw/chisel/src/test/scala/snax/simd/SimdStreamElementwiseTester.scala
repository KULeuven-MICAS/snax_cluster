package snax.simd

import scala.util.Random

import org.scalatest.flatspec.AnyFlatSpec
import chiseltest._
import snax.DataPathExtension.DataPathExtensionHarness

/** SimdStreamElementwise against an exact model: MUL = fma(x, B, +0), ADD = fma(x, 1, B), one narrow. */
class SimdStreamElementwiseTester extends AnyFlatSpec with ChiselScalatestTester {
  import SimdFp._

  val MUL = 0; val ADD = 1; val STICKY = 0x100

  def combine(op: Int, b: Int, x: Int): Int = {
    val bf = f16ToF32(b); val xf = f16ToF32(x)
    f32ToF16(if (op == MUL) Math.fma(xf, bf, 0.0f) else Math.fma(xf, 1.0f, bf))
  }

  def tasks(lanes: Int, rnd: Random): Seq[(SimdTask, Seq[BigInt])] = {
    def beats(n:     Int) = Seq.fill(n)(Seq.fill(lanes)(randF16(rnd, 60)))
    def lanewise(op: Int)(b: Seq[Int], x: Seq[Int]) = pack(b.zip(x).map { case (bi, xi) => combine(op, bi, xi) }, 16)
    Seq(MUL, ADD).flatMap { op =>
      val p = beats(24) // 12 pairs, B first
      val s = beats(33) // the B beat, then 32 x
      val t = beats(9)
      Seq(
        SimdTask(Seq[BigInt](2, op), p.map(pack(_, 16)), 12, s"pairs_op$op")           ->
          p.grouped(2).map { case Seq(b, x) => lanewise(op)(b, x) }.toSeq,
        SimdTask(Seq[BigInt](1, op | STICKY), s.map(pack(_, 16)), 32, s"sticky_op$op") ->
          s.tail.map(x => lanewise(op)(s.head, x)),
        SimdTask(Seq[BigInt](1, op), t.map(pack(_, 16)), 9, s"pass_op$op")             -> t.map(pack(_, 16))
      )
    }
  }

  def check(dw: Int, fpPipe: Int): Unit = {
    val lanes = dw / 16
    val all   = tasks(lanes, new Random(dw + fpPipe))
    test(new DataPathExtensionHarness(new HasSimdStreamElementwise(Seq("FMA_FP16"), 16, dw, fpPipe)))
      .withAnnotations(SimdStreamDriver.annotations) { dut =>
        for ((gap, stall, mode) <- Seq((0.0, 0.0, "full rate"), (0.3, 0.4, "stalled"))) {
          val res = SimdStreamDriver.run(dut, all.map(_._1), seed = 11, gap, stall)
          for (((t, expect), (got, held)) <- all.zip(res)) {
            for ((e, k) <- expect.zipWithIndex)
              assert(got(k) == e, f"${t.name} ($mode): beat $k is ${got(k)}%x, expected $e%x")
            if (mode == "full rate") assert(held == 0, s"${t.name}: input held off $held cycles at full rate")
          }
        }
      }
  }

  behavior of "SimdStreamElementwise"
  it should "combine pairs, a sticky B and a pass-through exactly (16 lanes)" in { check(256, 1) }
  it should "do the same with a combinational FMA, 8 lanes" in { check(128, 0) }
  it should "do the same with a two-stage FMA" in { check(256, 2) }
}
