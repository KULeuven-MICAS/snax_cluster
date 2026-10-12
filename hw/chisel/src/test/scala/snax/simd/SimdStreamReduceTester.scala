package snax.simd

import scala.util.Random

import org.scalatest.flatspec.AnyFlatSpec
import chiseltest._
import snax.DataPathExtension.DataPathExtensionHarness

/** SimdStreamReduce against a bit-exact model: the same partial assignment (beat k into partial k mod accLat), the same
  * fused accumulate, the same balanced folds. Every task runs twice, once at full rate -- where a task that should
  * stream must never hold its input off -- and once under random input gaps and output stalls.
  */
class SimdStreamReduceTester extends AnyFlatSpec with ChiselScalatestTester {
  import SimdFp._

  val MAX = 0; val ADD         = 1; val SUMSQ        = 2
  val TAP = 0x100; val FP32OUT = 0x200; val LANEWISE = 0x400

  /** The reduction of one row as the hardware computes it: (per-lane results, scalar). */
  def golden(op: Int, row: Seq[Seq[Int]], partials: Int, lanes: Int): (Seq[Float], Float) = {
    val part = Array.fill(partials, lanes)(0.0f)
    for {
      (beat, k) <- row.zipWithIndex
      i         <- 0 until lanes
    } {
      val p    = k % partials
      val x    = f16ToF32(beat(i))
      val seed = k < partials
      part(p)(i) = op match {
        case MAX   => if (seed) x else max32(x, part(p)(i))
        case ADD   => Math.fma(x, 1.0f, if (seed) 0.0f else part(p)(i))
        case SUMSQ => Math.fma(x, x, if (seed) 0.0f else part(p)(i))
      }
    }
    val node: (Float, Float) => Float = if (op == MAX) max32 else (a, b) => a + b
    val id      = if (op == MAX) Float.NegativeInfinity else 0.0f
    val written = math.min(row.length, partials)
    val perLane = (0 until lanes).map(i => tree((0 until partials).map(p => if (p < written) part(p)(i) else id))(node))
    (perLane, tree(perLane)(node))
  }

  case class Case(name: String, op: Int, flags: Int, rowLen: Int, rows: Int, mag: Int, fullRate: Boolean)

  def check(hasMax: Boolean, hasFma: Boolean, dw: Int, fpPipe: Int, treePipe: Int): Unit = {
    val lanes    = dw / 16
    val partials = 2 * fpPipe + 1
    val ops      = (if (hasFma) Seq("FMA_FP16") else Nil) ++ (if (hasMax) Seq("MAX_FP16") else Nil)
    val all      = Seq(
      Case("max_row8", MAX, 0, 8, 1, 40, true),
      Case("add_row64", ADD, 0, 64, 1, 4, true),
      Case("sumsq_row5", SUMSQ, 0, 5, 1, 2, true),
      Case("add_rows2x6", ADD, 0, 2, 6, 8, true),
      Case("max_rows1x5", MAX, 0, 1, 5, 40, true),
      Case("sumsq_rows3x4", SUMSQ, 0, 3, 4, 4, true),
      // a long row loads every partial with large values; the short rows after it must not see them
      Case("add_long", ADD, 0, 16, 1, 1000, true),
      Case("add_rows1x3_after_long", ADD, 0, 1, 3, 4, true),
      Case("add_rows2x3", ADD, 0, 2, 3, 4, true),
      Case("max_long", MAX, 0, 16, 1, 60000, true),
      Case("max_rows2x3_after_long", MAX, 0, 2, 3, 4, true),
      Case("max_lanewise_row64", MAX, LANEWISE, 64, 1, 40, true),
      Case("add_lanewise_rows2x4", ADD, LANEWISE, 2, 4, 8, true),
      Case("add_lanewise_tap_row33", ADD, LANEWISE | TAP, 33, 1, 4, true),
      Case("add_tap_row7", ADD, TAP, 7, 1, 8, true),
      Case("max_tap_rows4x3", MAX, TAP, 4, 3, 8, false), // the next row waits for the result: not full rate
      Case("sumsq_fp32out_row8", SUMSQ, FP32OUT, 8, 1, 4096, true),
      Case("add_fp32out_rows2x2", ADD, FP32OUT, 2, 2, 8, true)
    ).filter(c => if (c.op == MAX) hasMax else hasFma)

    val rnd   = new Random(dw * 131 + fpPipe * 17 + treePipe)
    val built = all.map { c =>
      val rowsData = Seq.fill(c.rows)(Seq.fill(c.rowLen)(Seq.fill(lanes)(randF16(rnd, c.mag))))
      val expect   = rowsData.flatMap { row =>
        val (perLane, scalar) = golden(c.op, row, partials, lanes)
        val result            =
          if ((c.flags & LANEWISE) != 0) pack(perLane.map(f32ToF16), 16)
          else if ((c.flags & FP32OUT) != 0) pack(Seq.fill(dw / 32)(bits32(scalar).toInt), 32)
          else pack(Seq.fill(lanes)(f32ToF16(scalar)), 16)
        (if ((c.flags & TAP) != 0) row.map(pack(_, 16)) else Nil) :+ result
      }
      (
        c,
        SimdTask(
          Seq(BigInt(c.rowLen), BigInt(c.op | c.flags)),
          rowsData.flatten.map(pack(_, 16)),
          expect.length,
          c.name
        ),
        expect
      )
    }

    test(new DataPathExtensionHarness(new HasSimdStreamReduce(ops, 16, dw, fpPipe, treePipe)))
      .withAnnotations(SimdStreamDriver.annotations) { dut =>
        for ((inGap, outStall, mode) <- Seq((0.0, 0.0, "full rate"), (0.3, 0.4, "stalled"))) {
          val res = SimdStreamDriver.run(dut, built.map(_._2), seed = 7, inGap, outStall)
          for (((c, _, expect), (got, held)) <- built.zip(res)) {
            for ((e, k) <- expect.zipWithIndex)
              assert(got(k) == e, f"${c.name} ($mode): output beat $k is ${got(k)}%x, expected $e%x")
            if (mode == "full rate" && c.fullRate)
              assert(held == 0, s"${c.name}: input held off for $held cycles at full rate")
          }
        }
      }
  }

  behavior of "SimdStreamReduce"
  it should "match the model at the cfg point (16 lanes, 3 partials)" in { check(true, true, 256, 1, 1) }
  it should "match the model with 5 partials and a combinational fold" in { check(true, true, 256, 2, 0) }
  it should "match the model as an FMA-only build, 8 lanes, 1 partial" in { check(false, true, 128, 0, 2) }
  it should "match the model as a MAX-only build, 32 lanes" in { check(true, false, 512, 1, 1) }
}
