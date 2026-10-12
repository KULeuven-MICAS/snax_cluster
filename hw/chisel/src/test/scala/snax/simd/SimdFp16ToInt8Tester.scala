package snax.simd

import scala.util.Random

import org.scalatest.flatspec.AnyFlatSpec
import chiseltest._
import snax.DataPathExtension.DataPathExtensionHarness

/** SimdFp16ToInt8 against an exact model of the PE (FP32 product, clamp, magic-number round, saturate) and of the three
  * byte layouts: pack, 4-beat interleave, and a tail beat passed through unquantised.
  */
class SimdFp16ToInt8Tester extends AnyFlatSpec with ChiselScalatestTester {
  import SimdFp._

  val ILV = 1 << 16

  def quant(x: Int, inv: Float): Int = {
    val scaled  = (inv.toDouble * f16ToF32(x).toDouble).toFloat // exact product, one rounding
    val clamped = math.max(math.min(scaled, 128.0f), -128.0f)
    val r       = (clamped + 12582912.0f)                       // 1.5 * 2^23: rounds to an integer, ties to even
    val i       = java.lang.Float.floatToRawIntBits(r) - 0x4b400000
    math.max(-127, math.min(127, i))
  }

  /** The output beats of one stream: groups of 2 (pack) or 4 (interleave) beats, a tail every `period` + 1. */
  def golden(in: Seq[Seq[Int]], inv: Float, ilv: Boolean, period: Int, lanes: Int): Seq[BigInt] = {
    val outBytes = 2 * lanes
    val out      = scala.collection.mutable.ArrayBuffer[BigInt]()
    var group    = scala.collection.mutable.ArrayBuffer[Seq[Int]]()
    var since    = 0
    for (beat <- in) {
      if (period != 0 && since == period) { out += pack(beat, 16); since = 0 }
      else {
        since += 1
        group += beat.map(quant(_, inv))
        val g = if (ilv) 4 else 2
        if (group.length == g) {
          val bytes = Array.fill(g / 2 * outBytes)(0)
          for {
            (qs, b) <- group.zipWithIndex
            (q, i)  <- qs.zipWithIndex
          } {
            val at = if (ilv) (i / (outBytes / 4)) * outBytes + (i % (outBytes / 4)) * 4 + b else b * lanes + i
            bytes(at) = q & 0xff
          }
          bytes.grouped(outBytes).foreach(bs => out += pack(bs.toSeq, 8))
          group = scala.collection.mutable.ArrayBuffer[Seq[Int]]()
        }
      }
    }
    out.toSeq
  }

  def check(dw: Int, fpPipe: Int): Unit = {
    val lanes   = dw / 16
    val latency = 1 + 2 + 3 * fpPipe // input register + Fp16ToInt8PE
    val rnd     = new Random(dw + 3 * fpPipe)
    def beats(n: Int) = Seq.fill(n)(Seq.fill(lanes)(randF16(rnd, 300)))
    val cases = Seq(
      ("pack", 0.5f, false, 0, 32),
      ("interleave", 0.4375f, true, 0, 32),
      ("pack_tail4", 1.0f, false, 4, 5 * 6),       // 6 rows of 4 beats + their tail
      ("interleave_tail8", 0.25f, true, 8, 9 * 4), // 4 rows of 8 beats + their tail
      ("pack_tail2", 0.75f, false, 2, 3 * 8)       // the shortest legal period: tails close together
    ).map { case (name, inv, ilv, period, n) =>
      val in     = beats(n)
      val expect = golden(in, inv, ilv, period, lanes)
      val csr1   = BigInt((if (ilv) ILV else 0) | period)
      (SimdTask(Seq(BigInt(bits32(inv)), csr1), in.map(pack(_, 16)), expect.length, name), expect)
    }
    test(new DataPathExtensionHarness(new HasSimdFp16ToInt8(dw, fpPipe, tailPassthrough = 1, interleave = 1)))
      .withAnnotations(SimdStreamDriver.annotations) { dut =>
        for ((gap, stall, mode) <- Seq((0.0, 0.0, "full rate"), (0.3, 0.4, "stalled"))) {
          val res = SimdStreamDriver.run(dut, cases.map(_._1), seed = 13, gap, stall)
          for (((t, expect), (got, held)) <- cases.zip(res)) {
            for ((e, k) <- expect.zipWithIndex)
              assert(got(k) == e, f"${t.name} ($mode): beat $k is ${got(k)}%x, expected $e%x")
            // A tail waits for the previous tail, so tails closer together than the PE latency hold the input.
            val period = (t.csr(1) & 0xffff).toInt
            if (mode == "full rate" && (period == 0 || period >= latency))
              assert(held == 0, s"${t.name}: input held off $held cycles at full rate")
          }
        }
      }
  }

  behavior of "SimdFp16ToInt8"
  it should "pack, interleave and pass tails exactly (16 lanes)" in { check(256, 1) }
  it should "do the same at 32 lanes with a deeper PE" in { check(512, 2) }
}
