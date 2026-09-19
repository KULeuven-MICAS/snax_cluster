package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Tier-1 for `Int32ToFp32` -- the INT32 accumulator's way onto a floating-point collective.
  *
  * The golden is `v.toFloat`, which IS the IEEE-754 round-to-nearest-even conversion, so this compares BIT
  * PATTERNS and not a tolerance. Three things are pinned:
  *
  *   1. every lane of a full beat converts exactly, over the whole INT32 range and at both pipe depths,
  *   2. the values that motivated the block -- flash-attention numerators around 3e6 -- are EXACT, i.e. the
  *      round-trip back to an integer is lossless, which is the property FP16 cannot offer at that magnitude,
  *   3. the map is width-preserving and streams at one beat per cycle, so putting it on a reader chain costs
  *      bandwidth nothing.
  */
class Int32ToFp32Tester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))
  private val dataWidth = 512
  private val lanes     = dataWidth / 32

  private def f32bits(v: Int):  BigInt = BigInt(java.lang.Float.floatToRawIntBits(v.toFloat).toLong & 0xffffffffL)
  private def lane(b: BigInt, i: Int): BigInt = (b >> (32 * i)) & ((BigInt(1) << 32) - 1)
  private def pack(v: Seq[Int]): BigInt = {
    var b = BigInt(0)
    for ((x, i) <- v.zipWithIndex) b |= (BigInt(x.toLong & 0xffffffffL) << (32 * i))
    b
  }

  /** stream `beats` through the extension, returning the converted beats in order */
  private def run(dut: DataPathExtensionHarness, beats: Seq[BigInt]): Seq[BigInt] = {
    dut.io.csr_i(0).poke(0.U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    dut.io.data_o.ready.poke(true)
    var got = Seq.empty[BigInt]
    var fed = 0
    var cyc = 0
    dut.io.data_i.valid.poke(true)
    dut.io.data_i.bits.poke(beats.head.U)
    while (got.length < beats.length && cyc < beats.length * 20 + 500) {
      val canFeed = fed < beats.length && dut.io.data_i.ready.peekBoolean()
      val outNow  = dut.io.data_o.valid.peekBoolean()
      val outBits = if (outNow) dut.io.data_o.bits.peekInt() else BigInt(0)
      dut.clock.step(1); cyc += 1
      if (canFeed) {
        fed += 1
        if (fed < beats.length) dut.io.data_i.bits.poke(beats(fed).U) else dut.io.data_i.valid.poke(false)
      }
      if (outNow) got = got :+ outBits
    }
    assert(got.length == beats.length, s"only ${got.length}/${beats.length} beats retired")
    got
  }

  for (numPipe <- Seq(0, 1)) {
    s"Int32ToFp32_exact_p$numPipe" should
      s"convert every lane bit-exactly at numPipe=$numPipe" in {
        test(new DataPathExtensionHarness(new HasInt32ToFp32(dataWidth, numPipe)))
          .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
            val rng = new Random(0x1f32 + numPipe)
            // corner cases first, then the whole range, then the flash-attention magnitude band
            val corners = Seq(0, 1, -1, Int.MaxValue, Int.MinValue, Int.MinValue + 1,
                              (1 << 24) - 1, 1 << 24, (1 << 24) + 1, -(1 << 24) - 1, 1 << 23, -(1 << 23))
            val wide    = Seq.fill(lanes * 6)(rng.nextInt())
            val flash   = Seq.fill(lanes * 4)(rng.between(-4000000, 4000000))
            val all     = corners ++ wide ++ flash
            val padded  = all ++ Seq.fill((lanes - all.length % lanes) % lanes)(0)
            val beats   = padded.grouped(lanes).map(g => pack(g)).toSeq
            val got     = run(dut, beats)
            for ((g, bi) <- got.zipWithIndex; i <- 0 until lanes) {
              val v = padded(bi * lanes + i)
              assert(lane(g, i) == f32bits(v),
                f"numPipe=$numPipe beat $bi lane $i: v=$v got 0x${lane(g, i)}%08x want 0x${f32bits(v)}%08x")
            }
            println(s"[Int32ToFp32] ${padded.length} values, numPipe=$numPipe -- every lane bit-exact")
          }
      }
  }

  "Int32ToFp32_flashRange" should "carry a flash-attention numerator with NO loss, where FP16 saturates" in {
    test(new DataPathExtensionHarness(new HasInt32ToFp32(dataWidth, 0)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        // sum P8.V8 over a KV tile: ~127 . l . 127. FP16's ceiling is 65504, so Int32ToFp16 saturates to +-Inf
        // for any l > 4. Here the conversion is not merely finite, it is EXACT: |v| < 2^24 has no round bits.
        val vals   = Seq(3175000, 12700, 1270000, 16777215, -3175000, -16777215) ++ Seq.fill(lanes - 6)(0)
        val got    = run(dut, Seq(pack(vals)))
        for (i <- 0 until 6) {
          val bits = lane(got.head, i)
          val back = java.lang.Float.intBitsToFloat(bits.toInt)
          assert(bits == f32bits(vals(i)), f"lane $i bits")
          assert(back.toLong == vals(i).toLong, s"lane $i: $back did not round-trip to ${vals(i)}")
        }
        println("[Int32ToFp32] flash numerators up to 2^24-1 convert EXACTLY (round-trip lossless)")
      }
  }

  "Int32ToFp32_throughput" should "stream one beat per cycle, width-preserving" in {
    test(new DataPathExtensionHarness(new HasInt32ToFp32(dataWidth, 1)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng   = new Random(0x7777)
        val n     = 64
        val beats = Seq.fill(n)(pack(Seq.fill(lanes)(rng.nextInt())))
        dut.io.csr_i(0).poke(0.U); dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.data_o.ready.poke(true)
        dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beats.head.U)
        var cyc = 0; var fed = 0; var got = 0; var warm = -1; var last = -1
        while (got < n && cyc < n * 20 + 500) {
          val canFeed = fed < n && dut.io.data_i.ready.peekBoolean()
          val outNow  = dut.io.data_o.valid.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) {
            fed += 1
            if (fed == 4) warm = cyc
            last            = cyc
            if (fed < n) dut.io.data_i.bits.poke(beats(fed).U) else dut.io.data_i.valid.poke(false)
          }
          if (outNow) got += 1
        }
        val util = (n - 4).toDouble / (last - warm)
        println(f"[Int32ToFp32] stream util = $util%.3f beat/cycle over $n beats")
        assert(util > 0.95, f"converter should not throttle the reader chain, got $util%.3f")
      }
  }
}
