package snax.DataPathJunction

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathExtension.FpHelpers
import snax.DataPathJunction.ElementwiseJunction.FMT_FP32
import snax.DataPathJunction.JunctionTestUtils._

/** \============================================================================================================
  * The monoid junction WITHOUT the ordered scan: the instance the split cluster ships.
  * \============================================================================================================
  *
  * The softmax statistics beat is `F = 2, S = 8`: eight slots of a key and one value, 16 FP32 lanes. An instance built
  * with `hasScan = false` is the same operator for that beat without the scan's delay line, and these tests hold it to
  * that:
  *
  *   - it produces the SAME BEATS, bit for bit, as the instance that has the scan, for every (m, l) word, for both key
  *     polarities and for short beats (identity slots), and it publishes the same identity;
  *   - what it cannot fold is REPORTED: the ordered scan, and any transport other than FP32;
  *   - it keeps the rate: one beat pair per cycle.
  *
  * The instance with the scan is the reference, so the comparison is against the netlist the other testers already pin
  * to double-precision goldens and frozen beats. Two `test(...)` builds in one scalatest case collide on the Verilator
  * build directory, so the reference and the instance under test are separate cases that share their results through
  * the class.
  */
class MonoidNoScanTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  /** the geometry word of the (m, l) statistics: n = 1, nExp = 1, sigma = 3, FP32 */
  private def statsWord(nValid: Int, keyPol: Int = 0): BigInt =
    (BigInt(keyPol) << 28) | (BigInt(3) << 26) | (BigInt(1) << 18) | (BigInt(FMT_FP32) << 12) | (BigInt(1) << 8) |
      BigInt(nValid)

  /** eight keys on lanes 0..7 and eight values on lanes 8..15 */
  private def statsBeat(keys: Seq[Double], vals: Seq[Double]): BigInt = {
    var b = BigInt(0)
    for (k <- 0 until 8) { b |= f32(keys(k)) << (32 * k); b |= f32(vals(k)) << (32 * (8 + k)) }
    b
  }

  private case class Case(label: String, csr: BigInt, a: BigInt, b: BigInt)

  /** The same stimulus for both instances: both key polarities, full and short beats. */
  private val cases: Seq[Case] = {
    val rng = new Random(0x57a7)
    for {
      pol    <- Seq(0, 1)
      nValid <- Seq(1, 3, 8)
      rep    <- 0 until 6
    } yield {
      def keys = Seq.fill(8)(rng.between(-6.0, 6.0))
      def vals = Seq.fill(8)(rng.between(0.25, 8.0))
      Case(s"pol=$pol nValid=$nValid #$rep", statsWord(nValid, pol), statsBeat(keys, vals), statsBeat(keys, vals))
    }
  }

  private var reference: Seq[(BigInt, BigInt)] = Seq() // (folded beat, published identity) per case

  private def run(dut: DataPathJunctionHarness): Seq[(BigInt, BigInt)] =
    cases.map { c =>
      val out = runPair(dut, c.csr, c.a, c.b)
      dut.io.csr_i(0).poke(c.csr.U); dut.io.enable_i.poke(true); dut.clock.step(2)
      (out, dut.io.identity_o.peekInt())
    }

  /** hold a CSR word with the operator armed, and read the (stream-constant) verdict */
  private def verdict(dut: DataPathJunctionHarness, csr: BigInt): Boolean = {
    dut.io.csr_i(0).poke(csr.U); dut.io.enable_i.poke(true); dut.clock.step(2)
    dut.io.cfgerr_o.peekBoolean()
  }

  private def word(n: Int, nExp: Int, sigma: Int, nValid: Int, fmt: Int = FMT_FP32, keyMul: Int = 0): BigInt =
    (BigInt(keyMul) << 29) | (BigInt(sigma) << 26) | (BigInt(nExp) << 18) | (BigInt(fmt) << 12) | (BigInt(n) << 8) |
      BigInt(nValid)

  "NoScan_reference" should "record the beats of the instance that has the scan, for the (m, l) words" in {
    test(new DataPathJunctionHarness(new HasMonoidJunction()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        reference = run(dut)
        assert(reference.length == cases.length)
        println(s"[NoScan] reference: ${cases.length} (m, l) beats recorded from the instance with the scan")
      }
  }

  "NoScan_bitExact" should "produce the same beats and the same identity as the instance with the scan" in {
    assert(reference.nonEmpty, "run NoScan_reference first")
    test(new DataPathJunctionHarness(new HasMonoidJunction(hasScan = false)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val got = run(dut)
        for (((c, (refOut, refId)), (out, id)) <- cases.zip(reference).zip(got)) {
          assert(out == refOut, f"${c.label}: beat differs\n  scan    ${refOut}%x\n  no scan ${out}%x")
          assert(id == refId, f"${c.label}: published identity differs")
        }
        println(s"[NoScan] ${cases.length} (m, l) beats are bit-identical to the instance with the scan")
      }
  }

  "NoScan_cfgerr" should "report the scan and any transport other than FP32, and nothing legal" in {
    test(new DataPathJunctionHarness(new HasMonoidJunction(hasScan = false)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        // the statistics words, and the widest layouts that fit 16 lanes, are legal
        assert(!verdict(dut, statsWord(8)), "the (m, l) word must not be flagged")
        assert(!verdict(dut, statsWord(8, keyPol = 1)), "the min-monoid statistics word is legal")
        assert(!verdict(dut, word(n = 15, nExp = 15, sigma = 0, nValid = 1)), "F = 16, S = 1 fits")
        assert(!verdict(dut, word(n = 3, nExp = 3, sigma = 2, nValid = 4)), "F = 4, S = 4 fits")
        // the ordered scan is not built: the bit is tied off, so the word would fold as the max-monoid
        assert(verdict(dut, statsWord(8) | (BigInt(1) << 29)), "keyMul must be reported without the scan")
        // a transport this operator does not build
        for (f <- Seq(FpHelpers.FMT_FP16, FpHelpers.FMT_BF16, FpHelpers.FMT_FP8))
          assert(verdict(dut, word(n = 1, nExp = 1, sigma = 3, nValid = 8, fmt = f)), s"fmt $f is not FP32")
        println("[NoScan] the ordered scan and every transport other than FP32 are reported by cfgerr")
      }
  }

  /** Stream `beats` back to back through an armed junction and return (first-out cycle, steady pairs per cycle). With
    * `stallOut`, the sink accepts only every other cycle, which is the control that proves the meter can read below 1.
    */
  private def stream(dut: DataPathJunctionHarness, csr: BigInt, beats: Seq[BigInt], stallOut: Boolean): (Int, Double) = {
    val n = beats.length
    dut.io.csr_i(0).poke(csr.U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    dut.io.a_i.valid.poke(true); dut.io.b_i.valid.poke(true)
    dut.io.a_i.bits.poke(beats(0).U); dut.io.b_i.bits.poke(beats(0).U)
    var cyc = 0; var fed = 0; var rec = 0; var first = -1; var warm = -1; var last = -1
    while (rec < n && cyc < n * 40 + 600) {
      dut.io.out_o.ready.poke(!stallOut || cyc % 2 == 0)
      val canFeed = fed < n && dut.io.a_i.ready.peekBoolean() && dut.io.b_i.ready.peekBoolean()
      val outNow  = dut.io.out_o.valid.peekBoolean() && dut.io.out_o.ready.peekBoolean()
      dut.clock.step(1); cyc += 1
      if (canFeed) {
        fed += 1
        if (fed == 8) warm = cyc
        last = cyc
        if (fed < n) { dut.io.a_i.bits.poke(beats(fed).U); dut.io.b_i.bits.poke(beats(fed).U) }
        else { dut.io.a_i.valid.poke(false); dut.io.b_i.valid.poke(false) }
      }
      if (outNow) { if (first < 0) first = cyc; rec += 1 }
    }
    dut.io.out_o.ready.poke(true)
    assert(rec == n, s"only $rec/$n beats retired")
    var w = 0; while (dut.io.busy_o.peekBoolean() && w < 400) { dut.clock.step(1); w += 1 }
    (first, if (last > warm) (n - 8).toDouble / (last - warm) else 0.0)
  }

  "NoScan_throughput" should "take one full (m, l) beat pair every cycle" in {
    val rng = new Random(0x7a11)
    test(new DataPathJunctionHarness(new HasMonoidJunction(hasScan = false)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val beats = Seq.fill(64)(statsBeat(Seq.fill(8)(rng.between(-6.0, 6.0)), Seq.fill(8)(rng.between(0.25, 8.0))))
        val csr          = statsWord(nValid = 8)
        val (first, utl) = stream(dut, csr, beats, stallOut = false)
        assert(utl > 0.97, f"the instance must accept a beat pair every cycle, got $utl%.3f")
        val (_, stalled) = stream(dut, csr, beats, stallOut = true)
        assert(stalled < 0.6, f"negative control, a sink that accepts every other cycle must read well below 1, got $stalled%.3f")
        println(f"[NoScan] first-out $first CC, steady $utl%.3f pairs/cycle (sink at half rate: $stalled%.3f), 8 slots x 2 fields = 16 live lanes")
      }
  }
}
