package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Tier-1 test for StreamElementwise: per-lane FP binary combine of operandCount=2 interleaved beats
  * (out(i) = A(i) op B(i), op in {MUL, ADD}). FP16 transport, FP32-internal. Drives the harness with the
  * two operands as consecutive beats (as the AGU interleaves them), compares HW FP16 output (widened) to an
  * FP32 golden with a small tolerance. Exercises both PATH A (computeLanes=lanes, parallel) and PATH B
  * (computeLanes<lanes, time-mux), plus a single-op build.
  */
class StreamElementwiseTester extends AnyFlatSpec with ChiselScalatestTester {

  val testWidth = 512 // 32 FP16 lanes
  val lanes     = testWidth / 16

  def f16bitsToF32(h: Int): Float = {
    val sign = if ((h & 0x8000) != 0) -1.0 else 1.0
    val exp  = (h >> 10) & 0x1f
    val mant = h & 0x3ff
    val v: Double =
      if (exp == 0) sign * mant * math.pow(2, -24)
      else if (exp == 0x1f) if (mant == 0) sign * Double.PositiveInfinity else Double.NaN
      else sign * (1024 + mant) * math.pow(2, exp - 25)
    v.toFloat
  }
  def f32ToF16bits(f: Float): Int = {
    val bits = java.lang.Float.floatToIntBits(f)
    val sign = (bits >>> 16) & 0x8000
    val rawe = (bits >>> 23) & 0xff
    val mant = bits & 0x7fffff
    if (rawe == 0xff) return sign | 0x7c00 | (if (mant != 0) 0x200 else 0)
    val exp = rawe - 127 + 15
    if (exp >= 0x1f) return sign | 0x7c00
    if (exp <= 0) {
      if (exp < -10) return sign
      val m = mant | 0x800000; val shift = 14 - exp
      val half = m >>> shift; val rem = m & ((1 << shift) - 1); val halfway = 1 << (shift - 1)
      var r = half; if (rem > halfway || (rem == halfway && (half & 1) == 1)) r += 1; return sign | r
    }
    var h = sign | (exp << 10) | (mant >>> 13)
    val rem = mant & 0x1fff; if (rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1)) h += 1; h
  }
  def packBeat(ls: Seq[Int]): BigInt =
    ls.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (h, i)) => acc | (BigInt(h & 0xffff) << (16 * i)) }

  // op: 0=MUL, 1=ADD. pairs: Seq of (A_beat, B_beat); each beat is `lanes` FP16-bit ints.
  def run(op: Int, pairs: Seq[(Seq[Int], Seq[Int])], computeLanes: Int,
          opList: Seq[String] = Seq("MUL_FP16", "ADD_FP16")): Seq[Seq[Float]] = {
    var outs = Seq[Seq[Float]]()
    test(new DataPathExtensionHarness(
      new HasStreamElementwise(dataWidth = testWidth, elementWidth = 16, computeLanes = computeLanes, op = opList)))
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) {
        dut =>
          dut.io.csr_i(0).poke(2.U) // operandCount = 2 (binary)
          if (dut.io.csr_i.length > 1) dut.io.csr_i(1).poke(op.U)
          dut.io.enable_i.poke(true)
          dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)

          val inBeats = pairs.flatMap { case (a, b) => Seq(a, b) }
          var threads = new chiseltest.internal.TesterThreadList(Seq())
          threads = threads.fork {
            dut.io.data_i.valid.poke(true)
            for (bt <- inBeats) {
              while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
              dut.io.data_i.bits.poke(packBeat(bt)); dut.clock.step(1)
            }
            dut.io.data_i.valid.poke(false)
          }
          threads = threads.fork {
            for (_ <- pairs.indices) {
              while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
              val out = dut.io.data_o.bits.peekInt()
              outs = outs :+ (0 until lanes).map(i => f16bitsToF32(((out >> (16 * i)) & 0xffff).toInt))
              dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
            }
          }
          threads.joinAndStep()
      }
    outs
  }

  def golden(op: Int, pairs: Seq[(Seq[Int], Seq[Int])]): Seq[Seq[Float]] =
    pairs.map { case (a, b) =>
      a.zip(b).map { case (x, y) =>
        val r = if (op == 1) f16bitsToF32(x) + f16bitsToF32(y) else f16bitsToF32(x) * f16bitsToF32(y)
        f16bitsToF32(f32ToF16bits(r))
      }
    }

  def check(name: String, op: Int, computeLanes: Int, opList: Seq[String] = Seq("MUL_FP16", "ADD_FP16")): Unit = {
    val rng   = new Random(0xE1E + op + computeLanes)
    val pairs = Seq.fill(4)(
      (Seq.fill(lanes)(f32ToF16bits(rng.between(-4, 4) + rng.nextInt(4) * 0.25f)),
       Seq.fill(lanes)(f32ToF16bits(rng.between(-4, 4) + rng.nextInt(4) * 0.25f))))
    val hw = run(op, pairs, computeLanes, opList)
    val gd = golden(op, pairs)
    var maxErr = 0.0f
    for ((hr, gr) <- hw.zip(gd); (h, g) <- hr.zip(gr)) {
      val tol = math.max(math.abs(g) * 0.01f, 0.02f)
      val e   = math.abs(h - g)
      if (e > maxErr) maxErr = e
      assert(e <= tol, s"$name mismatch: hw=$h golden=$g err=$e tol=$tol")
    }
    println(f"[StreamElementwise:$name cl=$computeLanes] op=$op maxErr=$maxErr%.4g")
  }

  // PATH A (computeLanes = lanes = 32, fully parallel)
  "StreamElementwise_mul_cl32" should "match A*B" in { check("mul", 0, 32) }
  "StreamElementwise_add_cl32" should "match A+B" in { check("add", 1, 32) }
  // PATH B (computeLanes = 8, time-mux over 4 sub-cycles)
  "StreamElementwise_mul_cl8" should "match A*B (time-mux)" in { check("mul", 0, 8) }
  "StreamElementwise_add_cl8" should "match A+B (time-mux)" in { check("add", 1, 8) }
  // single-op build (userCsrNum shrinks to 1; no op-select CSR)
  "StreamElementwise_mul_only" should "match A*B with a single-op build" in {
    check("mulonly", 0, 8, opList = Seq("MUL_FP16"))
  }
}
