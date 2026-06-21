package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Tier-1 test for StreamMap act(a*x+b). Uses a narrow datapath (8 FP16 lanes) so the EXP path's
  * per-lane FpExp stays cheap to build. Compares HW FP16 output (widened to f32) against an FP32 golden.
  */
class StreamMapTester extends AnyFlatSpec with ChiselScalatestTester {

  val testWidth = 512 // 32 FP16 lanes -> 8-lane time-mux over 4 sub-cycles (still only 8 FpExp built)
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
  def f32bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)
  def packBeat(ls: Seq[Int]): BigInt =
    ls.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (h, i)) => acc | (BigInt(h & 0xffff) << (16 * i)) }

  def run(a: Float, b: Float, act: Int, beats: Seq[Seq[Int]], computeLanes: Int = 8,
          func: Seq[String] = Seq("LINEAR_FP16", "EXP_FP16")): Seq[Seq[Float]] = {
    var outs = Seq[Seq[Float]]()
    test(new DataPathExtensionHarness(
      new HasStreamMap(dataWidth = testWidth, elementWidth = 16, computeLanes = computeLanes, func = func)))
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) {
        dut =>
          dut.io.csr_i(0).poke(f32bits(a).U)
          dut.io.csr_i(1).poke(f32bits(b).U)
          dut.io.csr_i(2).poke(act.U)
          dut.io.enable_i.poke(true)
          dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)

          var threads = new chiseltest.internal.TesterThreadList(Seq())
          threads = threads.fork {
            dut.io.data_i.valid.poke(true)
            for (bt <- beats) {
              while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
              dut.io.data_i.bits.poke(packBeat(bt)); dut.clock.step(1)
            }
            dut.io.data_i.valid.poke(false)
          }
          threads = threads.fork {
            for (_ <- beats.indices) {
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

  def golden(a: Float, b: Float, act: Int, beats: Seq[Seq[Int]]): Seq[Seq[Float]] =
    beats.map(_.map { h =>
      val t = a * f16bitsToF32(h) + b
      val r = if (act == 1) math.exp(t.toDouble).toFloat else t
      f16bitsToF32(f32ToF16bits(r))
    })

  def check(name: String, a: Float, b: Float, act: Int, computeLanes: Int = 8,
            func: Seq[String] = Seq("LINEAR_FP16", "EXP_FP16")): Unit = {
    val rng   = new Random(0xC0DE + act)
    val beats = Seq.fill(4)(Seq.fill(lanes)(f32ToF16bits(rng.between(-6, 6) + rng.nextInt(4) * 0.25f)))
    val hw    = run(a, b, act, beats, computeLanes, func)
    val gd    = golden(a, b, act, beats)
    var maxErr = 0.0f
    for ((hr, gr) <- hw.zip(gd); (h, g) <- hr.zip(gr)) {
      val tol = math.max(math.abs(g) * 0.01f, 0.02f)
      val e   = math.abs(h - g)
      if (e > maxErr) maxErr = e
      assert(e <= tol, s"$name mismatch: hw=$h golden=$g err=$e tol=$tol")
    }
    println(f"[StreamMap:$name cl=$computeLanes act=${func.mkString}] a=$a b=$b maxErr=$maxErr%.4g")
  }

  "StreamMap_affine" should "match a*x+b" in { check("affine", 2.0f, -1.5f, 0) }
  "StreamMap_exp" should "match exp(x - max)" in { check("exp", 1.0f, -3.0f, 1) }

  // --- specialized configs ---
  // computeLanes knob: 32 -> subCycles=1 (fully parallel), 16 -> subCycles=2.
  "StreamMap_affine_cl32" should "match (fully parallel)" in { check("affine", 2.0f, -1.5f, 0, computeLanes = 32) }
  "StreamMap_exp_cl16" should "match (subCycles=2)" in { check("exp", 1.0f, -3.0f, 1, computeLanes = 16) }
  // func=LINEAR: no FpExp built; affine only must still work.
  "StreamMap_affine_noexp" should "match with FpExp dropped" in {
    check("affine", 2.0f, -1.5f, 0, func = Seq("LINEAR_FP16"))
  }

  /** operandMode (csr(2) bits[9:8]): the scalar a is transformed once (1=1/a, 2=1/sqrt(a)) before the
    * per-lane a*x+b. Folds the reciprocal into the normalize map (softmax x/Σ, rmsnorm x*rms). */
  def checkOperand(name: String, aRaw: Float, b: Float, mode: Int): Unit = {
    val aEff = mode match {
      case 1 => 1.0f / aRaw
      case 2 => (1.0 / math.sqrt(aRaw.toDouble)).toFloat
      case _ => aRaw
    }
    val actCsr = mode << 8 // func LINEAR (bit0=0), operandMode in bits[9:8]
    val rng    = new Random(0xBEEF + mode)
    val beats  = Seq.fill(4)(Seq.fill(lanes)(f32ToF16bits(rng.between(-6, 6) + rng.nextInt(4) * 0.25f)))
    val hw     = run(aRaw, b, actCsr, beats)
    val gd     = beats.map(_.map(h => f16bitsToF32(f32ToF16bits(aEff * f16bitsToF32(h) + b))))
    var maxErr = 0.0f
    for ((hr, gr) <- hw.zip(gd); (h, g) <- hr.zip(gr)) {
      val tol = math.max(math.abs(g) * 0.02f, 0.03f) // NR recip ~machine eps + FP16 narrowing
      val e   = math.abs(h - g)
      if (e > maxErr) maxErr = e
      assert(e <= tol, s"$name mismatch: hw=$h golden=$g err=$e tol=$tol")
    }
    println(f"[StreamMap:$name] aRaw=$aRaw aEff=$aEff mode=$mode maxErr=$maxErr%.4g")
  }

  "StreamMap_recipA" should "match (1/a)*x" in { checkOperand("recipA", 8.0f, 0.0f, 1) }
  "StreamMap_rsqrtA" should "match (1/sqrt(a))*x" in { checkOperand("rsqrtA", 16.0f, 0.0f, 2) }
}
