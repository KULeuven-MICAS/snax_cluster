package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Tolerance-aware Tier-1 test for StreamReduce (the base DataPathExtensionTester does exact BigInt
  * compare, which doesn't fit FP tree-order + FP16-narrowing). Drives the same harness, then compares
  * the HW FP16 scalar (widened to f32) against an FP32 Scala golden within a relative tolerance.
  */
class StreamReduceTester extends AnyFlatSpec with ChiselScalatestTester {

  // ---- FP16 <-> FP32 helpers ----
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

  // round-to-nearest-even FP32 -> FP16 (used only to snap test inputs onto the FP16 grid)
  def f32ToF16bits(f: Float): Int = {
    val bits  = java.lang.Float.floatToIntBits(f)
    val sign  = (bits >>> 16) & 0x8000
    val rawe  = (bits >>> 23) & 0xff
    val mant  = bits & 0x7fffff
    if (rawe == 0xff) return sign | 0x7c00 | (if (mant != 0) 0x200 else 0)
    val exp = rawe - 127 + 15
    if (exp >= 0x1f) return sign | 0x7c00
    if (exp <= 0) {
      if (exp < -10) return sign
      val m       = mant | 0x800000
      val shift   = 14 - exp
      val half    = m >>> shift
      val rem     = m & ((1 << shift) - 1)
      val halfway = 1 << (shift - 1)
      var r       = half
      if (rem > halfway || (rem == halfway && (half & 1) == 1)) r += 1
      return sign | r
    }
    var h       = sign | (exp << 10) | (mant >>> 13)
    val rem     = mant & 0x1fff
    val halfway = 0x1000
    if (rem > halfway || (rem == halfway && (h & 1) == 1)) h += 1
    h
  }

  /** Build one 512-bit beat from 32 FP16 lane bit patterns (lane 0 in the low 16 bits). */
  def packBeat(lanesF16: Seq[Int]): BigInt =
    lanesF16.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (h, i)) => acc | (BigInt(h & 0xffff) << (16 * i)) }

  val lanes = 32

  /** Drive the DUT for one row and return the HW scalar (as f32) read from the output beat's low lane.
    * computeLanes>0 exercises the time-mux FSM (path B); ops selects which reductions are built. */
  def runReduce(op: Int, beats: Seq[Seq[Int]], computeLanes: Int = 32,
                ops: Seq[String] = Seq("MAX_FP16", "ADD_FP16", "SUMSQ_FP16")): Float = {
    var result: Float = 0.0f
    test(new DataPathExtensionHarness(new HasStreamReduce(computeLanes = computeLanes, op = ops, elementWidth = 16)))
      // Serial build: fpnew blackboxes (lzc.sv UNOPTFLAT) trip a Verilator PCH parallel-build race.
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.csr_i(0).poke(beats.length.U)
        dut.io.csr_i(1).poke(op.U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true)
        dut.clock.step(1)
        dut.io.start_i.poke(false)

        var threads = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          dut.io.data_i.valid.poke(true)
          for (b <- beats) {
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.io.data_i.bits.poke(packBeat(b))
            dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        threads = threads.fork {
          while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
          val out  = dut.io.data_o.bits.peekInt()
          val lane0 = (out & ((BigInt(1) << 16) - 1)).toInt
          result = f16bitsToF32(lane0)
          dut.io.data_o.ready.poke(true)
          dut.clock.step(1)
          dut.io.data_o.ready.poke(false)
        }
        threads.joinAndStep()
      }
    result
  }

  // ---- golden + check ----
  def goldenScalar(op: Int, vals: Seq[Float]): Float = op match {
    case 0 => vals.max                                  // MAX
    case 1 => vals.foldLeft(0.0f)(_ + _)                // ADD
    case 2 => vals.foldLeft(0.0f)((a, x) => a + x * x)  // SUMSQ
  }

  def checkOp(op: Int, opName: String, nBeats: Int, mag: Int, computeLanes: Int = 32,
              ops: Seq[String] = Seq("MAX_FP16", "ADD_FP16", "SUMSQ_FP16")): Unit = {
    val rng   = new Random(0x5EED + op)
    // FP16-representable inputs; magnitude kept small enough that the reduction stays in FP16 range
    // (SUMSQ over nBeats*32 squared terms must not overflow 65504).
    val beats = Seq.fill(nBeats)(Seq.fill(lanes) {
      val f = (rng.between(-mag, mag) + rng.nextInt(4) * 0.25f)
      f32ToF16bits(f)
    })
    val vals  = beats.flatten.map(f16bitsToF32)
    val golden = goldenScalar(op, vals)
    // narrow golden to FP16 grid for a fair compare with the HW (which narrows before output)
    val goldenF16 = f16bitsToF32(f32ToF16bits(golden))
    val hw        = runReduce(op, beats, computeLanes, ops)
    val tol       = math.max(math.abs(goldenF16) * 0.004f, 0.05f) // ~0.4% rel + small abs floor
    val err       = math.abs(hw - goldenF16)
    println(s"[StreamReduce:$opName cl=$computeLanes ops=${ops.mkString}] beats=$nBeats golden=$goldenF16 hw=$hw err=$err tol=$tol")
    assert(err <= tol, s"$opName mismatch: hw=$hw golden=$goldenF16 err=$err > tol=$tol")
  }

  "StreamReduce_MAX" should "match the FP golden" in { checkOp(0, "MAX", 8, 32) }
  "StreamReduce_ADD" should "match the FP golden" in { checkOp(1, "ADD", 8, 32) }
  "StreamReduce_SUMSQ" should "match the FP golden" in { checkOp(2, "SUMSQ", 8, 4) }

  // --- specialized configs ---
  // time-mux FSM (computeLanes=8 -> 4 sub-cycles/beat): MAX + ADD must still match.
  "StreamReduce_MAX_cl8" should "match (time-mux)" in { checkOp(0, "MAX", 8, 32, computeLanes = 8) }
  "StreamReduce_ADD_cl8" should "match (time-mux)" in { checkOp(1, "ADD", 8, 32, computeLanes = 8) }
  // single-op build (op CSR fixed, unused arithmetic dropped): ADD-only.
  "StreamReduce_ADDonly" should "match (op=ADD only)" in { checkOp(1, "ADD", 8, 32, ops = Seq("ADD_FP16")) }

  /** Tap mode (op | 0x100): the N input beats pass through unchanged, then one trailing scalar beat is
    * emitted (output = N+1 beats). The harness inserts -||> register cuts on both data ports, so use the
    * fork-based producer/consumer handshake. Returns all N+1 output beats (full 512-bit each). */
  def runReduceTap(op: Int, beats: Seq[Seq[Int]], computeLanes: Int = 32): Seq[BigInt] = {
    val outs = scala.collection.mutable.ArrayBuffer[BigInt]()
    test(new DataPathExtensionHarness(
      new HasStreamReduce(computeLanes = computeLanes, op = Seq("MAX_FP16", "ADD_FP16", "SUMSQ_FP16"), elementWidth = 16)))
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.csr_i(0).poke(beats.length.U)
        dut.io.csr_i(1).poke((op | 0x100).U) // tap bit[8]
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true)
        dut.clock.step(1)
        dut.io.start_i.poke(false)

        var threads = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          dut.io.data_i.valid.poke(true)
          for (b <- beats) {
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.io.data_i.bits.poke(packBeat(b))
            dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        threads = threads.fork {
          for (_ <- 0 to beats.length) { // N+1 output beats
            while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
            outs += dut.io.data_o.bits.peekInt()
            dut.io.data_o.ready.poke(true)
            dut.clock.step(1)
            dut.io.data_o.ready.poke(false)
          }
        }
        threads.joinAndStep()
      }
    outs.toSeq
  }

  def checkTap(op: Int, opName: String, nBeats: Int, mag: Int, computeLanes: Int = 32): Unit = {
    val rng   = new Random(0x7A9 + op)
    val beats = Seq.fill(nBeats)(Seq.fill(lanes) {
      val f = (rng.between(-mag, mag) + rng.nextInt(4) * 0.25f)
      f32ToF16bits(f)
    })
    val outs = runReduceTap(op, beats, computeLanes)
    assert(outs.length == nBeats + 1, s"$opName tap: expected ${nBeats + 1} beats, got ${outs.length}")
    for (i <- 0 until nBeats)
      assert(outs(i) == packBeat(beats(i)), s"$opName tap: passthrough beat $i changed")
    val vals      = beats.flatten.map(f16bitsToF32)
    val goldenF16 = f16bitsToF32(f32ToF16bits(goldenScalar(op, vals)))
    val hw        = f16bitsToF32((outs(nBeats) & 0xffff).toInt)
    val tol       = math.max(math.abs(goldenF16) * 0.004f, 0.05f)
    val err       = math.abs(hw - goldenF16)
    println(s"[StreamReduce:$opName:tap] beats=$nBeats passthrough OK; scalar golden=$goldenF16 hw=$hw err=$err tol=$tol")
    assert(err <= tol, s"$opName tap scalar mismatch: hw=$hw golden=$goldenF16 err=$err > tol=$tol")
  }

  "StreamReduce_ADD_tap" should "pass the row through and emit the sum" in { checkTap(1, "ADD", 8, 32) }
  "StreamReduce_MAX_tap" should "pass the row through and emit the max" in { checkTap(0, "MAX", 8, 32) }
  // tap in the time-mux FSM (computeLanes=8): passthrough beats + trailing scalar must still be correct.
  "StreamReduce_ADD_tap_cl8" should "tap-pass-through in the time-mux" in { checkTap(1, "ADD", 8, 32, computeLanes = 8) }
}
