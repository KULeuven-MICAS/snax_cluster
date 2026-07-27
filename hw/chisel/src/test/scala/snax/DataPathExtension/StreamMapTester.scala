package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Tier-1 test for StreamMap act(a*x+b). Uses a narrow datapath (8 FP16 lanes) so the EXP/SILU path's
  * per-lane FpActivation stays cheap to build. Compares HW FP16 output (widened to f32) against an FP32 golden.
  */
class StreamMapTester extends AnyFlatSpec with ChiselScalatestTester {

  val testWidth = 512 // 32 FP16 lanes -> 8-lane time-mux over 4 sub-cycles (still only 8 activation cores built)
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
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) {
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
      val r =
        if (act == 1) math.exp(t.toDouble).toFloat
        else if (act == 2) (t.toDouble / (1.0 + math.exp(-t.toDouble))).toFloat // silu = x*sigmoid(x)
        else t
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
  "StreamMap_silu" should "match x*sigmoid(x)" in {
    check("silu", 1.0f, 0.0f, 2, func = Seq("LINEAR_FP16", "EXP_FP16", "SILU_FP16"))
  }

  // --- specialized configs ---
  // computeLanes knob: 32 -> subCycles=1 (fully parallel), 16 -> subCycles=2.
  "StreamMap_affine_cl32" should "match (fully parallel)" in { check("affine", 2.0f, -1.5f, 0, computeLanes = 32) }
  "StreamMap_exp_cl16" should "match (subCycles=2)" in { check("exp", 1.0f, -3.0f, 1, computeLanes = 16) }
  // func=LINEAR: no FpActivation built; affine only must still work.
  "StreamMap_affine_noexp" should "match with the activation dropped" in {
    check("affine", 2.0f, -1.5f, 0, func = Seq("LINEAR_FP16"))
  }

  // --- streaming-pipeline stress tests (exercise the new continuous-issue FSM) ---

  /** Like [[run]] but the consumer holds ready low for `outStall` cycles first, then drains one beat at a
    * time. Feeds many beats so the credit counter saturates: input `ready` must drop (throttle) while the
    * output Queue fills to Qdepth, then recover as beats dequeue — with every beat still correct + in order.
    */
  def runStall(a: Float, b: Float, act: Int, beats: Seq[Seq[Int]], computeLanes: Int,
               func: Seq[String], outStall: Int): Seq[Seq[Float]] = {
    var outs = Seq[Seq[Float]]()
    test(new DataPathExtensionHarness(
      new HasStreamMap(dataWidth = testWidth, elementWidth = 16, computeLanes = computeLanes, func = func)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.csr_i(0).poke(f32bits(a).U); dut.io.csr_i(1).poke(f32bits(b).U); dut.io.csr_i(2).poke(act.U)
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
          dut.clock.step(outStall) // stall the consumer: force credit exhaustion + queue fill
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

  "StreamMap_backpressure" should "stay correct under a stalled consumer (credit throttle + queue fill)" in {
    val rng   = new Random(0xBEEF)
    val beats = Seq.fill(24)(Seq.fill(lanes)(f32ToF16bits(rng.between(-4, 4) + rng.nextInt(4) * 0.25f)))
    val (a, b) = (1.0f, -3.0f)
    val hw = runStall(a, b, 1, beats, 8, Seq("LINEAR_FP16", "EXP_FP16"), outStall = 48)
    val gd = golden(a, b, 1, beats)
    assert(hw.length == beats.length, s"lost beats: got ${hw.length}/${beats.length}")
    var maxErr = 0.0f
    for ((hr, gr) <- hw.zip(gd); (h, g) <- hr.zip(gr)) {
      val e = math.abs(h - g); if (e > maxErr) maxErr = e
      assert(e <= math.max(math.abs(g) * 0.01f, 0.02f), s"backpressure mismatch hw=$h golden=$g err=$e")
    }
    println(f"[StreamMap:backpressure] ${beats.length} beats, outStall=48, maxErr=$maxErr%.4g")
  }

  /** Drive input always-valid + output always-ready and count cycles to process `nbeats`. The streaming FSM
    * should reach ~subCycles cycles/beat (a one-time P-cycle fill); the old drain-per-beat FSM needed
    * subCycles + P + 2 per beat. `-||>` harness cuts are full-bandwidth, so they don't cap the rate. */
  def throughputCycles(computeLanes: Int, nbeats: Int, func: Seq[String], act: Int): Long = {
    val rng   = new Random(0x7)
    val beats = Seq.fill(nbeats)(Seq.fill(lanes)(f32ToF16bits(rng.between(-3, 3))))
    var cyc = 0L; var got = 0
    test(new DataPathExtensionHarness(
      new HasStreamMap(dataWidth = testWidth, elementWidth = 16, computeLanes = computeLanes, func = func)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.csr_i(0).poke(f32bits(2.0f).U); dut.io.csr_i(1).poke(f32bits(-1.5f).U); dut.io.csr_i(2).poke(act.U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.data_o.ready.poke(true)
        var fed = 0
        dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(packBeat(beats(0)))
        val maxCyc = nbeats.toLong * 40 + 200
        while (got < nbeats && cyc < maxCyc) {
          val canFeed = fed < nbeats && dut.io.data_i.ready.peekBoolean()
          val outNow  = dut.io.data_o.valid.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) {
            fed += 1
            dut.io.data_i.valid.poke(fed < nbeats)
            if (fed < nbeats) dut.io.data_i.bits.poke(packBeat(beats(fed)))
          }
          if (outNow) got += 1
        }
        assert(got == nbeats, s"throughput: only $got/$nbeats outputs after $cyc cycles")
      }
    cyc
  }

  "StreamMap_throughput" should "stream at ~subCycles cycles/beat (not subCycles+P+2)" in {
    val nbeats    = 64
    val cl        = 8
    val subCycles = lanes / cl // 4
    val cyc       = throughputCycles(cl, nbeats, Seq("LINEAR_FP16"), 0)
    val perBeat   = cyc.toDouble / nbeats
    println(f"[StreamMap:throughput] cl=$cl subCycles=$subCycles: $cyc cyc / $nbeats beats = $perBeat%.2f cyc/beat")
    // Streaming target ~subCycles (+ one-time fill). The old drain-per-beat design was subCycles+P+2 (~11).
    assert(cyc < nbeats * (subCycles + 2), s"not streaming: $cyc cyc for $nbeats beats (>= ${nbeats * (subCycles + 2)})")
  }

  // APP-LEVEL protocol test: emulate the xDMA controller across back-to-back tasks and require busy_o to
  // fall after each drain (the completion signal the controller waits on before the next task). The
  // value-only tests never check busy_o -- a stuck-busy would hang the real app but pass those.
  "StreamMap_multitask_busy" should "return busy_o low after each back-to-back task" in {
    test(new DataPathExtensionHarness(
      new HasStreamMap(dataWidth = testWidth, elementWidth = 16, computeLanes = 8,
                       func = Seq("LINEAR_FP16", "EXP_FP16", "SILU_FP16"))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.enable_i.poke(true)
        val rng = new Random(0xB05)
        def runTask(a: Float, b: Float, act: Int, nBeats: Int): Unit = {
          val beats = Seq.fill(nBeats)(Seq.fill(lanes)(f32ToF16bits(rng.between(-3, 3))))
          dut.io.csr_i(0).poke(f32bits(a).U); dut.io.csr_i(1).poke(f32bits(b).U); dut.io.csr_i(2).poke(act.U)
          dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
          var threads = new chiseltest.internal.TesterThreadList(Seq())
          threads = threads.fork {
            dut.io.data_i.valid.poke(true)
            for (bt <- beats) { while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1); dut.io.data_i.bits.poke(packBeat(bt)); dut.clock.step(1) }
            dut.io.data_i.valid.poke(false)
          }
          threads = threads.fork {
            for (_ <- beats.indices) { while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1); dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false) }
          }
          threads.joinAndStep()
          dut.io.data_o.ready.poke(true)
          var w = 0; while (dut.io.busy_o.peekBoolean() && w < 400) { dut.clock.step(1); w += 1 }
          dut.io.data_o.ready.poke(false)
          assert(!dut.io.busy_o.peekBoolean(), s"StreamMap busy_o stuck high after task (act=$act) -- completion bug")
        }
        runTask(2.0f, -1.5f, 0, 6); runTask(1.0f, -3.0f, 1, 6); runTask(1.0f, 0.0f, 2, 6); runTask(2.0f, -1.5f, 0, 6)
      }
  }
}
