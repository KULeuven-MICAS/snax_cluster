package snax.DataPathExtension

import java.io.File
import java.nio.file.Files

import scala.io.Source
import scala.sys.process._
import scala.util.Random

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** The decompressor and the format it is tested at, from the environment (defaults: the built design point).
  *
  *   - CREST_G: groupBeats, the codec's --g (default 64)
  *   - CREST_ESC: escWords, the codec's --emax (default 12)
  *   - CREST_PLANE_DEPTH: planeDepth (default 8)
  *   - CREST_K1, CREST_K2: k1Lanes, k2Lanes, the tier-1 / tier-2 entries placed per cycle (default 32, 16)
  *   - CREST_CASES: comma-separated case names to run (default: every case the codec lists)
  *   - CREST_ARMS: comma-separated gap:stall pairs for the random-stall arms (default 0.3:0.3,0:0.6,0.6:0)
  *   - CREST_CODEC_ARGS: extra arguments for the codec's `vectors` call, space-separated (e.g. --params 0,9,3,2)
  *   - CREST_SEED: added to every report-mode random seed (default 0), for repeats of the random arms
  *   - CREST_ZP: 1 builds the zero-point-aware decompressor (HasCrestDecompressor(zeroPoint = true)) and adds the
  *     codec's zero-point cases (`cases --zp`)
  *
  * Setting any of CREST_G, CREST_ESC, CREST_PLANE_DEPTH or CREST_ZP selects REPORT mode: every (case, arm) prints one
  * `CREST_RESULT` line, runs a plain copy of the same beats through the bypass under the same random stalls for
  * comparison, and the beats-per-cycle floor is printed instead of asserted. A wrong beat still fails the test, after
  * every line is printed.
  */
object CrestParams {
  private def env(k: String): Option[String] = sys.env.get(k).map(_.trim).filter(_.nonEmpty)

  val g:          Int                   = env("CREST_G").map(_.toInt).getOrElse(64)
  val esc:        Int                   = env("CREST_ESC").map(_.toInt).getOrElse(12)
  val planeDepth: Int                   = env("CREST_PLANE_DEPTH").map(_.toInt).getOrElse(8)
  val k1:         Int                   = env("CREST_K1").map(_.toInt).getOrElse(32)
  val k2:         Int                   = env("CREST_K2").map(_.toInt).getOrElse(16)
  val caseFilter: Option[Set[String]]   = env("CREST_CASES").map(_.split(",").map(_.trim).filter(_.nonEmpty).toSet)
  val arms:       Seq[(Double, Double)] = env("CREST_ARMS")
    .map(_.split(",").map(_.trim).filter(_.nonEmpty).toSeq.map { a =>
      val Array(gap, stall) = a.split(":").map(_.trim.toDouble)
      (gap, stall)
    })
    .getOrElse(Seq((0.3, 0.3), (0.0, 0.6), (0.6, 0.0)))
  val codecArgs:  Seq[String]           = env("CREST_CODEC_ARGS").map(_.split("\\s+").toSeq).getOrElse(Seq())
  val seed:       Long                  = env("CREST_SEED").map(_.toLong).getOrElse(0L)
  val zp:         Boolean               = env("CREST_ZP").contains("1")
  val report:     Boolean               =
    Seq("CREST_G", "CREST_ESC", "CREST_PLANE_DEPTH", "CREST_ZP", "CREST_K1", "CREST_K2").exists(env(_).isDefined)
}

/** Vectors from the reference codec, `doc/crest_decompressor/crest_codec.py`, so the RTL is checked against the encoder
  * itself: its synthetic edge cases and, when the environment variable CREST_WEIGHT_CACHE names the checkpoint cache,
  * chunks of published checkpoints (GPTQ / AWQ INT4, MXFP4, FP8, INT8, BF16). The codec encodes at CrestParams.g and
  * CrestParams.esc.
  */
object CrestVectors {
  val codec = new File("doc/crest_decompressor/crest_codec.py")

  case class Vectors(name: String, in: Seq[BigInt], out: Seq[BigInt], beats: Int)

  def cases: Seq[String] = {
    val listed =
      Process(Seq("python3", codec.getPath, "cases") ++ (if (CrestParams.zp) Seq("--zp") else Seq()))
        .!!.split("\n").map(_.trim).filter(_.nonEmpty).toIndexedSeq
    CrestParams.caseFilter match {
      case Some(want) =>
        val missing = want -- listed.toSet
        require(missing.isEmpty, s"CREST_CASES names cases the codec does not list: ${missing.mkString(", ")}")
        listed.filter(want.contains)
      case None       => listed
    }
  }

  def load(name: String): Vectors = {
    val dir            = Files.createTempDirectory("crest_" + name).toFile
    val cmd            = Seq("python3", codec.getPath, "vectors", "--case", name, "--out", dir.getPath) ++
      Seq("--g", CrestParams.g.toString, "--emax", CrestParams.esc.toString) ++ CrestParams.codecArgs
    require(Process(cmd).! == 0, s"crest_codec.py could not generate case $name")
    def hex(f: String) =
      Source.fromFile(new File(dir, f)).getLines().map(_.trim).filter(_.nonEmpty).map(BigInt(_, 16)).toSeq
    Vectors(
      name,
      hex("in.hex"),
      hex("out.hex"),
      Source.fromFile(new File(dir, "csr.txt")).getLines().next().trim.toInt
    )
  }
}

/** Drives one decompressor transfer through the extension harness and checks it beat for beat. */
object DecompressorDriver {
  import CrestVectors.Vectors

  /** One transfer: start with CSR 0 = N, offer the input words with probability 1 - inGap per cycle, take output beats
    * with probability 1 - outStall. Checks every beat, that all input words were taken by the time the N-th beat is
    * out, and that busy drops. Returns the cycles from the first offered word to the last beat taken, and the number
    * of failed checks: with strict (the default) a failed check fails at once, so the count is 0.
    */
  def transfer(
    dut:      DataPathExtensionHarness,
    v:        Vectors,
    inGap:    Double,
    outStall: Double,
    rnd:      Random,
    strict:   Boolean = true
  ): (Int, Int) = {
    var failed = 0
    def check(ok: Boolean, msg: => String): Unit =
      if (strict) assert(ok, msg)
      else if (!ok) {
        println(s"CREST_CHECK_FAIL $msg")
        failed += 1
      }
    dut.io.enable_i.poke(true)
    dut.io.csr_i(0).poke(v.beats)
    dut.io.start_i.poke(true)
    dut.clock.step()
    dut.io.start_i.poke(false)

    val limit  = 40 * (v.beats + v.in.length) + 1000
    var inIdx  = 0
    var outIdx = 0
    var cycles = 0
    while (outIdx < v.out.length) {
      val offer = inIdx < v.in.length && rnd.nextDouble() >= inGap
      dut.io.data_i.valid.poke(offer)
      if (offer) dut.io.data_i.bits.poke(v.in(inIdx))
      val take  = rnd.nextDouble()    >= outStall
      dut.io.data_o.ready.poke(take)
      if (offer && dut.io.data_i.ready.peekBoolean()) inIdx += 1
      if (take  && dut.io.data_o.valid.peekBoolean()) {
        val got = dut.io.data_o.bits.peekInt()
        check(
          got == v.out(outIdx),
          s"${v.name}: beat $outIdx is ${got.toString(16)}, expected ${v.out(outIdx).toString(16)}"
        )
        outIdx += 1
      }
      dut.clock.step()
      cycles += 1
      assert(cycles < limit, s"${v.name}: stuck at input word $inIdx / ${v.in.length}, beat $outIdx / ${v.beats}")
    }
    dut.io.data_i.valid.poke(false)
    dut.io.data_o.ready.poke(false)
    check(inIdx == v.in.length, s"${v.name}: ${v.in.length - inIdx} input words left when the last beat was out")
    var settle = 0
    while (dut.io.busy_o.peekBoolean() && settle < 8) { dut.clock.step(); settle += 1 }
    check(!dut.io.busy_o.peekBoolean(), s"${v.name}: busy still high after the last beat")
    (cycles, failed)
  }

  /** A plain copy of the case's output beats through the extension's bypass (enable low), under the same offer / take
    * process as transfer(). Returns the cycles from the first offered beat to the last beat taken, and the number of
    * beats that came out changed.
    */
  def copy(dut: DataPathExtensionHarness, v: Vectors, inGap: Double, outStall: Double, rnd: Random): (Int, Int) = {
    dut.io.enable_i.poke(false)
    dut.io.start_i.poke(false)
    val limit  = 40 * v.beats + 1000
    var inIdx  = 0
    var outIdx = 0
    var cycles = 0
    var failed = 0
    while (outIdx < v.out.length) {
      val offer = inIdx < v.out.length && rnd.nextDouble() >= inGap
      dut.io.data_i.valid.poke(offer)
      if (offer) dut.io.data_i.bits.poke(v.out(inIdx))
      val take  = rnd.nextDouble()     >= outStall
      dut.io.data_o.ready.poke(take)
      if (offer && dut.io.data_i.ready.peekBoolean()) inIdx += 1
      if (take  && dut.io.data_o.valid.peekBoolean()) {
        if (dut.io.data_o.bits.peekInt() != v.out(outIdx)) {
          println(s"CREST_CHECK_FAIL ${v.name}: bypass beat $outIdx changed")
          failed += 1
        }
        outIdx += 1
      }
      dut.clock.step()
      cycles += 1
      assert(cycles < limit, s"${v.name}: bypass stuck at beat $outIdx / ${v.beats}")
    }
    dut.io.data_i.valid.poke(false)
    dut.io.data_o.ready.poke(false)
    dut.clock.step(2)
    (cycles, failed)
  }
}

class CrestDecompressorTester extends AnyFlatSpec with ChiselScalatestTester {
  import DecompressorDriver.{copy, transfer}
  val P   = CrestParams
  val all = CrestVectors.cases.map(CrestVectors.load)
  if (!all.exists(_.name.startsWith("real_")))
    println("CrestDecompressorTester: CREST_WEIGHT_CACHE is not set, so only the synthetic cases run")
  if (P.report)
    println(
      s"CrestDecompressorTester: report mode, G = ${P.g}, escWords = ${P.esc}, planeDepth = ${P.planeDepth}, " +
        s"k1Lanes = ${P.k1}, k2Lanes = ${P.k2}, zeroPoint = ${P.zp}"
    )

  "CrestDecompressor" should "reproduce every reference-codec case back to back, under random stalls" in {
    test(
      new DataPathExtensionHarness(
        new HasCrestDecompressor(
          groupBeats = P.g,
          escWords   = P.esc,
          k1Lanes    = P.k1,
          k2Lanes    = P.k2,
          planeDepth = P.planeDepth,
          zeroPoint  = P.zp
        )
      )
    )
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
        dut.clock.setTimeout(0)
        val rnd    = new Random(5)
        var failed = 0

        // REPORT mode: the decompressor and the plain copy see the same random offer / take sequence.
        def measure(v: CrestVectors.Vectors, gap: Double, stall: Double, seed: Long): Int = {
          val (c, f)   = transfer(dut, v, gap, stall, new Random(seed), strict = false)
          val (cc, cf) = copy(dut, v, gap, stall, new Random(seed))
          failed += f + cf
          println(
            f"CREST_RESULT case=${v.name} G=${P.g} esc=${P.esc} pd=${P.planeDepth} gap=$gap%.2f stall=$stall%.2f " +
              f"seed=${P.seed} beats=${v.beats} words=${v.in.length} cycles=$c bpc=${v.beats.toDouble / c}%.5f " +
              f"copy_cycles=$cc copy_bpc=${v.beats.toDouble / cc}%.5f ok=${if (f + cf == 0) 1 else 0}"
          )
          c
        }

        for {
          ((gap, stall), a) <- P.arms.zipWithIndex
          (v, i)            <- all.zipWithIndex
        }
          if (P.report) measure(v, gap, stall, 1000L * (a + 1) + i + 100000L * P.seed)
          else transfer(dut, v, gap, stall, rnd)
        println(f"${"case"}%-18s ${"beats"}%6s ${"words in"}%9s ${"cycles"}%7s ${"beats/cycle"}%12s")
        for ((v, i) <- all.zipWithIndex) {
          val c = if (P.report) measure(v, 0.0, 0.0, i + 100000L * P.seed) else transfer(dut, v, 0.0, 0.0, rnd)._1
          println(f"${v.name}%-18s ${v.beats}%6d ${v.in.length}%9d $c%7d ${v.beats.toDouble / c}%12.3f")
          // Published checkpoints: one beat a cycle but for the beats with more than 32 tier-1 / 16 tier-2
          // escapes (GPTQ INT4: 1.4% of its beats) and the raw groups' header words.
          if (v.name.startsWith("real_")) {
            val floor = if (v.name == "real_int4_gptq") 0.97 else 0.975
            if (P.report) {
              if (v.beats.toDouble / c < floor)
                println(f"CREST_FLOOR case=${v.name} bpc=${v.beats.toDouble / c}%.4f below $floor (report only)")
            } else
              assert(v.beats.toDouble / c >= floor, s"${v.name}: ${v.beats} beats took $c cycles")
          }
        }
        assert(failed == 0, s"$failed failed checks (CREST_CHECK_FAIL lines above)")
      }
  }

  it should "pass data through unchanged when disabled" in {
    test(new DataPathExtensionHarness(new HasCrestDecompressor(groupBeats = 64, escWords = 12)))
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
        val rnd    = new Random(9)
        val data   = Seq.fill(64)(BigInt(512, rnd))
        dut.io.enable_i.poke(false)
        dut.io.data_o.ready.poke(true)
        var inIdx  = 0
        var outIdx = 0
        var cycles = 0
        while (outIdx < data.length && cycles < 1000) {
          dut.io.data_i.valid.poke(inIdx < data.length)
          if (inIdx < data.length) dut.io.data_i.bits.poke(data(inIdx))
          if (inIdx < data.length && dut.io.data_i.ready.peekBoolean()) inIdx += 1
          if (dut.io.data_o.valid.peekBoolean()) {
            assert(dut.io.data_o.bits.peekInt() == data(outIdx), s"bypass beat $outIdx changed")
            outIdx += 1
          }
          dut.clock.step()
          cycles += 1
        }
        assert(outIdx == data.length, "bypass lost beats")
      }
  }
}
