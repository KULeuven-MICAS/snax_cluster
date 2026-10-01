package snax.DataPathExtension

import java.io.File
import java.nio.file.Files

import scala.io.Source
import scala.sys.process._
import scala.util.Random

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** Vectors from the reference codec, `doc/crest_decompressor/crest_codec.py`, so the RTL is checked against the encoder
  * itself: its synthetic edge cases and, when the environment variable CREST_WEIGHT_CACHE names the checkpoint cache,
  * chunks of published checkpoints (GPTQ / AWQ INT4, MXFP4, FP8, INT8, BF16).
  */
object CrestVectors {
  val codec = new File("doc/crest_decompressor/crest_codec.py")

  case class Vectors(name: String, in: Seq[BigInt], out: Seq[BigInt], beats: Int)

  def cases: Seq[String] =
    Process(Seq("python3", codec.getPath, "cases")).!!.split("\n").map(_.trim).filter(_.nonEmpty).toIndexedSeq

  def load(name: String): Vectors = {
    val dir            = Files.createTempDirectory("crest_" + name).toFile
    require(
      Process(Seq("python3", codec.getPath, "vectors", "--case", name, "--out", dir.getPath)).! == 0,
      s"crest_codec.py could not generate case $name"
    )
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
    * out, and that busy drops. Returns the cycles from the first offered word to the last beat taken.
    */
  def transfer(dut: DataPathExtensionHarness, v: Vectors, inGap: Double, outStall: Double, rnd: Random): Int = {
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
        assert(
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
    assert(inIdx == v.in.length, s"${v.name}: ${v.in.length - inIdx} input words left when the last beat was out")
    var settle = 0
    while (dut.io.busy_o.peekBoolean() && settle < 8) { dut.clock.step(); settle += 1 }
    assert(!dut.io.busy_o.peekBoolean(), s"${v.name}: busy still high after the last beat")
    cycles
  }
}

class CrestDecompressorTester extends AnyFlatSpec with ChiselScalatestTester {
  import DecompressorDriver.transfer
  val all = CrestVectors.cases.map(CrestVectors.load)
  if (!all.exists(_.name.startsWith("real_")))
    println("CrestDecompressorTester: CREST_WEIGHT_CACHE is not set, so only the synthetic cases run")

  "CrestDecompressor" should "reproduce every reference-codec case back to back, under random stalls" in {
    test(new DataPathExtensionHarness(new HasCrestDecompressor(groupBeats = 64, escWords = 12)))
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
        dut.clock.setTimeout(0)
        val rnd = new Random(5)
        for {
          (gap, stall) <- Seq((0.3, 0.3), (0.0, 0.6), (0.6, 0.0))
          v            <- all
        }
          transfer(dut, v, gap, stall, rnd)
        println(f"${"case"}%-18s ${"beats"}%6s ${"words in"}%9s ${"cycles"}%7s ${"beats/cycle"}%12s")
        for (v         <- all) {
          val c = transfer(dut, v, 0.0, 0.0, rnd)
          println(f"${v.name}%-18s ${v.beats}%6d ${v.in.length}%9d $c%7d ${v.beats.toDouble / c}%12.3f")
          // Published checkpoints: one beat a cycle but for the beats with more than 32 tier-1 / 16 tier-2
          // escapes (GPTQ INT4: 1.4% of its beats) and the raw groups' header words.
          if (v.name.startsWith("real_")) {
            val floor = if (v.name == "real_int4_gptq") 0.97 else 0.975
            assert(v.beats.toDouble / c >= floor, s"${v.name}: ${v.beats} beats took $c cycles")
          }
        }
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
