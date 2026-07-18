package snax.DataPathExtension

import scala.collection.mutable.ArrayBuffer

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec
import firrtl2.options.TargetDirAnnotation

/** StreamDecompressRt: the in-transit DECOMPRESSOR (MX/FP8 -> FP16/BF16), the mirror of StreamCastRt. This
  * tester chains COMPRESS (reader-side StreamCastRt, BF16 -> MXFP4) -> the "link" -> DECOMPRESS (writer-side
  * StreamDecompressRt, MXFP4 -> BF16), the local reader->writer loopback that models compress-at-sender /
  * decompress-at-receiver, and reports the MEASURED link-beat ratio + the recovery error, for scaleBurst = 1
  * (v1, 2:1) and scaleBurst = 16 (packed scales, the real MXFP4 3.76:1).
  */
class StreamDecompressRtTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  def bf16bits(f: Float): Int = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xffff
  def bf16ToF32(hw: Int): Float = java.lang.Float.intBitsToFloat((hw & 0xffff) << 16)
  def packHalf(v: Seq[Int]): BigInt =
    v.zipWithIndex.foldLeft(BigInt(0)) { case (a, (h, i)) => a | (BigInt(h & 0xffff) << (16 * i)) }

  def setup(dut: DataPathExtensionHarness, csr: BigInt): Unit = {
    dut.io.csr_i(0).poke(csr.U); dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    dut.io.data_o.ready.poke(true)
  }

  /** Feed `inBeats`, collect the first `wantOut` output beats. */
  def drive(dut: DataPathExtensionHarness, inBeats: Seq[BigInt], wantOut: Int): Seq[BigInt] = {
    var fed = 0; var got = 0; var cyc = 0; val out = ArrayBuffer[BigInt]()
    while (got < wantOut && cyc < 20000) {
      val ready = dut.io.data_i.ready.peekBoolean()
      if (fed < inBeats.length && ready) { dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(inBeats(fed).U) }
      else dut.io.data_i.valid.poke(false)
      val ov = dut.io.data_o.valid.peekBoolean()
      val ob = if (ov) dut.io.data_o.bits.peekInt() else BigInt(0)
      dut.clock.step(1); cyc += 1
      if (fed < inBeats.length && ready) fed += 1
      if (ov) { out += ob; got += 1 }
    }
    assert(got == wantOut, s"expected $wantOut output beats, got $got"); out.toSeq
  }

  // MXFP4, srcFmt=BF16(1), dstFmt=MXFP4(7), R=4. burst input beats = 4*scaleBurst; compressed = scaleBurst+1.
  def roundTrip(scaleBurst: Int, tag: String): Unit = {
    val R = 4; val nBlk = R * scaleBurst // input BF16 beats (blocks) per burst
    // WIDE dynamic range WITHIN each block (powers of two spanning 8x): with block max = 16*const the scale is
    // 4*const, so the min element 2*const maps to 0.5 -- the E2M1 SUBNORMAL. The old {2,3,4,6} (max/min=3) never
    // scaled below 1.0, so it never exercised the subnormal and MISSED the FTZ bug the vsim loopback exposed.
    // All of {0.5,1,2,4} are exact E2M1 codes, so recovery stays bit-exact once the subnormal is encoded.
    val base = Seq(2f, 4f, 8f, 16f)
    val blocks = (0 until nBlk).map(b => Seq.tabulate(32)(i => base(i % 4) * (1 << (b % 3))))
    val inBeats = blocks.map(bk => packHalf(bk.map(bf16bits)))

    var linkBeats: Seq[BigInt] = Seq()
    test(new DataPathExtensionHarness(new HasStreamCastRt(scaleBurst = scaleBurst)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags, TargetDirAnnotation(s"test_run_dir/rt_c_$tag"))) { dut =>
        setup(dut, BigInt(1 | (7 << 2) | (1 << 5))) // srcFmt=BF16, dstFmt=MXFP4, emitScale
        linkBeats = drive(dut, inBeats, wantOut = scaleBurst + 1) // scaleBurst data beats + 1 packed scale beat
      }
    var outBeats: Seq[BigInt] = Seq()
    test(new DataPathExtensionHarness(new HasStreamDecompressRt(scaleBurst = scaleBurst)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags, TargetDirAnnotation(s"test_run_dir/rt_d_$tag"))) { dut =>
        setup(dut, BigInt(1 | (7 << 2) | (1 << 5))) // outFmt=BF16, srcFmt=MXFP4, hasScale
        outBeats = drive(dut, linkBeats, wantOut = nBlk) // scaleBurst*R BF16 beats
      }
    var maxErr = 0.0
    for (b <- 0 until nBlk; i <- 0 until 32) {
      val rec = bf16ToF32(((outBeats(b) >> (16 * i)) & 0xffff).toInt)
      maxErr = math.max(maxErr, math.abs(rec - blocks(b)(i)) / math.max(math.abs(blocks(b)(i)), 1e-9))
    }
    val ratio = nBlk.toDouble / linkBeats.length
    println(f"[Decompress roundtrip scaleBurst=$scaleBurst] $nBlk BF16 beats --compress--> ${linkBeats.length} link beats " +
            f"--decompress--> ${outBeats.length} BF16 beats;  MEASURED link ratio = $ratio%.2fx;  recover maxErr=$maxErr%.2e")
    assert(maxErr < 1e-6, s"round-trip recovery must be bit-exact for representable values (incl. the 0.5 subnormal); got $maxErr")
    assert(outBeats.length == nBlk, "decompress must recover all input beats")
  }

  "StreamDecompressRt_v1" should "recover BF16 through MXFP4 compress -> link -> decompress (scaleBurst=1, 2:1)" in {
    roundTrip(scaleBurst = 1, tag = "sb1")
  }

  "StreamDecompressRt_packed" should "reach the real MXFP4 3.76:1 link ratio with packed scales (scaleBurst=16)" in {
    roundTrip(scaleBurst = 16, tag = "sb16")
  }
}
