package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._
import fp_native._

/** StreamDecompressRt: the in-transit DECOMPRESSOR -- the MX/FP8 -> 16-bit (FP16/BF16) dequant MIRROR of
  * StreamCastRt. It recovers a low-precision (MX or FP8) stream back to FP16/BF16 as it moves. Placed on the
  * WRITER side of the xDMA, it pairs with a reader-side StreamCastRt to model compress-at-sender /
  * decompress-at-receiver over a (locally looped-back) link: the compressed MX beats cross the reader->writer
  * path (the "link"), and the decompressor expands them back at the writer -- so a single-cluster reader->writer
  * loopback yields real RTL numbers for the in-flight fixed-ratio compression (04 §7.2, 03 §2.6).
  *
  *   - EDGE problem, FP32-internal: widenMXRt(code, scale) -> FP32 -> narrowRt(FP32, outFmt) -> 16-bit. widenMX
  *     is EXACT (an MX element's <=3 mantissa bits fit BF16's 7 / FP16's 10), so MX->BF16 recovery is exact for
  *     the MX value -- the round-trip loss is ONLY the original quantization at the sender.
  *   - EXPANSION (mirror of StreamCast's contraction): one burst = `scaleBurst` MX data beats + ONE packed scale
  *     beat, expanding to `scaleBurst * R` output beats of 32 16-bit values (R=4 for MXFP4, 2 for an 8-bit grid).
  *     `scaleBurst` MUST match the compressor's (the scale beat carries `scaleBurst*R` E8M0 bytes, scale byte
  *     `dbIdx*R + blk` = block `blk` of data beat `dbIdx`).  scaleBurst=1 == the per-group [data, scale] form.
  *
  * CSR: csr0 bits[1:0]=outFmt (0=FP16,1=BF16); bits[4:2]=srcFmt (2=FP8, 3..7=MX element type); bit[5]=hasScale.
  */
class HasStreamDecompressRt(
  dataWidth:  Int = 512,
  fpPipe:     Int = 1,
  scaleBurst: Int = 1 // MUST equal the paired StreamCastRt's scaleBurst
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(moduleName = "StreamDecompressRt", userCsrNum = 1, dataWidth = dataWidth)

  def instantiate(clusterName: String): StreamDecompressRt =
    Module(new StreamDecompressRt(fpPipe, scaleBurst = scaleBurst) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamDecompressRt(
  fpPipeParam: Int     = 1,
  pipelined:   Boolean = true,
  scaleBurst:  Int     = 1
)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  require(scaleBurst >= 1 && scaleBurst * 4 <= extensionParam.dataWidth / 8,
          "StreamDecompressRt: scaleBurst*maxGrp scales must fit one scale beat (<= 64 for 512-bit)")
  val outLanes = extensionParam.dataWidth / 16 // 32: 16-bit output values per beat
  val maxCodes = extensionParam.dataWidth / 4  // 128: MXFP4 codes per input data beat
  val maxGrp   = 4                             // R in {2,4}
  val maxScl   = scaleBurst * maxGrp           // packed scale bytes per scale beat

  val csr0    = ext_csr_i(0)
  val outFmt  = csr0(1, 0) // 0=FP16, 1=BF16
  val srcFmt  = csr0(4, 2) // 2=FP8, 3..7=MX
  val hasScl  = csr0(5).asBool
  val is4bit  = srcFmt === FMT_MXFP4_E2M1.U
  val Rbeats  = Mux(is4bit, 4.U, 2.U) // output (16-bit) beats per input data beat

  // ---- buffered burst: scaleBurst MX data beats + up to maxScl block scales ----
  val dataBuf = Reg(Vec(scaleBurst, UInt(extensionParam.dataWidth.W)))
  val sclReg  = Reg(Vec(maxScl, UInt(8.W)))

  val sFill :: sScale :: sEmit :: Nil = Enum(3)
  val state   = RegInit(sFill)
  val fillCnt = RegInit(0.U(log2Ceil(scaleBurst).max(1).W)) // data beats buffered so far (0..scaleBurst-1)
  val dbIdx   = RegInit(0.U(log2Ceil(scaleBurst).max(1).W)) // which buffered data beat is being emitted
  val blk     = RegInit(0.U(log2Ceil(maxGrp).W))            // which 32-elem block within it (0..R-1)
  val sclIdx  = RegInit(0.U(log2Ceil(maxScl).max(1).W))     // running scale index == dbIdx*R + blk

  // decode block `blk` of the selected data beat with scale sclReg(sclIdx) -> one 16-bit output beat
  val curData  = dataBuf(dbIdx)
  val nibView  = curData.asTypeOf(Vec(maxCodes, UInt(4.W)))     // 128 MXFP4 nibbles
  val byteView = curData.asTypeOf(Vec(maxCodes / 2, UInt(8.W))) // 64 8-bit MX/FP8 codes
  def carrierOf(j: Int): UInt = {
    val idx4 = (blk * 32.U + j.U)(log2Ceil(maxCodes) - 1, 0)
    val idx8 = (blk * 32.U + j.U)(log2Ceil(maxCodes / 2) - 1, 0)
    Mux(is4bit, Cat(0.U(12.W), nibView(idx4)), Cat(0.U(8.W), byteView(idx8)))
  }
  val outBeat = Cat((0 until outLanes).reverse.map { j =>
    narrowRt(widenMXRt(carrierOf(j), sclReg(sclIdx), srcFmt, 0), outFmt, 0)(15, 0)
  })

  // ---- FSM: sFill (buffer scaleBurst data beats) -> sScale (1 scale beat) -> sEmit (scaleBurst*R beats) ----
  val Qdepth = 4
  val outQ   = Module(new Queue(UInt(extensionParam.dataWidth.W), entries = Qdepth))
  ext_data_i.ready := ((state === sFill) || (state === sScale)) && !ext_start_i
  val accept = ext_data_i.fire
  outQ.io.enq.valid := (state === sEmit) && !ext_start_i // backpressured by enq.ready (a stalling producer)
  outQ.io.enq.bits  := outBeat
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits
  val enqFire = outQ.io.enq.fire

  val sclView  = ext_data_i.bits.asTypeOf(Vec(extensionParam.dataWidth / 8, UInt(8.W)))
  val lastEmit = (dbIdx === (scaleBurst - 1).U) && (blk === (Rbeats - 1.U))
  when(ext_start_i) {
    state := sFill; fillCnt := 0.U; dbIdx := 0.U; blk := 0.U; sclIdx := 0.U
  }.otherwise {
    when(accept) {
      when(state === sFill) {
        dataBuf(fillCnt) := ext_data_i.bits
        when(fillCnt === (scaleBurst - 1).U) { fillCnt := 0.U; state := Mux(hasScl, sScale, sEmit) }
          .otherwise { fillCnt := fillCnt + 1.U }
      }.elsewhen(state === sScale) {
        for (k <- 0 until maxScl) sclReg(k) := sclView(k)
        state := sEmit
      }
    }
    when(state === sEmit && enqFire) {
      sclIdx := sclIdx + 1.U
      when(lastEmit) { state := sFill; dbIdx := 0.U; blk := 0.U; sclIdx := 0.U }
        .elsewhen(blk === (Rbeats - 1.U)) { blk := 0.U; dbIdx := dbIdx + 1.U }
        .otherwise { blk := blk + 1.U }
    }
  }

  ext_busy_o := (state =/= sFill) || (fillCnt =/= 0.U)
}
