package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._
import fp_native._

/** StreamCastRt: the in-transit QUANTIZER -- the Cast primitive of the streaming ISA. Converts a streamed row
  * from a 16-bit transport (FP16/BF16) DOWN to a low-precision transport (FP8 or an MX element type), deriving
  * the OCP-microscaling E8M0 block scale IN-TRANSIT. This is the dedicated home for MX quantization that was
  * factored OUT of StreamMapRt: recent models run activations in BF16 and quantize to MX/FP8 only at the GEMM
  * boundary, so this wide (up to 128-lane) low-precision path is LINEAR (widen + block-scale + narrow) with NO
  * activation LUT. Chain it after StreamMapRt for a fused epilogue in one streaming pass.
  *
  *   - FP32-internal, an EDGE problem: widen the 16-bit input to FP32, derive the block scale, narrow each
  *     element to the dst grid by that scale (widenRt + blockScaleE8M0 + narrowMXRt, all in FpHelpers).
  *   - ONE input beat = ONE MX block of 32 elements (16-bit src => 32/beat). The block scale = MAXABS over the
  *     beat (a small combinational max-tree over the 32 exponents). R = 512/dstElemW / 32 input beats fill one
  *     output beat: R=2 for an 8-bit dst (FP8/MXFP8/MXFP6), R=4 for MXFP4. So beats-in : beats-out = R : 1
  *     (data) or R : 2 (data + a trailing scale beat, for MX with emitScale set).
  *   - The R E8M0 scales of an output beat are emitted in the LOW R bytes of a trailing scale beat when
  *     emitScale is set (the OCP scale tensor is stored separately from the data). Plain FP8 dst has no scale.
  *
  * CSR layout: csr(0) bits[1:0]=srcFmt (0=FP16,1=BF16); bits[4:2]=dstFmt (2=FP8, 3=MXFP8-E5M2, 4=MXFP8-E4M3,
  * 5=MXFP6-E3M2, 6=MXFP6-E2M3, 7=MXFP4-E2M1); bit[5]=emitScale (emit the trailing block-scale beat).
  *
  * v1 scope: quantize (16-bit -> low precision). Dequant (widenMX back to FP16/BF16) and in-lane time-mux are
  * noted extensions; here every one of the 32 src lanes converts combinationally (narrow is ~1 FP unit/lane,
  * far cheaper than the activation LUT this replaces).
  *
  * v1 CONSTRAINTS (caller contract; each is a documented limitation, not a silent hazard):
  *   - Feed a MULTIPLE OF R input beats: a trailing partial group (< R blocks) is not flushed (it stays in
  *     `quarters` until the group completes). A row is a whole number of 32-elem MX blocks, so pad the block
  *     count up to a multiple of R (2 for 8-bit dst, 4 for MXFP4). An end-of-stream flush is future work.
  *   - srcFmt/dstFmt/emitScale must be STABLE per stream (read combinationally from csr0, like the sibling Rt
  *     blocks' fmt): changing them mid-group changes R/packing and corrupts the in-flight group.
  *   - Inputs are assumed FINITE (no inf/NaN). An inf/NaN lane widens to FP32 exp=255 and would dominate the
  *     block MAXABS, flushing the rest of the block to zero -- same finite-input assumption as fp32max elsewhere.
  *   - The OCP block scale aligns the block MAX to the element's top normal binade; a max element whose mantissa
  *     rounds up past the format max SATURATES to max-normal (narrowFin by construction; fpnew narrow for
  *     E5M2/E4M3 saturates too -- verified, no inf), i.e. the top element takes up to ~half-a-step of loss.
  */
class HasStreamCastRt(
  dataWidth:  Int = 512,
  fpPipe:     Int = 1,
  scaleBurst: Int = 1 // E8M0 scales packed per emitted scale beat (1 = v1: one scale beat per group; up to 16
  //                     for MXFP4 fills a 64-byte scale beat -> the real 3.76:1 ratio instead of v1's 2:1)
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(moduleName = "StreamCastRt", userCsrNum = 1, dataWidth = dataWidth)

  def instantiate(clusterName: String): StreamCastRt =
    Module(new StreamCastRt(fpPipe, scaleBurst = scaleBurst) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamCastRt(
  fpPipeParam: Int     = 1,
  pipelined:   Boolean = true,
  scaleBurst:  Int     = 1
)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  require(scaleBurst >= 1 && scaleBurst * 4 <= extensionParam.dataWidth / 8,
          "StreamCastRt: scaleBurst*maxGrp scales must fit one scale beat (<= 64 for 512-bit)")
  val srcLanes = extensionParam.dataWidth / 16 // 32: one input beat = one MX block of 32 (16-bit src)
  val maxDst   = extensionParam.dataWidth / 4  // 128: MXFP4 element count (the widest output packing)
  val maxGrp   = 4                             // R in {2,4}; group register holds up to 4 blocks
  val maxScl   = scaleBurst * maxGrp           // E8M0 scale bytes accumulated before one scale beat is emitted

  val csr0    = ext_csr_i(0)
  val srcFmt  = csr0(1, 0) // 0=FP16, 1=BF16
  val dstFmt  = csr0(4, 2) // 2=FP8, 3..7=MX
  val emitScl = csr0(5).asBool

  val is4bit = dstFmt === FMT_MXFP4_E2M1.U
  val Rbeats = Mux(is4bit, 4.U, 2.U) // input beats per output beat: 8-bit dst -> 2, MXFP4 -> 4

  // ---- per-input-beat combinational transform: widen 32 src -> FP32, derive block scale, narrow to dst ----
  // Read the INCOMING beat directly (the whole per-beat convert is combinational), so the block is captured on
  // the same cycle it is accepted -- no input register, no stale-beat hazard.
  val srcView = ext_data_i.bits.asTypeOf(Vec(srcLanes, UInt(16.W)))
  val wide    = VecInit((0 until srcLanes).map(i => widenRt(srcView(i), srcFmt, 0))) // exact 16-bit -> FP32
  // block scale = max FP32 (biased) exponent over the beat - emaxElem(dstFmt) (== FpHelpers.blockScaleE8M0)
  val maxExp  = wide.map(_(30, 23)).reduceLeft((x, y) => Mux(x >= y, x, y))
  val emaxE   = MuxLookup(dstFmt, 0.U(8.W))(Seq(
                  FMT_MXFP8_E5M2.U -> 15.U, FMT_MXFP8_E4M3.U -> 8.U,
                  FMT_MXFP6_E3M2.U -> 4.U,  FMT_MXFP6_E2M3.U -> 2.U, FMT_MXFP4_E2M1.U -> 2.U))
  val blkScale = Mux(maxExp > emaxE, maxExp - emaxE, 0.U(8.W)) // E8M0 code (bias 127), also fed to narrowMXRt
  // narrow each element to the dst grid by the block scale (plain FP8 -> narrowRt, ignores the scale)
  val code    = VecInit((0 until srcLanes).map(i => narrowMXRt(wide(i), blkScale, dstFmt, 0))) // 16b carriers

  // ---- R-beat accumulation: each input beat fills one 32-element block; emit when the output beat is full ----
  val quarters = Reg(Vec(maxGrp, Vec(srcLanes, UInt(16.W)))) // up to 4 blocks of 32 dst codes
  val sclSlot  = Reg(Vec(maxGrp, UInt(8.W)))                 // up to 4 E8M0 block scales
  val grp      = RegInit(0.U(log2Ceil(maxGrp).W))            // block index within the output group (0..R-1)
  val completing = grp === (Rbeats - 1.U)

  // this beat overlaid on the registered blocks (like StreamMap's outNow) so the completing beat emits now
  val qNow = WireInit(quarters); qNow(grp) := code
  val sNow = WireInit(sclSlot);  sNow(grp) := blkScale
  val allCodes = Wire(Vec(maxDst, UInt(16.W)))
  for (q <- 0 until maxGrp; i <- 0 until srcLanes) allCodes(q * srcLanes + i) := qNow(q)(i)
  val packed4  = Cat((0 until maxDst).map(i => allCodes(maxDst - 1 - i)(3, 0)))         // 128 nibbles (MXFP4)
  val packed8  = Cat((0 until maxDst / 2).map(i => allCodes(maxDst / 2 - 1 - i)(7, 0))) // 64 bytes (8-bit dst)
  val dataBeat = Mux(is4bit, packed4, packed8)

  // ---- SCALE PACKING: accumulate `scaleBurst` groups' E8M0 scales, emit ONE full scale beat per burst
  // (scales-last). The 64-byte scale beat is then FILLED (1 byte/block) instead of v1's 4-of-64, giving the
  // real MXFP4 3.76:1 (vs v1's 2:1). scaleBurst=1 == v1 (a scale beat every group). Scales are laid out
  // contiguously so scale byte `burstCnt*R + k` = block k of the burstCnt-th data beat (== the decompressor's read).
  val burstCnt    = RegInit(0.U(log2Ceil(scaleBurst).max(1).W))
  val isBurstLast = burstCnt === (scaleBurst - 1).U
  val sclAcc      = Reg(Vec(maxScl, UInt(8.W)))
  val sclAccNow   = WireInit(sclAcc) // this completing group's R scales overlaid at burstCnt*R
  for (k <- 0 until maxGrp) when(k.U < Rbeats) {
    sclAccNow((burstCnt * Rbeats + k.U)(log2Ceil(maxScl) - 1, 0)) := sNow(k)
  }
  val sclLow  = Cat((0 until maxScl).reverse.map(sclAccNow(_))) // maxScl bytes, index 0 -> low byte
  val sclBeat = if (8 * maxScl >= extensionParam.dataWidth) sclLow
                else Cat(0.U((extensionParam.dataWidth - 8 * maxScl).W), sclLow)

  // ---- output FSM: credit + Queue; emit the data beat on the completing accept, then a trailing scale beat ----
  val Qdepth = 4
  val outQ   = Module(new Queue(UInt(extensionParam.dataWidth.W), entries = Qdepth))
  val credit     = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))
  val sclPending = RegInit(false.B)
  val sclBeatReg = Reg(UInt(extensionParam.dataWidth.W))
  val needSlots  = Mux(emitScl && isBurstLast, 2.U, 1.U) // the scale beat is emitted only on the burst-last group

  // non-completing beats never emit (just fill a block); completing beats need output slots + no pending scale
  ext_data_i.ready := (!completing || (credit >= needSlots && !sclPending)) && !ext_start_i
  val accept = ext_data_i.fire

  val doDataEnq = completing && accept
  val doSclEnq  = sclPending // trailing scale beat has priority to drain (mutually exclusive with doDataEnq)
  outQ.io.enq.valid := (doDataEnq || doSclEnq) && !ext_start_i
  outQ.io.enq.bits  := Mux(doSclEnq, sclBeatReg, dataBeat)
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "StreamCastRt: output queue overflow (credit accounting bug)")
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits

  val enqFire = outQ.io.enq.fire
  val deq     = outQ.io.deq.fire
  when(ext_start_i) {
    grp := 0.U; credit := Qdepth.U; sclPending := false.B; burstCnt := 0.U
  }.otherwise {
    when(accept) {
      quarters(grp) := code; sclSlot(grp) := blkScale
      grp := Mux(completing, 0.U, grp + 1.U)
      when(completing) { // register this group's R scales into the burst accumulator; advance the burst
        for (k <- 0 until maxGrp) when(k.U < Rbeats) {
          sclAcc((burstCnt * Rbeats + k.U)(log2Ceil(maxScl) - 1, 0)) := sNow(k)
        }
        burstCnt := Mux(isBurstLast, 0.U, burstCnt + 1.U)
      }
    }
    when(doDataEnq && isBurstLast) { sclPending := emitScl; sclBeatReg := sclBeat }
      .elsewhen(doSclEnq) { sclPending := false.B }
    when(enqFire =/= deq) { credit := Mux(enqFire, credit - 1.U, credit + 1.U) }
  }

  ext_busy_o := (credit =/= Qdepth.U) || (grp =/= 0.U) || sclPending || (burstCnt =/= 0.U)
}
