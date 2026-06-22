// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._

/** Fp16ToInt8PE: symmetric per-tensor quantize of one FP16 lane to a signed int8.
  *
  *   q = sat_[-127,127]( round_rne( widen_fp32(x) * inv_scale ) )
  *
  * The mirror image of Int32ToFp16PE. FP16 transport in, FP32 internal (fpnew mixed-format
  * blackboxes), SInt(8) out. The fp32->int round uses the 1.5*2^23 magic-number add (no F2I unit),
  * the same trick as FpExp/FpSilu. The product is first clamped into [-128, 128] so the magic add
  * stays exact (|value| << 2^23) and the rounded integer lands in a tiny signed range; the result is
  * then saturated to the SYMMETRIC int8 range [-127, 127] (zero-point = 0).
  */
class Fp16ToInt8PE extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in        = Input(UInt(16.W))  // FP16
    val inv_scale = Input(UInt(32.W))  // FP32 (= 1 / quant_scale)
    val out       = Output(SInt(8.W))
  })

  import FpHelpers._

  val MAGIC = f32lit(12582912.0f) // 1.5 * 2^23 = 0x4B400000
  val HI    = f32lit(128.0f)      // pre-round clamp window (keeps the RNE exact, bounds iM)
  val LO    = f32lit(-128.0f)

  // widen fp16 -> fp32, then multiply by the FP32 inv_scale
  val xf     = widen(io.in, FP16)
  val scaled = fmul(xf, io.inv_scale)

  // clamp the fp32 product into [-128, 128] before the magic add
  val clamped = fp32max(fp32min(scaled, HI), LO)

  // round to nearest (ties to even) via the magic-number add; |iM| <= 128 after the clamp
  val rM = fadd(clamped, MAGIC)
  val iM = rM.asSInt - 0x4b400000.S

  // symmetric saturate to [-127, 127]
  val q = WireDefault(0.S(8.W))
  when(iM > 127.S) {
    q := 127.S(8.W)
  }.elsewhen(iM < (-127).S) {
    q := (-127).S(8.W)
  }.otherwise {
    q := iM(7, 0).asSInt // exact: |iM| <= 127 in this branch
  }

  io.out := q
}

/** Fp16ToInt8: stream FP16 -> INT8 quantize extension. Packs 2 input beats (32 FP16 = 512b each)
  * into 1 output beat (64 INT8 = 512b), reusing RescaleDown's beat-accumulation FSM. One FP32 CSR
  * carries inv_scale. Chains LAST in the reader extension list so a vector op (StreamMap /
  * StreamElementwise) computes fp16 and the cast narrows it to int8 before it leaves the stream.
  *
  * NOTE: the 2:1 packing requires an EVEN number of input beats (a lone trailing beat is latched but
  * never emitted). All current SIMD apps use row lengths that are a multiple of 64 FP16 elements.
  */
class Fp16ToInt8(
  in_elementWidth:  Int = 16,
  out_elementWidth: Int = 8
)(implicit extensionParam: DataPathExtensionParam)
    extends DataPathExtension {

  require(
    extensionParam.dataWidth % in_elementWidth == 0,
    s"Fp16ToInt8: dataWidth (${extensionParam.dataWidth}) must be a multiple of in_elementWidth ($in_elementWidth)"
  )
  require(
    in_elementWidth == 16 && out_elementWidth == 8,
    s"Fp16ToInt8: only FP16(16)->INT8(8) supported, got $in_elementWidth->$out_elementWidth"
  )

  val counter = Module(new snax.utils.BasicCounter(log2Ceil(in_elementWidth / out_elementWidth)) {
    override val desiredName = "Fp16ToInt8Counter"
  })
  counter.io.ceil  := (in_elementWidth / out_elementWidth).asUInt
  counter.io.reset := ext_start_i
  counter.io.tick  := ext_data_i.fire
  ext_busy_o       := counter.io.value =/= 0.U(1.W)

  val inv_scale = WireInit(ext_csr_i(0).asUInt)

  // regs hold the int8 results of the earlier phase(s); out_wires hold the current phase live.
  val regs = RegInit(
    VecInit(
      Seq.fill((extensionParam.dataWidth / out_elementWidth) - (extensionParam.dataWidth / in_elementWidth))(
        0.S(out_elementWidth.W)
      )
    )
  )

  val out_wires = Wire(Vec(extensionParam.dataWidth / in_elementWidth, SInt(out_elementWidth.W)))

  val PEs = for (i <- 0 until extensionParam.dataWidth / in_elementWidth) yield {
    val PE = Module(new Fp16ToInt8PE {
      override def desiredName = extensionParam.moduleName + "_fp16_to_int8_pe"
    })
    PE.io.in        := ext_data_i.bits((i + 1) * in_elementWidth - 1, i * in_elementWidth)
    PE.io.inv_scale := inv_scale

    when(ext_data_i.fire) {
      when(counter.io.value =/= ((in_elementWidth / out_elementWidth).U - 1.U)) {
        regs(counter.io.value * (extensionParam.dataWidth / in_elementWidth).U + i.U) := PE.io.out
      }
    }
    out_wires(i) := PE.io.out
  }

  ext_data_o.bits  := VecInit(regs ++ out_wires).asTypeOf(ext_data_o.bits)
  ext_data_o.valid := ext_data_i.fire && counter.io.value === ((in_elementWidth / out_elementWidth).U - 1.U)
  ext_data_i.ready := ext_data_o.ready
}

class HasFp16ToInt8(
  in_elementWidth:  Int = 16,
  out_elementWidth: Int = 8,
  dataWidth:        Int = 512
) extends HasDataPathExtension {
  require(
    in_elementWidth == 16 && out_elementWidth == 8,
    s"HasFp16ToInt8: only FP16(16)->INT8(8) supported, got $in_elementWidth->$out_elementWidth"
  )

  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "Fp16ToInt8", // -> READER_EXT_FP16TOINT8 (keep stable: never width-encode the name)
      userCsrNum = 1,            // inv_scale (FP32 bits)
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): Fp16ToInt8 =
    Module(
      new Fp16ToInt8(in_elementWidth, out_elementWidth) {
        override def desiredName = clusterName + namePostfix
      }
    )
}
