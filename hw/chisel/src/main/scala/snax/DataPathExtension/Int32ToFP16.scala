// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>

package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** Int32ToFp16
  *
  * Converts a signed 32-bit integer (SInt) to IEEE-754 half-precision float (UInt(16.W)).
  *   - Rounds to nearest, ties to even.
  *   - Saturates to +/- Infinity on overflow.
  *   - Zero maps exactly to +0 or -0 (sign bit from input).
  */
class Int32ToFp16PE extends Module {
  val io = IO(new Bundle {
    val in  = Input(SInt(32.W))
    val out = Output(UInt(16.W)) // IEEE-754 fp16
  })

  // Constants for fp16
  val expBias = 15.U(5.W) // bias for 5-bit exponent
  val infExp  = 31.U(5.W) // exponent field for Inf/NaN (11111b)

  val sign = io.in(31)
  val abs  = Mux(io.in < 0.S, (-io.in).asUInt, io.in.asUInt)

  // Default output = 0
  val result = WireDefault(0.U(16.W))

  when(abs === 0.U) {
    // ±0
    result := Cat(sign, 0.U(15.W))
  }.otherwise {
    // -----------------------------
    // 1) Find MSB index of abs
    // -----------------------------
    // msbIndex in [0..31], 0 = LSB, 31 = MSB
    val rev      = Reverse(abs)
    val lsbIndex = PriorityEncoder(rev) // index of first 1 in reversed
    val msbIndex = 31.U - lsbIndex      // index of highest 1 in original

    // Unbiased exponent (actual power of two for normalized 1.x * 2^exp)
    val expUnbiased = msbIndex // 0..31, 5 bits

    // -----------------------------
    // 2) Normalize magnitude
    // -----------------------------
    val shiftAmt = 31.U - msbIndex // how much to left-shift so MSB -> bit 31
    val magNorm  = (abs << shiftAmt)(31, 0)

    // -----------------------------
    // 3) Extract fraction + Guard/Round/Sticky
    // -----------------------------
    val frac     = magNorm(30, 21) // 10 fraction bits
    val guard    = magNorm(20)
    val roundBit = magNorm(19)
    val sticky   = magNorm(18, 0).orR

    val lsb       = frac(0)
    val increment = guard && (roundBit || sticky || lsb) // round-to-nearest-even

    // -----------------------------
    // 4) Rounding: use wider adder to detect mantissa carry
    // -----------------------------
    // 11-bit sum: [carry | 10-bit fraction]
    val fracPlus     = Cat(0.U(1.W), frac) + increment
    val mantOverflow = fracPlus(10) // carry out
    val fracRounded  = Mux(mantOverflow, 0.U(10.W), fracPlus(9, 0))

    // -----------------------------
    // 5) Exponent with bias + mantissa overflow adjustment
    // -----------------------------
    val expPreWide     = expUnbiased +& expBias                    // 6-bit result
    val expIncWide     = expPreWide + 1.U
    val expRoundedWide = Mux(mantOverflow, expIncWide, expPreWide) // 6 bits

    // -----------------------------
    // 6) Handle overflow to infinity
    // -----------------------------
    val overflow = expRoundedWide >= infExp

    when(overflow) {
      // Saturate to +/- Infinity
      result := Cat(sign, Fill(5, 1.U(1.W)), 0.U(10.W))
    }.otherwise {
      val expField = expRoundedWide(4, 0) // take lower 5 bits
      result := Cat(sign, expField, fracRounded)
    }
  }

  io.out := result
}

class Int32ToFp16Converter(
  dataWidth:          Int      = 512,
  in_elementWidth:    Int      = 32,
  out_elementWidth:   Int      = 16,
  extra_loops_choice: Seq[Int] = Seq(1, 2) // 
)(implicit extensionParam: DataPathExtensionParam)
    extends DataPathExtension {

  // set requirements
  require(dataWidth % 32 == 0, "dataWidth must be multiple of 32")
  require(
    in_elementWidth == 32,
    s"Int32ToFp16Converter: in_elementWidth must be 32, got $in_elementWidth"
  )
  require(
    out_elementWidth == 16,
    s"Int32ToFp16Converter: out_elementWidth must be 16, got $out_elementWidth"
  )
  val numPEs = dataWidth / 32

  // -------------------------
  // Counter for controlling conversion
  // -------------------------
  // the counter counts up to (in_elementWidth / out_elementWidth) * extra_loop
  // assume 32 is enough for extra_loop
  val counter = Module(new snax.utils.BasicCounter(log2Ceil(in_elementWidth / out_elementWidth * 32)) {
    override val desiredName = "Int32ToFp16Converter" + "_counter"
  })

  // create ROM for extra_loops_choice
  val extra_loops_rom = VecInit(extra_loops_choice.map(_.U))
  // get extra_loop from csr
  val extra_loop      = extra_loops_rom(ext_csr_i(0)).asUInt // number of batches needed to get a full dataWidth conversion output
  // conversion numbers to get the effective dataWidth output if all PEs are used
  val numConversions  = (in_elementWidth / out_elementWidth).U

  // the total number of conversions needed to get the full dataWidth output
  counter.io.ceil  := numConversions * extra_loop
  counter.io.reset := ext_start_i
  counter.io.tick  := ext_data_i.fire
  ext_busy_o       := counter.io.value =/= 0.U

  // -------------------------
  // PE Array
  // -------------------------
  val peArray = Seq.fill(numPEs) {
    Module(new Int32ToFp16PE() {
      override def desiredName = extensionParam.moduleName + "_int32_to_fp16_pe"
    })
  }

  val pe_inputs  = Wire(Vec(numPEs, SInt(32.W)))
  val pe_outputs = Wire(Vec(numPEs, UInt(16.W)))

  // -------------------------
  // Effective PEs per phase
  // -------------------------
  val effectivePEs = (numPEs.U) / extra_loop

  for (i <- 0 until numPEs) {
    when(i.U < effectivePEs){
      pe_inputs(i)     := ext_data_i.bits(32 * (i + 1) - 1, 32 * i).asSInt
      peArray(i).io.in := pe_inputs(i)
      pe_outputs(i)    := peArray(i).io.out
    }.otherwise{
      pe_inputs(i)     := 0.S(32.W)
      peArray(i).io.in := 0.S(32.W)
      pe_outputs(i)    := 0.U(16.W)
    }
  }

  // -------------------------
  // Output Registers to store the converted results
  // -------------------------
  val regs = RegInit(
    VecInit(
      Seq.fill((extensionParam.dataWidth / out_elementWidth))(
        0.U(out_elementWidth.W)
      )
    )
  )

  // -------------------------
  // Per-phase indexing
  // -------------------------
  // for batch, upper loop
  // for phase, inner loop
  val phase   = counter.io.value % numConversions // which phase we are in
  val batchId = counter.io.value / numConversions // which batch of outputs

  val update_previous_regs = ext_data_i.fire && (counter.io.value =/= (numConversions * extra_loop - 1.U))
  val update_final_regs = ext_data_i.valid && (counter.io.value === (numConversions * extra_loop - 1.U))

  for (i <- 0 until numPEs) {
    // dynamic index in regs for this PE
    val reg_index = batchId * numConversions * effectivePEs + phase * effectivePEs + i.U

    // Only the first 'effectivePEs' PEs store into regs in each phase
    val useThisPE = i.U < effectivePEs

    when((update_previous_regs || update_final_regs) && useThisPE) {
        regs(reg_index) := pe_outputs(i)
    }
  }

  // -------------------------
  // Output construction
  // -------------------------
  // concatenate all regs to form output
  ext_data_o.bits := Cat(regs.reverse).asTypeOf(ext_data_o.bits)

  val data_valid = RegNext(update_final_regs)
  val keep_data_valid = RegInit(false.B)
  keep_data_valid := ext_data_o.valid && !ext_data_o.ready

  ext_data_o.valid := data_valid || keep_data_valid

  ext_data_i.ready := !(ext_data_o.valid && !ext_data_o.ready) && !keep_data_valid
}

class HasInt32ToFp16Converter(dataWidth: Int = 512) extends HasDataPathExtension {
  // The length of row, col, and elementWidth should be the same
  require(dataWidth % 32 == 0, "dataWidth must be multiple of 32")

  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = s"Int32ToFp16Converter_${dataWidth}",
      userCsrNum = 1,
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): Int32ToFp16Converter =
    Module(
      new Int32ToFp16Converter(dataWidth) {
        override def desiredName = clusterName + namePostfix
      }
    )
}
