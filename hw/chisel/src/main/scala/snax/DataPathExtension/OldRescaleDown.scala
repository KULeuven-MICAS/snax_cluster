package snax.DataPathExtension

import chisel3._
import chisel3.util._

class RescalePECtrl extends Bundle {

  /** @input
    *   input_zp_i, input zero point
    * @input
    *   output_zp_i, output zero point
    * @input
    *   multiplier_i, scaling factor
    * @input
    *   shift_i, shift number
    * @input
    *   max_int_i, maximum number for clamping
    * @input
    *   min_int_i, minimum number for clamping
    * @input
    *   double_round_i, if double round
    */
  val input_zp_i     = (SInt(32.W))
  val output_zp_i    = (SInt(32.W))
  // ! this port has different data width
  val multiplier_i   = (SInt(32.W))
  val shift_i        = (SInt(32.W))
  val max_int_i      = (SInt(32.W))
  val min_int_i      = (SInt(32.W))
  val double_round_i = (Bool())

  val len = (UInt(32.W))
}

// processing element input and output declaration
class RescalePEIO extends Bundle {
  val ctrl_i = Input(new RescalePECtrl)
  val input_i  = Flipped(Decoupled(SInt(32.W)))
  val output_o = Decoupled(SInt(8.W))
}

// processing element module.
// see specification: https://gist.github.com/jorendumoulin/83352a1e84501ec4a7b3790461fee2bf for more details
class RescalePE extends Module with RequireAsyncReset {
  val io = IO(new RescalePEIO)

  val var0_0 = WireInit((0.S((64).W)))
  val var0   = WireInit((0.S((64).W)))
  val var1   = WireInit(0.S((32).W))
  val var2   = WireInit(0.S((32).W))
  val var3   = WireInit(0.S((32).W))

  // for clamp
  val overflow  = WireInit(0.B)
  val underflow = WireInit(0.B)

  // post processing operations
  var0_0 := (io.input_i.bits - io.ctrl_i.input_zp_i)
  var0 := var0_0 * io.ctrl_i.multiplier_i
  var1 := (var0 >> (io.ctrl_i.shift_i.asUInt - 1.U))(31, 0).asSInt

  when(io.ctrl_i.double_round_i) {
    var2 := (Mux(
      var1 >= 0.S,
      var1 + 1.S,
      var1 - 1.S
    ) >> 1.U) + io.ctrl_i.output_zp_i
  }.otherwise {
    var2 := (var1 >> 1.U) + io.ctrl_i.output_zp_i
  }

  // clamping
  overflow  := var2 > io.ctrl_i.max_int_i
  underflow := var2 < io.ctrl_i.min_int_i
  var3      := Mux(
    overflow,
    io.ctrl_i.max_int_i,
    Mux(underflow, io.ctrl_i.min_int_i, var2)
  )

  io.output_o.bits := var3(7, 0).asSInt

  // combination block, output valid when input data is valid
  io.output_o.valid := io.input_i.valid
  // combination block, input ready when output is ready
  io.input_i.ready  := io.output_o.ready
}

class HasOldRescaleDown(in_elementWidth: Int = 32, out_elementWidth: Int = 8) extends HasDataPathExtension {
  implicit val extensionParam:          DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "OldRescaleDown",
      userCsrNum = 4,
      dataWidth  = 512
    )
  def instantiate(clusterName: String): OldRescaleDown            =
    Module(
      new OldRescaleDown(in_elementWidth, out_elementWidth) {
        override def desiredName = clusterName + namePostfix
      }
    )
}

class OldRescaleDown(
  in_elementWidth:  Int = 32,
  out_elementWidth: Int = 8
)(implicit extensionParam: DataPathExtensionParam)
    extends DataPathExtension {
  // Exact Version of RescaleDown with no optimizations for area efficiency, not meant to be used in production, only for testing purposes
  require(
    extensionParam.dataWidth % in_elementWidth == 0,
    s"RescaleDown: dataWidth (${extensionParam.dataWidth}) must be a multiple of in_elementWidth ($in_elementWidth)"
  )

  require(
    in_elementWidth % out_elementWidth == 0,
    s"RescaleDown: in_elementWidth ($in_elementWidth) must be a multiple of out_elementWidth ($out_elementWidth)"
  )

  val counter = Module(new snax.utils.BasicCounter(log2Ceil(in_elementWidth / out_elementWidth)) {
    override val desiredName = "RescaleDownCounter"
  })
  counter.io.ceil := (in_elementWidth / out_elementWidth).asUInt
  counter.io.reset := ext_start_i
  counter.io.tick  := ext_data_i.fire
  ext_busy_o       := counter.io.value =/= 0.U(1.W)

  val input_zp   = WireInit(ext_csr_i(0).asSInt)
  val multiplier = WireInit(ext_csr_i(1).asSInt)
  val output_zp  = WireInit(ext_csr_i(2).asSInt)
  val shift      = WireInit(ext_csr_i(3).asSInt)

  val regs = RegInit(
    VecInit(
      Seq.fill((extensionParam.dataWidth / out_elementWidth) - (extensionParam.dataWidth / in_elementWidth))(
        0.S(out_elementWidth.W)
      )
    )
  )

  val out_wires = Wire(Vec(extensionParam.dataWidth / in_elementWidth, SInt(out_elementWidth.W)))

  val PEs = for (i <- 0 until extensionParam.dataWidth / in_elementWidth) yield {
    val PE = Module(new RescalePE {
      // override val desiredName = "RescaleDownPE"
    })
    // Create a DecoupledIO wire for each PE
    val inputWire = Wire(Decoupled(SInt(in_elementWidth.W)))
    inputWire.bits := ext_data_i.bits((i + 1) * in_elementWidth - 1, i * in_elementWidth).asSInt
    inputWire.valid := ext_data_i.valid
    PE.io.input_i <> inputWire

    PE.io.ctrl_i.input_zp_i     := input_zp.asSInt
    PE.io.ctrl_i.multiplier_i   := multiplier.asSInt
    PE.io.ctrl_i.output_zp_i    := output_zp.asSInt
    PE.io.ctrl_i.shift_i        := shift.asSInt
    PE.io.ctrl_i.max_int_i      := 127.S(8.W)
    PE.io.ctrl_i.min_int_i      := -128.S(8.W)
    PE.io.ctrl_i.double_round_i := true.B
    PE.io.ctrl_i.len            := 32.U

    when(ext_data_i.fire) {
      when(counter.io.value =/= ((in_elementWidth / out_elementWidth).U - 1.U)) {
        regs(counter.io.value * (extensionParam.dataWidth / in_elementWidth).U + i.U) := PE.io.output_o.bits.asSInt
      }
    }
    out_wires(i) := PE.io.output_o.bits.asSInt
    PE.io.output_o.ready := ext_data_o.ready
  }

  ext_data_o.bits  := VecInit(regs ++ out_wires).asTypeOf(ext_data_o.bits)
  ext_data_o.valid := ext_data_i.fire && counter.io.value === ((in_elementWidth / out_elementWidth).U - 1.U)
  ext_data_i.ready := ext_data_o.ready // Check if this can be more efficient
}

object OldRescaleDownEmitter extends App {
  val svString = getVerilogString(new DataPathExtensionHost(
    extensionList = Seq(new HasOldRescaleDown(32, 8))
  ))
  println(svString)
}
