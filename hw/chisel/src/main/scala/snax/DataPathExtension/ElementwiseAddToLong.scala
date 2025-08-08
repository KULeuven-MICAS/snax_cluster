package snax.DataPathExtension

import chisel3._
import chisel3.util._

class HasElementwiseAddToLong(
  in_elementWidth: Int = 8,
  out_elementWidth: Int = 32,
  dataWidth:    Int = 512
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = s"ElementwiseAddToLongBit${in_elementWidth}To${out_elementWidth}",
      userCsrNum = 1,
      dataWidth  = dataWidth match {
        case 0 => in_elementWidth
        case _ => dataWidth
      }
    )

  def instantiate(clusterName: String): ElementwiseAddToLong =
    Module(
      new ElementwiseAddToLong(in_elementWidth, out_elementWidth) {
        override def desiredName = clusterName + namePostfix
      }
    )
}

object OutputCtrlEnumObj {
  object OutputCtrlEnum extends ChiselEnum {
    val WaitingInput, Busy = Value
  }
}




class AccumulateStage(
  in_elementWidth: Int,
  out_elementWidth: Int
)(implicit extensionParam: DataPathExtensionParam)
    extends DataPathExtension {

  val io = IO(new Bundle {
    val data_i     = Flipped(Valid(UInt(extensionParam.dataWidth.W)))
    val data_o     = Output(UInt((out_elementWidth / in_elementWidth * extensionParam.dataWidth).W)) //TODO: replace by 2D-vector
    val previous_valid = previous_valid
    val output_ready = Input(Bool())
    val self_valid = Output(Bool())
    val self_ready = Output(Bool())
  })

}







class OutputCtrl(
  in_elementWidth: Int = 8,
  out_elementWidth: Int = 32
)(implicit extensionParam: DataPathExtensionParam)
    extends DataPathExtension {
  val io = IO(new Bundle {
    val accumulate_valid = Input(Bool())
    val next_ready       = Input(Bool())
    val self_valid   = Output(Bool())
    val self_ready   = Output(Bool())
    val output_counter_value = Output(UInt(log2Ceil(out_elementWidth / in_elementWidth).W))
  })
    import OutputCtrlEnumObj.OutputCtrlEnum._

    val counter_active = Wire(Bool())
    val reset_counter  = Wire(Bool())

    // Counter to record the steps
    val output_counter = Module(new snax.utils.BasicCounter(8) {
        override val desiredName = "Output_Counter"
    })
    output_counter.io.ceil := (out_elementWidth / in_elementWidth - 1).U(8.W) //TODO: check if inclusive
    output_counter.io.reset := reset_counter
    output_counter.io.tick  := counter_active & next_ready //Only go up when handshaking is done

    output_counter_value := output_counter.io.value

    val state = RegInit(WaitingInput)

    switch(state) {
      is(WaitingInput) {
        when(io.accumulate_valid) {
          state := Busy
        }
      }
      is(Busy) {
        when(output_counter.io.value === (out_elementWidth / in_elementWidth - 1).U(8.W)) {
            when(io.accumulate_valid){
                state := Busy
            }.otherwise {
          state := WaitingOutput
            }
        }.otherwise{
          state := Busy
        }
      }
    }

    switch(state) {
        is(WaitingInput) {
      io.self_valid := false.B
        reset_counter := true.B
        counter_active := false.B
        io.self_ready := true.B
      }
      is(Busy) {
        when(output_counter.io.value === (out_elementWidth / in_elementWidth - 1).U(8.W)) {
          io.self_valid := true.B
          reset_counter := false.B
          counter_active := true.B
          io.self_ready := true.B
        }.otherwise {
          io.self_valid := true.B
          reset_counter := false.B
          counter_active := true.B
          io.self_ready := false.B
        }
      }

  }

class OutputStage(
  in_elementWidth: Int,
  out_elementWidth: Int
)(implicit extensionParam: DataPathExtensionParam)
    extends DataPathExtension { // TODO; Should not extend DataPathExtension
  
  val io = IO(new Bundle {
    val data_i     = Flipped(Valid(UInt((out_elementWidth / in_elementWidth * extensionParam.dataWidth).W))) // TODO: replace by 2D-vector
    val data_o     = Output(UInt(extensionParam.dataWidth.W))
    val accumulate_valid = Input(Bool())
    val next_ready = Input(Bool())
    val self_ready = Output(Bool())
  })

    val output_ctrl = Module(
        new OutputCtrl(in_elementWidth, out_elementWidth) {
        override def desiredName = "OutputCtrl"
        }
    )
    output_ctrl.io.accumulate_valid := io.accumulate_valid
    output_ctrl.io.next_ready       := io.next_ready

    io.self_ready := output_ctrl.io.self_ready

    // For each input element, on output element is needed
    val output_regs = RegInit(
        VecInit(
            Seq.fill(in_elementWidth / out_elementWidth, extensionParam.dataWidth / out_elementWidth)(0.U(out_elementWidth.W))
        )
    )

    //TODO: make logic for updating the registers


    ext_data_o.bits := Cat(output_regs(output_counter.io.value).reverse)
}

























class ElementwiseAddToLong(
  in_elementWidth: Int,
  out_elementWidth: Int
)(implicit extensionParam: DataPathExtensionParam)
    extends DataPathExtension {
  require(
    extensionParam.dataWidth % in_elementWidth == 0,
    s"Data width ${extensionParam.dataWidth} must be a multiple of element width $in_elementWidth"
  )
  require(
    extensionParam.dataWidth % out_elementWidth == 0,
    s"Data width ${extensionParam.dataWidth} must be a multiple of output element width $out_elementWidth"
  )

  // Counter to record the steps
  val counter = Module(new snax.utils.BasicCounter(8) {
    override val desiredName = "ElementwiseAddToLongCounter"
  })
  counter.io.ceil := ext_csr_i(0)
  counter.io.reset := ext_start_i
  counter.io.tick  := ext_data_i.fire
  ext_busy_o       := counter.io.value =/= 0.U

  // The wire to connect the output result
  val ext_data_o_bits = Wire(
    Vec(extensionParam.dataWidth / elementWidth, UInt(elementWidth.W))
  )

  // The register to hold the sum of each Element
  val regs = RegInit(
    VecInit(
      Seq.fill(extensionParam.dataWidth / elementWidth)(0.U(elementWidth.W))
    )
  )

  val input_data = ext_data_i.bits.asTypeOf(
    Vec(extensionParam.dataWidth / elementWidth, UInt(elementWidth.W))
  )
  for (i <- 0 until extensionParam.dataWidth / elementWidth) {
    ext_data_o_bits(i) := regs(i) + input_data(i)
  }

  when(ext_data_i.fire) {
    when(counter.io.value === (ext_csr_i(0) - 1.U(8.W))) {
      // Reset the registers when the counter reaches the end
      for (i <- 0 until extensionParam.dataWidth / elementWidth) {
        regs(i) := 0.U
      }
    } otherwise {
      // Add the input data to the corresponding register
      for (i <- 0 until extensionParam.dataWidth / elementWidth) {
        regs(i) := regs(i) + input_data(i)
      }
    }
  }

  ext_data_o.bits  := Cat(ext_data_o_bits.reverse)
  ext_data_o.valid := (counter.io.value === (ext_csr_i(0) - 1.U)(7, 0)) & ext_data_i.fire
  ext_data_i.ready := ext_data_o.ready
}
