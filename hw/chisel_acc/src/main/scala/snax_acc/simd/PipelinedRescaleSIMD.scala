package snax_acc.simd

import chisel3._
import chisel3.util._
import chisel3.VecInit
import snax_acc.utils.DecoupledCut._

// Rescale SIMD module
// This module implements this spec: specification: https://gist.github.com/jorendumoulin/83352a1e84501ec4a7b3790461fee2bf in parallel
class PipelinedRescaleSIMD(params: RescaleSIMDParams)
    extends RescaleSIMD(params) {

  // state: pe in-compute or not
  val sNotCOMP :: sCOMP :: Nil = Enum(2)
  val compstate = RegInit(sNotCOMP)
  val ncompstate = WireInit(sNotCOMP)

  compstate := ncompstate

  chisel3.dontTouch(compstate)
  when(cstate === sBUSY) {
    switch(compstate) {
      is(sNotCOMP) {
        when(io.data.input_i.fire) {
          ncompstate := sCOMP
        }.otherwise {
          ncompstate := sNotCOMP
        }
      }
      is(sCOMP) {
        when(io.data.output_o.fire && !io.data.input_i.fire) {
          ncompstate := sNotCOMP
        }.elsewhen(io.data.input_i.fire) {
          ncompstate := sCOMP
        }.otherwise {
          ncompstate := sCOMP
        }
      }
    }
  }.otherwise {
    ncompstate := sNotCOMP
  }

  val lane_comp_cycle = params.dataLen / params.laneLen

  // lane computation process counter
  val pe_input_valid = WireInit(0.B)
  val pipe_input_counter = RegInit(0.U(32.W))

  pe_input_valid := io.data.input_i.fire || (compstate === sCOMP && pipe_input_counter =/= lane_comp_cycle.U)
  when(pe_input_valid && pipe_input_counter =/= lane_comp_cycle.U - 1.U) {
    pipe_input_counter := pipe_input_counter + 1.U
  }.elsewhen(pe_input_valid && pipe_input_counter === lane_comp_cycle.U - 1.U) {
    pipe_input_counter := 0.U
  }.elsewhen(compstate === sNotCOMP || cstate === sIDLE) {
    pipe_input_counter := 0.U
  }

  // store the input data for params.laneLen - 1 lane
  val input_data_reg = RegInit(
    0.U(((params.dataLen - params.laneLen) * params.inputType).W)
  )
  when(io.data.input_i.fire) {
    input_data_reg := io.data.input_i.bits(
      params.dataLen * params.inputType - 1,
      params.laneLen * params.inputType
    )
  }.elsewhen(compstate === sCOMP && lane.map(_.io.input_i.valid).reduce(_ && _)) {
    input_data_reg := input_data_reg >> (params.inputType.U * params.laneLen.U)
  }

  val current_input_data = Wire(Decoupled(UInt((params.laneLen * params.inputType).W)))
  current_input_data.bits := Mux(
    io.data.input_i.fire,
    io.data.input_i.bits(params.laneLen * params.inputType - 1, 0),
    input_data_reg(params.laneLen * params.inputType - 1, 0)
  )
  current_input_data.valid := pe_input_valid

  val delayed_three_cycle_current_input_data = Wire(Decoupled(UInt((params.laneLen * params.inputType).W)))
  current_input_data -\\\> delayed_three_cycle_current_input_data

  // give each RescalePE right control signal and data
  // collect the result of each RescalePE
  for (i <- 0 until params.laneLen) {
      lane(i).io.input_i.bits := delayed_three_cycle_current_input_data.bits(
        (i + 1) * params.inputType - 1,
        i * params.inputType
      ).asSInt
      lane(i).io.input_i.valid := delayed_three_cycle_current_input_data.valid
      // fake ready, always ready inside the pipeline
      lane(i).io.output_o.ready := 1.B
  }

  delayed_three_cycle_current_input_data.ready := lane.map(_.io.input_i.ready).reduce(_ && _) && (cstate === sBUSY)

  // lane output valid process counter
  val pipe_out_counter = RegInit(0.U(16.W))
  val lane_output_valid = lane.map(_.io.output_o.valid).reduce(_ && _)
  when(
    compstate === sCOMP && lane_output_valid && pipe_out_counter =/= lane_comp_cycle.U - 1.U
  ) {
    pipe_out_counter := pipe_out_counter + 1.U
  }.elsewhen(
    compstate === sCOMP && lane_output_valid && pipe_out_counter === lane_comp_cycle.U - 1.U
  ) {
    pipe_out_counter := 0.U
  }.elsewhen(compstate === sNotCOMP || cstate === sIDLE) {
    pipe_out_counter := 0.U
  }

  // collect the valid output data
  val output_data_reg = RegInit(0.U((params.dataLen * params.outputType).W))
  when(lane_output_valid) {
    output_data_reg := Cat(
      Cat(result.reverse),
      output_data_reg(
        params.dataLen * params.outputType - 1,
        params.laneLen * params.outputType
      )
    )
  }

  // valid for new input if pipeline is empty and output is not stalled
  val keep_output = RegInit(0.B)
  val output_stall = WireInit(0.B)
  output_stall := io.data.output_o.valid & !io.data.output_o.ready
  keep_output := output_stall

  io.data.input_i.ready := !keep_output && !output_stall && cstate === sBUSY && (pipe_input_counter === 0.U && pipe_out_counter === 0.U)

  // concat every result to a big data bus for output
  // if is keep sending output, send the stored result
  when(lane_output_valid && pipe_out_counter === lane_comp_cycle.U - 1.U) {
    io.data.output_o.bits := Cat(
      Cat(result.reverse),
      output_data_reg(
        params.dataLen * params.outputType - 1,
        params.laneLen * params.outputType
      )
    )
  }.otherwise {
    io.data.output_o.bits := output_data_reg
  }

  // first valid from RescalePE or keep sending valid if receiver side is not ready
  io.data.output_o.valid := (lane_output_valid && pipe_out_counter === lane_comp_cycle.U - 1.U) || keep_output

}

// Scala main function for generating system verilog file for the Rescale SIMD module
object PipelinedRescaleSIMD extends App {
  emitVerilog(
    new PipelinedRescaleSIMD(PipelinedConfig.rescaleSIMDConfig),
    Array("--target-dir", "generated/simd")
  )
}
