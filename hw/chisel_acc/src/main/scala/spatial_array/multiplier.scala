package snax_acc.spatial_array

import chisel3._

class MultiplierIO(
  inputAElemWidth:  Int,
  inputBElemWidth:  Int,
  outputCElemWidth: Int
) extends Bundle {
  val in_a  = Input(UInt(inputAElemWidth.W))
  val in_b  = Input(UInt(inputBElemWidth.W))
  val out_c = Output(UInt(outputCElemWidth.W))
}

class Multiplier(
  opType:           Int,
  inputAElemWidth:  Int,
  inputBElemWidth:  Int,
  outputCElemWidth: Int
) extends Module
    with RequireAsyncReset {
  val io = IO(new MultiplierIO(inputAElemWidth, inputBElemWidth, outputCElemWidth))
  require(
    opType == OpType.UIntUIntOp || opType == OpType.SIntSIntOp ||
      opType == OpType.Float16Int4Op || opType == OpType.Float16Float16Op
  )
  require(
    inputAElemWidth > 0 && inputBElemWidth > 0 && outputCElemWidth > 0,
    "Element widths must be greater than 0"
  )
  if (opType == OpType.UIntUIntOp) {
    io.out_c := io.in_a * io.in_b
  } else if (opType == OpType.SIntSIntOp) {
    io.out_c := (io.in_a.asTypeOf(SInt(outputCElemWidth.W)) * io.in_b.asTypeOf(
      SInt(outputCElemWidth.W)
    )).asUInt
  } else if (opType == OpType.Float16Int4Op) {
    val fpMulInt = Module(
      new FPMULIntBlackBox("fp_mul_int", inputAElemWidth, inputBElemWidth, outputCElemWidth)
    )
    fpMulInt.io.operand_a_i := io.in_a
    fpMulInt.io.operand_b_i := io.in_b
    io.out_c                := fpMulInt.io.result_o
    // assert(
    //   inputAElemWidth == 16 && inputBElemWidth == 4 && outputCElemWidth == 32,
    //   "For Float16Int4Op, input widths must be 16, 4 and output width must be 32"
    // )
  } else if (opType == OpType.Float16Float16Op) {
    val fpMulfp = Module(
      new FPMULFP("fp_mul", inputAElemWidth, inputBElemWidth, outputCElemWidth)
    )
    fpMulfp.io.operand_a_i  := io.in_a
    fpMulfp.io.operand_b_i  := io.in_b
    io.out_c                := fpMulfp.io.result_o
    assert(
      inputAElemWidth == 16 && inputBElemWidth == 16 && outputCElemWidth == 32,
      "For Float16Float16Op, input widths must be 16, 16 and output width must be 32"
    )
  } else {
    // TODO: add support for other types
    // For now, just set the output to 0
    io.out_c := 0.U
  }
}

object MultiplierEmitterUInt extends App {
  emitVerilog(
    new Multiplier(OpType.UIntUIntOp, 8, 4, 16),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object MultiplierEmitterSInt extends App {
  emitVerilog(
    new Multiplier(OpType.SIntSIntOp, 8, 4, 16),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object MultiplierEmitterFloat16Int4 extends App {
  emitVerilog(
    new Multiplier(OpType.Float16Int4Op, 16, 4, 32),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object MultiplierEmitterFloat16Float16 extends App {
  emitVerilog(
    new Multiplier(OpType.Float16Float16Op, 16, 16, 32),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object MultiplierEmitters {

// int Multiplier for evaluation
  def emitInt2_Int2_Int4(): Unit = {
    val tag = "Int2_Int2_Int4_eva"
    emitVerilog(
      new Multiplier(OpType.SIntSIntOp, 2, 2, 4),
      Array("--target-dir", s"/users/micas/xyi/no_backup/opengemm_journal_exp/pe_syn_scripts/rtl_src_code/Multiplier/$tag")
    )
  }

  def emitInt4_Int4_Int8(): Unit = {
    val tag = "Int4_Int4_Int8_eva"
    emitVerilog(
      new Multiplier(OpType.SIntSIntOp, 4, 4, 8),
      Array("--target-dir", s"/users/micas/xyi/no_backup/opengemm_journal_exp/pe_syn_scripts/rtl_src_code/Multiplier/$tag")
    )
  }

  def emitInt8_Int8_Int16(): Unit = {
    val tag = "Int8_Int8_Int16_eva"
    emitVerilog(
      new Multiplier(OpType.SIntSIntOp, 8, 8, 16),
      Array("--target-dir", s"/users/micas/xyi/no_backup/opengemm_journal_exp/pe_syn_scripts/rtl_src_code/Multiplier/$tag")
    )
  }

  def emitInt16_Int4_Int32(): Unit = {
    val tag = "Int16_Int4_Int32_eva"
    emitVerilog(
      new Multiplier(OpType.SIntSIntOp, 16, 4, 32),
      Array("--target-dir", s"/users/micas/xyi/no_backup/opengemm_journal_exp/pe_syn_scripts/rtl_src_code/Multiplier/$tag")
    )
  }

  def emitInt16_Int16_Int32(): Unit = {
    val tag = "Int16_Int16_Int32_eva"
    emitVerilog(
      new Multiplier(OpType.SIntSIntOp, 16, 16, 32),
      Array("--target-dir", s"/users/micas/xyi/no_backup/opengemm_journal_exp/pe_syn_scripts/rtl_src_code/Multiplier/$tag")
    )
  }

  def emitInt32_Int32_Int64(): Unit = {
    val tag = "Int32_Int32_Int64_eva"
    emitVerilog(
      new Multiplier(OpType.SIntSIntOp, 32, 32, 64),
      Array("--target-dir", s"/users/micas/xyi/no_backup/opengemm_journal_exp/pe_syn_scripts/rtl_src_code/Multiplier/$tag")
    )
  }

// fp-int Multiplier for evaluation
  def emitFloat16_Int1_Float32(): Unit = {
    val tag = "Float16_Int1_Float32_eva"
    emitVerilog(
      new Multiplier(OpType.Float16Int4Op, 16, 1, 32),
      Array("--target-dir", s"/users/micas/xyi/no_backup/opengemm_journal_exp/pe_syn_scripts/rtl_src_code/Multiplier/$tag")
    )
  }

  def emitFloat16_Int2_Float32(): Unit = {
    val tag = "Float16_Int2_Float32_eva"
    emitVerilog(
      new Multiplier(OpType.Float16Int4Op, 16, 2, 32),
      Array("--target-dir", s"/users/micas/xyi/no_backup/opengemm_journal_exp/pe_syn_scripts/rtl_src_code/Multiplier/$tag")
    )
  }

  def emitFloat16_Int3_Float32(): Unit = {
    val tag = "Float16_Int3_Float32_eva"
    emitVerilog(
      new Multiplier(OpType.Float16Int4Op, 16, 3, 32),
      Array("--target-dir", s"/users/micas/xyi/no_backup/opengemm_journal_exp/pe_syn_scripts/rtl_src_code/Multiplier/$tag")
    )
  }

  def emitFloat16_Int4_Float32(): Unit = {
    val tag = "Float16_Int4_Float32_eva"
    emitVerilog(
      new Multiplier(OpType.Float16Int4Op, 16, 4, 32),
      Array("--target-dir", s"/users/micas/xyi/no_backup/opengemm_journal_exp/pe_syn_scripts/rtl_src_code/Multiplier/$tag")
    )
  }

// fp-fp Multiplier for evaluation
  def emitFloat16_Float16_Float32(): Unit = {
    val tag = "Float16_Float16_Float32_eva"
    emitVerilog(
      new Multiplier(OpType.Float16Float16Op, 16, 16, 32),
      Array("--target-dir", s"/users/micas/xyi/no_backup/opengemm_journal_exp/pe_syn_scripts/rtl_src_code/Multiplier/$tag")
    )
  }

}

object RunAllMultiplierEmitters extends App {
  import MultiplierEmitters._

  println("Running all multiplier emitters...")

  // emitInt2_Int2_Int4()
  // emitInt4_Int4_Int8()
  // emitInt8_Int8_Int16()
  // emitInt16_Int4_Int32()
  // emitInt16_Int16_Int32()
  // emitInt32_Int32_Int64()

  // emitFloat16_Int1_Float32()
  // emitFloat16_Int2_Float32()
  // emitFloat16_Int3_Float32()
  // emitFloat16_Int4_Float32()

  emitFloat16_Float16_Float32()

  println("All multiplier emitters completed.")
}
