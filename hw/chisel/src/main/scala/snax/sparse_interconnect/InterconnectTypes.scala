package snax.sparse_interconnect

import chisel3._

/** Chisel Enum for Atomic Memory Operation (AMO) types, corresponding to reqrsp_pkg::amo_op_e in SystemVerilog. */
object AmoOp extends ChiselEnum {
  val None = Value(0.U)
  val Swap = Value(1.U)
  val Add  = Value(2.U)
  val And  = Value(3.U)
  val Or   = Value(4.U)
  val Xor  = Value(5.U)
  val Max  = Value(6.U)
  val Maxu = Value(7.U)
  val Min  = Value(8.U)
  val Minu = Value(9.U)
  val LR   = Value(10.U)
  val SC   = Value(11.U)
}

/** TCDM Request Type */
class TcdmReq(
  addrWidth: Int,
  dataWidth: Int,
  strbWidth: Int,
  userWidth: Int
) extends Bundle {
  val addr  = UInt(addrWidth.W)
  val write = Bool()
  val amo   = AmoOp()
  val data  = UInt(dataWidth.W)
  val strb  = UInt(strbWidth.W)
  val user  = UInt(userWidth.W)
}

/** TCDM Response Type */
class TcdmRsp(
  dataWidth: Int
) extends Bundle {
  val data = UInt(dataWidth.W)
}
