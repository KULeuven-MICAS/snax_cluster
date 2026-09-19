package snax.xdma.xdmaFrontend

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathJunction.HasMonoidJunction
import snax.readerWriter.ReaderWriterParam
import snax.xdma.DesignParams._

/** Liveness of the two mode FSMs' WAIT states in `XDMADataSwitch`.
  *
  * Both modes leave their busy state on the same condition:
  *
  * {{{
  *   is(stateChainedWriteWait) { when(~io.fromRemote.valid) { nextState  := stateIdle } }
  *   is(gWait)                 { when(~io.fromRemote.valid) { gNextState := gIdle    } }
  * }}}
  *
  * `io.fromRemote` is the single shared remote-data port -- it carries every arriving beat, whichever transfer it
  * belongs to -- and the port's own flow control is gated on the level those states are asserting:
  *
  * {{{
  *   io.fromRemote.ready := remoteSplitter.io.in.ready && io.writerBusy
  * }}}
  *
  * So `~io.fromRemote.valid` does not mean "my transfer's data is done". It means "nobody anywhere is offering a
  * beat". The mode therefore stays up for as long as SOMETHING is offered, whether or not this transfer can consume
  * it -- and `io.writerBusy` (which is `isChainedWrite | isGather | writerBusyRaw`) stays up with it.
  *
  * That level is what `XDMADataPath` publishes as `fromRemoteAccompaniedCfg.readyToTransfer`, and it is also what
  * `XDMACtrl`'s destination FSM waits on before popping the next cfg:
  *
  * {{{
  *   is(sBusy) { when(~io.localXDMACfg.writerBusy) { currentCfgDst.ready := true.B } }
  * }}}
  *
  * so a mode that will not drop pins the whole node: the AXI adapter's receive window never closes, its grant
  * manager never rearms, and the next cfg is never popped.
  *
  * WHAT THIS TEST ASKS, precisely: can the wait state be pinned by a beat its own consumer refuses? It offers one
  * beat after the local writer has finished, with the writer no longer accepting -- and checks whether the mode
  * ever returns to idle. It asserts the LIVENESS property (it must), so a pinning FSM fails here.
  */
class XDMADataSwitchWaitExitTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))
  private val dataWidth = 512

  private def param = new XDMAParam(
    cfgParam          = new XDMAConfigParam(addrWidth = 32, dataWidth = 32),
    axiParam          = new XDMAAXIParam,
    crossClusterParam = new XDMACrossClusterParam,
    rwParam           = new ReaderWriterParam,
    extParam          = Seq(),
    junctionParam     = Seq(new HasMonoidJunction(dataWidth = dataWidth))
  )

  private def idle(dut: XDMADataSwitch): Unit = {
    dut.io.readerLocalLoopback.poke(false)
    dut.io.writerLocalLoopback.poke(false)
    dut.io.writerRemoteLoopback.poke(false)
    dut.io.writerStart.poke(false)
    dut.io.writerBusyRaw.poke(false)
    dut.io.readerBusy.poke(false)
    dut.io.junctionCfg.enable.poke(0)
    dut.io.junctionCfg.userCsr(0).poke(0)
    dut.io.localIn.valid.poke(false)
    dut.io.fromRemote.valid.poke(false)
    dut.io.localOut.ready.poke(false)
    dut.io.toRemote.ready.poke(false)
    dut.clock.step(2)
  }

  /** Step up to `maxCyc` cycles and report the first cycle on which `isChainedWrite | isGather` was low. */
  private def cyclesUntilModeClears(dut: XDMADataSwitch, maxCyc: Int = 1200): Option[Int] = {
    for (c <- 0 until maxCyc) {
      val chained = dut.io.isChainedWrite.peekBoolean()
      val gather  = dut.io.isGather.peekBoolean()
      if (!chained && !gather) return Some(c)
      dut.clock.step(1)
    }
    None
  }

  "XDMADataSwitch_CHAINWRITE_waitExit" should "not be pinned by a beat its consumer refuses" in {
    test(new XDMADataSwitch(param, dataWidth, "sw")).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      idle(dut)

      // Enter CHAINWRITE: a next hop exists and this node writes locally too.
      dut.io.writerRemoteLoopback.poke(true)
      dut.io.writerBusyRaw.poke(true)
      dut.clock.step(2)
      assert(dut.io.isChainedWrite.peekBoolean(), "CHAINWRITE state never entered")

      // The local writer finishes -> the FSM drops into stateChainedWriteWait, still asserting the mode.
      dut.io.writerBusyRaw.poke(false)
      dut.clock.step(1)
      assert(dut.io.isChainedWrite.peekBoolean(), "expected to be parked in the WAIT state")

      // A beat is offered that this transfer's consumers will not take: the local writer has finished, so
      // `localOut` is not draining. In CHAINWRITE the splitter must place a copy in BOTH sinks, so it cannot
      // accept the beat at all.
      dut.io.fromRemote.bits.poke(0.U)
      dut.io.fromRemote.valid.poke(true)
      dut.io.localOut.ready.poke(false)
      dut.io.toRemote.ready.poke(true)

      val cleared = cyclesUntilModeClears(dut)
      println(s"[Switch/waitExit] CHAINWRITE mode cleared after: ${cleared.map(_.toString).getOrElse("NEVER")}")
      assert(
        cleared.isDefined,
        "CHAINWRITE wait state is pinned by an unconsumable beat: `isChainedWrite` never drops, so " +
          "`io.writerBusy` never drops, so `fromRemoteAccompaniedCfg.readyToTransfer` never falls and " +
          "XDMACtrl never pops the next destination cfg. The node is wedged."
      )
    }
  }

  // NO CHAINGATHER ARM, deliberately. The same probe does not reach `gWait` on the gather path: `gActive` exits
  // on `~gatherWorking`, which includes `junctionHost.io.busy`, so a junction holding an undrained operand pins
  // the mode one state EARLIER -- and that is legitimate backpressure (a fold cannot retire until its result has
  // somewhere to go), not the shared-port hazard. `drainComplete` is wired into `gWait` as well for symmetry, but
  // no test here claims the gather path was ever pinned by it. `XDMADataSwitchTester`'s "GATHER x2" already pins
  // the property that matters there: the mode clears fully between two consecutive gathers.
}
