package snax.simd

import scala.reflect.runtime.currentMirror
import scala.tools.reflect.ToolBox

import play.api.libs.json._
import snax.DataPathExtension.HasDataPathExtension
import snax.readerWriter.ReaderWriterParam

/** Turns a `snax_simd_cfg` JSON blob into a `SimdParam`.
  *
  * Split out of `SimdTopGen` so that the combined-elaboration path in `snax.xdma.xdmaTop.XDMATopGen` can build
  * the same parameters without duplicating this code. The extension list is instantiated by reflection, exactly
  * as the xDMA does it, so adding an extension stays a cfg-only change.
  */
object SimdCfgParser {

  private lazy val toolbox = currentMirror.mkToolBox()

  private def renderExtensionArgs(extensionName: String, args: JsValue): String =
    args match {
      case obj: JsObject =>
        obj.fields.map { case (paramName, paramValue) =>
          val rendered = paramValue.validate[Int] match {
            case JsSuccess(intValue, _) => intValue.toString
            case JsError(_)             =>
              paramValue.validate[Seq[Int]] match {
                case JsSuccess(seqValue, _) => s"Seq(${seqValue.mkString(",")})"
                case JsError(_)             =>
                  paramValue.validate[String] match {
                    case JsSuccess(strValue, _) => "\"" + strValue + "\""
                    case JsError(_)             =>
                      paramValue.validate[Seq[String]] match {
                        case JsSuccess(seqStr, _) => s"Seq(${seqStr.map(s => "\"" + s + "\"").mkString(",")})"
                        case JsError(_)           =>
                          throw new IllegalArgumentException(
                            s"Invalid SIMD extension parameter $extensionName.$paramName: expected Int, Seq[Int], " +
                              "String, or Seq[String]"
                          )
                      }
                  }
              }
          }
          s"$paramName = $rendered"
        }.mkString(", ")
      case _ =>
        throw new IllegalArgumentException(
          s"Invalid SIMD extension parameters for $extensionName: expected an object"
        )
    }

  private def instantiateExtension(extensionName: String, extensionArgs: String): HasDataPathExtension =
    toolbox
      .compile(toolbox.parse(s"""
import snax.DataPathExtension._
return new $extensionName($extensionArgs)
      """))()
      .asInstanceOf[HasDataPathExtension]

  private def extensionsOf(obj: JsObject): Seq[HasDataPathExtension] =
    obj.fields.filter { case (k, _) => k.startsWith("Has") }.toSeq.map { case (k, v) =>
      instantiateExtension(k, renderExtensionArgs(k, v))
    }

  /** `reader_extensions` accepts two shapes, and declaration order is chain order in both.
    *
    *   - OBJECT  `{ HasA: {...}, HasB: {...} }` — every existing cfg, kept working unchanged.
    *   - ARRAY   `[ {HasA: {...}}, {HasB: {...}}, {HasA: {...}} ]`
    *
    * The array exists because a JSON object cannot repeat a key, so the object form can never place the
    * SAME operator at two points in the chain. That is not a hypothetical: the chain is a fixed linear
    * order, so an operator that some kernels need BEFORE the pointwise map and others need AFTER it has to
    * be instantiated twice. Dequantise-then-activate wants an elementwise combine ahead of the map;
    * SwiGLU wants one behind it.
    */
  def extensions(parsedSimdCfg: JsValue): Seq[HasDataPathExtension] =
    (parsedSimdCfg \ "reader_extensions").toOption match {
      case Some(arr: JsArray)  =>
        arr.value.toSeq.flatMap {
          case o: JsObject => extensionsOf(o)
          case _           => Seq.empty
        }
      case Some(obj: JsObject) => extensionsOf(obj)
      case _                   => Seq.empty
    }

  /** The number of 64-bit TCDM channels on each side. Defaults to the DMA beat width so a cfg that omits the
    * knob matches the xDMA's geometry.
    */
  def numChannel(parsedSimdCfg: JsValue, tcdmDataWidth: Int, axiDataWidth: Int): Int =
    (parsedSimdCfg \ "num_channel").asOpt[Int].getOrElse(axiDataWidth / tcdmDataWidth)

  def apply(parsedSimdCfg: JsValue, tcdmDataWidth: Int, axiDataWidth: Int, tcdmSize: Int): SimdParam = {
    val nCh = numChannel(parsedSimdCfg, tcdmDataWidth, axiDataWidth)

    // The reader and the writer share the beat width: the extension chain is one stream and there is no width
    // converter between them. An asymmetric (dual-operand) reader is a separate design step.
    val readerParam = new ReaderWriterParam(
      spatialBounds        = List(nCh),
      temporalDimension    = (parsedSimdCfg \ "reader_agu_temporal_dimension").as[Int],
      tcdmDataWidth        = tcdmDataWidth,
      tcdmSize             = tcdmSize,
      numChannel           = nCh,
      addressBufferDepth   = (parsedSimdCfg \ "reader_buffer").as[Int],
      dataBufferDepth      = (parsedSimdCfg \ "reader_buffer").as[Int],
      configurableChannel  = true,
      configurableByteMask = false,
      // Bid with FIFO occupancy, in the same currency as the GEMM streamer. The xDMA's
      // higherStaticPriority=true would otherwise let a data mover outrank a compute engine at every bank.
      dynamicPriority      = true,
      higherStaticPriority = false
    )

    val writerParam = new ReaderWriterParam(
      spatialBounds        = List(nCh),
      temporalDimension    = (parsedSimdCfg \ "writer_agu_temporal_dimension").as[Int],
      tcdmDataWidth        = tcdmDataWidth,
      tcdmSize             = tcdmSize,
      numChannel           = nCh,
      addressBufferDepth   = (parsedSimdCfg \ "writer_buffer").as[Int],
      dataBufferDepth      = (parsedSimdCfg \ "writer_buffer").as[Int],
      configurableChannel  = true,
      configurableByteMask = true,
      dynamicPriority      = true,
      higherStaticPriority = false
    )

    new SimdParam(
      cfgParam    = new SimdConfigParam(
        addrWidth = 32,
        dataWidth = (parsedSimdCfg \ "cfg_io_width").asOpt[Int].getOrElse(32)
      ),
      readerParam = readerParam,
      writerParam = writerParam,
      extParam    = extensions(parsedSimdCfg)
    )
  }
}
