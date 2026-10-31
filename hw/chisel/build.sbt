// See README.md for license details.

ThisBuild / scalaVersion := "2.13.14"
ThisBuild / version      := "0.1.0"
ThisBuild / organization := "be.kuleuven.esat.micas"

val chiselVersion   = "6.4.0"
val playJSONVersion = "3.0.4"

lazy val fpUnits  = ProjectRef(file("../chisel_acc/subprojects/chisel-float"), "chiselFloat")
lazy val fpNative = ProjectRef(file("../chisel_acc/subprojects/fp-native"), "fpNative")

lazy val root = (project in file("."))
  .settings(
    name := "snax-streamer",
    libraryDependencies ++= Seq(
      "org.scala-lang"     % "scala-compiler" % "2.13.14",
      "org.chipsalliance" %% "chisel"         % chiselVersion,
      "edu.berkeley.cs"   %% "chiseltest"     % "6.0.0" % "test",
      "org.playframework" %% "play-json"      % playJSONVersion
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-Ymacro-annotations",
      "-Wunused" // Enable unused import fixes

    ),
    addCompilerPlugin(
      "org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full
    )
  )
  .dependsOn(fpUnits, fpUnits % "compile->test", fpNative)

// -oD prints each test's duration in the sbt log.
Test / testOptions += Tests.Argument(TestFrameworks.ScalaTest, "-oD")

// SNAX_TEST_SHARD picks the suites one CI job runs. A named shard runs the suites listed for it; "rest" runs every
// other suite, so a new suite lands there by default. Unset or empty runs all suites.
val testShards = Map(
  "stream-reduce" -> Set("snax.DataPathExtension.StreamReduceTester"),
  "xdma-app"      -> Set("snax.DataPathExtension.XdmaAppTester"),
  "stream-map"    -> Set("snax.DataPathExtension.StreamMapTester", "snax.DataPathExtension.StreamElementwiseTester")
)

Test / testOptions ++= {
  val shardNames = testShards.keys.toSeq.sorted :+ "rest"
  sys.env.getOrElse("SNAX_TEST_SHARD", "") match {
    case ""                                  => Nil
    case "rest"                              =>
      val named = testShards.values.flatten.toSet
      Seq(Tests.Filter(suite => !named(suite)))
    case shard if testShards.contains(shard) => Seq(Tests.Filter(testShards(shard)))
    case shard                               =>
      sys.error(s"Unknown SNAX_TEST_SHARD '$shard'. Use one of: ${shardNames.mkString(", ")}, or leave it unset.")
  }
}
