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
    ),
    // Run each test suite in its OWN forked JVM.
    //
    // chiseltest spawns a thread per forked test thread and does not reliably reap them, so running the whole
    // suite in a single JVM accumulates threads across suites until the process hits the host's `ulimit -u` and
    // dies mid-run with "OutOfMemoryError: unable to create native thread" -- after ~30 suites on a 5166-thread
    // limit. That aborts `sbt test` without a per-suite result, which reads like a hang rather than a failure.
    // One JVM per suite bounds the live thread count to a single suite's worth; the cost is one JVM start per
    // suite. Serial, because several suites drive Verilator builds that are already parallel internally.
    Test / fork               := true,
    Test / testForkedParallel := false,
    Test / javaOptions ++= Seq("-Xmx8G"),
    Test / testGrouping := (Test / definedTests).value.map { suite =>
      Tests.Group(name = suite.name, tests = Seq(suite), runPolicy = Tests.SubProcess((Test / forkOptions).value))
    }
  )
  .dependsOn(fpUnits, fpUnits % "compile->test", fpNative)
