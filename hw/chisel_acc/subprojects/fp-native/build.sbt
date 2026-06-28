// See README.md for license details.
// Native-Chisel floating-point units (FpAdd/FpMul/FpFma) — 1:1 ports of the fpnew SV datapath,
// replacing the SystemVerilog BlackBoxes on the xDMA path. Depends on chisel-float for FpType.

ThisBuild / scalaVersion := "2.13.14"
ThisBuild / version      := "0.1.0"
ThisBuild / organization := "MICAS (KU Leuven)"

Test / parallelExecution := true
Test / fork              := true

val chiselVersion = "6.4.0"

lazy val chiselFloat = ProjectRef(file("../chisel-float"), "chiselFloat")

lazy val fpNative = (project in file("."))
  .settings(
    name := "fp-native",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel"     % chiselVersion,
      "org.scalatest"     %% "scalatest"  % "3.2.19" % "test",
      "edu.berkeley.cs"   %% "chiseltest" % "6.0.0"  % "test"
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-Ymacro-annotations",
      "-Wunused"
    ),
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full)
  )
  .dependsOn(chiselFloat)
