// See README.md for license details.

ThisBuild / scalaVersion               := "2.13.14"
ThisBuild / version                    := "0.1.0"
ThisBuild / organization               := "be.kuleuven.esat.micas"
ThisBuild / scalafixScalaBinaryVersion := "2.13"

val chiselVersion = "6.4.0"

lazy val chisel       = ProjectRef(file("hw/chisel"), "chisel")
lazy val chisel_acc   = ProjectRef(file("hw/chisel_acc"), "chisel_acc")
lazy val chisel_float = ProjectRef(file("hw/chisel_acc/subprojects/chisel-float"), "chisel_float")

lazy val root = (project in file("."))
  .settings(
    name := "snax-cluster",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel"     % chiselVersion,
      "edu.berkeley.cs"   %% "chiseltest" % "6.0.0" % "test"
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
  .dependsOn(chisel_float, chisel, chisel_acc)
