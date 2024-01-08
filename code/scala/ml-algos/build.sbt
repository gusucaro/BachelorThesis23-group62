val scala3Version = "2.13.6"

enablePlugins(JmhPlugin)
lazy val root = project
  .in(file("."))
  .settings(
    name := "ml-algos",
    version := "0.1.0-SNAPSHOT",

    scalaVersion := scala3Version,


    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "0.7.29" % Test,
      "org.apache.spark" %% "spark-core" % "3.5.0",
      "org.apache.spark" %% "spark-sql" % "3.5.0",
      "org.apache.spark" %% "spark-mllib" % "3.5.0",
      "org.apache.spark" %% "spark-hive" % "3.5.0",
      "com.github.haifengl" %% "smile-scala" % "3.0.1",
        "org.scalanlp" %% "breeze" % "1.2",
       "com.github.haifengl" % "smile-core" % "2.6.0",
       "com.github.haifengl" % "smile-nlp" % "2.6.0"
    )



  )
