package com.example.mlalgos

import org.openjdk.jmh.runner.Runner
import org.openjdk.jmh.runner.options.OptionsBuilder
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.openjdk.jmh.annotations._
import org.apache.spark.sql.DataFrame
import java.lang.Runtime
import java.lang.management.ManagementFactory
import java.lang.management.MemoryMXBean
import java.util.concurrent.TimeUnit
import com.sun.management.OperatingSystemMXBean

@State(Scope.Thread)
class LinearRegressionBenchmark {

  var spark: SparkSession = _
  var trainingData: DataFrame = _
  var lr: LinearRegression = _

  

  @Setup(Level.Trial)
  def setUp(): Unit = {
    spark = SparkSession.builder()
     .appName("ml-algos")
     .config("spark.master", "local")
     //.enableHiveSupport()
     .getOrCreate()

     spark.sparkContext.setLogLevel("WARN")

    val df = spark.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("../ml-algos/src/main/scala/com/example/mlalgos/Real_estate.csv")

    val X = df.drop("Y house price of unit area")
    val y = X.withColumnRenamed("X4 number of convenience stores" , "label")

    val assembler = new VectorAssembler()
      .setInputCols(Array("X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station", "X5 latitude", "X6 longitude"))
      .setOutputCol("features")

    val output = assembler.transform(y)
    val Array(trainingData, _) = output.randomSplit(Array(0.7, 0.3))
    this.trainingData = trainingData

     lr = new LinearRegression()
     }

  @Benchmark
  @OutputTimeUnit(TimeUnit.MILLISECONDS)
  def fitModel(): Unit = {
    val lrModel = lr.fit(trainingData)
      }

  @TearDown(Level.Trial)
  def tearDown(): Unit = {
    spark.stop()
  }
}

object Main1 {
  def main(args: Array[String]): Unit = {
    // Use JMH to run the LinearRegressionBenchmark
     new Runner(new OptionsBuilder().include(classOf[LinearRegressionBenchmark].getSimpleName).build).run()
  }
}