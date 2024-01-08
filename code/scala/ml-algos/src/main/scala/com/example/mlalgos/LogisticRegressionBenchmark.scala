package com.example.mlalgos

import org.openjdk.jmh.runner.Runner
import org.openjdk.jmh.runner.options.OptionsBuilder
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.openjdk.jmh.annotations._
import org.apache.spark.sql.DataFrame

import java.lang.management.ManagementFactory
import java.lang.management.MemoryMXBean
import java.util.concurrent.TimeUnit

@State(Scope.Thread)
class LogisticRegressionBenchmark {

  var spark: SparkSession = _
  var trainingData: DataFrame = _
  var lr: LogisticRegression = _

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
      .load("../ml-algos/src/main/scala/com/example/mlalgos/heart.csv")


   val y = df.withColumnRenamed("target", "label")
   val X = df.drop("target")


   // Vectorize the features
   val assembler = new VectorAssembler()
     .setInputCols(X.columns)
     .setOutputCol("features")

   val output = assembler.transform(y)

   // Split the data into training and test datasets
   val Array(trainingData, _) = output.randomSplit(Array(0.7, 0.3))
    this.trainingData = trainingData

 // Initialize the LogisticRegression model
    lr = new LogisticRegression()


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

object Main2 {
  def main(args: Array[String]): Unit = {
    // Use JMH to run the LogisticRegressionBenchMark
     new Runner(new OptionsBuilder().include(classOf[LogisticRegressionBenchmark].getSimpleName).build).run()
  }
}