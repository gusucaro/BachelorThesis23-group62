package com.example.mlalgos

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import java.lang.Runtime
import java.lang.management.ManagementFactory
import java.lang.management.ThreadMXBean
import java.lang.management.MemoryMXBean
import java.lang.management.MemoryUsage
import com.sun.management.OperatingSystemMXBean

object Main {

 def main(args: Array[String]): Unit = {

   val spark = SparkSession.builder()
     .appName("ml-algos")
     .config("spark.master", "local[1]")
//     .config("spark.eventLog.enabled", "true")
//    .config("spark.eventLog.dir", "../ml-algos/src/main/scala/")
     .enableHiveSupport()
     .getOrCreate()

   val df = spark.read
     .format("csv")
     .option("header", "true")
     .option("inferSchema", "true")
     .load("../ml-algos/src/main/scala/com/example/mlalgos/Real_estate.csv")



// drop the last column
val X = df.drop("Y house price of unit area")


//renaming our target/label column 
val y = X.withColumnRenamed("X4 number of convenience stores" , "label")



val assembler = new VectorAssembler()
 .setInputCols(Array("X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station", "X5 latitude", "X6 longitude"))
 .setOutputCol("features")

val output = assembler.transform(y)

val Array(trainingData, testData) = output.randomSplit(Array( 0.7, 0.3)) 

val lr = new LinearRegression()


val runtime = Runtime.getRuntime()
//System.gc()
//Thread.sleep(100)
val startMemoryUsage =runtime.totalMemory() - runtime.freeMemory()
val lrModel = lr.fit(trainingData)

///System.gc()
///Thread.sleep(100)


val endMemoryUsage = runtime.totalMemory() - runtime.freeMemory()
val memoryUsed = endMemoryUsage - startMemoryUsage

 // Predict
 val predictions = lrModel.transform(testData)


val evaluatorRMSE = new RegressionEvaluator()
 .setMetricName("rmse")
 .setLabelCol("label")
 .setPredictionCol("prediction")

val rmse = evaluatorRMSE.evaluate(predictions)

val evaluatorMAE = new RegressionEvaluator()
 .setMetricName("mae")
 .setLabelCol("label")
 .setPredictionCol("prediction")

val mae = evaluatorMAE.evaluate(predictions)

val evaluatorMSE = new RegressionEvaluator()
 .setMetricName("mse")
 .setLabelCol("label")
 .setPredictionCol("prediction")

val mse = evaluatorMSE.evaluate(predictions)

println(s"Memory Used (MB) ${memoryUsed/( 1024 * 1024)}")
println("Mean Absolute Error (MAE) on test data = " + mae)
println("Mean Squared Error (MSE) on test data = " + mse)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)


   spark.stop()
 }
}
