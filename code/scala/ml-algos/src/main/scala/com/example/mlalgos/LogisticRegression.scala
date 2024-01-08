import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions._
import java.lang.Runtime
import java.lang.management.ManagementFactory
import java.lang.management.ThreadMXBean
import java.lang.management.MemoryMXBean
import java.lang.management.MemoryUsage
import java.lang.management.OperatingSystemMXBean

object HeartDiseaseClassifier {

 def main(args: Array[String]): Unit = {

   // Initiate Spark Session
   val spark = SparkSession.builder()
     .appName("Heart Disease Classifier")
     .config("spark.master", "local")
     .enableHiveSupport()
     .getOrCreate()

   // Read the data
   val df = spark.read
     .option("header", "true")
     .option("inferSchema", "true")
     .csv("../ml-algos/src/main/scala/com/example/mlalgos/heart.csv")

   // Separate features and target variable

   val y = df.withColumnRenamed("target", "label")
   val X = df.drop("target")


   // Vectorize the features
   val assembler = new VectorAssembler()
     .setInputCols(X.columns)
     .setOutputCol("features")

   val output = assembler.transform(y)

   // Split the data into training and test datasets
   val Array(trainingData, testData) = output.randomSplit(Array(0.7, 0.3))

 // Initialize the LogisticRegression model
   val lr = new LogisticRegression()

   val runtime = Runtime.getRuntime()

//  System.gc()
//  Thread.sleep(100)
val startMemoryUsage = runtime.totalMemory - runtime.freeMemory()
val lrModel = lr.fit(trainingData)
  
 /// System.gc()
 /// Thread.sleep(100)
   val endMemoryUsage = runtime.totalMemory - runtime.freeMemory()
   val memoryUsed = endMemoryUsage - startMemoryUsage


   // Make predictions on the test data
   val predictions = lrModel.transform(testData)

   // Evaluate the model
   val evaluator = new MulticlassClassificationEvaluator()
     .setLabelCol("label")
     .setPredictionCol("prediction")
     .setMetricName("accuracy")

   val accuracy = evaluator.evaluate(predictions)

println(s"Memory Used (MB) ${memoryUsed/( 1024 * 1024)}")
println("Accuracy: " + accuracy)

   // Stop the Spark Session
   spark.stop()
 }
}