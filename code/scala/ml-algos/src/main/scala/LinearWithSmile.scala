
import breeze.numerics._
import breeze.stats._
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.data.vector.DoubleVector
import smile.io.Read
import smile.math.MathEx
import smile.regression._
import java.util.concurrent.TimeUnit
import java.lang.Runtime

object linearWithSmile {

 def main(args: Array[String]): Unit= {

 val data = Read.csv("../ml-algos/src/main/scala/Real_estate.csv")

 // Print column names
 data.names().foreach(println)

 val formula = Formula.lhs("V5") // Replace with the correct column name
 val x = formula.x(data)
 val y = formula.y(data)

 val totalSize = data.size()
 val trainSize = math.round(0.7 * totalSize).toInt
 val testSize = totalSize - trainSize
 val split = trainSize

 val xTrain = Array.tabulate(trainSize)(i => x.get(i))
 val yTrain = Array.tabulate(trainSize)(i => y.get(i).toString.toDouble)

 val xTest = Array.tabulate(testSize)(i => x.get(i + split))
 val yTest = Array.tabulate(testSize)(i => y.get(i + split).toString.toDouble)

val runtime = Runtime.getRuntime

val startMemUsage = runtime.totalMemory() - runtime.freeMemory()
 val start = System.nanoTime()
 val lmModel = lm(formula, data) // Use lm instead of ols
 val end = System.nanoTime()

 val endMemUsage = runtime.totalMemory()  -  runtime.freeMemory()

println(s"Start memory usage: $startMemUsage bytes")
println(s"End memory usage: $endMemUsage bytes")


 val duration = TimeUnit.MILLISECONDS.convert((end - start), TimeUnit.NANOSECONDS)

 println(s"Training time: $duration milliseconds")

 val predictions = lmModel.predict(xTest)

 

 }
}