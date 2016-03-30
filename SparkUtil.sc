import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression.{LabeledPoint, RegressionModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD


/*
  Return:
  Array[(train set, test set)], length = nFold
 */

def stratifiedKFold(d: RDD[LabeledPoint], nFold: Int): Array[(RDD[LabeledPoint], RDD[LabeledPoint])] = {
  val keys = d.map(_.label).distinct.collect
  keys.map{ key =>
    d.filter(k => k.label == key)
  } map { dl =>
    MLUtils.kFold(dl, nFold, 0)
  } reduce { (a1, a2) =>
    a1 zip a2 map { case (a1t, a2t) =>
      (a1t._1 union a2t._1, a1t._2 union a2t._2)
    }
  }
}

def cvReg(arr: Array[(RDD[LabeledPoint], RDD[LabeledPoint])],
          regressor: { def run(input: RDD[LabeledPoint]): RegressionModel}): Array[Double] = {
  val res = arr.map{ case (train, test) =>
    val model = regressor.run(train)
    val pao = test.map{case LabeledPoint(label, features) =>
      (model.predict(features), label)
    }
    val metric = new RegressionMetrics(pao)
    metric.r2
  }
  res
}


