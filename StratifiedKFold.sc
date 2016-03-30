import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD


/*
  Return:
  Array[(train set, test set)], length = nFold
 */

def StratifiedKFold(d: RDD[LabeledPoint], nFold: Int): Array[(RDD[LabeledPoint], RDD[LabeledPoint])] = {
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
