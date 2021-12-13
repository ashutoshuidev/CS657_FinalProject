from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id, col, avg, floor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier,RandomForestClassifier, GBTClassifier, NaiveBayes, LinearSVC, LogisticRegression
from pyspark.ml.feature import IndexToString,StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("657Project").getOrCreate()
featureData = spark.read.option("header", "true").option("inferSchema", "true").parquet("features.parquet")
featureData = featureData.filter(featureData.signal_id < 8712)
featureData = featureData.orderBy('signal_id')
data = spark.read.option("header", "true").option("inferSchema", "true").csv("metadata_train.csv")
print(featureData.count())

finalData = data.join(featureData,data.signal_id ==  featureData.signal_id,"inner")
finalData.show(100)
labelIndexer = StringIndexer(inputCol="target", outputCol="indexedLabel").fit(finalData)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(finalData)
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = finalData.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
lr = LogisticRegression(labelCol="target",regParam = 0.01, maxIter=100)

# Fit the model
lrModel = lr.fit(trainingData)

# Compute predictions for test data
predictions = lrModel.transform(testData)

# Select example rows to display.
predictions.select("features", "target", "prediction").show(10)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(accuracy)
print("Test Error = %g " % (1.0 - accuracy))
fpr = lrModel.summary.roc.select('FPR').collect()
tpr = lrModel.summary.roc.select('TPR').collect()

plt.plot(fpr, tpr)
plt.show()

auc = roc_auc_score(y, probs)
print('AUC: %.3f' % auc)
