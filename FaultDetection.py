from __future__ import print_function
import findspark
findspark.init()
from pyspark.sql import SparkSession, SQLContext, Row
import seaborn as sns
from pyspark.sql.functions import col, mean, monotonically_increasing_id, floor
from pyspark.sql.types import StructType,StructField, StringType
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler,IndexToString, StringIndexer, VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier,RandomForestClassifier, GBTClassifier, LogisticRegression, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import matplotlib.pyplot as plt
if __name__ == "__main__":

    # Create a SparkSession (Note, the config section is only for Windows!)
    spark = SparkSession.builder.master('local[*]').config('spark.executor.memory', '12g').config('spark.driver.memory', '12g').config('spark.driver.maxResultSize', '12g').config("spark.cores.max", "6").appName("FaultDetection").getOrCreate()
    #spark = SparkSession.builder.appName("RecommenderSystem").getOrCreate()
    
    # Load up data as dataframe
    data = spark.read.option("header", "true").option("inferSchema", "true").csv("metadata_train.csv")
    
    featureData = spark.read.option("header", "true").option("inferSchema", "true").parquet("finalfeatures.parquet")
    
    # Setting up data
    
    finalData = data.join(featureData,data.signal_id ==  featureData.signal_id,"inner")
    #finalData.show(1000)
    print((finalData.count(), len(finalData.columns)))
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="target", outputCol="indexedLabel").fit(finalData)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=10).fit(finalData)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = finalData.randomSplit([0.7, 0.3])
    
    ################################# Decision Tree Classifier #######################################################

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test accuracy = %g" % (accuracy))
    print("Test Error = %g " % (1.0 - accuracy))
    
    roc_eval = BinaryClassificationEvaluator(labelCol="indexedLabel").evaluate(predictions)
    print("Test Area Under ROC Curve =  %g" % (roc_eval))
    
    
    #################################################################################################################
    
    ################################# Random Forest Classifier #######################################################
    
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="target", outputCol="indexedLabel").fit(finalData)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(finalData)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = finalData.randomSplit([0.7, 0.3])
    
    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=5)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "target", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test accuracy = %g" % (accuracy))
    print("Test Error = %g" % (1.0 - accuracy))
    
    roc_eval = BinaryClassificationEvaluator(labelCol="indexedLabel").evaluate(predictions)
    print("Test Area Under ROC Curve =  %g" % (roc_eval))
    
    
    #################################################################################################################
    
    ################################# Gradient-boosted tree Classifier #######################################################
    
    # Train a GBT model.
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=5)

    # Chain indexers and GBT in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test accuracy = %g" % (accuracy))
    print("Test Error = %g" % (1.0 - accuracy))
    
    roc_eval = BinaryClassificationEvaluator(labelCol="indexedLabel").evaluate(predictions)
    print("Test Area Under ROC Curve =  %g" % (roc_eval))
    
    
    #################################################################################################################
    
    ################################# Linear SVM Classifier ###############################################
    
    lsvc = LinearSVC(maxIter=10, regParam=0.5, labelCol="target")
    
    # Fit the model
    lsvcModel = lsvc.fit(trainingData)
    # Compute predictions for test data
    predictions = lsvcModel.transform(testData)

    # Show the computed predictions and compare with the original labels
    predictions.select("features", "target", "prediction").show(10)

    # Define the evaluator method with the corresponding metric and compute the classification error on test data
    evaluator = MulticlassClassificationEvaluator(labelCol="target").setMetricName('accuracy')
    accuracy = evaluator.evaluate(predictions) 

    # Show the accuracy
    print("Test accuracy = %g" % (accuracy))
    print("Test Error = %g" % (1.0 - accuracy))
    
    roc_eval = BinaryClassificationEvaluator(labelCol="target").evaluate(predictions)
    print("Test Area Under ROC Curve =  %g" % (roc_eval))
    
    #################################################################################################################
    
    
    ################################# Hybrid Random Forest Classifier  ###############################################
    
    assembler = VectorAssembler(inputCols=["features", "phase"],outputCol="features_hybrid")

    finalData = assembler.transform(finalData)
    
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="target", outputCol="indexedLabel").fit(finalData)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(finalData)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = finalData.randomSplit([0.7, 0.3])
    
     # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=15)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "target", "features_hybrid").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test accuracy = %g" % (accuracy))
    print("Test Error = %g" % (1.0 - accuracy))
    
    roc_eval = BinaryClassificationEvaluator(labelCol="indexedLabel").evaluate(predictions)
    print("Test Area Under ROC Curve =  %g" % (roc_eval))
    
    
    #################################################################################################################
    
    ################################# logistic Regression Classifier  ###############################################
    
    # Train a DecisionTree model.
    lr = LogisticRegression(labelCol="target",regParam = 0.5, maxIter=100)

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
    print("Test accuracy = %g" % (accuracy))
    print("Test Error = %g " % (1.0 - accuracy))
    
    roc_eval = BinaryClassificationEvaluator(labelCol="target").evaluate(predictions)
    print("Test Area Under ROC Curve =  %g" % (roc_eval))
    
    fpr = lrModel.summary.roc.select('FPR').collect()
    tpr = lrModel.summary.roc.select('TPR').collect()

    plt.plot(fpr, tpr)
    plt.show()
    
    #################################################################################################################
    
    
    spark.stop()






