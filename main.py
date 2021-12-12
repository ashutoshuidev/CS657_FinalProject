from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id, col, avg, floor
from pyspark.ml.feature import VectorAssembler

if __name__ == "__main__":
    spark = SparkSession.builder.appName("657Project").getOrCreate()
    
    dataList = []
    for i in range(0,5):
        data = spark.read.option("header", "true").option("inferSchema", "true").csv("train_processed" + str(i) + ".csv")
        data = data.select(data.columns[1:])
        dataList.append(data)
    
    for i in range(0,5):
        dataList[i] = dataList[i].withColumn("index", monotonically_increasing_id())
        indexes = dataList[i].select(col("index"))
        dataList[i] = dataList[i].withColumn("index", dataList[i]['index']-indexes.head()['index']+2000*i)
        if i!=0:
            dataList[0] = dataList[0].union(dataList[i])
    data = dataList[0].orderBy('index').drop(col('index'))
    
    assembler = VectorAssembler(inputCols=[x for x in data.columns],outputCol="features")

    features = assembler.transform(data)
    features = features.select("features")
    
    featureData = features.withColumn("signal_id", monotonically_increasing_id())
    indexes = featureData.select(col("signal_id"))
     
    featureData = featureData.withColumn('signal_id', featureData['signal_id']-indexes.head()['signal_id'])
    featureData.show()
    
    
    