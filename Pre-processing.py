from __future__ import print_function
import findspark
findspark.init()
from pyspark.sql import SparkSession, SQLContext, Row
import seaborn as sns
from pyspark.sql.functions import col, mean, monotonically_increasing_id, floor,row_number
from pyspark.sql.types import StructType,StructField, StringType
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.window import Window

if __name__ == "__main__":

    # Create a SparkSession (Note, the config section is only for Windows!)
    spark = SparkSession.builder.master('local[*]').config('spark.executor.memory', '12g').config('spark.driver.memory', '12g').config('spark.driver.maxResultSize', '12g').config("spark.cores.max", "6").appName("FaultDetection").getOrCreate()
    
    # Load up data as dataframe
    data = spark.read.option("header", "true").option("inferSchema", "true").csv("metadata_train.csv")
    
    ################################# Visualization of train data ###################################################
    
    notFaulty = data.select('signal_id').where(data.target == 0).count()
    faulty = data.select('signal_id').where(data.target == 1).count()
    
    # 8187 -  signals are not faulty, while 525 are faulty
    print(notFaulty, faulty)
    
    # phase wise distribution of faulty vs not faulty signals
    notFaultyPhase0 = data.select('signal_id').where((data.target == 0) & (data.phase == 0)).count()
    faultyPhase0 = data.select('signal_id').where((data.target == 1) & (data.phase == 0)).count()
    print(notFaultyPhase0, faultyPhase0)
    
    notFaultyPhase1 = data.select('signal_id').where((data.target == 0) & (data.phase == 1)).count()
    faultyPhase1 = data.select('signal_id').where((data.target == 1) & (data.phase == 1)).count()
    print(notFaultyPhase1, faultyPhase1)
    
    notFaultyPhase2 = data.select('signal_id').where((data.target == 0) & (data.phase == 2)).count()
    faultyPhase2 = data.select('signal_id').where((data.target == 1) & (data.phase == 2)).count()
    print(notFaultyPhase2, faultyPhase2)
    
    
    meta_data =  data.toPandas()
    sns.set(style="darkgrid")
    sns.countplot(x = 'target',hue = 'phase',data = meta_data)
    
    #################################################################################################################
    
    
    ################################# Feature Extraction ############################################################

    # Create an empty dataframe with a schema
    schema = StructType([StructField('features', VectorUDT(), True)])
    finalfeatures = spark.createDataFrame([], schema)

    for i in range(0,18):
        signalData = spark.read.option("header", "true").option("inferSchema", "true").parquet("train.parquet")
        signalData = signalData.select(signalData.columns[500*i:500*(i+1)])
        print(500*i,500*(i+1) )
        signalData = signalData.withColumn("index", monotonically_increasing_id())
        
        indexes = signalData.select(col("index"))
        print(indexes)
        minindex = indexes.head()['index']
        if (indexes.tail(1)[0]['index']-minindex != 799999):
            print("ID assign error")
            spark.stop()
            exit()

        n_aggregate_columns = 80000
        signalData = signalData.withColumn('index', signalData['index']-minindex)
        signalData = signalData.withColumn('index', floor(signalData['index']/n_aggregate_columns)).groupBy('index').avg().orderBy('index')
        print((signalData.count(), len(signalData.columns)))
        signalData =  signalData.drop(col("avg(index)"))
        signalDataWithFeatures = spark.createDataFrame(signalData.toPandas().set_index("index").transpose())
        assembler = VectorAssembler(inputCols=[x for x in signalDataWithFeatures.columns],outputCol="features")

        features = assembler.transform(signalDataWithFeatures)
        features = features.select("features")

        features.show()
        print((features.count(), len(features.columns)))
        finalfeatures = finalfeatures.union(features)
    
    finalfeatures = finalfeatures.withColumn('signal_id', row_number().over(Window.orderBy(monotonically_increasing_id()))-1)
    #finalfeatures.show(1000)
    print((finalfeatures.count(), len(finalfeatures.columns)))
    finalfeatures.write.parquet("finalfeatures.parquet")
    #################################################################################################################
    
    spark.stop()