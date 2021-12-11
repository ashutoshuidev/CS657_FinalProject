from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id, col, avg, floor

if __name__ == "__main__":
    spark = SparkSession.builder.appName("657Project").getOrCreate()

    data = spark.read.option("header", "true").option("inferSchema", "true").parquet("train.parquet")
    # data = data.select(data.columns[:15])
    
    data = data.withColumn("index", monotonically_increasing_id())
    data.show(5)
    
    indexes = data.select(col("index"))
    
    if (indexes.tail(1)[0]['index']-indexes.head()['index'] != 799999):
        print("ID assign error")
        spark.stop()
        exit()
       
    n_aggregate_columns = 50000
    
    data = data.withColumn('index', data['index']-indexes.head()['index'])
    data.show(5)
    
    data = data.withColumn('index', floor(data['index']/n_aggregate_columns)).groupBy('index').avg().orderBy('index')
    data.show(16)
    
    
    spark.stop()
