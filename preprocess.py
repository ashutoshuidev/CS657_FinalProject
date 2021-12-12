from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id, col, avg, floor

if __name__ == "__main__":
    spark = SparkSession.builder.appName("657Project").getOrCreate()

    for i in range(0,5):
        data = spark.read.option("header", "true").option("inferSchema", "true").parquet("train.parquet")
        data = data.select(data.columns[2000*i:2000*(i+1)])
        
        data = data.withColumn("index", monotonically_increasing_id())
        data.show(5)
        
        indexes = data.select(col("index"))
        minindex = indexes.head()['index']
        
        if (indexes.tail(1)[0]['index']-minindex != 799999):
            print("ID assign error")
            spark.stop()
            exit()
           
        n_aggregate_columns = 80000
        
        data = data.withColumn('index', data['index']-minindex)
        data.show(5)
        
        data = data.withColumn('index', floor(data['index']/n_aggregate_columns)).groupBy('index').avg().orderBy('index')
        data.show(10)
        data = data.drop(col('avg(index)'))
        
        data.toPandas().set_index("index").transpose().to_csv("train_processed" + str(i) + ".csv")
    spark.stop()
