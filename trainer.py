import pyspark
import numpy as np
from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT
from transforms import Transforms

class SparkConfig:
    appName = "CIFAR"
    receivers = 4
    host = "local"
    stream_host = "192.168.1.12"
    port = 6100
    batch_interval = 2

from dataloader import DataLoader

class Trainer:
    def __init__(self, 
                 model, 
                 split:str, 
                 spark_config:SparkConfig, 
                 transforms: Transforms) -> None:

        self.model = model
        self.split = split
        self.sparkConf = spark_config
        self.transforms = transforms
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]",f"{self.sparkConf.appName}")
        self.ssc = StreamingContext(self.sc,self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc,self.ssc,self.sqlContext,self.sparkConf,self.transforms)
        # Accumulators for prediction statistics
        self.test_accuracy = 0
        self.test_loss = 0
        self.test_precision = 0
        self.test_recall = 0
        self.test_f1 = 0
        self.batch_count = 0
        self.total_samples = 0
        self.cm = np.zeros((10, 10), dtype=int)

    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__train__)

        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        if not rdd.isEmpty():
            schema = StructType([
                StructField("image", VectorUDT(), True),
                StructField("label", IntegerType(), True)])

            df = self.sqlContext.createDataFrame(rdd, schema)
            
            predictions, accuracy, precision, recall, f1 = self.model.train(df)

            print("="*10)
            print(f"Predictions = {predictions}")
            print(f"Accuracy = {accuracy}")
            print(f"Precision = {precision}")
            print(f"Recall = {recall}")
            print(f"F1 Score = {f1}")
            print("="*10)
        
        print("Total Batch Size of RDD Received :",rdd.count())
        print("+"*20)

    def predict(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__predict__)

        self.ssc.start()
        try:
            self.ssc.awaitTerminationOrTimeout(30)  # Dừng sau 30s, bạn có thể điều chỉnh
        finally:
            self.ssc.stop(stopSparkContext=False, stopGraceFully=True)
            self.__write_predict_results()

def __predict__(self, rdd: pyspark.RDD) -> DataFrame:     
    if not rdd.isEmpty():
        schema = StructType([
            StructField(name="image", dataType=VectorUDT(), nullable=True),
            StructField(name="label", dataType=IntegerType(), nullable=True)
        ])
        
        df = self.sqlContext.createDataFrame(rdd, schema)

        # Giả sử self.model.predict(df) trả về các giá trị như sau:
        accuracy, loss, precision, recall, f1, cm = self.model.predict(df)

        self.batch_count += 1
        self.test_accuracy += accuracy
        self.test_loss += loss
        self.test_precision += precision
        self.test_recall += recall
        self.test_f1 += f1
        self.cm += cm

        print(f"Test Accuracy: {accuracy}")
        print(f"Test Loss: {loss}")
        print(f"Test Precision: {precision}")
        print(f"Test Recall: {recall}")
        print(f"Test F1 Score: {f1}")
        print(f"Confusion Matrix:\n{cm}")
    
    print(f"Batch: {self.batch_count}")
    print("Total Batch Size of RDD Received:", rdd.count())
    print("-" * 40)
