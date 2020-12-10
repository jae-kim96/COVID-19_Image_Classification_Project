# Used this video to download Spark on my personal computer. I had trouble using it in the
# class Docker Image.
# https://medium.com/@GalarnykMichael/install-spark-on-windows-pyspark-4498a5d8d66c

import numpy as np
import pandas as pd

# Just copying some stuff from this website
# https://towardsdatascience.com/transfer-learning-with-pyspark-729d49604d45
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from functools import reduce


# Starting Spark Cluster
sc = SparkContext(appName = 'COVID_Classification')
spark = SparkSession(sc).builder.appName('XRayProcessing').getOrCreate()

# Paths for the preprocessed images
covidPath = '../images/processed/covid19'
normalPath = '../images/processed/normal'
pneumoniaPath = '../images/processed/pneumonia'

# Read all the image data into dataframes
normal = spark.read.format('image').load(normalPath).withColumn('label', lit(0))
covid19 = spark.read.format('image').load(covidPath).withColumn('label', lit(1))
pneumonia = spark.read.format('image').load(pneumoniaPath).withColumn('label', lit(2))

dataframes = [normal, covid19, pneumonia]
# Union of the three dataframes vertically
df = reduce(lambda x, y: x.union(y), dataframes)

# Print Information on the DF
df.printSchema() 
df.select('image').show()# Schema of the Spark DF
grouped = df.groupBy('label').count() # Count of the labels that come up
grouped.show()

# Split the data-frame
train, test = df.randomSplit([0.8, 0.2])

# df.toPandas().to_csv('full_image_dataframe.csv')


####### ONLY WORKS UNTIL HERE ########




from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
# model: InceptionV3
# extracting feature from images
featurizer = DeepImageFeaturizer(inputCol = "image", outputCol = "features", modelName = "InceptionV3")
# used as a multi class classifier
lr = LogisticRegression(maxIter = 5, regParam = 0.03, elasticNetParam = 0.5, labelCol = "label")
# define a pipeline model
sparkdn = Pipeline(stages = [featurizer, lr])
spark_model = sparkdn.fit(train) # start fitting or training


# # EVALUATE THE MODEL
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# evaluate the model with test set
evaluator = MulticlassClassificationEvaluator() 
tx_test = spark_model.transform(test)
print('F1-Score ', evaluator.evaluate(tx_test, {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(tx_test,{evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(tx_test, {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(tx_test, {evaluator.metricName: 'accuracy'}))