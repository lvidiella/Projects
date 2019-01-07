#Laura Vidiella del Blanco
#Final Project MET CS 777 - Big Data Analytics

from pyspark.sql import SQLContext

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: app_name <file>", file=sys.stderr)
        exit(-1)

     #Import red wine
     df = spark.read.csv('winequality-red.csv', header = True, inferSchema = True, sep=";")
     df.printSchema()

     #Import white wine
     df2 = spark.read.csv('winequality-white.csv', header = True, inferSchema = True, sep=";")


     df3 = df.union(df2)
     df3.show(5)

     df = df3

     import six
     for i in df.columns:
          if not( isinstance(df.select(i).take(1)[0][0], six.string_types)):
              print( "Correlation to quality for ", i, df.stat.corr('quality',i))

     #('Correlation to quality for ', 'fixed acidity', -0.07674320790962014)
     #('Correlation to quality for ', 'volatile acidity', -0.26569947761146706)
     #('Correlation to quality for ', 'citric acid', 0.085531717183678)
     #('Correlation to quality for ', 'residual sugar', -0.0369804845857698)
     #('Correlation to quality for ', 'chlorides', -0.2006655004351014)
     #('Correlation to quality for ', 'free sulfur dioxide', 0.055463058616632414)
     #('Correlation to quality for ', 'total sulfur dioxide', -0.04138545385560937)
     #('Correlation to quality for ', 'density', -0.3058579060694188)
     #('Correlation to quality for ', 'pH', 0.019505703714435507)
     #('Correlation to quality for ', 'sulphates', 0.038485445876513875)
     #('Correlation to quality for ', 'alcohol', 0.44431852000765354)
     #('Correlation to quality for ', 'quality', 1.0)



     from pyspark.ml.feature import VectorAssembler
     vectorAssembler = VectorAssembler(inputCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'], outputCol = 'features')
     v_df = vectorAssembler.transform(df)
     v_df = v_df.select(['features', 'quality'])
     v_df.show(10)

     #+--------------------+-------+
     #|            features|quality|
     #+--------------------+-------+
     #|[7.4,0.7,0.0,1.9,...|      5|
     #|[7.8,0.88,0.0,2.6...|      5|
     #|[7.8,0.76,0.04,2....|      5|
     #|[11.2,0.28,0.56,1...|      6|
     #|[7.4,0.7,0.0,1.9,...|      5|
     #|[7.4,0.66,0.0,1.8...|      5|
     #|[7.9,0.6,0.06,1.6...|      5|
     #|[7.3,0.65,0.0,1.2...|      7|
     #|[7.8,0.58,0.02,2....|      7|
     #|[7.5,0.5,0.36,6.1...|      5|
     #+--------------------+-------+
     #only showing top 10 rows


     splits = v_df.randomSplit([0.7, 0.3])
     train_df = splits[0]
     test_df = splits[1]

     from pyspark.ml.regression import LinearRegression
     lr = LinearRegression(featuresCol = 'features', labelCol='quality', maxIter=10, regParam=0.3, elasticNetParam=0.8)
     lr_model = lr.fit(train_df)
     print("Coefficients: " + str(lr_model.coefficients))
     #Coefficients: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.11809926946]

     print("Intercept: " + str(lr_model.intercept))
     #Intercept: 4.571174663

     trainingSummary = lr_model.summary

     print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
     #RMSE: 0.830703

     print("r2: %f" % trainingSummary.r2)
     #r2: 0.115904


     train_df.describe().show()
     #+-------+------------------+
     #|summary|           quality|
     #+-------+------------------+
     #|  count|              4519|
     #|   mean| 5.810798849302943|
     #| stddev|0.8835761043715878|
     #|    min|                 3|
     #|    max|                 9|
     #+-------+------------------+

     lr_predictions = lr_model.transform(test_df)
     lr_predictions.select("prediction","quality","features").show(5)

     #+-----------------+-------+--------------------+
     #|       prediction|quality|            features|
     #+-----------------+-------+--------------------+
     #|  6.1773247276489|      6|[5.0,0.4,0.5,4.3,...|
     #|  6.1773247276489|      7|[5.1,0.42,0.0,1.8...|
     #|6.094655239027107|      7|[5.1,0.51,0.18,2....|
     #|6.011985750405315|      7|[5.2,0.48,0.04,1....|
     #|6.047415531243226|      6|[5.2,0.645,0.0,2....|
     #+-----------------+-------+--------------------+
     #only showing top 5 rows


     from pyspark.ml.evaluation import RegressionEvaluator
     lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                      labelCol="quality",metricName="r2")
     print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
     #R Squared (R2) on test data = 0.119998

     print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
     Root Mean Squared Error (RMSE) on test data = 0.796393

     print("numIterations: %d" % trainingSummary.totalIterations)
     #numIterations: 11

     print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
     #objectiveHistory: [0.49999999999999645, 0.4956103053191332, 0.48691695060326384, 0.48643028608959576, 0.48636104546224795, 
     #0.4863522163711371, 0.4863510905458467, 0.4863509469883082, 0.4863509286828313, 0.48635092634863936, 0.4863509260510017]


     from pyspark.ml.regression import DecisionTreeRegressor
     dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'quality')
     dt_model = dt.fit(train_df)
     dt_predictions = dt_model.transform(test_df)
     dt_evaluator = RegressionEvaluator(
         labelCol="quality", predictionCol="prediction", metricName="rmse")
     rmse = dt_evaluator.evaluate(dt_predictions)
     print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
     #Root Mean Squared Error (RMSE) on test data = 0.715905

     print(dt_model.featureImportances)
     #(11,[1,2,3,4,5,6,7,9,10],[0.186443160835,0.0123855335534,0.0431085308425,0.00761099305703,
     #0.0862635095428,0.0147151635426,0.0236281098531,0.0556970400348,0.570147958739])


     #Gradient-boosted tree regression
     from pyspark.ml.regression import GBTRegressor
     gbt = GBTRegressor(featuresCol = 'features', labelCol = 'quality', maxIter=10)
     gbt_model = gbt.fit(train_df)
     gbt_predictions = gbt_model.transform(test_df)
     gbt_predictions.select('prediction', 'quality', 'features').show(5)

     gbt_evaluator = RegressionEvaluator(
         labelCol="quality", predictionCol="prediction", metricName="rmse")
     rmse = gbt_evaluator.evaluate(gbt_predictions)
     print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
     #Root Mean Squared Error (RMSE) on test data = 0.690916

     #In the combined dataset, it looks like Gradient-boosted tree regression is the best performing one as well
     sc.stop()
