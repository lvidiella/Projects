#Laura Vidiella del Blanco
#Final Project MET CS 777 - Big Data Analytics

from pyspark.sql import SQLContext

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: app_name <file>", file=sys.stderr)
        exit(-1)

     #White wine linear regression

     df = spark.read.csv('winequality-white.csv', header = True, inferSchema = True, sep=";")

     import six
     for i in df.columns:
          if not( isinstance(df.select(i).take(1)[0][0], six.string_types)):
              print( "Correlation to quality for ", i, df.stat.corr('quality',i))

     #('Correlation to quality for ', 'fixed acidity', -0.11366283071301823)
     #('Correlation to quality for ', 'volatile acidity', -0.19472296892113428)
     #('Correlation to quality for ', 'citric acid', -0.009209090883975582)
     #('Correlation to quality for ', 'residual sugar', -0.09757682889469343)
     #('Correlation to quality for ', 'chlorides', -0.20993441094675971)
     #('Correlation to quality for ', 'free sulfur dioxide', 0.008158067123435954)
     #('Correlation to quality for ', 'total sulfur dioxide', -0.1747372175970627)
     #('Correlation to quality for ', 'density', -0.3071233127347343)
     #('Correlation to quality for ', 'pH', 0.09942724573666432)
     #('Correlation to quality for ', 'sulphates', 0.053677877132792026)
     #('Correlation to quality for ', 'alcohol', 0.4355747154613734)
     #('Correlation to quality for ', 'quality', 1.0)

     from pyspark.ml.feature import VectorAssembler
     vectorAssembler = VectorAssembler(inputCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'], outputCol = 'features')
     v_df = vectorAssembler.transform(df)
     v_df = v_df.select(['features', 'quality'])
     v_df.show(10)
     #+--------------------+-------+
     #|            features|quality|
     #+--------------------+-------+
     #|[7.0,0.27,0.36,20...|      6|
     #|[6.3,0.3,0.34,1.6...|      6|
     #|[8.1,0.28,0.4,6.9...|      6|
     #|[7.2,0.23,0.32,8....|      6|
     #|[7.2,0.23,0.32,8....|      6|
     #|[8.1,0.28,0.4,6.9...|      6|
     #|[6.2,0.32,0.16,7....|      6|
     #|[7.0,0.27,0.36,20...|      6|
     #|[6.3,0.3,0.34,1.6...|      6|
     #|[8.1,0.22,0.43,1....|      6|
     #+--------------------+-------+
     #only showing top 10 rows

     #Let's get the splits

     splits = v_df.randomSplit([0.7, 0.3])
     train_df = splits[0]
     test_df = splits[1]

     from pyspark.ml.regression import LinearRegression
     lr = LinearRegression(featuresCol = 'features', labelCol='quality', maxIter=10, regParam=0.3, elasticNetParam=0.8)
     lr_model = lr.fit(train_df)
     print("Coefficients: " + str(lr_model.coefficients))
     #Coefficients: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.108463373249]

     print("Intercept: " + str(lr_model.intercept))
     #Intercept: 4.74329157251

     trainingSummary = lr_model.summary

     print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
     #RMSE: 0.826781

     print("numIterations: %d" % trainingSummary.totalIterations)
     #numIterations: 11

     print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
     #objectiveHistory: [0.49999999999999645, 0.49803321109465737, 0.488301984685466, 0.48787776578652203, 0.4877994889265423, 0.48779778730481504, 
     #0.48779745885889453, 0.48779739546247036, 0.48779738322573046, 0.48779738086381125, 0.48779738040791243]

     #+--------------------+
     #|           residuals|
     #+--------------------+
     #|-0.08823740079789832|
     #|  1.8683772499023403|
     #|-0.16416176207248068|
     #| -0.6109985585005209|
     #| 0.03107230977644626|
     #|  -0.882156991624031|
     #|  0.7382212020030554|
     #| -1.2183934486971832|
     #|  0.9768406231517437|
     #|  1.0310723097764463|
     #| 0.13953568302584962|
     #|-0.03400571417319...|
     #| 0.18292103232561097|
     #| 0.23715271895031353|
     #| -0.8785415458490542|
     #| -0.8785415458490542|
     #| -0.8387716423242697|
     #|0.009379635126565589|
     #|   1.085303996401148|
     #|-0.05027522016060715|
     #+--------------------+
     #only showing top 20 rows


     predictions = lr_model.transform(test_df)
     predictions.select("prediction","quality","features").show(4)
     #+-----------------+-------+--------------------+
     #|       prediction|quality|            features|
     #+-----------------+-------+--------------------+
     #| 6.13162275009766|      8|[3.9,0.225,0.4,4....|
     #|6.044852051498137|      7|[4.2,0.17,0.36,1....|
     #|5.610998558500521|      3|[4.2,0.215,0.23,5...|
     #|6.066544726148018|      7|[4.4,0.54,0.09,5....|
     #+-----------------+-------+--------------------+
     #only showing top 4 rows


     #Decision Tree Regression:
     from pyspark.ml.evaluation import RegressionEvaluator

     from pyspark.ml.regression import DecisionTreeRegressor
     dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'quality')
     dt_model = dt.fit(train_df)
     dt_predictions = dt_model.transform(test_df)
     dt_evaluator = RegressionEvaluator(
         labelCol="quality", predictionCol="prediction", metricName="rmse")
     rmse = dt_evaluator.evaluate(dt_predictions)
     print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

     #Root Mean Squared Error (RMSE) on test data = 0.789392

     print(dt_model.featureImportances)
     #(11,[0,1,2,3,4,5,6,7,8,9,10],[0.0161345534247,0.226477933279,0.0196112073563,0.0171371559714,0.00952117178485,0.104527785965,0.00675556878503,0.0340917497346,0.0115079025446,0.0297641649,0.524470806254])


     #Gradient-boosted tree regression
     from pyspark.ml.regression import GBTRegressor
     gbt = GBTRegressor(featuresCol = 'features', labelCol = 'quality', maxIter=10)
     gbt_model = gbt.fit(train_df)
     gbt_predictions = gbt_model.transform(test_df)
     gbt_predictions.select('prediction', 'quality', 'features').show(5)

     #+-----------------+-------+--------------------+
     #|       prediction|quality|            features|
     #+-----------------+-------+--------------------+
     #|6.593587842097541|      8|[3.9,0.225,0.4,4....|
     #|7.069291566571429|      7|[4.2,0.17,0.36,1....|
     #|5.626861271951191|      3|[4.2,0.215,0.23,5...|
     #|6.621183767885232|      7|[4.4,0.54,0.09,5....|
     #|4.673763486325942|      5|[4.6,0.445,0.0,1....|
     #+-----------------+-------+--------------------+

     gbt_evaluator = RegressionEvaluator(
         labelCol="quality", predictionCol="prediction", metricName="rmse")
     rmse = gbt_evaluator.evaluate(gbt_predictions)
     print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
     #Root Mean Squared Error (RMSE) on test data = 0.759619
     
     sc.stop()









