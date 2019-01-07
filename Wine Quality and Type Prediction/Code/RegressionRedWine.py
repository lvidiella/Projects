#Laura Vidiella del Blanco
#Final Project MET CS 777 - Big Data Analytics

from pyspark.sql import SQLContext

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: app_name <file>", file=sys.stderr)
        exit(-1)

    #Let's test it with logistic regression now
    df = spark.read.csv('winequality-red.csv', header = True, inferSchema = True, sep=";")

    df.printSchema()

    import six
    for i in df.columns:
        if not( isinstance(df.select(i).take(1)[0][0], six.string_types)):
            print( "Correlation to quality for ", i, df.stat.corr('quality',i))

    #('Correlation to quality for ', 'fixed acidity', 0.12405164911322263)
    #('Correlation to quality for ', 'volatile acidity', -0.3905577802640061)
    #('Correlation to quality for ', 'citric acid', 0.22637251431804048)
    #('Correlation to quality for ', 'residual sugar', 0.013731637340065798)
    #('Correlation to quality for ', 'chlorides', -0.12890655993005293)
    #('Correlation to quality for ', 'free sulfur dioxide', -0.05065605724427597)
    #('Correlation to quality for ', 'total sulfur dioxide', -0.18510028892653774)
    #('Correlation to quality for ', 'density', -0.17491922778336474)
    #('Correlation to quality for ', 'pH', -0.0577313912053826)
    #('Correlation to quality for ', 'sulphates', 0.25139707906925995)
    #('Correlation to quality for ', 'alcohol', 0.4761663240011364)
    #('Correlation to quality for ', 'quality', 1.0)

    #The correlation coefficient ranges from –1 to 1. When it is close to 1, 
    #it means that there is a strong positive correlation

    #For white wines:
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


    #Let's prepare data for Machine Learning, having two columns only - features and label("quality"):

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
    #+--------------------+-------+
    #only showing top 4 rows

    #Let's get the splits

    splits = v_df.randomSplit([0.7, 0.3])
    train_df = splits[0]
    test_df = splits[1]

    from pyspark.ml.regression import LinearRegression
    lr = LinearRegression(featuresCol = 'features', labelCol='quality', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_df)
    print("Coefficients: " + str(lr_model.coefficients))
    #Coefficients: [0.0,-0.265554239871,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.115616629797]

    print("Intercept: " + str(lr_model.intercept))
    #Intercept: 4.58766409067

    #Summary of the model
    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    #RMSE: 0.752109

    print("r2: %f" % trainingSummary.r2)
    #r2: 0.155683

    train_df.describe().show()
    #+-------+------------------+
    #|summary|           quality|
    #+-------+------------------+
    #|  count|              1099|
    #|   mean| 5.655141037306643|
    #| stddev|0.8188903889785253|
    #|    min|                 3|
    #|    max|                 8|
    #+-------+------------------+


    #lr_predictions = lr_model.transform(test_df)
    #lr_predictions.select("prediction","quality","features").show(5)

    from pyspark.ml.evaluation import RegressionEvaluator
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                     labelCol="quality",metricName="r2")
    print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
    #R Squared (R2) on test data = 0.168567


    test_result = lr_model.evaluate(test_df)
    print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
    #Root Mean Squared Error (RMSE) on test data = 0.711674

    print("numIterations: %d" % trainingSummary.totalIterations)
    #numIterations: 11

    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    #objectiveHistory: [0.5, 0.49514774963371877, 0.48491582980929276, 0.48433044683175874, 0.48429382096139084, 0.4842903994499468, 
    #0.48428997284971775, 0.48428996515933626, 0.4842899648370882, 0.48428996480091463, 0.48428996479488884]


    trainingSummary.residuals.show()
    #+--------------------+
    #|           residuals|
    #+--------------------+
    #| -1.9641537362751134|
    #|  0.0802139288739534|
    #|  0.9052358729205787|
    #|-0.10538629667426669|
    #|-0.05382855995809166|
    #|  1.9052358729205787|
    #| -1.5307733788676519|
    #| -0.6410789238671111|
    #|  0.9514825248393315|
    #|   1.056314047285551|
    #|   1.076230615275886|
    #|-0.11600846626911299|
    #|  0.9647602368328885|
    #|  1.1017841435752782|
    #|  0.3870677445281121|
    #| 0.16363817437466555|
    #|  1.1310575122639825|
    #| -0.5690152704402127|
    #|  -0.957903158092849|
    #| -0.9184582999266908|
    #+--------------------+
    #only showing top 20 rows

    predictions = lr_model.transform(test_df)
    predictions.select("prediction","quality","features").show(4)
    #+------------------+-------+--------------------+
    #|        prediction|quality|            features|
    #+------------------+-------+--------------------+
    #|5.8363618256253345|      6|[5.0,0.74,0.0,1.2...|
    #| 5.942746459329606|      6|[5.1,0.47,0.02,1....|
    #| 5.566359728041501|      5|[5.2,0.32,0.25,1....|
    #| 6.116008466269113|      6|[5.2,0.34,0.0,1.8...|
    #+------------------+-------+--------------------+
    #only showing top 4 rows


    #Decision Tree Regression:

    from pyspark.ml.regression import DecisionTreeRegressor
    dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'quality')
    dt_model = dt.fit(train_df)
    dt_predictions = dt_model.transform(test_df)
    dt_evaluator = RegressionEvaluator(
        labelCol="quality", predictionCol="prediction", metricName="rmse")
    rmse = dt_evaluator.evaluate(dt_predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    #Root Mean Squared Error (RMSE) on test data = 0.647972

    dt_model.featureImportances
    print(dt_model.featureImportances)
    #(11,[0,1,2,3,4,5,6,8,9,10],[0.0169003713331,0.177762351722,0.00341827433315,0.0311491728453,0.0250867312299,0.0294384717424,0.0549983365993,0.00459567993679,0.204812916148,0.45183769411])

    #It seems like alcohol is the most important one to predict the wine quality in our data, followed by volatile acidity
    #which is what we could assume as well from the correlation 

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
    #Root Mean Squared Error (RMSE) on test data = 0.639156

    #It looks like Gradient-boosted tree regression was the best performing one, lower values of RMSE indicate better fit.
    
    sc.stop()
