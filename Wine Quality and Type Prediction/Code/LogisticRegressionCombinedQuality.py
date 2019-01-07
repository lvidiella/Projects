if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: app_name <file>", file=sys.stderr)
        exit(-1)

    from pyspark.sql import SQLContext

    #Import red wine
    df = spark.read.csv('winequality-red.csv', header = True, inferSchema = True, sep=";")
    df.printSchema()

    #Import white wine
    df2 = spark.read.csv('winequality-white.csv', header = True, inferSchema = True, sep=";")

    #result = df.union(df2)
    #result.show(5)

    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+
    #|fixed acidity|volatile acidity|citric acid|residual sugar|chlorides|free sulfur dioxide|total sulfur dioxide|density|  pH|sulphates|alcohol|quality|
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+
    #|          7.4|             0.7|        0.0|           1.9|    0.076|               11.0|                34.0| 0.9978|3.51|     0.56|    9.4|      5|
    #|          7.8|            0.88|        0.0|           2.6|    0.098|               25.0|                67.0| 0.9968| 3.2|     0.68|    9.8|      5|
    #|          7.8|            0.76|       0.04|           2.3|    0.092|               15.0|                54.0|  0.997|3.26|     0.65|    9.8|      5|
    #|         11.2|            0.28|       0.56|           1.9|    0.075|               17.0|                60.0|  0.998|3.16|     0.58|    9.8|      6|
    #|          7.4|             0.7|        0.0|           1.9|    0.076|               11.0|                34.0| 0.9978|3.51|     0.56|    9.4|      5|
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+
    #only showing top 5 rows

    result.count()
    #6497

    from pyspark.sql import functions as F
    from pyspark.sql.functions import lit

    df = df.withColumn("tasty", lit(F.when(df.quality >= 7, 1).otherwise(0)))
    df.show(5)
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+----+
    #|fixed acidity|volatile acidity|citric acid|residual sugar|chlorides|free sulfur dioxide|total sulfur dioxide|density|  pH|sulphates|alcohol|quality|Type|
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+----+
    #|          7.4|             0.7|        0.0|           1.9|    0.076|               11.0|                34.0| 0.9978|3.51|     0.56|    9.4|      5| red|
    #|          7.8|            0.88|        0.0|           2.6|    0.098|               25.0|                67.0| 0.9968| 3.2|     0.68|    9.8|      5| red|
    #|          7.8|            0.76|       0.04|           2.3|    0.092|               15.0|                54.0|  0.997|3.26|     0.65|    9.8|      5| red|
    #|         11.2|            0.28|       0.56|           1.9|    0.075|               17.0|                60.0|  0.998|3.16|     0.58|    9.8|      6| red|
    #|          7.4|             0.7|        0.0|           1.9|    0.076|               11.0|                34.0| 0.9978|3.51|     0.56|    9.4|      5| red|
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+----+

    df2 = df2.withColumn("tasty", lit(F.when(df2.quality >= 7, 1).otherwise(0)))
    df2.show(5)
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+-----+
    #|fixed acidity|volatile acidity|citric acid|residual sugar|chlorides|free sulfur dioxide|total sulfur dioxide|density|  pH|sulphates|alcohol|quality| Type|
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+-----+
    #|          7.4|             0.7|        0.0|           1.9|    0.076|               11.0|                34.0| 0.9978|3.51|     0.56|    9.4|      5|white|
    #|          7.8|            0.88|        0.0|           2.6|    0.098|               25.0|                67.0| 0.9968| 3.2|     0.68|    9.8|      5|white|
    #|          7.8|            0.76|       0.04|           2.3|    0.092|               15.0|                54.0|  0.997|3.26|     0.65|    9.8|      5|white|
    #|         11.2|            0.28|       0.56|           1.9|    0.075|               17.0|                60.0|  0.998|3.16|     0.58|    9.8|      6|white|
    #|          7.4|             0.7|        0.0|           1.9|    0.076|               11.0|                34.0| 0.9978|3.51|     0.56|    9.4|      5|white|
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+-----+
    #only showing top 5 rows

    df3 = df.union(df2)
    df3.show(5)
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+----+
    #|fixed acidity|volatile acidity|citric acid|residual sugar|chlorides|free sulfur dioxide|total sulfur dioxide|density|  pH|sulphates|alcohol|quality|Type|
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+----+
    #|          7.4|             0.7|        0.0|           1.9|    0.076|               11.0|                34.0| 0.9978|3.51|     0.56|    9.4|      5| red|
    #|          7.8|            0.88|        0.0|           2.6|    0.098|               25.0|                67.0| 0.9968| 3.2|     0.68|    9.8|      5| red|
    #|          7.8|            0.76|       0.04|           2.3|    0.092|               15.0|                54.0|  0.997|3.26|     0.65|    9.8|      5| red|
    #|         11.2|            0.28|       0.56|           1.9|    0.075|               17.0|                60.0|  0.998|3.16|     0.58|    9.8|      6| red|
    #|          7.4|             0.7|        0.0|           1.9|    0.076|               11.0|                34.0| 0.9978|3.51|     0.56|    9.4|      5| red|
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+----+
    #only showing top 5 rows

    df = df3 

    df = df.select('fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality','tasty') 
    cols = df.columns
    df.printSchema()


    from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
    stages = []
    label_stringIdx = StringIndexer(inputCol = 'tasty', outputCol = 'label')
    stages += [label_stringIdx]
    numericCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    assemblerInputs = numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    #df = pipelineModel.transform(df)
    selectedCols = ['label', 'features'] + cols
    df = df.select(selectedCols)
    train, test = df.randomSplit([0.7, 0.3], seed = 2018)

    print("Training Dataset Count: " + str(train.count()))
    #Training Dataset Count: 4540                                                  

    print("Test Dataset Count: " + str(test.count()))
    #Test Dataset Count: 1957


    #Logistic Regression

    from pyspark.ml.classification import LogisticRegression
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
    lrModel = lr.fit(train)

    import matplotlib.pyplot as plt
    import numpy as np

    beta = np.sort(lrModel.coefficients)
    print(beta)

    #Use the coefficient to determine whether a change in a predictor variable makes the event more likely or less likely
    #βi > 0 implies eβi > 1 and the odds and probability increase with Xi
    #βi < 0 implies eβi < 1 and the odds and probability decrease with Xi

    #[ -1.39748038e+00  -4.83354279e-01  -9.44424563e-02  -4.27232204e-03
    #  -1.20525474e-03  -8.46089422e-05   1.43218910e-04   2.63551590e-02
    #   2.99958573e-02   3.45812040e-02   3.87686567e-02   9.25371397e-02]

    plt.plot(beta)
    #[<matplotlib.lines.Line2D object at 0x10e234fd0>]
    #[<matplotlib.lines.Line2D object at 0x106edad50>]

    plt.ylabel('Beta Coefficients')
    #<matplotlib.text.Text object at 0x10d26bd90>
    #<matplotlib.text.Text object at 0x106ee3090>
    plt.title("Combined Quality Beta Coefficients")

    plt.show()

    predictions = lrModel.transform(test)
    predictions.select('alcohol', 'quality', 'density', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    print('Test Area Under ROC', evaluator.evaluate(predictions))
    #Test Error = 0.0211162
    #('Test Area Under ROC', 0.9788837841314185)


    trainingSummary = lrModel.summary

    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)


    #Decision Tree Classifier

    from pyspark.ml.classification import DecisionTreeClassifier
    dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
    dtModel = dt.fit(train)
    predictions = dtModel.transform(test)
    predictions.select('alcohol', 'quality', 'density', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    #Test Error = 0 
    #Test Area Under ROC: 1.0


    #Random Forest Classifier

    from pyspark.ml.classification import RandomForestClassifier
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)
    predictions.select('alcohol', 'quality', 'density', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    #Test Error = 0 
    #Test Area Under ROC: 1.0



    #Gradient-Boosted Classifier

    from pyspark.ml.classification import GBTClassifier
    gbt = GBTClassifier(maxIter=10)
    gbtModel = gbt.fit(train)
    predictions = gbtModel.transform(test)
    predictions.select('alcohol', 'quality', 'density', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    #Test Error = 0 
    #Test Area Under ROC: 1.0
    
    sc.stop()


