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

    df = df.withColumn("winetype", lit('red'))
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

    df2 = df.withColumn("winetype", lit('white'))
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

    #1 for red, 0 for white
    df4 = df3.withColumn("numericaltype", lit(F.when((df3.winetype == "red"), 1).otherwise(0)))

    df = df4.select('fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality','winetype', 'numericaltype') 
    cols = df.columns
    df.printSchema()


    df.show(5)
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+--------+-------------+
    #|fixed acidity|volatile acidity|citric acid|residual sugar|chlorides|free sulfur dioxide|total sulfur dioxide|density|  pH|sulphates|alcohol|quality|winetype|numericaltype|
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+--------+-------------+
    #|          7.4|             0.7|        0.0|           1.9|    0.076|               11.0|                34.0| 0.9978|3.51|     0.56|    9.4|      5|     red|            1|
    #|          7.8|            0.88|        0.0|           2.6|    0.098|               25.0|                67.0| 0.9968| 3.2|     0.68|    9.8|      5|     red|            1|
    #|          7.8|            0.76|       0.04|           2.3|    0.092|               15.0|                54.0|  0.997|3.26|     0.65|    9.8|      5|     red|            1|
    #|         11.2|            0.28|       0.56|           1.9|    0.075|               17.0|                60.0|  0.998|3.16|     0.58|    9.8|      6|     red|            1|
    #|          7.4|             0.7|        0.0|           1.9|    0.076|               11.0|                34.0| 0.9978|3.51|     0.56|    9.4|      5|     red|            1|
    #+-------------+----------------+-----------+--------------+---------+-------------------+--------------------+-------+----+---------+-------+-------+--------+-------------+
    #only showing top 5 rows

    from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
    stages = []
    label_stringIdx = StringIndexer(inputCol = 'numericaltype', outputCol = 'label')
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
    #Training Dataset Count: 2247                                                  

    print("Test Dataset Count: " + str(test.count()))
    #Test Dataset Count: 951

    #Logistic Regression

    from pyspark.ml.classification import LogisticRegression
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
    lrModel = lr.fit(train)

    import matplotlib.pyplot as plt
    import numpy as np

    beta = np.sort(lrModel.coefficients)
    print(beta)

    #[ -2.01590795e-03  -1.17867958e-03  -3.42493735e-04  -7.27838006e-06
    #  -2.39349772e-06   5.29788426e-06   8.90171468e-06   1.22404953e-05
    #   1.63694460e-05   6.34920843e-05   1.22165857e-04   4.84716269e-04]

    plt.plot(beta)
    #[<matplotlib.lines.Line2D object at 0x10e234fd0>]
    #[<matplotlib.lines.Line2D object at 0x106edad50>]

    plt.ylabel('Beta Coefficients')
    #<matplotlib.text.Text object at 0x10d26bd90>
    #<matplotlib.text.Text object at 0x106ee3090>

    plt.title("Combined Wine Beta Coefficients for Red/White")

    plt.show()

    predictions = lrModel.transform(test)
    predictions.select('winetype', 'numericaltype', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    #Test Error = 0.528097 
    print('Test Area Under ROC', evaluator.evaluate(predictions))
    #('Test Area Under ROC', 0.4719029487293015)

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
    predictions.select('winetype', 'numericaltype', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    #Test Error = 0.532746 
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    #Test Area Under ROC: 0.467254051182


    #Random Forest Classifier

    from pyspark.ml.classification import RandomForestClassifier
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)
    predictions.select('winetype', 'numericaltype', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    #Test Error = 0.76452 
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    #Test Area Under ROC: 0.235479943328


    #Gradient-Boosted Classifier

    from pyspark.ml.classification import GBTClassifier
    gbt = GBTClassifier(maxIter=10)
    gbtModel = gbt.fit(train)
    predictions = gbtModel.transform(test)
    predictions.select('winetype', 'numericaltype', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    #Test Error = 0.803254 
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    #Test Area Under ROC: 0.196745771717

    sc.stop()


