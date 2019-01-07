#Laura Vidiella del Blanco
#Final Project MET CS 777 - Big Data Analytics

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: app_name <file>", file=sys.stderr)
        exit(-1)

  #To start with the project, let's play around with the data and see what it shows.
  #In particular, we are going to look at 'quality' as we will create a column called 'tasty'
  #which will be used to predict in the Binary Classification
  #The data can be found here: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ 

  from pyspark.sql import SQLContext
  df = spark.read.csv('winequality-red.csv', header = True, inferSchema = True, sep=";")
  df.printSchema()

  #Below we can see the first 5 rows of the column 'quality'
  df.select('quality').show(5)
  #+-------+
  #|quality|
  #+-------+
  #|      5|
  #|      5|
  #|      5|
  #|      6|
  #|      5|
  #+-------+

  #Let's extract the number of distinct values that quality has 
  df.select('quality').distinct().count()
  6   

  #Let's count the number of each distinct value
  df.groupBy('quality').count().show()
  #+-------+-----+
  #|quality|count|
  #+-------+-----+
  #|      6|  638|
  #|      3|   10|
  #|      5|  681|
  #|      4|   53|
  #|      8|   18|
  #|      7|  199|
  #+-------+-----+

  #Let's get a crosstab of alcohol and quality to see the correlation
  df.crosstab('alcohol','quality').show()
  #+----------------+---+---+---+---+---+---+
  #| alcohol_quality|  3|  4|  5|  6|  7|  8|
  #+----------------+---+---+---+---+---+---+
  #|            9.05|  0|  1|  0|  0|  0|  0|
  #|             9.1|  0|  2| 14|  7|  0|  0|
  #|            10.2|  1|  0| 21| 20|  4|  0|
  #|             9.4|  0|  2| 79| 22|  0|  0|
  #|           10.55|  0|  0|  0|  1|  1|  0|
  #|            10.0|  0|  4| 29| 25|  8|  1|
  #|9.23333333333333|  0|  0|  0|  1|  0|  0|
  #|             9.9|  1|  1| 25| 18|  4|  0|
  #|            14.0|  0|  0|  0|  4|  1|  2|
  #|            9.25|  0|  0|  0|  1|  0|  0|
  #|            12.1|  0|  0|  1|  4|  8|  0|
  #|            9.55|  0|  0|  1|  1|  0|  0|
  #|10.0333333333333|  0|  0|  0|  2|  0|  0|
  #|             8.8|  0|  0|  2|  0|  0|  0|
  #|             8.7|  0|  0|  0|  2|  0|  0|
  #|            10.9|  1|  3| 13| 27|  5|  0|
  #|13.5666666666667|  0|  0|  0|  0|  1|  0|
  #|            11.1|  0|  1|  7| 15|  4|  0|
  #|            11.4|  0|  2|  9| 18|  2|  1|
  #|             9.3|  0|  2| 44| 13|  0|  0|
  #+----------------+---+---+---+---+---+---+

  #Let's get the stats for quality
  df.describe('quality').show()
  #+-------+------------------+
  #|summary|           quality|
  #+-------+------------------+
  #|  count|              1599|
  #|   mean|5.6360225140712945|
  #| stddev|0.8075694397347051|
  #|    min|                 3|
  #|    max|                 8|
  #+-------+------------------+

  #Now, let's create the column 'tasty' with values 1 or 0. 1 For wines ranked higher or qual 
  #to 7 and 0 for the rest
  from pyspark.sql import functions as F
  from pyspark.sql.functions import lit

  df = df.withColumn("tasty", lit(F.when(df.quality >= 7, 1).otherwise(0)))
  df.printSchema()

  #Let's see how many tasty we have for each value
  df.groupBy('tasty').count().show()
  #+-----+-----+
  #|tasty|count|
  #+-----+-----+
  #|    1|  217|
  #|    0| 1382|
  #+-----+-----+

  #We're going to use all columns for our analysis:
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


  #The above code are taken from databricks’ official site and it indexes each categorical column 
  #using the StringIndexer, then converts the indexed categories into one-hot encoded variables. 
  #The resulting output has the binary vectors appended to the end of each row. We use the StringIndexer 
  #again to encode our labels to label indices. Next, we use the VectorAssembler to combine 
  #all the feature columns into a single vector column.


  #Pipeline

  #We use Pipeline to chain multiple Transformers and Estimators together to specify our machine learning 
  #workflow. A Pipeline’s stages are specified as an ordered array.


  from pyspark.ml import Pipeline
  pipeline = Pipeline(stages = stages)
  pipelineModel = pipeline.fit(df)
  df = pipelineModel.transform(df)
  #df = pipelineModel.transform(df)
  selectedCols = ['label', 'features'] + cols
  df = df.select(selectedCols)
  train, test = df.randomSplit([0.7, 0.3], seed = 2018)

  print("Training Dataset Count: " + str(train.count()))
  #Training Dataset Count: 7764                                                    

  print("Test Dataset Count: " + str(test.count()))
  #Test Dataset Count: 3398


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

  #[ -1.94008940e-02  -5.04077385e-03  -3.46652194e-03  -8.50823211e-04
  #  -1.04755658e-05  -9.42565262e-06   1.06574581e-04   1.63912224e-04
  #   8.63328053e-04   2.10066347e-03   2.56801343e-03   2.84811175e-03]

  plt.plot(beta)
  #[<matplotlib.lines.Line2D object at 0x10e234fd0>]
  #[<matplotlib.lines.Line2D object at 0x106edad50>]

  plt.ylabel('Beta Coefficients')
  #<matplotlib.text.Text object at 0x10d26bd90>
  #<matplotlib.text.Text object at 0x106ee3090>
  plt.title("Red Wine Beta Coefficients")

  plt.show()

  predictions = lrModel.transform(test)
  predictions.select('alcohol', 'quality', 'density', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
  #+-------+-------+-------+-----+--------------------+----------+--------------------+
  #|alcohol|quality|density|label|       rawPrediction|prediction|         probability|
  #+-------+-------+-------+-----+--------------------+----------+--------------------+
  #|   13.6|      6| 0.9902|  0.0|[1.85928746002546...|       0.0|[0.86521387409435...|
  #|   12.5|      6|0.99258|  0.0|[1.86323623786571...|       0.0|[0.86567371217858...|
  #|   12.5|      6|0.99444|  0.0|[1.86257277694081...|       0.0|[0.86559654440800...|
  #|   13.0|      5| 0.9937|  0.0|[1.86376142452199...|       0.0|[0.86573477059251...|
  #|   11.4|      5| 0.9942|  0.0|[1.86476612291344...|       0.0|[0.86585151189531...|
  #|   11.4|      5|0.99346|  0.0|[1.86634860101260...|       0.0|[0.86603521456034...|
  #|   10.9|      6|0.99483|  0.0|[1.86311957313505...|       0.0|[0.86566014550572...|
  #|   10.3|      6|0.99721|  0.0|[1.86293177860920...|       0.0|[0.86563830488145...|
  #|   11.7|      6| 0.9944|  0.0|[1.86107961773678...|       0.0|[0.86542273666214...|
  #|   11.2|      5|0.99385|  0.0|[1.86528498110524...|       0.0|[0.86591176722096...|
  #+-------+-------+-------+-----+--------------------+----------+--------------------+
  #only showing top 10 rows

  #rawPrediction is the raw output of the logistic regression classifier (array with length equal to the number of classes)
  #probability is the result of applying the logistic function to rawPrediction (array of length equal to that of rawPrediction)
  #prediction is the argument where the array probability takes its maximum value, and it gives the most probable label (single number)


  from pyspark.ml.evaluation import BinaryClassificationEvaluator
  evaluator = BinaryClassificationEvaluator()
  accuracy = evaluator.evaluate(predictions)
  print("Test Error = %g " % (1.0 - accuracy))
  print('Test Area Under ROC', evaluator.evaluate(predictions))
  #('Test Area Under ROC', 0.9840593434343433)


  #Decision Tree Classifier

  from pyspark.ml.classification import DecisionTreeClassifier
  dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
  dtModel = dt.fit(train)
  predictions = dtModel.transform(test)
  predictions.select('alcohol', 'quality', 'density', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

  #+-------+-------+-------+-----+-------------+----------+-----------+
  #|alcohol|quality|density|label|rawPrediction|prediction|probability|
  #+-------+-------+-------+-----+-------------+----------+-----------+
  #|   13.6|      6| 0.9902|  0.0|  [986.0,0.0]|       0.0|  [1.0,0.0]|
  #|   12.5|      6|0.99258|  0.0|  [986.0,0.0]|       0.0|  [1.0,0.0]|
  #|   12.5|      6|0.99444|  0.0|  [986.0,0.0]|       0.0|  [1.0,0.0]|
  #|   13.0|      5| 0.9937|  0.0|  [986.0,0.0]|       0.0|  [1.0,0.0]|
  #|   11.4|      5| 0.9942|  0.0|  [986.0,0.0]|       0.0|  [1.0,0.0]|
  #|   11.4|      5|0.99346|  0.0|  [986.0,0.0]|       0.0|  [1.0,0.0]|
  #|   10.9|      6|0.99483|  0.0|  [986.0,0.0]|       0.0|  [1.0,0.0]|
  #|   10.3|      6|0.99721|  0.0|  [986.0,0.0]|       0.0|  [1.0,0.0]|
  #|   11.7|      6| 0.9944|  0.0|  [986.0,0.0]|       0.0|  [1.0,0.0]|
  #|   11.2|      5|0.99385|  0.0|  [986.0,0.0]|       0.0|  [1.0,0.0]|
  #+-------+-------+-------+-----+-------------+----------+-----------+
  #only showing top 10 rows

  evaluator = BinaryClassificationEvaluator()
  accuracy = evaluator.evaluate(predictions)
  print("Test Error = %g " % (1.0 - accuracy))
  print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

  #Test Area Under ROC: 1.0


  #Random Forest Classifier

  from pyspark.ml.classification import RandomForestClassifier
  rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
  rfModel = rf.fit(train)
  predictions = rfModel.transform(test)
  predictions.select('alcohol', 'quality', 'density', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
  #+-------+-------+-------+-----+--------------------+----------+--------------------+
  #|alcohol|quality|density|label|       rawPrediction|prediction|         probability|
  #+-------+-------+-------+-----+--------------------+----------+--------------------+
  #|   13.6|      6| 0.9902|  0.0|          [20.0,0.0]|       0.0|           [1.0,0.0]|
  #|   12.5|      6|0.99258|  0.0|[19.8222222222222...|       0.0|[0.99111111111111...|
  #|   12.5|      6|0.99444|  0.0|[19.2687105165366...|       0.0|[0.96343552582683...|
  #|   13.0|      5| 0.9937|  0.0|          [20.0,0.0]|       0.0|           [1.0,0.0]|
  #|   11.4|      5| 0.9942|  0.0|[19.9191782810685...|       0.0|[0.99595891405342...|
  #|   11.4|      5|0.99346|  0.0|[19.1699507389162...|       0.0|[0.95849753694581...|
  #|   10.9|      6|0.99483|  0.0|[19.9233449477351...|       0.0|[0.99616724738675...|
  #|   10.3|      6|0.99721|  0.0|[19.9630831016373...|       0.0|[0.99815415508186...|
  #|   11.7|      6| 0.9944|  0.0|[18.8222222222222...|       0.0|[0.94111111111111...|
  #|   11.2|      5|0.99385|  0.0|[19.8533346596693...|       0.0|[0.99266673298346...|
  #+-------+-------+-------+-----+--------------------+----------+--------------------+
  #only showing top 10 rows

  evaluator = BinaryClassificationEvaluator()
  accuracy = evaluator.evaluate(predictions)
  print("Test Error = %g " % (1.0 - accuracy))
  print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
  #Test Area Under ROC: 1.0


  #Gradient-Bossted Classifier

  from pyspark.ml.classification import GBTClassifier
  gbt = GBTClassifier(maxIter=10)
  gbtModel = gbt.fit(train)
  predictions = gbtModel.transform(test)
  predictions.select('alcohol', 'quality', 'density', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

  #predictions = gbtModel.transform(test)
  #predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
  #+-------+-------+-------+-----+--------------------+----------+--------------------+
  #|alcohol|quality|density|label|       rawPrediction|prediction|         probability|
  #+-------+-------+-------+-----+--------------------+----------+--------------------+
  #|   13.6|      6| 0.9902|  0.0|[1.32590267922033...|       0.0|[0.93412217565278...|
  #|   12.5|      6|0.99258|  0.0|[1.32590267922033...|       0.0|[0.93412217565278...|
  #|   12.5|      6|0.99444|  0.0|[1.32590267922033...|       0.0|[0.93412217565278...|
  #|   13.0|      5| 0.9937|  0.0|[1.32590267922033...|       0.0|[0.93412217565278...|
  #|   11.4|      5| 0.9942|  0.0|[1.32590267922033...|       0.0|[0.93412217565278...|
  #|   11.4|      5|0.99346|  0.0|[1.32590267922033...|       0.0|[0.93412217565278...|
  #|   10.9|      6|0.99483|  0.0|[1.32590267922033...|       0.0|[0.93412217565278...|
  #|   10.3|      6|0.99721|  0.0|[1.32590267922033...|       0.0|[0.93412217565278...|
  #|   11.7|      6| 0.9944|  0.0|[1.32590267922033...|       0.0|[0.93412217565278...|
  #|   11.2|      5|0.99385|  0.0|[1.32590267922033...|       0.0|[0.93412217565278...|
  #+-------+-------+-------+-----+--------------------+----------+--------------------+
  #only showing top 10 rows


  evaluator = BinaryClassificationEvaluator()
  accuracy = evaluator.evaluate(predictions)
  print("Test Error = %g " % (1.0 - accuracy))
  print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
  #Test Area Under ROC: 1.0
  
  sc.stop()



