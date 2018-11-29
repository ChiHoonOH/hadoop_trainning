# 데이터를 스파크 datframe api를 사용해 모델에 사용할 특징 행렬을 만든다
# 트윗의 내용으로 계산한 TF-IDF 기반 단어 백터
# 감성 단어 리스트롤 계산한 긍정/부정 감성 점수
# 월, 요일, 시각등 시간관련 특정 변수
# 코드 진행시 
'''
pyspark에서는 
SparkContext 삭제해야만 새로 만들수 있다
sc.stop()
del sc

spark-submit로 제출할때는 삭제할 필요 없다

즉 구동방식에 따라 다름
'''

import re, string
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import StringType, ArrayType, FloatType
import pyspark.sql.functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler, IDF, RegexTokenizer, HashingTF
from pyspark.ml import Pipeline

# SparkContext 신규 생성
# executor의 메모리 설정
#SparkContext.setSystemProperty('spark.executor.memory', '8g')
SparkContext.setSystemProperty('spark.executor.memory', '500')
conf = SparkConf()
#conf.set('spark.executor.instances', 8)
# 하나의 Spark executor = 하나의 YARN container
# 각 executor가 사용하는 thread의 수
conf.set('spark.executor.instances', 3)
# join 수행시 자동으로 브로드캐스트 되는 최대 크기를 400MB로 확대(기본값은 10MB)
conf.set('spark.sql.autoBroadcastJoinThreshold', 400*1024*1024)  # 400MB for broadcast join
sc = SparkContext('yarn-client', 'app', conf=conf)

# 하이브 세팅
from pyspark.sql import HiveContext
hc = HiveContext(sc)
hc.sql("use demo")

# 특정 변수 생성 쿼리에 PySpark UDF(User Defined Function) 생성
# Define PySpark UDF to tokenize text into words with various other specialized procesing
punct = re.compile('[%s]' % re.escape(string.punctuation))
def tok_str(text, ngrams=1, minChars=2):
    # 화이트 스페이스를 스페이스 문자 문자 하나로 치환
    text   = re.sub(r'\s+', ' ', text) 		     
    # split into tokens and change to lower case
    # 텍스트(문장)를 단어로 분할하고 모든 영문자는 소문자로 만듬
    # unicode 처리
    tokens = map(unicode, text.lower().split(' '))     
    # 길이가 두글자 미만 경우, @로 시작하는 문자(트위터 사용자명) 제외
    tokens = filter(lambda x: len(x)>=minChars and x[0]!='@', tokens)     
    # remove short words and usernames
    # url 주소는 모둔 URL 문자로 치환하여 리스트 처리
    tokens = ["URL" if t[:4]=="http" else t for t in tokens]      
    # 단어에 구두점 제거
    tokens = [punct.sub('', t) for t in tokens]
    if ngrams==1:
        return tokens
    else:
        return tokens + [' '.join(tokens[i:i+ngrams]) for i in xrange(len(tokens)-ngrams+1)]
# 함수 정의 완료 (User Define Function)
# spark에서 sql을 날릴 때, 사용자가 커스텀하게 만든 함수를 사용할 수 있게 하는 방법
# UDF는 우리가 필요한 새로운 컬럼 기반의 함수를 생성
tokenize = F.udf(lambda s: tok_str(unicode(s),ngrams=2), ArrayType(StringType()))

# 감성 점수 테이블 로드
wv = hc.table('sentiment_words').collect()
wordlist = dict([(r.word,r.score) for r in wv])

# 단어 리스토루벝 감성 점수를 계산하는 UDF 정의
def pscore(words):
    scores = filter(lambda x: x>0, [wordlist[t] for t in words if t in wordlist])
    return 0.0 if len(scores)==0 else (float(sum(scores))/len(scores))
pos_score = F.udf(lambda w: pscore(w), FloatType())

# 네가티브 점수 계산
def nscore(words):
    scores = filter(lambda x: x<0, [wordlist[t] for t in words if t in wordlist])
    return 0.0 if len(scores)==0 else (float(sum(scores))/len(scores))
neg_score = F.udf(lambda w: nscore(w), FloatType()) 

# 특징 행렬 생성
tw1 = hc.sql("""
SELECT text, query, polarity, date,
       regexp_extract(date, '([0-9]{2}):([0-9]{2}):([0-9]{2})', 1) as hour,
       regexp_extract(date, '(Sun|Mon|Tue|Wed|Thu|Fri|Sat)', 1) as dayofweek,
       regexp_extract(date, '(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', 1) as month
FROM tweets 
""")
tw2 = tw1.filter("polarity != 2").withColumn('words', tokenize(tw1['text']))
tw3 = (tw2.select("user", "hour", "dayofweek", "month", "words",
            	  F.when(tw2.polarity == 4, "Pos").otherwise("Neg").alias("sentiment"),
            	  pos_score(tw2["words"]).alias("pscore"), 	
	    	  neg_score(tw2["words"]).alias("nscore")))
tw3.registerTempTable("fm")

# 분류 모델 구축
# 모델링 매개변수
numFeatures = 5000
minDocFreq  = 50
numTrees    = 1000

# 머신 러닝 파이프라인 구축
inx1 = StringIndexer(inputCol="hour", outputCol="hour-inx")
inx2 = StringIndexer(inputCol="month", outputCol="month-inx")
inx3 = StringIndexer(inputCol="dayofweek", outputCol="dow-inx")
inx4 = StringIndexer(inputCol="sentiment", outputCol="label")
hashingTF = HashingTF(numFeatures=numFeatures, inputCol="words", outputCol="hash-tf")
idf  = IDF(minDocFreq=minDocFreq, inputCol="hash-tf", outputCol="hash-tfidf")
va   = VectorAssembler(inputCols =["hour-inx", "month-inx", "dow-inx", "hash-tfidf", "pscore", "nscore"], outputCol="features")
rf   = RandomForestClassifier(numTrees=numTrees, maxDepth=4, maxBins=32, labelCol="label", seed=42)
p    = Pipeline(stages=[inx1, inx2, inx3, inx4, hashingTF, idf, va, rf])

# 훈련용 테스용 데이터 분류
(trainSet, testSet) = hc.table("fm").randomSplit([0.7, 0.3])
trainData = trainSet.cache()
testData  = testSet.cache()
# 훈련
model     = p.fit(trainData) 

# 모델 정밀도, 재현율, 정확도를 평가할 함수 구성
def eval_metrics(lap):
    tp = float(len(lap[(lap['label']==1) & (lap['prediction']==1)]))
    tn = float(len(lap[(lap['label']==0) & (lap['prediction']==0)]))
    fp = float(len(lap[(lap['label']==0) & (lap['prediction']==1)]))
    fn = float(len(lap[(lap['label']==1) & (lap['prediction']==0)]))
    precision = tp / (tp+fp)
    recall    = tp / (tp+fn)
    accuracy  = (tp+tn) / (tp+tn+fp+fn)
    return {'precision': precision, 'recall': recall, 'accuracy': accuracy}

# 테스트 데이터로 예측
results = model.transform(testData) 
lap     = results.select("label", "prediction").toPandas()

m = eval_metrics(lap)
print (m)
