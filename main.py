import pandas as pd
import math
from sklearn.model_selection import train_test_split

class Lookup:
    def __init__(self, _trainData):
        self.wordDict={}
        self.numWords=0
        self.countByWord=[[],[],[],[]]
        self.totalCounts=[0,0,0,0]
        self.logProbs=[[],[],[],[]]
        self.wordStrings=[]
        self.trainData = _trainData
    def addUnitClassification(self, unit, sentiment):
        sentimentIndex=SENTIMENT_DICT[sentiment]
        if (not (unit in self.wordDict)):
            self.wordDict[unit]=self.numWords
            self.numWords+=1
            for si in range(4):
                self.countByWord[si].append(1)
                self.totalCounts[si]+=1
            self.wordStrings.append(unit)
        wordIndex=self.wordDict[unit]
        self.countByWord[sentimentIndex][wordIndex]+=1
        self.totalCounts[sentimentIndex]+=1
    def addListClassification(self, wordsRow, sentiment):
        for word in wordsRow:
            self.addUnitClassification(word,sentiment)
    def buildClassification(self):
        for sentiment in SENTIMENTS:
            sentData=self.trainData.loc[self.trainData[LABEL]==sentiment]
            for row in sentData.iterrows():
                wordsRow=getWordsRow(row[1])
                self.addListClassification(wordsRow,sentiment)
        for wordIndex in range(self.numWords):
            for sentimentIndex in range(4):
                self.logProbs[sentimentIndex].append(math.log(self.countByWord[sentimentIndex][wordIndex]/self.totalCounts[sentimentIndex]))
    def outputSignificantWords(self):
        for wi in range(self.numWords):
            avgLogProb=0
            for si in range(4):
                avgLogProb+=self.logProbs[si][wi]
            avgLogProb/=4
            for si in range(4):
                if (self.logProbs[si][wi] > (avgLogProb+2)):
                    print(self.wordStrings[wi],SENTIMENTS[si],round(self.logProbs[si][wi]-avgLogProb,2))

class Prob():
    def __init__(self, totalProbs, rowId):
        self.totalProbs=totalProbs
        self.rowId=rowId
        self.maxProb=0
        self.maxSentiment=0
        for si in range(4):
            if (totalProbs[si]>self.maxProb):
                self.maxProb=totalProbs[si]
                self.maxSentiment=si

class Classifier():
    def __init__(self, _classifyData, targetDistribution):
        self.probsByRow=[]
        self.numRows=len(_classifyData)
        self.targetAmounts=[self.numRows,0,0,0]
        self.scores=[]
        self.classifyData=_classifyData.reset_index(drop=True)
        for si in range(1,4):
            self.targetAmounts[si]=round(targetDistribution[si]*self.numRows)
            self.targetAmounts[0]-=self.targetAmounts[si]
        for row in range(self.numRows):
            self.scores.append(0)
    def classifyRow(self,row):
        wordsRow=getWordsRow(row)
        totalLogProbs=[0,0,0,0]
        for word in wordsRow:
            if (not (word in lookup.wordDict)):
                continue
            wi=lookup.wordDict[word]
            for si in range(4):
                totalLogProbs[si]+=lookup.logProbs[si][wi]
        maxLogProb=max(totalLogProbs)
        sumProbs=0
        totalProbs=[]
        for logProb in totalLogProbs:
            prob=math.exp(logProb-maxLogProb)
            totalProbs.append(prob)
            sumProbs+=prob
        for i in range(4):
            totalProbs[i]=totalProbs[i]/sumProbs
        return totalProbs
    def score(self):
        for row in self.classifyData.iterrows():
            totalProbs=self.classifyRow(row[1])
            self.probsByRow.append(Prob(totalProbs,row[0]))
    def fitToTarget(self):
        self.probsByRow.sort(key=lambda row: row.maxProb)
        self.probsByRow.reverse()
        for row in self.probsByRow:
            if (self.targetAmounts[row.maxSentiment]>0):
                self.targetAmounts[row.maxSentiment]-=1
                self.scores[row.rowId]=row.maxSentiment
            else:
                self.targetAmounts[0]-=1
                self.scores[row.rowId]=0
    def convertSentiments(self):
        sentiments=[]
        for row in range(self.numRows):
            sentiment=SENTIMENTS[self.scores[row]]
            sentiments.append(sentiment)
        return sentiments
    def getResults(self):
        self.score()
        self.fitToTarget()
        self.classifyData.loc[:,LABEL]=pd.Series(self.convertSentiments())

def getDataFrame(filename):
    trainFile=open(filename,mode="r",encoding="utf-8")
    trainLines=trainFile.read().split("\n")
    trainData=pd.DataFrame([line.split("\t") for line in trainLines[1:]])
    trainData.columns=trainLines[0].split("\t")
    return trainData

def getWordsCell(cell):
    return list(word.lower() for word in cell.split())

def scrapePunctuation(word):
    newWord=""
    punctuation=[]
    for char in word:
        if (char.isalpha()):
            newWord+=char
        else:
            punctuation.append(char)
    return [newWord,punctuation]

def getWordsRow(row):
    wordsRow=[]
    for turn in TURN_NAMES:
        words=getWordsCell(row[turn])
        for word in words:
            [newWord,punctuation]=scrapePunctuation(word)
            if (len(newWord)>0):
                wordsRow.append(newWord)
            for punc in punctuation:
                wordsRow.append(punc)
    return wordsRow

def calcAccuracy(trueOutput, testOutput):
    trueLabels=trueOutput.loc[:,LABEL].tolist()
    testLabels=testOutput.loc[:,LABEL].tolist()
    truePositive=[0,0,0,0]
    falsePositive=[0,0,0,0]
    falseNegative=[0,0,0,0]
    for index in range(len(trueLabels)):
        if (trueLabels[index]==testLabels[index]):
            sentIndex = SENTIMENT_DICT[trueLabels[index]]
            truePositive[sentIndex]+=1
        else:
            sentIndexTrue = SENTIMENT_DICT[trueLabels[index]]
            sentIndexTest = SENTIMENT_DICT[testLabels[index]]
            falseNegative[sentIndexTrue]+=1
            falsePositive[sentIndexTest]+=1
    tp=sum(truePositive[1:])
    fp=sum(falsePositive[1:])
    fn=sum(falseNegative[1:])
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    harmonicMean=2/(1/precision+1/recall)
    return harmonicMean
    

def formatResult(classifyData):
    outFile=open(OUT_NAME, mode="w", encoding="utf-8")
    outFile.write(classifyData.to_csv(sep="\t",index=False))

TRAIN_NAME="train.txt"
DEV_NAME="devwithoutlabels.txt"
OUT_NAME="test.txt"

SENTIMENTS=["others","happy","sad","angry"]
SENTIMENT_DICT={"others": 0, "happy": 1, "sad": 2, "angry": 3}
TARGET_DISTRIBUTION_DEV=[0.88,0.04,0.04,0.04]
TARGET_DISTRIBUTION_TEST=[0.5,1/6,1/6,1/6]
TURN_NAMES=["turn1","turn2","turn3"]
LABEL="label"

origData=getDataFrame(TRAIN_NAME)
[trainData,testData]=train_test_split(origData,test_size=0.2)
devData=getDataFrame(DEV_NAME)

lookup=Lookup(trainData)
lookup.buildClassification()
#lookup.outputSignificantWords()

testClassifier=Classifier(testData, TARGET_DISTRIBUTION_TEST)
testClassifier.getResults()
testOutput=testClassifier.classifyData
testAccuracy=calcAccuracy(testData,testOutput)
print(testAccuracy)

devClassifier=Classifier(devData, TARGET_DISTRIBUTION_DEV)
devClassifier.getResults()
devOutput=devClassifier.classifyData
formatResult(devOutput)












