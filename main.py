import pandas as pd
import math
from sklearn.model_selection import train_test_split
from random import random
from time import time

class Lookup:
    def __init__(self, _trainData):
        self.wordDict={}
        self.numWords=0
        self.countByWord=[[],[],[],[]]
        self.totalCounts=[0,0,0,0]
        self.logProbs=[[],[],[],[]]
        self.wordStrings=[]
        self.trainData = _trainData
        self.turnMultiplier = [1,1,1]
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
    def __init__(self, _classifyData, _lookup, targetDistribution):
        self.probsByRow=[]
        self.numRows=len(_classifyData)
        self.targetAmounts=[self.numRows,0,0,0]
        self.scores=[]
        self.lookup=_lookup
        self.classifyData=_classifyData.reset_index(drop=True)
        for si in range(1,4):
            self.targetAmounts[si]=round(targetDistribution[si]*self.numRows)
            self.targetAmounts[0]-=self.targetAmounts[si]
        for row in range(self.numRows):
            self.scores.append(0)
    def classifyRow(self,row):
        totalLogProbs=[0,0,0,0]
        for turn in TURN_NAMES:
            turnIndex=TURN_NAMES_DICT[turn]
            wordsTurn=getWordsTurn(row[turn])
            for word in wordsTurn:
                if (not (word in self.lookup.wordDict)):
                    continue
                wi=self.lookup.wordDict[word]
                for si in range(4):
                    totalLogProbs[si]+=(self.lookup.logProbs[si][wi]*self.lookup.turnMultiplier[turnIndex])
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

def cutData(origTestData):
    testData=origTestData.reset_index(drop=True)
    keepColumn=[]
    for i in range(len(testData)):
        keepColumn.append(random() < CUT_FACTOR)
    testData.loc[:,"keep"]=pd.Series(keepColumn)
    return testData[(testData[LABEL]=="others") | (testData["keep"])].drop("keep",axis=1)

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

def getWordsTurn(wordTurn):
    wordsTurn=[]
    words=getWordsCell(wordTurn)
    for word in words:
        [newWord,punctuation]=scrapePunctuation(word)
        if (len(newWord)>0):
            wordsTurn.append(newWord)
        for punc in punctuation:
            wordsTurn.append(punc)
    return wordsTurn

def getWordsRow(row):
    wordsRow=[]
    for turn in TURN_NAMES:
        wordsTurn=getWordsTurn(row[turn])
        for word in wordsTurn:
            wordsRow.append(word)
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

def simulateSingleSplit(modifiedTurnMultiplier):
    [trainData,testData]=train_test_split(origData,test_size=0.2)
    testData=cutData(testData)
    lookup=Lookup(trainData)
    lookup.buildClassification()
    lookup.turnMultiplier=modifiedTurnMultiplier
    testClassifier=Classifier(testData, lookup, TARGET_DISTRIBUTION_DEV)
    testClassifier.getResults()
    testOutput=testClassifier.classifyData
    testAccuracy=calcAccuracy(testData,testOutput)
    return testAccuracy

def simulateAverageAccuracy(numIterations,modifiedTurnMultiplier):
    startTime=time()
    avg=0
    for i in range(numIterations):
        avg+=simulateSingleSplit(modifiedTurnMultiplier)
    endTime=time()
    print("Time elapsed",endTime-startTime)
    return avg/numIterations

def trainTurnMultiplier(numIterations, upperValue):
    modifiedTurnMultiplier=[1,1,1]
    for i in range(upperValue):
        bestAccuracy=0
        bestIndex=0
        for turnIndex in range(3):
            modifiedTurnMultiplier[turnIndex]+=1
            curAccuracy=simulateAverageAccuracy(numIterations,modifiedTurnMultiplier)
            if (curAccuracy>bestAccuracy):
                bestAccuracy=curAccuracy
                bestIndex=turnIndex
            modifiedTurnMultiplier[turnIndex]-=1
        modifiedTurnMultiplier[bestIndex]+=1
        print(modifiedTurnMultiplier)
    return modifiedTurnMultiplier

def produceFinalOutput(turnMultiplier):
    lookup=Lookup(origData)
    lookup.buildClassification()
    lookup.turnMultiplier=turnMultiplier
    devClassifier=Classifier(devData, lookup, TARGET_DISTRIBUTION_DEV)
    devClassifier.getResults()
    devOutput=devClassifier.classifyData
    formatResult(devOutput)

def formatResult(classifyData):
    outFile=open(OUT_NAME, mode="w", encoding="utf-8")
    outFile.write(classifyData.to_csv(sep="\t",index=False))

TRAIN_NAME="train.txt"
DEV_NAME="devwithoutlabels.txt"
OUT_NAME="test.txt"

SENTIMENTS=["others","happy","sad","angry"]
SENTIMENT_DICT={"others": 0, "happy": 1, "sad": 2, "angry": 3}
TARGET_DISTRIBUTION_DEV=[0.88,0.04,0.04,0.04]
#TARGET_DISTRIBUTION_TEST=[0.5,1/6,1/6,1/6]
CUT_FACTOR=0.04*0.5/(0.88*1/6)
TURN_NAMES=["turn1","turn2","turn3"]
TURN_NAMES_DICT={"turn1": 0, "turn2": 1, "turn3": 2}
LABEL="label"
RANDOM="random"

origData=getDataFrame(TRAIN_NAME)
devData=getDataFrame(DEV_NAME)

actualTurnMultiplier=trainTurnMultiplier(10,10)
produceFinalOutput(actualTurnMultiplier)









