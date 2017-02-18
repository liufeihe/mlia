# -*- coding:utf-8 -*-

import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how','to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] # 1 is abusive
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 # one word appearring more times can have no meaning
        else:
            print "word:%s is not in my Vocabulary"%word
    return returnVec


def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 # one word appearring more times can have meaning
        else:
            print "word:%s is not in my Vocabulary"%word
    return returnVec


def trainNB0(trainMatrix, trainClasses):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainClasses)/float(numTrainDocs) # class 1 's prob
    p0Num = np.ones(numWords)
    p0Denom = 2.0
    p1Num = np.ones(numWords)
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainClasses[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom) # each word 's prob in class 1
    p0Vect = np.log(p0Num/p0Denom) # each word 's prob in class 0
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1-pClass1)
    if p1 > p0:
        return 1
    return 0


def test():
    posts, classes = loadDataSet()
    myVocab = createVocabList(posts)
    trainMat = []
    for post in posts:
        trainMat.append(setOfWords2Vec(myVocab, post))
    print trainMat
    p0V, p1V, pAb = trainNB0(trainMat, classes)
    testEntry = ['love', 'my', 'dalmation']
    print classifyNB(setOfWords2Vec(myVocab, testEntry), p0V, p1V, pAb)
    testEntry = ['fuck', 'my', 'stupid']
    print classifyNB(setOfWords2Vec(myVocab, testEntry), p0V, p1V, pAb)


if __name__ == '__main__':
    test()




















