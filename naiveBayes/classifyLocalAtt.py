# -*- coding: utf-8 -*-
import classifyText
import classifySpam
import numpy as np
import feedparser


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict={}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[0:30]


def localWords(feed1, feed0):
    import feedparser
    docList=[]; classList=[]; fullText=[]
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = classifySpam.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = classifySpam.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = classifyText.createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0]) #remove the most frequently occuring words
    trainingSet = range(2*minLen)
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(classifyText.bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = classifyText.trainNB0(np.array(trainMat), np.array(trainClasses))
    errCnt = 0
    for docIndex in testSet:
        wordVector = classifyText.bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyText.classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errCnt += 1
    print 'error rate: ', float(errCnt)/len(testSet)
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY=[];topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append(vocabList[i], p0V[i])
        if p1V[i] > -6.0:
            topNY.append(vocabList[i], p1V[i])
    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    for item in sortedNY:
        print item[0]


def test():
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocaList,pSF,pNY = localWords(ny, sf)


if __name__ == '__main__':
    test()