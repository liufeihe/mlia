# -*- coding:utf-8 -*-
import classifyText
import numpy as np


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]


def spamTest():
    docList=[]; classList=[]; fullText=[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = classifyText.createVocabList(docList)
    trainingSet = range(50)
    testSet=[]
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(classifyText.setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1v,pSpam = classifyText.trainNB0(np.array(trainMat), np.array(trainClasses))
    #print p0V,p1v, pSpam
    errCnt = 0
    for docIndex in testSet:
        wordVector = classifyText.setOfWords2Vec(vocabList, docList[docIndex])
        if classifyText.classifyNB(np.array(wordVector),p0V,p1v,pSpam) != classList[docIndex]:
            errCnt += 1
    print 'error rate: ', float(errCnt)/len(testSet)


def test():
    for i in range(100):
        spamTest()


if __name__ == '__main__':
    test()