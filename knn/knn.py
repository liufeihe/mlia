# -*- coding:utf-8 -*-

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import os


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(str(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet,ranges, minVals


def personClassTest():
    ratio = 0.1
    mat, labels = file2matrix('knn/datingTestSet.txt')
    normMat,ranges,mins=autoNorm(mat)
    m = normMat.shape[0]
    numTestVecs = int(m*ratio)
    errs = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],labels[numTestVecs:m],3)
        print "the classifier came back with:%s, real answer:%s"%(classifierResult, labels[i])
        if (classifierResult != labels[i]):
            errs += 1
    print "total error rate is:%f"%(errs/float(numTestVecs))
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(mat[:,1], mat[:, 2])
    #plt.show()


def img2Vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    labels = []
    trainingFileList = os.listdir('./trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        labels.append(classNumStr)
        trainingMat[i,:] = img2Vector('./trainingDigits/%s'%fileNameStr)
    testFileList = os.listdir('./testDigits')
    errs = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('./testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, labels, 3)
        print "the classifier came back with:%d, the real:%d"%(classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errs +=1
    print "total errs:%d"%errs
    print "error ratio:%f"%(errs/float(mTest))


def createDataSet():
    group = np.array([[1.0,1.1], [1.0,1.0], [0, 0], [0, 0.1]])
    labels = ['A','A','B','B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]#get the size
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances =  sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices =  distances.argsort()  #after sorted, get axis val
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) +1
    sortedClassCount =  sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #print sortedClassCount
    return sortedClassCount[0][0]


def simpleTest():
    dataSet, labels = createDataSet()
    inX = [0.2, 0.5]
    k = 3
    classX = classify0(inX, dataSet, labels, k)
    print classX


if __name__ == '__main__':
    #simpleTest()
    #personClassTest()
    handwritingClassTest()