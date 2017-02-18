# -*- coding: utf-8 -*-
from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 1, 'yes'],
               [1, 1, 0, 'no'],
               [1, 0, 1, 'no'],
               [0, 1, 0, 'no'],
               [0, 1, 0, 'no']]
    labels = ['no surfacing', 'flippers','tail']
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestFeature = -1
    for i in range(numFeatures):
        featList = [ex[i] for ex in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for val in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, val)
            subEnt = calcShannonEnt(subDataSet)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * subEnt
        if newEntropy < baseEntropy:
            baseEntropy = newEntropy
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [ex[-1] for ex in dataSet]
    #print classList
    if classList.count(classList[0]) == len(classList):#class is all the same
        return classList[0]
    if len(dataSet[0]) == 1:# there is no feature to split ,only leaves
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del (labels[bestFeat])
    featValues = [ex[bestFeat] for ex in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        #print subLabels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def test2():
    fr = open('./lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, labels)
    print lensesTree
    storeTree(lensesTree, './tree_lenses')


def test():
    myDat, labels = createDataSet()
    print myDat,labels
    #print len(myDat), len(myDat[0])
    #print chooseBestFeatureToSplit(myDat)
    #print majorityCnt(['n','n','n','n'])
    trees =  createTree(myDat, labels)
    #storeTree(trees, 'trees_1')
    #print grabTree('trees_1')
    print classify(trees, ['no surfacing', 'flippers', 'tail'], [1, 1, 1])


if __name__ == '__main__':
    test2()