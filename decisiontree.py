# For python2
# This program is written by Jun Peng. 02/06/2015

from sys import argv
from math import log
import numpy as np
from scipy import stats


# Item stores the value-label pair from training data and validating data
class Item:
    def __init__(self, value, label):
        self.value = value
        self.label = label

    def __str__(self):
        return 'value: ' + self.value + ' label: ' + self.label


# DTNode is used to construct the decision tree
class DTNode:
    def __init__(self):
        self.isLeaf = False
        self.targetIndex = None   # indicate which index would be test in this node
        self.descendants = {}   # stores the child for this DTNode
        self.label = None    # Label is used to record the decision label when the node is a leaf
        self.stats = None    # record the count of positive label and negative label at this node


# construct Item from training data and testing data
# @return a collection of Items
def parseData(dataArray):
    items = []
    for data in dataArray:
        items.append(Item(data[0], data[1]))
    return items


# count labels in arrayItems
# @return the counted result (eg. {'+':10, '-':15})
def countingLabels(arrayItems):
    res = {}
    for item in arrayItems:
        if item.label in res:
            res[item.label] += 1
        else:
            res[item.label] = 1
    return res


# @return the most common label in arrayItems
def labelWithMaxCount(labelCountDict):
    res, maxCount = None, None
    for label in labelCountDict:   # find the most common label in current node
        if maxCount is None or labelCountDict[label] > maxCount:
            res, maxCount = label, labelCountDict[label]
    return res


# Given the list of occurrence frequency (eg. [1,2,3]), calculate the entropy = sum( -prob * log2 prob)
# @ return entropy
def entropy(listOfFrequency):
    totalCount = sum(listOfFrequency)
    listOfProbability = [x/float(totalCount) for x in listOfFrequency]
    listOfEntropy = [ -x*log(x, 2) if x != 0 else 0 for x in listOfProbability]
    return sum(listOfEntropy)


# @ return (index, informationGain)
def informationGain(arrayItems, index, stats):
    partitionCount = {}
    for item in arrayItems:
        if item.value[index] in partitionCount:
            if item.label in partitionCount[item.value[index]]:
                partitionCount[item.value[index]][item.label] += 1
            else:
                partitionCount[item.value[index]][item.label] = 1
        else:
            partitionCount[item.value[index]] = {}
            partitionCount[item.value[index]][item.label] = 1
    rootEntropy = entropy(stats.values())
    childEntropyList = [entropy(partitionCount[key].values()) for key in partitionCount]
    childFrequency = [sum(partitionCount[key].values()) for key in partitionCount]
    totalCount = sum(childFrequency)
    return (index, rootEntropy - sum(map(lambda x, y: float(x)/totalCount*y, childFrequency, childEntropyList)))


# arrayItems -- array of class Item. Construct the complete decision tree.
# @return tree root
def DTBuilderForID3(arrayItems):
    root = DTNode()
    if not arrayItems or len(arrayItems) == 0: return "empty array"
    label, count = arrayItems[0].label, 0
    for item in arrayItems:
        if item.label != label:
            break
        count += 1
    if count == len(arrayItems):
        root.label, root.isLeaf = label, True
        root.stats = {label:count}
        return root
    stats = countingLabels(arrayItems)
    root.stats = stats
    root.label = labelWithMaxCount(stats)
    # listOfInformationGain records the infoGain which is used later to determine which attr is chosen
    listOfInformationGain = [informationGain(arrayItems, index, stats) for index in range(len(arrayItems[0].value))]
    # sort listOfInformationGain
    listOfInformationGain.sort(key=lambda x: x[1], reverse=True)
    # find the max information gain
    (targetIndex, _) = listOfInformationGain[0]
    # partition records the partition based on the selected attr
    partition = {}
    for item in arrayItems:
        key = item.value[targetIndex]
        item.value = item.value[:targetIndex] + item.value[targetIndex + 1:]
        if key in partition:
            partition[key].append(item)
        else:
            partition[key] = [item]
    root.targetIndex = targetIndex
    for key in partition:
        root.descendants[key] = DTBuilderForID3(partition[key])
    return root


# Given the list of occurrence frequency (eg. [1,2,3])
# Calculate the impurity (misclassification error) = 1 - max (pi)
# @ return misclassification error rate
def impurity(listOfFrequency):
    return 1 - max(listOfFrequency)/float(sum(listOfFrequency))


def impurityGain(arrayItems, index, stats):
    partitionCount = {}
    for item in arrayItems:
        if item.value[index] in partitionCount:
            if item.label in partitionCount[item.value[index]]:
                partitionCount[item.value[index]][item.label] += 1
            else:
                partitionCount[item.value[index]][item.label] = 1
        else:
            partitionCount[item.value[index]] = {}
            partitionCount[item.value[index]][item.label] = 1
    rootImp = impurity(stats.values())
    childImpList = [impurity(partitionCount[key].values()) for key in partitionCount]
    childFrequency = [sum(partitionCount[key].values()) for key in partitionCount]
    totalCount = sum(childFrequency)
    return (index, rootImp - sum(map(lambda x, y: float(x)/totalCount*y, childFrequency, childImpList)))

# builder decision tree by using misclassification error method
def DTBuilderForImp(arrayItems):
    root = DTNode()
    if not arrayItems or len(arrayItems) == 0: return "empty array"
    label, count = arrayItems[0].label, 0
    for item in arrayItems:
        if item.label != label:
            break
        count += 1
    if count == len(arrayItems):
        root.label, root.isLeaf = label, True
        root.stats = {label:count}
        return root
    stats = countingLabels(arrayItems)
    root.stats = stats
    root.label = labelWithMaxCount(stats)
    # listOfImpurityGain records the impGain which is used later to determine which attr is chosen
    listOfImpurityGain = [impurityGain(arrayItems, index, stats) for index in range(len(arrayItems[0].value))]
    # sort listOfImpurityGain
    listOfImpurityGain.sort(key=lambda x: x[1], reverse=True)
    # find the max information gain
    (targetIndex, _) = listOfImpurityGain[0]
    # partition records the partition based on the selected attr
    partition = {}
    for item in arrayItems:
        key = item.value[targetIndex]
        item.value = item.value[:targetIndex] + item.value[targetIndex + 1:]
        if key in partition:
            partition[key].append(item)
        else:
            partition[key] = [item]
    root.targetIndex = targetIndex
    for key in partition:
        root.descendants[key] = DTBuilderForImp(partition[key])
    return root





# calculate chi square
# @return chi square
def chiSquareCalc(chiSquareArray):
    row, col = len(chiSquareArray), len(chiSquareArray[0])
    expectedArray = [[0 for x in range(col-1)] for x in range(row-1)]
    res = 0
    for i in range(row-1):
        for j in range(col-1):
            expectedArray[i][j] = chiSquareArray[i][col-1]*chiSquareArray[row-1][j]/float(chiSquareArray[row-1][col-1])
            res += (expectedArray[i][j]-chiSquareArray[i][j])**2 /float(expectedArray[i][j])
    return res


# prune decision tree using chi square test
def chiSquarePrune(root, confidenceLevel):
    if root.isLeaf:
        return
    alpha = 1 - confidenceLevel
    attrArray, chiSquareArray = [], []
    DOF = (len(root.descendants) - 1) * (len(root.stats)-1)   # DOF -- degree of freedom
    # attrArray records the all label in current node
    for item in root.stats:
        attrArray.append(item)
    # construct the chiSquareArray according to the attrArray
    # Then use the chiSquareArray to calculate the chi square
    for item in root.descendants.values():
        res = []
        for attr in attrArray:
            if attr in item.stats:
                res.append(item.stats[attr])
            else:
                res.append(0)
        res.append(sum(res))
        chiSquareArray.append(res)
    res = []
    for attr in attrArray:
        if attr in root.stats:
            res.append(root.stats[attr])
        else:
            res.append(0)
    res.append(sum(res))
    chiSquareArray.append(res)
    chiSquare = chiSquareCalc(chiSquareArray)     # calculate the chi square
    pValue = 1 - stats.chi2.cdf(chiSquare, DOF)   # using library function to calculate the p-value
    if pValue > alpha:  # if p-value is greater than alpha, then prune current node with setting the node to leaf
        root.isLeaf = True
        root.descendants = {}
    if not root.isLeaf:
        for value in root.descendants.values():
            chiSquarePrune(value, confidenceLevel)


# use validating data to test accuracy
# @ return True if prediction matches the validating item label else False
def validateDT(root, item):
    if root.isLeaf:
        return root.label == item.label
    index = root.targetIndex
    testedAttr = item.value[index]
    item.value = item.value[:index] + item.value[index+1:]
    if testedAttr in root.descendants:
        return validateDT(root.descendants[testedAttr], item)
    else:
        # if no testedAttr in descendants, return the most common label
        return root.label == item.label


def validation(treeRoot, validatingItems):
    correct = 0    # use to record the number of correct prediction
    for item in validatingItems:
        if validateDT(treeRoot, item):
            correct += 1
    return correct/float(len(validatingItems))



"""
# helper function. use to print the decision tree
def printDT(root):
    # use queue to help print decision tree in level order
    queue = [(root, 0)]
    while queue:
        (node, level) = queue.pop(0)
        if node.isLeaf:
            print("Level: %d Leaf: %s stats:%s" % (level, node.label, node.stats))
        else:
            print("Level: %d InternalNode Index: %s stats for internal node: %s" % (level, node.targetIndex, node.stats))
            for item in node.descendants:
                queue.append((node.descendants[item], level+1))
"""


script, trainingFilename, validatingFilename, confidenceLevel = argv
trainingData, validatingData = np.genfromtxt(trainingFilename, dtype='str'), np.genfromtxt(validatingFilename,
                                                                                           dtype='str')
trainingItemsForID3, validatingItemsForID3 = parseData(trainingData), parseData(validatingData)
trainingItemsForImp, validatingItemsForImp = parseData(trainingData), parseData(validatingData)
treeRootForID3 = DTBuilderForID3(trainingItemsForID3)
treeRootForImp = DTBuilderForImp(trainingItemsForImp)
#printDT(treeRoot)
chiSquarePrune(treeRootForID3, float(confidenceLevel))
chiSquarePrune(treeRootForImp, float(confidenceLevel))
print "Confidence Level: " + confidenceLevel
print "Correct percentage for ID3: %f" % validation(treeRootForID3, validatingItemsForID3)
print "Correct percentage for Imp: %f" % validation(treeRootForImp, validatingItemsForImp)

