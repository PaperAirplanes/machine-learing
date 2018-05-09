import numpy as np
import operator

def createDataSet():
    # 6 feature
    group = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
    # 6 labels
    labels = ['爱情片','爱情片','爱情片','动作片','动作片','动作片']
    return group, labels

def classify_knn_0(inX, dataSet, labels, k):
    # 返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    # 每列求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，得到实际距离
    distances = sqDistances**0.5
    # 返回从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    # 记录类别次数的字典
    print(sortedDistIndices)
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # 返回指定键入的值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # itemgetter(1) 根据字典的值进行排序；
        # itemgetter(2) 根据字典的键进行排序
        # reverse = True 降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

group, labels = createDataSet()
test_0 = [100, 20]
test_class = classify_knn_0(test_0, group, labels, 3)
print(test_class)
