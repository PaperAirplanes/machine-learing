
import numpy as np
import operator


"""
函数说明：打开文件，1：不喜欢， 2：魅力一般， 3：极具魅力
Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类标签向量
Modify：
    2018-05-09
"""
def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    # 得到文件数
    numberOfLines = len(arrayOlines)
    # 创建返回的numpy矩阵
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        # .strip(rm), 当rm为空时，默认删除空白符('\n','\r','\t',' ')
        line = line.strip()
        # .split 将字符串根据'\t'分隔符进行切片
        listFromLine = line.split('\t')
        # 将数据的前3列提取出来存入returnMat的numpy的特征矩阵中
        returnMat[index, :] = listFromLine[0:3]
        # 根据文本的中的最后一列表示喜欢的程度进行分类1.不喜欢，2.魅力一般，3.极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

"""
函数说明：knn算法，分类器
Parameters:
    inX - 测试集（特征）
    dataSet - 训练集
    labels - 分类标签
    k - knn算法选取的距离最小点的个数
Returns:
    sortedClassCount[0][0] - 分类结果
Modify：
    2018-05-09
"""
def classify_knn_0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances*0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCout = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCout[0][0]

"""
函数说明：特征集归一化
Parameters:
    dataSet - 训练集（特征矩阵）
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据的范围
    
    2018-05-09
"""
def autoNorm(dataSet):
    # minVals返回每列特征的最小值
    minVals = dataSet.min(0)
    # maxVals返回每列特征的最大值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    # 取所有数据作为测试集的比例
    hoRatio = 0.80
    # 特征矩阵归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCount = 0

    for i in range(numTestVecs):
        # 后numTestVecs-end数据作为训练集，前0-numTestVecs作为测试集
        classifierResult = classify_knn_0(normMat[i,:], normMat[numTestVecs:m,:],
             datingLabels[numTestVecs:m], 4)
        print("分类结果：%s\t真实类别：%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("错误率：%f%%" % float(errorCount/float(numTestVecs)*100))

def classifyPerson():
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    ffMiles = float(input("每年获得的飞行常客里程数："))
    precentTats = float(input("玩视频游戏所耗时间百分比："))
    iceCream = float(input("的冰淇淋的公升数"))
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, precentTats, iceCream])
    norminArr = (inArr - minVals) / ranges
    classiferResult = classify_knn_0(norminArr, normMat, datingLabels, 3)
    print("你可能%s这个人" % (resultList[classiferResult - 1]))

datingClassTest()