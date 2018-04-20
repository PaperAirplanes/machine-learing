from numpy import *

def loadDataSet():                          # 预处理数据
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/ (1+exp(-inX))

def gradAscent(dataMatIn, classLabels):     # 梯度上升
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(weights):              # 绘图
    import matplotlib.pyplot as plt
    import numpy as np
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
   # fig = plt.figure()
    plt.subplot(111)
    plt.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    plt.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
   # y = (-weights[0]-weights[1]*x)/weights[2]
    y = []
    for i in range(len(x)):
        y.append(float((-weights[0]-weights[1]*x[i])/weights[2]))
    plt.plot(x, y, '-b')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()