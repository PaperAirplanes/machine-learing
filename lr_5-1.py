import logRegres
dataArr,labelMat=logRegres.loadDataSet()
weights = logRegres.gradAscent(dataArr, labelMat)
print(weights)
logRegres.plotBestFit(weights)