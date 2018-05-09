import numpy as np
import collections

def createDataSet():
    # 6 feature
    group = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
    # 6 labels
    labels = ['爱情片','爱情片','爱情片','动作片','动作片','动作片']
    return group, labels

def classfy_knn(inx, dataset, labels, k):
    # distance
    dist = np.sum((inx - dataset)**2, axis=1)**0.5
    # k nearest labels
    k_labels = [labels[index] for index in dist.argsort()[0 : k]]
    # most labels
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label


group, labels = createDataSet()
test_1 = [100, 20]
test_class = classfy_knn(test_1, group, labels, 4)
print(test_class)