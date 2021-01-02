#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import munkres as mk


def dis(x: np.ndarray, y: np.ndarray) -> float:
    """欧氏距离"""
    return np.sqrt(np.sum((x - y) ** 2))


def loadData(filename: str) -> (np.ndarray, np.ndarray):
    """
    读取数据
    """
    tagList = []
    featMatrix = []
    with open(filename, "r") as f:
        for line in f.readlines():
            resVec = list(map(float, line.strip().split(',')))
            tagList.append(int(resVec[0]))
            featMatrix.append(resVec[1:])
    return np.array(tagList), np.array(featMatrix)


def randCentreIdx(n: int, k: int) -> np.ndarray:
    """
    从0...n中随机选k个数
    """
    randIdx = np.arange(n)
    np.random.shuffle(randIdx)
    return randIdx[0:k]


def findNearestCentre(centres: np.ndarray, sample: np.ndarray) -> (int, float):
    """
    从所有中心中找举例sample最近的。
    """
    minDis = float("inf")
    minIdx = -1
    for i in range(centres.shape[0]):
        curDis = dis(centres[i], sample)
        if curDis < minDis:
            minIdx = i
            minDis = curDis
    return minIdx, minDis


def kMeans(featMatrix: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
    """
    k-Means 主函数，输入特征矩阵和 k 即可。
    """
    n = featMatrix.shape[0]
    tagList = np.zeros(n)
    disList = np.zeros(n)
    centres = featMatrix[randCentreIdx(n, k)]
    clusterChanged = True  # 表示所有点的类是否存在发生变化的，如果没有，则说明收敛稳定下来了
    while clusterChanged:
        clusterChanged = False
        for i in range(n):  # 遍历所有点，更新类和距离
            idx, curDis = findNearestCentre(centres, featMatrix[i])
            if idx != tagList[i]:
                tagList[i] = idx
                clusterChanged = True
            disList[i] = curDis
        for i in range(k):  # 更新各个聚类的中心
            centres[i] = np.mean(featMatrix[np.nonzero(tagList == i)], axis=0)
    return tagList, disList


def drawRes(x: int, y: int, featMat: np.ndarray, resLst: np.ndarray, k: int, acc: float, sse: float) -> None:
    """
    选择数据集中的x，y属性进行绘图
    """
    xLst, yLst = featMat[:, x], featMat[:, y]
    colors = ['red', 'blue', 'yellow']
    for i in range(1, k + 1):
        plt.scatter(xLst[np.nonzero(resLst == i)],
                    yLst[np.nonzero(resLst == i)],
                    marker='o', color=colors[i - 1],
                    s=40, label="Class" + str(i))
    plt.title("SSE = {0:.3f}, Acc = {1:.3f}".format(sse, acc))
    plt.legend(loc='best')
    # plt.show()
    plt.savefig("./src/{}_{}_{}.png".format(x, y, k))


def bestMap(L1, L2):
    """
    用来将两个列表的元素进行最大匹配。
    """
    Lab1 = np.unique(L1)
    nL1 = len(Lab1)
    Lab2 = np.unique(L2)
    nL2 = len(Lab2)
    n = max(nL1, nL2)
    G = np.zeros((n, n))
    for i in range(nL1):
        indClass1 = L1 == Lab1[i]
        indClass1 = indClass1.astype(float)
        for j in range(nL2):
            indClass2 = L2 == Lab2[j]
            indClass2 = indClass2.astype(float)
            G[i, j] = np.sum(indClass2 * indClass1)
    m = mk.Munkres()
    idx = np.array(m.compute(-G.T))
    c = idx[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nL2):
        newL2[L2 == Lab2[i]] = Lab1[c[i]]
    return newL2


if __name__ == "__main__":
    tagLst, featMat = loadData("./src/NormalizedData.csv")
    k = 3
    # shuffle the dataset
    featMatrix, tmpTagLst = sklearn.utils.shuffle(featMat, tagLst)
    # do K-means
    resTagLst, resDisLst = kMeans(featMatrix, k)
    # map tags
    resTagLst = bestMap(tmpTagLst, resTagLst)
    # compute acc
    acc = np.sum(resTagLst == tmpTagLst) / featMatrix.shape[0]
    # compute sse
    sse = np.sum(resDisLst ** 2)
    print("ACC = {}, SSE = {}".format(acc, sse))
    # draw
    drawRes(5, 6, featMatrix, resTagLst, k, acc, sse)
