# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from typing import Tuple
from tester import predictTest
from predictor import *
from loader import loadData
import pandas as pd


def draw(X: list, Y: list,
         xLabel: str, yLabel: str, name: str, num: int, markerSize: int) -> None:
    plt.figure(num)
    plt.plot(X, Y, color="black", markersize=markerSize, marker='.')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    # plt.legend(loc='best')
    plt.title(name)
    plt.savefig("{}.png".format(name))


def drawTopK_u2u() -> None:
    _, _, userRankMat, testSet = loadData()
    topKLst = list(range(1, 335))
    sseLst = []
    for topK in topKLst:
        sse, _ = predictTest(user2user(userRankMat, topK), testSet, "")
        sseLst.append(sse)
    draw(topKLst, sseLst, "Top K", "SSE", "参数Top K与平方误差和的关系折线图(1,335)", 1, 1)
    draw(topKLst[50:], sseLst[50:], "Top K", "SSE", "参数Top K与平方误差和的关系折线图(50,335)", 2, 1)


def drawHashNumber_u2u() -> None:
    _, _, userRankMat, testSet = loadData()
    threshold = 2.5
    hashFuncNumber = range(100, 2001, 50)
    sseLst = []
    for hashFuncNum in hashFuncNumber:
        sse, _ = predictTest(
            user2user(userRankMat, topK=105, threshold=threshold,
                      minHashParas=(hashFuncNum, 0, 2 ** 32 - 1, 4294967311)), testSet, "")
        print(sse)
        sseLst.append(sse)
    draw(hashFuncNumber, sseLst, "Number of hash functions", "SSE", "hash函数数量与平方误差和的关系折线图", 1, 10)


def drawHashNumber_i2i() -> None:
    _, movieTagMat, userRankMat, testSet = loadData()
    sseLst = []
    for hashFuncNum in range(1, 21):
        sse, _ = predictTest(
            item2item(userRankMat, movieTagMat, topK=20000,
                      minHashParas=(hashFuncNum, 0, 2 ** 32 - 1, 4294967311)),
            testSet, "")
        print(sse)
        sseLst.append(sse)
    draw(list(range(1, 21)), sseLst, "Number of hash functions", "SSE", "hash函数数量与平方误差和的关系折线图", 1, 10)


def drawTopK_i2i() -> None:
    _, movieTagMat, userRankMat, testSet = loadData()
    # topKLst = list(range(1, 101))
    topKLst = list(range(1, 1301, 20))
    sseLst = []
    for topK in topKLst:
        sse, _ = predictTest(
            item2item(userRankMat, movieTagMat, topK=topK),
            testSet, "")
        print(sse)
        sseLst.append(sse)
    draw(topKLst, sseLst, "Top K", "SSE", "参数Top K与平方误差和的关系折线图(1,1300)", 1, 1)
    # draw(topKLst[50:], sseLst[50:], "Top K", "SSE", "参数Top K与平方误差和的关系折线图(50,335)", 2, 1)


def drawCmp() -> None:
    df1 = pd.read_csv("./res/rec1")
    df2 = pd.read_csv("./res/rec3")
    df2 = df2[df2.columns[:-1]]
    # res = pd.read_csv("./res/recommendRes")
    res = pd.merge(df1, df2, on="class")
    print(res)

    x = res["class"]
    y1 = res["i2i"]
    y2 = res["u2u"]
    y3 = res["real"]
    bar_width = 0.3
    x1 = list(range(len(x)))
    x2 = [i + bar_width for i in x1]
    x3 = [i + bar_width * 2 for i in x1]

    rect_1 = plt.bar(x1, y1, width=0.3, label='基于内容', color='blue')
    rect_2 = plt.bar(x2, y2, width=0.3, label='基于用户', color='green')
    rect_3 = plt.bar(x3, y3, width=0.3, label='用户喜爱', color='red')

    plt.xticks(x2, x)
    plt.xlabel("类别")
    plt.ylabel("词频")

    plt.gcf().autofmt_xdate()
    plt.legend(loc='best')
    plt.savefig("recommendRes0.png")
