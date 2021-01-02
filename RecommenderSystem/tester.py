# -*- coding:utf-8 -*-
from typing import List, Tuple

from predictor import *
from loader import loadData
from time import time


def predictTest(predictor: Predictor, testSet: List[Tuple[int, int, float]], resFile: str) \
        -> (float, List[Tuple[int, float, float]]):
    # 预测器通用测试函数
    sse: float = 0.0
    results: List[Tuple[int, float, float]] = []
    for testCase in testSet:
        uid, mid, expectedRating = testCase
        actualRating = predictor.doPredict(uid, mid)
        results.append((uid, expectedRating, actualRating))
        sse += (expectedRating - actualRating) ** 2

    if resFile != "":
        with open(resFile, "w") as f:
            for res in results:
                f.write("user{} expected: {:0.1f} actual: {:0.2f}\n".format(*res))
            f.write("SSE = {:0.2f}\n".format(sse))

    return sse, results


def testUser2User(rankMat: np.ndarray, testCases: List[Tuple[int, int, float]]) -> None:
    topK: int = 94
    hashFuncNum: int = 50
    threshold: float = 2.5
    resFile: str = "./res/user2user"
    # test basic user-to-user
    print("test user2user:")
    begin = time()
    sse, _ = predictTest(user2user(rankMat, topK), testCases, resFile)
    end = time()
    print("sse = {}".format(sse))
    print("time usage: {}\n".format(end - begin))
    # test user-to-user with minHash
    print("test user2user with minhash:")
    begin = time()
    sse, _ = predictTest(user2user(rankMat, topK, threshold=threshold, minHashParas=(hashFuncNum, 0, 2048, 2333)),
                         testCases, resFile + "_minhash")
    end = time()
    print("sse = {}".format(sse))
    print("time usage: {}\n".format(end - begin))


def testItem2Item(rankMat: np.ndarray, movieTagMat: np.ndarray, testCases: List[Tuple[int, int, float]]) -> None:
    hashFuncNum: int = 12
    topK: int = 20000  # 这里给一个很大的值是为了保证是任务书中要求实现的，而不是老师推荐的优化方式
    resFile: str = "./res/item2item"
    # test basic item-to-item
    print("test item2item:")
    begin = time()
    sse, _ = predictTest(item2item(rankMat, movieTagMat, topK), testCases, resFile)
    end = time()
    print("sse = {}".format(sse))
    print("time usage: {}\n".format(end - begin))
    # test item-to-item with minHash
    print("test item2item with minhash:")
    begin = time()
    sse, _ = predictTest(item2item(rankMat, movieTagMat, topK, minHashParas=(hashFuncNum, 0, 2048, 2333)), testCases,
                         resFile + "_minhash")
    end = time()
    print("sse = {}".format(sse))
    print("time usage: {}\n".format(end - begin))


def doTest() -> None:
    # 总体测试
    begin = time()
    movies, movieTagMat, userRankMat, testSet = loadData()
    testUser2User(userRankMat, testSet)
    testItem2Item(userRankMat, movieTagMat, testSet)
    end = time()
    print("total time usage: {}\n".format(end - begin))
