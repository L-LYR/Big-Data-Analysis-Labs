#!/usr/bin/python3
# -*- coding: utf-8 -*-

from tester import *
from loader import loadData
from recommender import Recommender
import json
from collections import defaultdict


def doRecommender() -> None:
    """
    对所有用户进行推荐，输出到recommend文件夹中，结果以json文件的格式保存
    """
    begin = time()
    movies, movieTagMat, userRankMat, _ = loadData()
    predictor = item2item(userRankMat, movieTagMat, topK=100)
    recommender = Recommender(movieTagMat, userRankMat, movies, predictor)
    resFilePrefix: str = "./recommend/user"
    for i in range(userRankMat.shape[0]):
        with open(resFilePrefix + str(i + 1) + ".json", "w") as f:
            f.write(json.dumps(recommender.doRecommend(i, 50), indent=4, separators=(',', ':')))
    end = time()
    print("total time usage: {}".format(end - begin))


def analyzeBestFitUser():
    """
    对测试集中最佳预测的用户进行深入的探究
    """
    movies, movieTagMat, userRankMat, testCases = loadData()
    user2userPredictor = user2user(userRankMat, topK=94)
    item2itemPredictor = item2item(userRankMat, movieTagMat, topK=50)

    # do test
    _, results = predictTest(user2userPredictor, testCases, "")
    userAvgSSE = defaultdict(float)
    for res in results:
        userAvgSSE[res[0]] += (res[2] - res[1]) ** 2
    sse = list(userAvgSSE.items())
    sse.sort(key=lambda x: x[1])
    # best-fit user
    uid, minSSE = sse[0]
    print("(uid, smallest SSE): ({}, {})".format(uid, minSSE))

    # do recommend
    # 使用不同的推荐系统进行结果对比
    # recommender = Recommender(movieTagMat, userRankMat, movies, user2userPredictor)
    recommender = Recommender(movieTagMat, userRankMat, movies, item2itemPredictor)
    recommendMovies = recommender.doRecommend(uid, 50)["recommended_movies"]
    print("recommended movies:")
    for m, r in recommendMovies.items():
        print(str(movies[r[0]]), r[1])
    print("")
    # compare
    print("His or her favorite movies:")
    userRank = userRankMat[uid]
    idx = np.argsort(-userRank)[:50]
    for i in idx:
        print(str(movies[i]), userRank[i])


if __name__ == "__main__":
    doTest()
    # doRecommender()
    analyzeBestFitUser()
