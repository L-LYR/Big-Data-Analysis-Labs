#!/usr/bin/python3
# -*- coding: utf-8 -*-

from tester import *
from draw import *
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
    user2userPredictor = user2user(userRankMat, topK=105)
    item2itemPredictor = item2item(userRankMat, movieTagMat, topK=20)

    # do test
    # _, results = predictTest(user2userPredictor, testCases, "")
    # _, results = predictTest(item2itemPredictor, testCases, "")
    # userAvgSSE = defaultdict(float)
    # for res in results:
    #     userAvgSSE[res[0]] += (res[2] - res[1]) ** 2
    # sse = list(userAvgSSE.items())
    # sse.sort(key=lambda x: x[1])
    # # best-fit user
    # uid, minSSE = sse[0]
    # print("(uid, smallest SSE): ({}, {})".format(uid, minSSE))
    uid = 480
    # do recommend
    # 使用不同的推荐系统进行结果对比
    # recommender = Recommender(movieTagMat, userRankMat, movies, user2userPredictor)
    recommender = Recommender(movieTagMat, userRankMat, movies, item2itemPredictor)
    recommendMovies = recommender.doRecommend(uid, 50)["recommended_movies"]
    print("recommended movies:")
    recommendedCategory = defaultdict(int)
    for m, r in recommendMovies.items():
        for genre in movies[r[0]].genres:
            recommendedCategory[genre] += 1
    for k, v in sorted(recommendedCategory.items(), key=lambda d: d[0], reverse=True):
        print("{}: {}".format(k, v))
    print("")
    # compare
    print("His or her favorite movies:")
    userRank = userRankMat[uid]
    idx = np.argsort(-userRank)[:50]
    userLikeCategory = defaultdict(int)
    for i in idx:
        for genre in movies[i].genres:
            userLikeCategory[genre] += 1
    for k, v in sorted(userLikeCategory.items(), key=lambda d: d[0], reverse=True):
        print("{}: {}".format(k, v))

    print("")
    for k, v in recommendedCategory.items():
        if k in userLikeCategory:
            print("{},{},{}".format(k, v, userLikeCategory[k]))
        else:
            print("{},{},0".format(k, v))
    for k, v in userLikeCategory.items():
        if k not in recommendedCategory:
            print("{},0,{}".format(k, v))


if __name__ == "__main__":
    # doTest()
    # doRecommender()
    # analyzeBestFitUser()
    # drawTopK_i2i()
    # drawHashNumber_i2i()
    drawCmp()
    pass
