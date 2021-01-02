# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple
from predictor import Predictor
import numpy as np


class Recommender(object):
    def __init__(self, movieTagMat: np.ndarray, userRankMat: np.ndarray, movies: Dict, predictor: Predictor) -> None:
        self.movieTagMat = movieTagMat
        self.userRankMat = userRankMat
        self.movies = movies
        self.predictor = predictor

    def doRecommend(self, uid: int, topK: int) -> Dict:
        idx = set(np.nonzero(self.userRankMat[uid] > 0)[0])  # 用户看过的电影集合
        # 对用户没看过的电影进行预测
        rankPrediction: List[Tuple[int, float]] \
            = [(i, self.predictor.doPredict(uid, i))
               for i in self.movies.keys() if i not in idx]
        # 推荐前k个预测高分的电影
        rankPrediction.sort(reverse=True, key=lambda x: x[1])
        recommendedMovies = [(self.movies[mid].name, (mid, rating)) for mid, rating in rankPrediction[:topK]]
        return {
            "user_id": uid + 1,
            "recommended_movies": dict(recommendedMovies)
        }
