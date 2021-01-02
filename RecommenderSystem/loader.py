# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np


class movie:
    # 电影信息
    def __init__(self, id: int, name: str, genres: Set[str]):
        self.id = id
        self.name: str = name
        self.genres: Set[str] = genres

    def __str__(self):
        return "name:{} genres:{}".format(self.name, ",".join(self.genres))


def loadData() -> (Dict[int, movie], np.ndarray, np.ndarray, List[Tuple[int, int, float]]):
    # 读取文件，返回电影集合、用户评级矩阵、电影标签矩阵、测试集
    movieSrc: str = "./src/movies.csv"
    ratingSrc: str = "./src/ratings.csv"
    trainSetSrc: str = "./src/train_set.csv"
    testSetSrc: str = "./src/test_set.csv"

    def loadMovies() -> (Dict[int, movie], Dict[int, int]):
        df = pd.read_csv(movieSrc, sep=',')
        movies: Dict[int, movie] = {}
        midDict: Dict[int, int] = {}
        # 对电影的ID进行了一次重新编号，保证从0连续映射
        idx: int = 0
        for row in df.itertuples(index=False, name="movie"):
            mid, name = getattr(row, "movieId"), getattr(row, "title")
            genres = getattr(row, "genres").split('|')
            if mid not in movies:
                movies[idx] = movie(mid, name, set(genres))
                midDict[mid] = idx
                idx += 1
        return movies, midDict

    def loadMovieTrainSet(movies: Dict[int, movie]) -> np.ndarray:
        categories: Dict[str, int] = {}
        idx: int = 0
        for (_, v) in movies.items():
            for c in v.genres:
                if c not in categories:
                    categories[c] = idx
                    idx += 1
        movieTagMat = np.zeros((len(movies), idx), dtype=int)
        for (k, v) in movies.items():
            for t in v.genres:
                movieTagMat[k][categories[t]] = 1
        return movieTagMat

    def loadUserTrainSet(midDict: Dict[int, int]) -> np.ndarray:
        df = pd.read_csv(trainSetSrc, sep=',')
        adjMat = np.zeros((df["userId"].max(), len(midDict.values())), dtype=float)
        for row in df.itertuples(index=False, name="edge"):
            uid, mid, rating = \
                getattr(row, "userId") - 1, getattr(row, "movieId"), getattr(row, "rating")
            adjMat[uid][midDict[mid]] = rating
        return adjMat

    def loadTestSet(midDict: Dict[int, int]) -> List[Tuple[int, int, float]]:
        df = pd.read_csv(testSetSrc, sep=',')
        es: List[Tuple[int, int, float]] = []
        for row in df.itertuples(index=False, name='edge'):
            es.append((getattr(row, "userId") - 1,
                       midDict[getattr(row, "movieId")],
                       getattr(row, "rating")))
        return es

    movies, midDict = loadMovies()
    moviesTagMat = loadMovieTrainSet(movies)
    userRankMat = loadUserTrainSet(midDict)
    testSet = loadTestSet(midDict)
    return movies, moviesTagMat, userRankMat, testSet
