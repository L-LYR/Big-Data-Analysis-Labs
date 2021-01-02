# -*- coding: utf-8 -*-

import abc
from minhash import minHash
import numpy as np


class Predictor(metaclass=abc.ABCMeta):
    """预测抽象基类"""

    @abc.abstractmethod
    def doPredict(self, uid: int, mid: int) -> float:
        pass


def pccMat(dataMat: np.ndarray) -> np.ndarray:
    """
    皮尔森相似度矩阵
    """
    rowStdDev = dataMat.std(axis=1, ddof=0)
    stdDevMat = rowStdDev * rowStdDev[:, np.newaxis]
    return np.cov(dataMat) / stdDevMat


def jscMat(dataMat: np.ndarray) -> np.ndarray:
    """
    近似杰卡德相似度矩阵
    """
    n, m = dataMat.shape
    sMat = np.zeros((n, n))
    for i in range(0, n):
        sMat[i] = np.sum(np.array(dataMat[i] == dataMat, dtype=int), axis=1)
    return sMat / m


def minHashSimMat(dataMat: np.ndarray, threshold: float, **kwargs) -> np.ndarray:
    """
    针对minhash计算的相似度矩阵
    """
    effMat = np.array(dataMat > threshold, dtype=int)
    sigMat = minHash(*kwargs["minHashParas"]).genSignatures(effMat)
    return jscMat(sigMat)


def cosMat(dataMat: np.ndarray) -> np.ndarray:
    """
    余弦相似度矩阵
    """
    norm = np.linalg.norm(dataMat, axis=1, keepdims=True)
    return np.dot(dataMat, dataMat.transpose()) / (norm * norm.transpose())


def TFIDFSimMat(dataMat: np.ndarray) -> np.ndarray:
    """
    TF-IDF算法获得的相似度矩阵
    """
    N = dataMat.shape[0]
    IDF = np.log10(N / dataMat.sum(axis=0))
    TF = dataMat / dataMat.sum(axis=1, keepdims=True)
    TFIDFMat = TF * IDF
    return cosMat(TFIDFMat)


class user2user(Predictor):
    """
    用户对用户的预测器
    """

    def __init__(self, userRankMat: np.ndarray, topK: int, **kwargs) -> None:
        self.userRankMat = userRankMat
        self.k = topK
        if len(kwargs) > 0:  # 是否使用minhash
            self.simMat = minHashSimMat(self.userRankMat, kwargs["threshold"], minHashParas=kwargs["minHashParas"])
        else:
            self.simMat = pccMat(self.userRankMat)

    def doPredict(self, uid: int, mid: int) -> float:
        topKLst = np.argsort(-self.simMat[uid])[1:self.k + 1]  # 选择K个近似的用户
        simVec = self.simMat[uid, topKLst]
        rVec = self.userRankMat[topKLst, mid]
        idx = rVec > 0
        sSim = np.sum(simVec[idx])
        if sSim == 0:
            return 0
        return np.sum(simVec[idx] * rVec[idx]) / np.sum(simVec[idx])


class item2item(Predictor):
    def __init__(self, userRankMat: np.ndarray, movieTagMat: np.ndarray, topK: int, **kwargs) -> None:
        self.userRankMat = userRankMat
        self.movieTagMat = movieTagMat
        self.k = topK
        if len(kwargs) > 0:
            self.simMat = minHashSimMat(self.movieTagMat, 0, minHashParas=kwargs["minHashParas"])
        else:
            self.simMat = TFIDFSimMat(self.movieTagMat)

    def doPredict(self, uid: int, mid: int) -> float:
        idx = np.where((self.simMat[mid] > 0) & (self.userRankMat[uid] > 0))[0]
        if len(idx) > self.k:  # 选择K个近似的电影，老师推荐这样做，因为这样做强化了相似电影的分数权重
            idx = idx[range(self.k)]
        simVec = self.simMat[mid][idx]
        sSim = np.sum(simVec)
        if sSim == 0:
            return 0.0
        rVec = self.userRankMat[uid][idx]
        return np.dot(simVec, rVec) / sSim
