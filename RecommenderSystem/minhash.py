# -*- coding: utf-8 -*-

import random
from typing import Callable, List, Tuple, Set
import numpy as np

minLowerBound = 0
maxUpperBound = 2 ** 32 - 1
aPrime = 4294967311


class minHash(object):
    def __init__(self, k: int, lowerBound: int = minLowerBound,
                 upperBound: int = maxUpperBound, nextPrime: int = aPrime) -> None:
        self.lb = lowerBound
        self.ub = upperBound
        self.c = nextPrime
        self.n = k
        self.coe = self.__genRandCoefficients(k)

    def __genRandCoefficients(self, k: int) -> List[Tuple[int, int]]:
        # 生成k个不同的随机数
        def genRandParaLst(k: int) -> List[int]:
            res: Set = set()
            while k > 0:
                curRandNum = random.randint(self.lb, self.ub)
                while curRandNum in res:
                    curRandNum = random.randint(self.lb, self.ub)
                res.add(curRandNum)
                k -= 1
            return list(res)

        # 生成k组不同的参数值
        aLst = genRandParaLst(k)
        bLst = genRandParaLst(k)
        return [(a, b) for a, b in zip(aLst, bLst)]

    # def genTmpSignatures(self, effMat: np.ndarray) -> np.ndarray:
    #     sigs: List[List[int]] = []
    #     idxMat = [list(np.nonzero(row > 0)[0]) for row in effMat]
    #     for row in idxMat:
    #         sig: List[int] = []
    #         for a, b in self.coe:
    #             hashFunc: Callable[[int], int] = lambda x: (a * x + b) % self.c
    #             sig.append(sorted(map(hashFunc, row))[0])
    #         sigs.append(sig)
    #     return np.array(sigs)

    def genSignatures(self, effMat: np.ndarray) -> np.ndarray:
        sigs: List[List[int]] = []
        idxMat = [np.nonzero(row > 0)[0] for row in effMat]
        for a, b in self.coe:
            hashFunc: Callable[[int], int] = lambda x: (a * x + b) % self.c
            # 采用numpy的ufunc进行加速
            doHash = np.frompyfunc(hashFunc, 1, 1)
            # 进行signature的生成
            sig = [doHash(row).min() for row in idxMat]
            sigs.append(sig)
        # 这里必须转置，每次生成的是同一个hash函数的一列
        return np.array(sigs).transpose()
