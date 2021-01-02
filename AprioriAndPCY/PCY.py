#!/usr/bin/python
# -*- encoding: utf-8 -*-

from collections import defaultdict
import Apriori as ap
import hashlib
from time import time


def hashPairs(baskets: list, Lk: set, bucketSize: int, minSupport: float, hashFunc) -> int:
    """
    将每一对hash到桶中，根据最小支持度，返回bitmap
    """
    bitmap: int = 0
    buckets = defaultdict(int)
    for basket in baskets:
        for item in Lk:
            if item.issubset(basket):
                for anotherItem in (basket - item):
                    pair = frozenset([anotherItem]) | item
                    bucketNo = hashFunc(pair) % bucketSize
                    buckets[bucketNo] += 1
    for k, v in buckets.items():
        curSupport = float(v) / len(baskets)
        if curSupport >= minSupport:
            bitmap += 1 << k
    return bitmap


def genCkByBitMap(Lk: set, bitmap: int, bucketSize: int, k: int, hashFunc) -> set:
    """
    依据bitmap获取k项候选集，即PCY优化
    """
    Ck = set()
    sz = len(Lk)
    tmpLst = list(Lk)
    for i in range(sz):
        for j in range(i + 1, sz):
            nxtItem = frozenset(tmpLst[i] | tmpLst[j])
            hashVal = hashFunc(nxtItem) % bucketSize
            if len(nxtItem) == k and (bitmap & (1 << hashVal)) != 0:
                Ck.add(nxtItem)
    return Ck


def pcy(baskets: list, minSupport: float, minConfidence: float,
        maxK: int, bucketSize: int, hashFunc) -> (list, int, defaultdict):
    C1 = ap.genC1(baskets)
    L1, sup1 = ap.genFreqSet(baskets, C1, minSupport)

    # 生成2项候选集时，进行PCY优化
    bitmap1 = hashPairs(baskets, L1, bucketSize, minSupport, hashFunc)
    Ck = genCkByBitMap(L1, bitmap1, bucketSize, 2, hashFunc)
    L = [set(), L1]
    sup = [defaultdict(float), sup1]

    k = 2
    while True:
        Lk, supk = ap.genFreqSet(baskets, Ck, minSupport)
        L.append(Lk)
        sup.append(supk)
        if k == maxK:
            break
        k += 1
        Ck = ap.genCk(L[k - 1], k)
    return sup, bitmap1, ap.genRules(L, sup, minConfidence)


if __name__ == '__main__':
    minConf = 0.5
    minSup = 0.005
    maxk = 4
    bucketSize = 10009
    idMap, dataSet = ap.loadData("./src/Groceries.csv")
    itemBaskets = list(map(frozenset, dataSet))


    def hashFuncForPair(pair: set) -> int:
        sha1 = hashlib.sha1()
        for item in pair:
            sha1.update(idMap[item].encode("utf-8"))
            # hashVal *= hash(idMap[item])
            # hashVal += hash(idMap[item])
        return int(sha1.hexdigest(), 16)


    startTime = time()
    sups, bitmap1, rules = pcy(itemBaskets, minSup, minConf, maxk, bucketSize, hashFuncForPair)
    endTime = time()
    print("Time cost: {}".format(endTime - startTime))

    with open("./res/pcy_result", "w")as f:
        for i in range(maxk + 1):
            f.write("{}-set size: {}\n".format(i, len(sups[i])))
        f.write("number of rules: {}\n".format(len(rules)))
        f.write("number of buckets: {}\n".format(bucketSize))
        f.write("bitmap for L2: {:0^{}b}\n".format(bitmap1, bucketSize))
        for sup in sups:
            for k, v in dict(sup).items():
                f.write("{}: {}\n".format(set(map(idMap.get, k)), v))
        for k, v in dict(rules).items():
            f.write("{} => {}: {}\n".format(set(map(idMap.get, k[0])),
                                            set(map(idMap.get, k[1])), v))
