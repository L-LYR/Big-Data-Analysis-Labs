#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict
from time import time


def loadData(filename: str) -> (dict, list):
    basketsLst = []
    idRevMap = {}
    idMap = {}
    idx = 0
    df = pd.read_csv(filename, encoding="utf-8")
    baskets = pd.DataFrame(df, columns=['items'])
    for _, basket in baskets.iterrows():
        items = basket.item()[1:-1].split(',')
        for item in items:
            if item not in idRevMap:
                idMap[idx] = item
                idRevMap[item] = idx
                idx += 1
        basketsLst.append([idRevMap[item] for item in items])
    return idMap, basketsLst


def genFreqSet(dataSet: list, Ck: set, minSupport: float) -> (set, defaultdict):
    # 获取k项频繁集
    Lk = set()
    cnt = defaultdict(int)
    supports = defaultdict(float)
    for txn in dataSet:
        for item in Ck:
            if item.issubset(txn):
                cnt[item] += 1
    for item, c in cnt.items():
        curSupport = float(c) / len(dataSet)
        if curSupport >= minSupport:
            Lk.add(item)
            supports[item] = curSupport
    return Lk, supports


def genC1(dataSet: list) -> set:
    # 获取一项候选集
    return set([frozenset([item]) for record in dataSet for item in record])


def genCk(Lk_1: set, k: int) -> set:
    # 获取k项候选集
    # 待优化：减少集合并集操作
    Ck = set()
    sz = len(Lk_1)
    LstLk_1 = list(Lk_1)
    for i in range(sz):
        for j in range(i + 1, sz):
            nxtItem = LstLk_1[i] | LstLk_1[j]
            if len(nxtItem) == k:
                Ck.add(frozenset(nxtItem))
    return Ck


def genRules(L: list, support: list, minConf: float) -> defaultdict:
    # 生成关联规则
    rules = defaultdict(float)
    subSets = list(L[1])
    for k in range(2, len(L)):
        for freqSet in L[k]:
            for toSet in subSets:
                if toSet.issubset(freqSet):
                    fromSet = freqSet - toSet
                    conf = support[len(freqSet)][freqSet] / support[len(fromSet)][fromSet]
                    if conf >= minConf:
                        rules[(fromSet, toSet)] = conf
            subSets.append(freqSet)
    return rules


def apriori(dataSet: list, minSupport: float, minConfidence: float, maxK: int) -> (list, defaultdict):
    dataSet = list(map(frozenset, dataSet))
    C1 = genC1(dataSet)
    L1, sup1 = genFreqSet(dataSet, C1, minSupport)

    L = [set(), L1]
    sup = [defaultdict(float), sup1]
    k = 2
    while k <= maxK:
        Ck = genCk(L[k - 1], k)
        # print(len(Ck))
        Lk, supk = genFreqSet(dataSet, Ck, minSupport)
        L.append(Lk)
        sup.append(supk)
        k += 1
    return sup, genRules(L, sup, minConfidence)


if __name__ == '__main__':
    # params
    minConf = 0.5
    minSup = 0.005
    maxk = 3
    idMap, dataLst = loadData("./src/Groceries.csv")
    # do apriori
    startTime = time()
    supports, rules = apriori(dataLst, minSup, minConf, maxk)
    endTime = time()
    print("Time Cost: {}".format(endTime - startTime))
    with open("./res/apriori_result", "w")as f:
        for i in range(maxk + 1):
            f.write("{}-set size: {}\n".format(i, len(supports[i])))
        f.write("number of rules: {}\n".format(len(rules)))
        for sup in supports:
            for k, v in dict(sup).items():
                f.write("{}: {}\n".format(set(map(idMap.get, k)), v))
        for k, v in dict(rules).items():
            f.write("{} => {}: {}\n".format(set(map(idMap.get, k[0])),
                                            set(map(idMap.get, k[1])), v))
