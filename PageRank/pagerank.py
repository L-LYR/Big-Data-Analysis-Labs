#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import *


def distance(x: np.ndarray, y: np.ndarray) -> float:
    """欧氏距离"""
    return np.sqrt(np.sum((x - y) ** 2))
    # return np.sqrt(np.sum(np.abs(x - y)))


def page_rank(adj_matrix: np.ndarray, eps: float, beta: float) -> np.ndarray:
    """
    page_rank 主函数
    :param adj_matrix: column stochastic matrix 随机转移矩阵
    :param eps: 误差限
    :param beta: 随机跳转概率
                With probability \beta,  follow a link at random
                With probability 1-\beta, jump to some random page
    :return: rank vector
    """
    tp_vec = np.array([1 / adj_matrix.shape[0]] * adj_matrix.shape[0])
    iter_vec = tp_vec
    iter_idx = 0
    cur_eps = 1

    while cur_eps > eps:
        last = iter_vec
        # 迭代
        iter_vec = beta * np.dot(adj_matrix, last) + (1 - beta) * tp_vec
        # 归一化
        iter_vec = iter_vec / np.sum(iter_vec)
        # 计算误差
        iter_idx += 1
        cur_eps = distance(last, iter_vec)
        print("{0} iteration: {1}".format(iter_idx, cur_eps))

    return iter_vec


def get_name(filename: str) -> Dict:
    """
    该函数将真实姓名与ID进行映射
    :param filename: 姓名与ID对应的csv文件路径
    :return: 字典 {name:str -> ID:int}
    """
    name_dict = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            id, name = map(str, line.strip().split(','))
            name_dict[int(id)] = name
    return name_dict


def read_csv(filename: str) -> (Set, Dict, Dict):
    """
    读取数据集文件，并进行ID的正反映射
    :param filename: 数据集文件路径
    :return: (边集, 正向ID映射, 反向ID映射)
        边集即为本数据集中的全部有向边构成的集合；
        正向ID映射即为真实ID映射到一个从0开始的连续ID空间，方便与转移矩阵的下标对应
        反向ID映射即为正向ID映射的逆，方便输出结果
    """
    edges = set()
    id_cnt = 0
    id_dict = {}
    rev_id_dict = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            _, sent_id, recv_id = map(int, line.strip().split(','))
            if sent_id not in id_dict:
                id_dict[sent_id] = id_cnt
                rev_id_dict[id_cnt] = sent_id
                id_cnt += 1
            if recv_id not in id_dict:
                id_dict[recv_id] = id_cnt
                rev_id_dict[id_cnt] = recv_id
                id_cnt += 1
            edges.add((id_dict[sent_id], id_dict[recv_id]))
    return edges, id_dict, rev_id_dict


def gen_adj_matrix(edges: Set, dim: int) -> np.ndarray:
    """
    生成初始的转移矩阵
    :param edges: 边集
    :param dim: 转移矩阵最大维度
    :return: 转移矩阵
    """
    adj_mat = np.zeros((dim, dim))
    out_deg = np.zeros(dim)
    for i, e in enumerate(edges):
        if adj_mat[e[1]][e[0]] == 0:
            adj_mat[e[1]][e[0]] = 1.0
            out_deg[e[0]] += 1.0
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[0]):
            if out_deg[j] > 0:
                adj_mat[i][j] /= out_deg[j]
    return adj_mat


if __name__ == "__main__":
    # load data set
    name_file = "./src/Persons.csv"
    name_dict = get_name(name_file)
    input_file = "./src/sent_receive.csv"
    es, id_dict, rev_id_dict = read_csv(input_file)
    # get stochastic matrix and do page rank
    adj_mat = gen_adj_matrix(es, len(id_dict))
    res_vec = page_rank(adj_mat, 1e-8, 0.85)
    # sort and output
    res = [(i, val) for i, val in enumerate(res_vec)]
    sorted(res, key=lambda x: x[1])
    result_file = open("./res/rank.csv", "w")
    for i, val in sorted(res, key=lambda x: x[1], reverse=True):
        result_file.write("{0}, {1}\n".format(name_dict[rev_id_dict[i]], val))
    result_file.close()
