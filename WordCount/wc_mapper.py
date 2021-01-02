#!/usr/bin/python3
# -*- encoding=UTF-8 -*-

from time import time
from collections import defaultdict
import multiprocessing

# 用来控制输出到控制台的信息
# python的多进程这里需要保护一下
# 不然有时候会导致输出冲突（小概率）
from typing import List

output_lock = multiprocessing.Lock()


def basic_mapper(input_files: List[str], ID: int) -> str:
    """
    基础部分的 mapper ，仅输出 "单词<\t>1" 到单独的文件中
    :param input_files: 分配给该 mapper 的输入文件名列表
    :param ID: 该 mapper （进程）的名字
    :return: 返回该 mapper 输出文件名
    """
    start_time = time()

    tar_file = "./basic_map_res/part_" + str(ID)
    tar = open(tar_file, "w")

    for in_file in input_files:
        with open(in_file, "r", errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                words = line.strip().split(", ")
                for word in words:
                    tar.write("{0}\t{1}\n".format(word, 1))

    tar.close()
    end_time = time()
    output_lock.acquire()
    print("map: process{0} running time: {1}".format(ID, end_time - start_time))
    output_lock.release()
    return tar_file


def shuffle(word_list: list, output_num: int, ID: int) -> List[str]:
    """
    单机（单进程）简单合并函数
    :param word_list: 本机（进程） mapper 的结果
    :param output_num: shuffle 阶段输出文件个数
    :param ID: 本机（进程）的名字
    :return: 返回 shuffle 阶段输出结果文件名
    """
    comb = defaultdict(int)
    for word in word_list:
        cur, cnt = word.split('\t')
        comb[cur] += int(cnt)

    tar_files_name = []
    for i in range(output_num):
        tar_files_name.append("./advanced_map_res/shuffled_part_" + str(ID) + "_" + str(i))

    tar_files = []
    for file in tar_files_name:
        tar_files.append(open(file, "w"))

    # 将 word 进行 hash 并分配到shuffle的其中一个结果文件中
    for (k, v) in comb.items():
        tar_files[hash(k) % output_num].write("{0}\t{1}\n".format(k, v))

    for file in tar_files:
        file.close()
    return tar_files_name


def advanced_mapper(input_files: List[str], output_num: int, ID: int) -> List[str]:
    """
    进阶版本的 mapper , 前半部分和基础版本一样，输出 "单词<\t>1"，之后调用了一次 combine 函数。
    :param input_files: 分配给该 mapper 的输入文件名列表
    :param output_num: 在 map 阶段之后进行 shuffle 的结果输出个数
    :param ID: 该 mapper （进程）的ID
    :return: 返回 shuffle 输出结果文件名列表
    """
    start_time = time()

    word_list = []
    for in_file in input_files:
        with open(in_file, "r", errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                words = line.strip().split(", ")
                for word in words:
                    word_list.append("{0}\t{1}".format(word, 1))

    tar_files_name = shuffle(word_list, output_num, ID)

    end_time = time()
    output_lock.acquire()
    print("map: process{0} running time: {1}".format(ID, end_time - start_time))
    output_lock.release()
    return tar_files_name
