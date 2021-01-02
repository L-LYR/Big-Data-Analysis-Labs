#!/usr/bin/python3
# -*- encoding=UTF-8 -*-
from typing import List

from wc_mapper import *
from wc_reducer import *

import multiprocessing
import random
import os
from math import ceil
from time import time


def sim_mapreduce_basic(input_files: List[str],
                        map_task_num: int,
                        reduce_task_num: int) -> None:
    """
    基础版本 mapreduce
    :param input_files: 输入文件名列表
    :param map_task_num: mapper 数量
    :param reduce_task_num:  reducer 数量
    """
    start_time = time()
    # map
    with multiprocessing.Pool(map_task_num) as map_pool:
        map_args = [([in_file], i + 1) for i, in_file in enumerate(input_files)]
        map_results = map_pool.starmap(basic_mapper, map_args)
    # shuffle
    # just shuffle randomly in basic version
    # random.shuffle(map_results)
    # divide map_res_files into 3 groups
    reduce_task_input_num = ceil(map_task_num / reduce_task_num)
    map_results = [map_results[n:n + reduce_task_input_num]
                   for n in range(0, map_task_num, reduce_task_input_num)]
    # reduce
    with multiprocessing.Pool(reduce_task_num) as reduce_pool:
        reduce_args = [(in_files, i + 1, "./basic_reduce_res") for i, in_files in enumerate(map_results)]
        reduce_results = reduce_pool.starmap(basic_reducer, reduce_args)
    # combine
    # combination is just the same with reduce
    _ = basic_reducer(reduce_results, 0, "./basic_reduce_res")
    end_time = time()
    print("total running time: {0}".format(end_time - start_time))


def sim_mapreduce_advanced(input_files: List[str],
                           map_task_num: int,
                           reduce_task_num: int) -> None:
    """
    进阶版 mapreduce
    :param input_files: 输入文件名列表
    :param map_task_num: mapper 数量
    :param reduce_task_num: reducer 数量
    :return:
    """
    start_time = time()
    # map
    with multiprocessing.Pool(map_task_num) as map_pool:
        map_args = [([in_file], reduce_task_num, i + 1) for i, in_file in enumerate(input_files)]
        map_results = map_pool.starmap(advanced_mapper, map_args)
    map_results = [file for files in map_results for file in files]
    reduce_input_files = []
    for i in range(reduce_task_num):
        reduce_input_files.append([file for file in map_results if file.endswith(str(i))])
    # reduce
    with multiprocessing.Pool(reduce_task_num) as reduce_pool:
        reduce_args = [(in_files, i + 1, "./advanced_reduce_res") for i, in_files in enumerate(reduce_input_files)]
        reduce_results = reduce_pool.starmap(basic_reducer, reduce_args)
    # combine
    # combination is just the same with reduce
    _ = basic_reducer(reduce_results, 0, "./advanced_reduce_res")
    end_time = time()
    print("total running time: {0}".format(end_time - start_time))


if __name__ == "__main__":
    # preparation
    src_path = "./lab-data/src"
    src_prefix = "source0"
    input_files = ["/".join([src_path, src_prefix + str(i)]) for i in range(1, 10)]
    # choose task type
    task_types = ["basic", "advanced"]
    task_type = task_types[0]
    # mkdir
    if not os.path.exists(task_type + "_map_res"):
        os.mkdir(task_type + "_map_res")
    if not os.path.exists(task_type + "_reduce_res"):
        os.mkdir(task_type + "_reduce_res")
    # do task
    if task_type == "basic":
        sim_mapreduce_basic(input_files, 9, 3)
    elif task_type == "advanced":
        sim_mapreduce_advanced(input_files, 9, 3)
