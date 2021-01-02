#!/usr/bin/python3
# -*- encoding=UTF-8 -*-

from collections import defaultdict
from time import time
import multiprocessing

# 见wc_mapper.py，同理
output_lock = multiprocessing.Lock()


def basic_reducer(input_files: [str], ID: int, output_prefix: str) -> str:
    start_time = time()

    word_dic = defaultdict(int)
    for in_file in input_files:
        with open(in_file, "r", errors="ignore") as f:
            for line in f.readlines():
                cur = line.split('\t')
                word_dic[cur[0]] = word_dic[cur[0]] + int(cur[1])

    tar_file = output_prefix + "/part_" + str(ID)
    with open(tar_file, "w") as tar:
        for (k, v) in sorted(word_dic.items(), key=lambda x: x[0]):
            tar.write("{0}\t{1}\n".format(k, v))

    end_time = time()
    output_lock.acquire()
    print("reduce: process{0} running time: {1}".format(ID, end_time - start_time))
    output_lock.release()
    return tar_file
