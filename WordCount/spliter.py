#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re

punctuation = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'


def removePunctuation(text: str) -> str:
    text = re.sub(r'[{}]+'.format(punctuation), ' ', text)
    text = re.sub(r'[^\x20-\x7F]+', ' ', text)
    return text.strip()


if __name__ == '__main__':
    limit = 9

    parts = [[] for _ in range(9)]
    with open("./src/text.txt", "r")as f:
        i = 0
        for line in f.readlines():
            line = removePunctuation(line)
            parts[i].append(', '.join(line.split()))
            i = (i + 1) % limit

    for i, part in enumerate(parts):
        with open("./src/source0" + str(i + 1), "w") as f:
            f.writelines(part)
