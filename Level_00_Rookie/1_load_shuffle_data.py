"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/09/30
    @Desc  : 数据打乱的思路（采用第二种）
    1. 准备epoch个训练集，确保每个训练集中样本顺序互不相同，每个epoch都读取不同的训练集
    【看似比较蠢，但也是一种训练策略（在不同的训练集中人为地添加数据扰动，增强模型鲁棒性）】
    2. 生成全局索引后，使用random模块打乱索引，再按批依次获取batch_idx
"""
import math
import random
import numpy as np


def read_data(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield line.strip()


def parse_data(path):
    all_texts = []
    all_labels = []
    for line in read_data(path):
        splitted = line.split('\t')
        if len(splitted) != 2:
            continue
        text, label = splitted

        try:
            label = int(label)
            all_texts.append(text)
            all_labels.append(label)
        except (TypeError, ValueError):
            print("标签错误!")
    assert (len(all_texts) == len(all_labels)), "数据没对齐"
    return all_texts, all_labels


def main(features, labels):
    epochs = 5
    batch_size = 2

    # batch_num = len(features) // batch_size + 1    # 可能会多加一次
    # ** 保证整除情况下batch_num不会加1
    batch_num = math.ceil(len(features) / batch_size)
    for e in range(epochs):
        print(f"epoch-{e + 1}" + "*" * 40)

        # shuffle
        rnd_idx = [i for i in range(len(features))]
        random.shuffle(rnd_idx)

        for batch_idx in range(batch_num):
            rnd_batch_idx = rnd_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_fea = [features[i] for i in rnd_batch_idx]
            batch_lab = [labels[i] for i in rnd_batch_idx]
            print(batch_fea)
            print(batch_lab)


if __name__ == '__main__':
    path = "data/train1.txt"
    all_texts, all_labels = parse_data(path)
    main(all_texts, all_labels)
