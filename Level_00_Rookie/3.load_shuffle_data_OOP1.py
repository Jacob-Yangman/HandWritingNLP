"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/09/30
    @Desc  : 
"""
import math
import random


class Dataset:
    """存数据"""

    def __init__(self, features, labels, batch_size):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size

    def __iter__(self):
        """去哪里取数据？"""
        loader = DataLoader(self)
        return loader  # 返回迭代器对象


class DataLoader:
    """取数据"""

    def __init__(self, dataset):
        self._index = 0
        self.dataset = dataset

        # shffule
        self.rnd_idx = [i for i in range(len(self.dataset.features))]
        random.shuffle(self.rnd_idx)

    def __next__(self):
        """如何取数据？"""
        if self._index >= len(self.dataset.features):
            raise StopIteration

        batch_rnd_idx = self.rnd_idx[self._index: self._index + self.dataset.batch_size]
        batch_features = [self.dataset.features[i] for i in batch_rnd_idx]
        batch_labels = [self.dataset.labels[i] for i in batch_rnd_idx]
        self._index += self.dataset.batch_size

        return batch_features, batch_labels


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


if __name__ == '__main__':
    path = "data/train1.txt"
    all_texts, all_labels = parse_data(path)

    epochs = 10
    batch_size = 4

    for e in range(epochs):
        print(f"epoch-{e + 1}" + "*" * 40)

        train_dataSet = Dataset(all_texts, all_labels, batch_size)
        print("")
        for batch_fea, batch_lab in train_dataSet:
            print(batch_fea)
            print(batch_lab)
