"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/09/30
    @Desc  : 
"""
import math
import random
from torch.utils.data import Dataset, DataLoader


class DataSetIterator:
    def __init__(self, dataset, batch_size):
        self._index = 0
        self.dataset = dataset
        self.batch_size = batch_size

        # shuffle
        self.rdm_idx = [i for i in range(len(self.dataset.features))]
        random.shuffle(self.rdm_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index > len(self.dataset.features):
            raise StopIteration

        batch_rdm_idx = self.rdm_idx[self._index: self._index + self.dataset.batch_size]

        batch_feature = [self.dataset.features[i] for i in batch_rdm_idx]
        batch_label = [self.dataset.labels[i] for i in batch_rdm_idx]

        self._index += self.dataset.batch_size

        return batch_feature, batch_label


class DataSet:
    def __init__(self, features, labels, batch_size):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size

    def __iter__(self):
        return DataSetIterator(self, self.batch_size)


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
    epochs = 10
    batch_size = 4

    # batch_num = len(features) // batch_size + 1    # 可能会多加一次
    batch_num = math.ceil(len(features) / batch_size)
    for e in range(epochs):
        print(f"epoch-{e + 1}" + "*" * 40)

        train_dataSet = DataSet(features, labels, batch_size)
        train_dataLoader = DataSetIterator(train_dataSet, batch_size)

        for batch_fea, batch_lab in train_dataSet:
            # rnd_batch_idx = rnd_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            # batch_fea = [features[i] for i in rnd_batch_idx]
            # batch_lab = [labels[i] for i in rnd_batch_idx]
            print(batch_fea)
            print(batch_lab)


if __name__ == '__main__':
    path = "data/train1.txt"
    all_texts, all_labels = parse_data(path)
    main(all_texts, all_labels)
