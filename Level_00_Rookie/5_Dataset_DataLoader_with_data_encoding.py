"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/09/30
    @Desc  :
    1. 对所有样本和标签OneHot编码
    2. 为Dataset类添加__getitem__()方法，改变dataset的索引方法
    2. 限制特征向量长度，执行相应的文本截断和填充
    3. 最终返回由batch_size条样本拼接成的二维ndarray
"""
import numpy as np
import random


def read_data(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield line.strip()


def parse_data(path):
    all_texts = []
    all_labels = []
    for line in read_data(path):
        splitted = line.split()
        if len(splitted) != 2:
            continue
        text, label = splitted

        # try:
        #     label = int(label)
        #     all_texts.append(text)
        #     all_labels.append(label)
        # except (TypeError, ValueError):
        #     print("标签错误!")
        all_texts.append(text)
        all_labels.append(label)
    assert (len(all_texts) == len(all_labels)), "数据没对齐"
    return all_texts, all_labels


def build_text2index(all_text):
    txt2idx = dict()
    txt2idx['PAD'] = 0
    for text in all_text:
        for c in text:
            txt2idx[c] = txt2idx.get(c, len(txt2idx))
    return txt2idx


def build_label2index(all_label):
    lab2idx = dict()
    for idx, label in enumerate(set(all_label)):
        lab2idx.update([(label, idx)])

    return lab2idx


class Dataset:
    """存数据"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __iter__(self):
        """去哪里取数据？"""
        loader = DataLoader(self, batch_size)
        return loader  # 返回迭代器对象

    def __getitem__(self, item):
        text = self.features[item][:text_limit]  # 样本截断
        label = self.labels[item]

        # 样本和标签编码
        text = [text2index.get(c) for c in text]
        label = label2index.get(label)

        # 样本填充0
        if len(text) < 10:
            text.extend([0] * (text_limit - len(text)))

        return text, label


class DataLoader:
    """取数据"""

    def __init__(self, dataset, batch_size, shuffle=True):
        self._index = 0
        self.dataset = dataset
        self.batch_size = batch_size

        # shuffle

        self.rnd_idx = [i for i in range(len(self.dataset.features))]
        if shuffle:
            random.shuffle(self.rnd_idx)

    def __iter__(self):
        return self

    def __next__(self):
        """如何取数据？"""
        if self._index >= len(self.dataset.features):
            raise StopIteration

        batch_rnd_idx = self.rnd_idx[self._index: self._index + self.batch_size]

        # batch_features = [self.dataset[i] for i in batch_rnd_idx]
        # batch_labels = [self.dataset.labels[i] for i in batch_rnd_idx]

        # [self.dataset[i] for i in batch_rnd_idx]

        # batch_features = [self.dataset[i][0] for i in batch_rnd_idx]
        # batch_labels = [self.dataset[i][1] for i in batch_rnd_idx]
        batch_data = [self.dataset[i] for i in batch_rnd_idx]

        batch_features, batch_labels = zip(*batch_data)
        batch_features = np.array(batch_features)
        batch_labels = np.array(batch_labels)

        self._index += self.batch_size

        return batch_features, batch_labels


if __name__ == '__main__':
    path = "data/train0.txt"
    all_texts, all_labels = parse_data(path)

    text2index = build_text2index(all_texts)
    label2index = build_label2index(all_labels)

    epochs = 20
    batch_size = 4
    text_limit = 15

    dataset = Dataset(all_texts, all_labels)

    for e in range(epochs):
        print(f"epoch-{e + 1}" + "*" * 40)

        dataloader = DataLoader(dataset, batch_size, shuffle=False)
        for batch_fea, batch_lab in dataloader:
            print(batch_fea)
            print(batch_lab)
