"""
encoding: utf-8
@Author: Jacob Y
@Date  : 11/23/2024
@Desc  : 以batch_size为单位，每次送入多句的currWordsOneHot和neighborsOneHot拼成的矩阵
"""
import os
import random

import jieba
import numpy as np
from tqdm import tqdm
import pickle
from datetime import datetime
import logging

logging.getLogger('jieba').setLevel(logging.ERROR)

OutPutDir = r"./output"
# 创建词向量和词表输出路径，如果已存在则跳过
os.makedirs(OutPutDir, exist_ok=True)

MODEL = "cbow"  # 或skip-gram


def mysoftmax(x):
    # xMax = np.max(x, axis=-1, keepdims=True)
    # x = x - xMax
    ex = np.exp(x)
    exSum = np.sum(ex, axis=-1, keepdims=True)
    return ex / exSum


def loadStopWords(filepath):
    with open(filepath, encoding="utf-8") as f:
        return f.readlines()


def parseData(filepath, sample=None):
    global stopWords
    with open(filepath, encoding="gbk") as f:
        allWords = list()
        for i, line in tqdm(enumerate(f, start=1), desc="Loading Data..."):
            lineCut = jieba.lcut(line)
            lineCut = [w for w in lineCut if w not in stopWords]
            if lineCut: allWords.append(lineCut)

    return allWords


def getWords2Idx(allWords):
    words2idx = dict(UNK=0)
    for lineWords in allWords:
        for word in lineWords:
            words2idx[word] = words2idx.get(word, len(words2idx))
    return words2idx, list(words2idx)


class MyDataset:
    def __init__(self, allWords):
        self.allWords = allWords

    def __len__(self):
        return len(self.allWords)

    def __getitem__(self, item):
        currLineWords = self.allWords[item]
        currLineWordsOneHot, currNeighborsOneHot = [], []
        for currId, currWord in enumerate(currLineWords):

            currOneHot = wordsOneHotMtx[words2Idx.get(currWord)]
            currOneHot = np.array(currOneHot).reshape(1, -1)

            neighbors = getNeighbors(currLineWords, currId)
            neighborsIdx = [words2Idx.get(w) for w in neighbors]
            neighborsOneHot = wordsOneHotMtx[neighborsIdx]
            # 将当前词OneHot向量复制len(neighbors)份
            currOneHot = currOneHot.repeat(len(neighbors), 0)

            currLineWordsOneHot.append(currOneHot)
            currNeighborsOneHot.append(neighborsOneHot)
        currLineWordsOH = np.vstack(currNeighborsOneHot)
        currNeighborsOH = np.vstack(currNeighborsOneHot)

        return currLineWordsOH, currNeighborsOH


class MyDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = [i for i in range(len(self.dataset))]


    def __iter__(self):
        self._index = 0
        if self.shuffle:
            random.shuffle(self.idx)

        return self

    def __next__(self):
        if self._index > len(self.dataset):
            raise StopIteration

        batch_idx = self.idx[self._index : self._index + self.batch_size]

        batch_data = [self.dataset[i] for i in batch_idx]

        batchCurrWordsOH, batchNeighborsOH = zip(*batch_data)

        batchCurrWordsOH = np.array(batchCurrWordsOH)
        batchNeighborsOH = np.array(batchNeighborsOH)

        # batchCurrWordsOHMean = np.mean(batchCurrWordsOH, axis=1)
        # batchNeighborsOHMean = np.mean(batchNeighborsOH, axis=1)

        self._index += self.batch_size

        return batchCurrWordsOH, batchNeighborsOH




class MyModel():
    global wordsNum, embLen

    def __init__(self):
        self.W1 = np.random.normal(size=(wordsNum, embLen))
        self.W2 = np.random.normal(size=(embLen, wordsNum))
        self.softmax = mysoftmax

    def forward(self, x, y):
        """
        :param x: (bs, n, wordsNum)
        :param y: (bs, n, wordsNum)
        :return: loss
        """
        global lr
        self.x = x
        self.y = np.mean(y, axis=1)
        self.height = self.x.shape[1]
        self.hiddenLayer = x @ self.W1  # (bs, n, embLen)
        predict = self.hiddenLayer @ self.W2  # (bs, n, wordsNum)
        self.predMean = np.mean(predict, axis=1)  # (bs, wordsNum)
        self.proba = self.softmax(self.predMean)  # (bs, wordsNum)

        epsilon = 1e-10
        loss = -np.mean(y * np.log(self.proba + epsilon))

        return loss

    def backward(self):
        G = deltaPred = self.proba - self.y  # (bs, wordsNum)

        dpred = np.tile(np.expand_dims(G, axis=1), (1, self.height, 1))     # (bs, n, wordsNum)
        deltaW2 = self.hiddenLayer.reshape(-1, self.hiddenLayer.shape[-1]).T @ dpred.reshape(-1, dpred.shape[-1])  # (embLen, wordsNum)
        deltaHidden = dpred @ self.W2.T  # (n, embLen)
        deltaW1 = self.x.reshape(-1, self.x.shape[-1]).T @ deltaHidden.reshape(-1, deltaHidden.shape[-1])  # (wordsNum, embLen)

        self.W1 -= lr * deltaW1
        self.W2 -= lr * deltaW2


def getNeighbors(currLine, currID):
    global nGrams
    leftN = currLine[max(0, currID - nGrams): currID]
    rightN = currLine[currID + 1: currID + nGrams + 1]
    return leftN + rightN


def saveEmbs():
    # word2vec版本
    scriptName = os.path.splitext(os.path.basename(__file__))[0]
    # 获取当前时间并格式化
    currentTime = datetime.now().strftime("%m%d_%H%M")
    # 组合文件名
    fileName = f"{scriptName}_{currentTime}.pkl"
    # 规范化路径
    filePath = os.path.join(OutPutDir, fileName)
    filePath = os.path.normpath(filePath)
    with open(filePath, "wb") as file:
        pickle.dump((model.W1, words2Idx), file)


if __name__ == '__main__':

    # all words without stopwords
    stopWords = loadStopWords("./data/hit_stopwords.txt")
    stopWords.extend(["。", "，", "；", "（", "）"])
    allWords = parseData("./data/数学原始数据.csv", sample=100)

    # words2idx, idx2words
    words2Idx, idx2Words = getWords2Idx(allWords)

    # hyper params
    lr = 0.1
    epoch = 3
    batch_size = 1

    step = 3
    nGrams = 12

    wordsNum = len(words2Idx)
    embLen = 200

    # global OneHot Matrix
    wordsOneHotMtx = np.eye(wordsNum)

    myDataset = MyDataset(allWords)
    myDataLoader = MyDataLoader(myDataset, batch_size=batch_size, shuffle=False)


    model = MyModel()

    for e in range(epoch):
        print("\n" + f"Epoch No.{e + 1}".center(60, "-"))
        for currWordsOneHot, neighborsOneHot in tqdm(myDataLoader):
            if MODEL == "cbow":
                loss = model.forward(neighborsOneHot, currWordsOneHot)
            elif MODEL == "skip-gram":
                loss = model.forward(currWordsOneHot, neighborsOneHot)
            model.backward()


        print(f"\nLoss >> {loss:.4f}")
    saveEmbs()
