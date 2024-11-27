"""
encoding: utf-8
@Author: Jacob Y
@Date  : 11/26/2024
@Desc  : 基于PyTorch的Dataset和DataLoader和手写Model实现批量Word2vec训练过程
        max(batch_size) = 1
"""
import os
import pickle
import random
from datetime import datetime

import numpy as np
from tqdm import tqdm

random.seed(1126)
from torch.utils.data import Dataset, DataLoader
import torch
import jieba
import logging

logging.getLogger('jieba').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

OutPutDir = r"./output"
# 创建词向量和词表输出路径，如果已存在则跳过
os.makedirs(OutPutDir, exist_ok=True)

MODEL = "skip-gram"


class LegendofConderHeroes(Dataset):
    def __init__(self, dataPath, sample, stopWordsPath, step, nGram, negNum):
        self.rawData = self.readData(dataPath, sample)
        self.stopWords = set(
            self.readData(stopWordsPath) + ['\u3000', '，', '。', '；', '------------------------------------------'])
        self.cutData = self.parseData()
        self.words2Idx, self.idx2Words = self.tokenize()
        self.triples = self.structTriples(step, nGram, negNum)


    def readData(self, path, sample=None):
        try:
            with open(path, "r", encoding="gbk") as f:
                content = f.read().replace(' ', '').replace("\t", '').splitlines()

        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().replace(' ', '').replace("\t", '').splitlines()
        return content[:sample] if sample else content

    def parseData(self):
        newData = list()
        for line in self.rawData:
            lineCut1 = jieba.lcut(line)
            lineCut2 = [w for w in lineCut1 if w not in self.stopWords]
            newData.append(lineCut2)
        return newData

    def tokenize(self):
        words2Idx = dict(UNK=0)
        for lineWords in self.cutData:
            for word in lineWords:
                words2Idx[word] = words2Idx.get(word, len(words2Idx))

        return words2Idx, list(words2Idx)


    def negRandomSample(self, centerID, neighborsID, negNum):
        # sampleRange = set(self.idx2Words) ^ set(neighbors + [self.word2Idx[centerID]])
        sampleRange = set(self.words2Idx.values()) ^ set([centerID] + neighborsID)

        negativesID = random.sample(sampleRange, negNum)
        # negativesID = [self.word2Idx[w] for w in negatives]
        negSamples = [(centerID, negID, 0) for negID in negativesID]
        return negSamples

    def structTriples(self, step, nGram, negNum):
        triples = list()
        for lineWords in tqdm(self.cutData, desc="Generating triples..."):
            for curr in range(0, len(lineWords), step):
                centerID = self.words2Idx[lineWords[curr]]
                neighbors = (lineWords[max(curr - nGram, 0): curr]
                             + lineWords[curr + 1: curr + nGram + 1])
                neighborsID = [self.words2Idx[w] for w in neighbors]
                triples.extend([(centerID, neiID, 1) for neiID in neighborsID])

                negaSamples = self.negRandomSample(centerID, neighborsID, negNum)
                triples.extend(negaSamples)
        return triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, item):
        currTriple = self.triples[item]

        return currTriple


class Model:
    def __init__(self, wordsNum, embeddingDim, lr):
        self.W1 = np.random.normal(size=(wordsNum, embeddingDim))
        self.W2 = np.random.normal(size=(embeddingDim, wordsNum))
        self.lr = lr

    def sigmoid(self, data):
        data = np.clip(data, -20, 20)
        return 1 / (1 + np.exp(-data))

    def forward(self, centerID, otherID, label):
        pred = self.W1[centerID] @ self.W2[:, otherID]
        p = self.sigmoid(pred)

        loss = -np.mean(label * np.log(p) +
                        (1 - label) * np.log(1 - p))
        self.p = p
        self.label = label
        self.centerID = centerID
        self.otherID = otherID

        return loss

    def backward(self):
        G = self.p - self.label                 # (1, 1)
        deltaW2 = self.W1[self.centerID].T @ G        # (embDim, 1)
        deltaW1 = G @ self.W2[:, self.otherID].T            # (1, embDim)

        self.W2[:, self.otherID] -= deltaW2 * self.lr
        self.W1[self.centerID] -= deltaW1 * self.lr

    def saveEmbs(self, words2Idx):
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
    dataPath = r"./data/金庸-射雕英雄传txt精校版.txt"
    stopWordsPath = r"./data/hit_stopwords.txt"
    dataset = LegendofConderHeroes(dataPath, sample=500, stopWordsPath=stopWordsPath, step=1, nGram=3, negNum=8)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    embeddingDim = 128
    wordsNum = len(dataset.words2Idx)

    lr = 0.1
    model = Model(wordsNum, embeddingDim, lr=lr)


    epoch = 3
    for e in range(epoch):
        print("\n" + f"Epoch >> {e + 1}".center(80, "-"))
        for centerWord, otherWord, label in tqdm(dataloader, desc="Training..."):
            centerWord, otherWord, label = centerWord.numpy(), otherWord.numpy(), label.numpy()
            loss = model.forward(centerWord, otherWord, label)
            model.backward()
        print(f"Loss >> {loss:.5f}")

    model.saveEmbs(dataset.words2Idx)