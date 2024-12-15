# coding=utf-8
"""
@Author: Jacob Y
@Date  : 11/26/2024
@Desc  : 基于PyTorch的Dataset和DataLoader和手写Model实现批量Word2vec训练过程
        真正实现支持batch_size>1的批量训练
"""
import os
import pickle
import random
from datetime import datetime

import numpy as np
from tqdm import tqdm

random.seed(1126)
np.random.seed(1126)
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


class LocalDataset(Dataset):
    def __init__(self, dataPath, sample, stopWordsPath, step, nGram, negNum):
        self.rawData = self.readData(dataPath, sample)
        self.stopWords = set(
            self.readData(stopWordsPath) + ['\u3000', '，', '。', '；', '------------------------------------------'])
        cutData = self.parseData()
        self.words2Idx, self.idx2Words, self.wordsIds, self.allLineIndices = self.tokenize(cutData)
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
        for line in tqdm(self.rawData,
                         desc="Cutting...",
                         colour="GREEN"):
            # if len(line) < 15:
            #     continue
            lineCut1 = jieba.lcut(line)
            lineCut2 = [w for w in lineCut1 if w not in self.stopWords]
            if lineCut2: newData.append(lineCut2)
        return newData

    def tokenize(self, cutData):
        words2Idx = dict(UNK=0)
        allLineIndices = list()
        for lineWords in cutData:
            lineIndices = list()
            for word in lineWords:
                words2Idx[word] = words2Idx.get(word, len(words2Idx))
                lineIndices.append(words2Idx[word])
            allLineIndices.append(lineIndices)
        return words2Idx, list(words2Idx), list(words2Idx.values()), allLineIndices


    def negRandomSample(self, centerID, neighborsID, negNum):
        # sampleRange = set(self.idx2Words) ^ set(neighbors + [self.word2Idx[centerID]])
        # sampleRange = set(self.words2Idx.values()) ^ set([centerID] + neighborsID)
        negativesID = set()
        while len(negativesID) < negNum:
            currNeg = random.choice(self.wordsIds)
            if currNeg not in set(neighborsID + [centerID]):
                negativesID.add(currNeg)

        negSamples = list(zip([centerID] * negNum, negativesID, [0] * negNum))
        return negSamples

    def structTriples(self, step, nGram, negNum):
        triples = list()
        for lineWordsIds in tqdm(self.allLineIndices, desc="Generating triples..."):
            # lineWordsIds = [self.words2Idx.get(w) for w in lineWords]
            for i in range(0, len(lineWordsIds), step):
                centerID = lineWordsIds[i]
                neighborsID = (lineWordsIds[max(i - nGram, 0): i]
                             + lineWordsIds[i + 1: i + nGram + 1])

                neiNum = len(neighborsID)
                triples.extend(zip([centerID] * neiNum, neighborsID, [1] * neiNum))

                negSamples = self.negRandomSample(centerID, neighborsID, negNum)

                triples.extend(negSamples)
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

        mat1 = self.W1[centerID]                # (bs, embDim)
        mat2 = self.W2[:, otherID].T            # (bs, embDim)

        # # 计算 mat1 和 mat2 每行的最大值
        # max_vals_mat1 = np.max(mat1, axis=-1, keepdims=True)
        # max_vals_mat2 = np.max(mat2, axis=-1, keepdims=True)
        #
        # # 减去每行的最大值
        # mat1_stable = mat1 - max_vals_mat1
        # mat2_stable = mat2 - max_vals_mat2
        #
        # # 计算 mat1 * mat2
        # dot_product = mat1_stable * mat2_stable

        dotProduct = mat1 * mat2

        pred = np.sum(dotProduct, axis=-1)     # (bs, )

        p = self.sigmoid(pred)                  # (bs, )

        loss = -np.sum(label * np.log(p) +
                        (1 - label) * np.log(1 - p))

        self.p = p
        self.label = label
        self.centerID = centerID
        self.otherID = otherID
        self.mat1 = mat1
        self.mat2 = mat2
        # self.max_vals_mat1 = max_vals_mat1
        # self.max_vals_mat2 = max_vals_mat2  # 保存每行的最大值

        return loss

    def backward(self):

        G = deltaPred = self.p - self.label             # (bs, )

        G = G.reshape(-1, 1)

        # deltaHidden = np.repeat(G.reshape(-1, 1), embeddingDim, 1)   # (bs, embDim)

        # # 重新计算 mat1 * mat2 并减去每行的最大值
        # mat1_stable = self.mat1 - self.max_vals_mat1
        # mat2_stable = self.mat2 - self.max_vals_mat2
        #
        # dot_product_stable = mat1_stable * mat2_stable

        deltaMat2 = G * self.mat1       # (bs, embDim)
        deltaMat1 = G * self.mat2       # (bs, embDim)

        # for i in range(len(self.centerID)):
        #     self.W1[self.centerID[i]] -= deltaMat1[i] * self.lr
        #     self.W2[:, self.otherID[i]] -= deltaMat2[i] * self.lr

        self.W1[self.centerID] -= deltaMat1 * self.lr
        self.W2[:, self.otherID] -= deltaMat2.T * self.lr

        # self.mat2 -= deltaMat2 * self.lr
        # self.mat1 -= deltaMat1 * self.lr

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
    # dataPath = r"./data/数学原始数据.csv"
    dataPath = r"./data/金庸-射雕英雄传txt精校版.txt"
    stopWordsPath = r"./data/hit_stopwords.txt"

    epoch = 30
    batch_size = 128

    dataset = LocalDataset(dataPath, sample=None, stopWordsPath=stopWordsPath, step=2, nGram=10, negNum=30)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddingDim = 256
    wordsNum = len(dataset.words2Idx)

    lr = 0.1
    model = Model(wordsNum, embeddingDim, lr=lr)

    for e in range(epoch):
        print("\n" + f"Epoch >> {e + 1}".center(80, "-"))
        for centerWord, otherWord, label in tqdm(dataloader, desc="Training..."):
            centerWord, otherWord, label = centerWord.numpy(), otherWord.numpy(), label.numpy()
            loss = model.forward(centerWord, otherWord, label)
            model.backward()

        print(f"\nLoss >> {loss}\t\tlr >> {lr}")


    model.saveEmbs(dataset.words2Idx)