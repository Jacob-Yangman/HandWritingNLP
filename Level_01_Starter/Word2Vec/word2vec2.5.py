"""
encoding: utf-8
@Author: Jacob Y
@Date  : 11/24/2024
@Desc  : 负采样1.5  基于Dataset和DataLoader实现批量运算

优化日志：采用基于词频的负采样策略，提高模型对高频词关联度的捕获敏感度

编码细节：
1. 采样时选取词频的P(w)^(3/4)
2. 取中心词index对应的W1中的行
3. 取样本词index对应的W2中的列

负采样策略：随机从正样本和全体样本的补集中抽取负样本

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
            self.readData(stopWordsPath) + ['\u3000', '，', '。', '；', '', '------------------------------------------'])
        self.cutData = self.parseData()
        self.words2Idx, self.idx2Words = self.tokenize()
        self.wordsFreqs = self.getFreqence()
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

    def getFreqence(self):
        wordsFreq = dict()
        for lineWords in self.cutData:
            for word in lineWords:
                wordsFreq[word] = wordsFreq.get(word, 0) + 1
        return wordsFreq



    def negRandomSample(self, centerID, neighborsID, negNum):
        # sampleRange = set(self.idx2Words) ^ set(neighbors + [self.word2Idx[centerID]])
        sampleRange = set(self.words2Idx.values()) ^ set([centerID] + neighborsID)

        negativesID = random.sample(sampleRange, negNum)
        # negativesID = [self.word2Idx[w] for w in negatives]
        negSamples = [(centerID, negID, 0) for negID in negativesID]
        return negSamples

    def negFreqSample(self, centerID, neighborsID, negNum):
        adjusted_probs = {}
        for word, freq in self.wordsFreqs.items():
            adjusted_probs[word] = freq ** (3 / 4)

        # 归一化调整后的概率
        total_adjusted_prob = sum(adjusted_probs.values())
        normalized_probs = {word: prob / total_adjusted_prob for word, prob in adjusted_probs.items()}

        # sampleRange = sorted(self.wordsFreqs.items(), reverse=True, key=lambda x: x[1])[:int(len(self.wordsFreqs) * (3 / 4))]
        negPreSample = random.choices(list(normalized_probs.keys()), weights=normalized_probs.values(),
                                          k=negNum)

        negativesID = [self.words2Idx[sam] for sam in negPreSample]
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

                negaSamples = self.negFreqSample(centerID, neighborsID, negNum)
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

        loss = -np.mean(label * np.log(p) +
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

        deltaHidden = np.repeat(G.reshape(-1, 1), embeddingDim, 1)   # (bs, embDim)

        # # 重新计算 mat1 * mat2 并减去每行的最大值
        # mat1_stable = self.mat1 - self.max_vals_mat1
        # mat2_stable = self.mat2 - self.max_vals_mat2
        #
        # dot_product_stable = mat1_stable * mat2_stable

        deltaMat2 = deltaHidden * self.mat1       # (bs, embDim)
        deltaMat1 = deltaHidden * self.mat2       # (bs, embDim)

        for i in range(len(self.centerID)):
            self.W1[self.centerID[i]] -= deltaMat1[i] * self.lr
            self.W2[:, self.otherID[i]] -= deltaMat2[i] * self.lr

        # self.W1[self.centerID] -= deltaMat1 * self.lr
        # self.W2[:, self.otherID] -= deltaMat2.T * self.lr

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
        return filePath


if __name__ == '__main__':
    dataPath = r"./data/数学原始数据.csv"
    stopWordsPath = r"./data/hit_stopwords.txt"

    epoch = 30
    batch_size = 16

    dataset = LocalDataset(dataPath, sample=None, stopWordsPath=stopWordsPath, step=5, nGram=10, negNum=20)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddingDim = 256
    wordsNum = len(dataset.words2Idx)

    lr = 0.01
    model = Model(wordsNum, embeddingDim, lr=lr)



    for e in range(epoch):
        print("\n" + f"Epoch >> {e + 1}".center(80, "-"))
        for centerWord, otherWord, label in tqdm(dataloader, desc="Training..."):
            centerWord, otherWord, label = centerWord.numpy(), otherWord.numpy(), label.numpy()
            loss = model.forward(centerWord, otherWord, label)
            model.backward()

        print(f"\nLoss >> {loss}\t\tlr >> {lr}")

        if loss < 0.1 and e % 3 == 0:
            lr *= 0.3
            continue

        if loss < 0.5 and e % 2 == 0:
            lr *= 0.9


    savedPath = model.saveEmbs(dataset.words2Idx)
    print(f"Model has been saved at {savedPath}")