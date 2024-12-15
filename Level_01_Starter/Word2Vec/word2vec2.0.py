# coding=utf-8
"""
@Author: Jacob Y
@Date  : 11/23/2024
@Desc  : 负采样1.0

核心：重构任务：
不再用一个词预测另一个词，而是预测两个词是否相关，原训练任务被重构为二分类任务
标签：0表示不相关，1表示相关
损失函数：变为二元交叉熵
激活函数：改为Sigmoid，减小运算量
编码细节：
1. 使用三元组构建特征和标签
2. 传入每个三元组中词语对应的OneHot向量
3. 取中心词onehot非零元素下标对应的W1中的行
4. 取样本词onehot非零元素下标对应的W2中的列

负采样策略：随机从正样本和全体样本的补集中抽取负样本

"""

import os
import jieba
import random
import numpy as np
from tqdm import tqdm
import pickle
from datetime import datetime
import logging

logging.getLogger('jieba').setLevel(logging.ERROR)

OutPutDir = r"./output"
# 创建词向量和词表输出路径，如果已存在则跳过
os.makedirs(OutPutDir, exist_ok=True)

MODEL = "skip-gram"

# def mysoftmax(x):
#     xMax = np.max(x, axis=-1, keepdims=True)
#     x = x - xMax
#     ex = np.exp(x)
#     exSum = np.sum(ex, axis=-1, keepdims=True)
#     return ex / exSum


def mysigmoid(x):
    x = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))

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


class MyModel():
    global wordsNum, embLen

    def __init__(self):
        self.W1 = np.random.normal(size=(wordsNum, embLen))
        self.W2 = np.random.normal(size=(embLen, wordsNum))
        self.sigmoid = mysigmoid

    def forward(self, currWordOH, correspondOH, label):
        """
        :param currWordOH: (1, wordsNum)
        :param correspondOH: (1, wordsNum)
        :return: loss
        """
        global lr
        self.currWordOH = currWordOH
        self.correspondOH = correspondOH
        self.label = label

        self.hiddenLayer = self.W1[np.argmax(self.currWordOH), None]  # (1, embLen)
        predict = self.hiddenLayer @ (self.W2[:, np.argmax(self.correspondOH), None])  # (1, 1)
        self.proba = self.sigmoid(predict)  # (1, 1)

        loss = -np.mean(self.label * np.log(self.proba) +
                        (1 - self.label) * np.log(1 - self.proba))
        return loss

    def backward(self):
        G = deltaPred = self.proba - self.label
        deltaW2 = self.hiddenLayer.T @ G  # (embLen, wordsNum)
        deltaHidden = G @ self.W2[:, np.argmax(self.correspondOH), None].T  # (n, embLen)
        deltaW1 = deltaHidden  # (wordsNum, embLen)

        self.W1[np.argmax(self.currWordOH), None] -= lr * deltaW1
        self.W2[:, np.argmax(self.correspondOH), None] -= lr * deltaW2


def getNeighbors(currLine, currID):
    global nGrams
    leftN = currLine[max(0, currID - nGrams): currID]
    rightN = currLine[currID + 1: currID + nGrams + 1]
    return leftN + rightN


def getTriples(allWords):
    global step, negNum, words2Idx
    triples = list()
    for lineWords in tqdm(allWords, desc="Generating Triples..."):
        for currId in range(0, len(lineWords), step):
            currWord = lineWords[currId]

            neighbors = getNeighbors(lineWords, currId)
            for nei in neighbors:
                triples.append((currWord, nei, 1))

            negativeRange = list(set(words2Idx) ^ set([currWord] + neighbors))

            negativeSamples = random.sample(negativeRange, negNum)

            for neg in negativeSamples:
                triples.append((currWord, neg, 0))

    return triples



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
    stopWords = loadStopWords("data/hit_stopwords.txt")
    stopWords.extend(["。", "，", "；", "（", "）"])
    allWords = parseData("data/数学原始数据.csv", sample=100)

    # words2idx, idx2words
    words2Idx, idx2Words = getWords2Idx(allWords)

    # hyper params
    lr = 0.1
    epoch = 3

    step = 2
    nGrams = 6
    negNum = 12

    wordsNum = len(words2Idx)
    embLen = 200

    # global OneHot Matrix
    wordsOneHotMtx = np.eye(wordsNum)

    triples = getTriples(allWords)

    model = MyModel()

    for e in range(epoch):
        print("\n" + f"Epoch No.{e + 1}".center(60, "-"))
        for data in tqdm(triples, desc="Training..."):
            currWord = data[0]
            correspondWord = data[1]
            label = data[-1]

            currWordOH = wordsOneHotMtx[words2Idx.get(currWord)]
            correspondWordOH = wordsOneHotMtx[words2Idx.get(correspondWord)]

            loss = model.forward(currWordOH, correspondWordOH, label)
            model.backward()
        print(f"\nLoss >> {loss:.4f}")
    saveEmbs()

