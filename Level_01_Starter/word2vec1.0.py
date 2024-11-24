"""
encoding: utf-8
@Author: Jacob Y
@Date  : 11/22/2024
@Desc  : 原汁原味Word2Vec
单轮训练时间预估：2~6h
"""
import os
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
    xMax = np.max(x, axis=-1)
    x -= xMax
    ex = np.exp(x)
    exSum = np.sum(ex, axis=-1)
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


class MyModel():
    global wordsNum, embLen

    def __init__(self):
        self.W1 = np.random.normal(size=(wordsNum, embLen))
        self.W2 = np.random.normal(size=(embLen, wordsNum))
        self.softmax = mysoftmax

    def forward(self, x, y):
        """
        :param x: (1, wordsNum)
        :param y: (1, wordsNum)
        :return: loss
        """
        global lr
        self.x, self.y = x, y
        self.hiddenLayer = x @ self.W1  # (1, embLen)
        predict = self.hiddenLayer @ self.W2  # (1, wordsNum)
        self.proba = self.softmax(predict)  # (1, wordsNum)
        epsilon = 1e-10
        loss = -np.mean(y * np.log(self.proba))

        return loss

    def backward(self):
        G = deltaPred = self.proba - self.y  # (1, wordsNum)
        deltaW2 = self.hiddenLayer.T @ G  # (embLen, wordsNum)
        deltaHidden = G @ self.W2.T  # (1, embLen)
        deltaW1 = self.x.T @ deltaHidden  # (wordsNum, embLen)

        self.W1 -= lr * deltaW1
        self.W2 -= lr * deltaW2


def getNeighbors(currLine, currIdx):
    global nGrams
    leftN = currLine[max(0, currIdx - nGrams): currIdx]
    rightN = currLine[currIdx + 1: currIdx + nGrams + 1]
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
    nGrams = 5
    wordsNum = len(words2Idx)
    embLen = 200

    # global OneHot Matrix
    wordsOneHotMtx = np.eye(wordsNum)

    model = MyModel()

    for e in range(epoch):
        print("\n" + f"Epoch No.{e + 1}".center(60, "-"))
        for lineWords in tqdm(allWords):
            for currIdx, currWord in enumerate(lineWords):

                currOneHot = wordsOneHotMtx[words2Idx.get(currWord)]
                currOneHot = np.array(currOneHot).reshape(1, -1)

                neighbors = getNeighbors(lineWords, currIdx)
                for neighbor in neighbors:
                    neighborOneHot = wordsOneHotMtx[words2Idx.get(neighbor)]
                    neighborOneHot = np.array(neighborOneHot).reshape(1, -1)

                    if MODEL == "cbow":
                        loss = model.forward(neighborOneHot, currOneHot)
                    elif MODEL == "skip-gram":
                        loss = model.forward(currOneHot, neighborOneHot)
                    model.backward()
        print(f"\nLoss >> {loss:.4f}")
    saveEmbs()