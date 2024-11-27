"""
encoding: utf-8
@Author: Jacob Y
@Date  : 11/23/2024
@Desc  : 借助torch的Dataset和DataLoader实现批量化送入多句的currWordsOneHot和neighborsOneHot拼成的矩阵
单轮训练时间预估：3~5m
"""
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import jieba
# import numpy as np
from tqdm import tqdm
import pickle
from datetime import datetime
import logging

logging.getLogger('jieba').setLevel(logging.ERROR)

OutPutDir = r"./output"
# 创建词向量和词表输出路径，如果已存在则跳过
os.makedirs(OutPutDir, exist_ok=True)

MODEL = "skip-gram"  # 或skip-gram





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


class MyDataset(Dataset):
    def __init__(self, allWords):
        self.allWords = allWords

    def __len__(self):
        return len(self.allWords)

    def __getitem__(self, item):
        currLineWords = self.allWords[item]
        currLineWordsOneHot, currNeighborsOneHot = [], []
        # for currId, currWord in enumerate(currLineWords):
        for currId in range(0, len(currLineWords), step):
            currWord = currLineWords[currId]
            currOneHot = wordsOneHotMtx[words2Idx.get(currWord)].reshape(1, -1)
            # currOneHot = torch.tensor(currOneHot).reshape(1, -1)

            neighbors = getNeighbors(currLineWords, currId)
            neighborsIdx = [words2Idx.get(w) for w in neighbors]
            neighborsOneHot = wordsOneHotMtx[neighborsIdx]
            # 将当前词OneHot向量复制len(neighbors)份
            currOneHot = currOneHot.repeat(len(neighbors), 1)

            currLineWordsOneHot.append(currOneHot)
            currNeighborsOneHot.append(neighborsOneHot)
        currLineWordsOH = torch.vstack(currLineWordsOneHot)
        currNeighborsOH = torch.vstack(currNeighborsOneHot)

        return currLineWordsOH, currNeighborsOH



class MyModel(nn.Module):
    global wordsNum, embLen

    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(wordsNum, embLen)
        self.W2 = nn.Linear(embLen, wordsNum)
        self.softmax = nn.Softmax(dim=-1)
        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        """
        :param x: (bs, n, wordsNum)
        :param y: (bs, n, wordsNum)
        :return: loss
        """
        global lr
        hidden = self.W1(x)
        pred = self.W2(hidden)
        p = self.softmax(pred)

        loss = self.lossFn(p, y)
        return loss

def collateFn(batchData):
    batchCurrWordOH, batchNeighborsOH = zip(*batchData)
    return torch.vstack(batchCurrWordOH), torch.vstack(batchNeighborsOH)



def getNeighbors(currLine, currID):
    global nGrams
    leftN = currLine[max(0, currID - nGrams): currID]
    rightN = currLine[currID + 1: currID + nGrams + 1]
    return leftN + rightN


def saveEmbs(W1):
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
        pickle.dump((W1, words2Idx), file)


if __name__ == '__main__':

    # all words without stopwords
    stopWords = loadStopWords("data/hit_stopwords.txt")
    stopWords.extend(["。", "，", "；", "（", "）"])
    allWords = parseData("data/数学原始数据.csv", sample=100)

    # words2idx, idx2words
    words2Idx, idx2Words = getWords2Idx(allWords)

    # hyper params
    lr = 0.01
    epoch = 3
    batch_size = 4

    step = 2
    nGrams = 10

    wordsNum = len(words2Idx)
    embLen = 200

    # global OneHot Matrix
    wordsOneHotMtx = torch.eye(wordsNum)

    myDataset = MyDataset(allWords)
    myDataLoader = DataLoader(myDataset, batch_size=batch_size, shuffle=True, collate_fn=collateFn)

    # model  optimizer
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        print("\n" + f"Epoch No.{e + 1}".center(60, "-"))
        for currWordsOneHot, neighborsOneHot in tqdm(myDataLoader):
            if MODEL == "cbow":
                loss = model.forward(neighborsOneHot, currWordsOneHot)
            elif MODEL == "skip-gram":
                loss = model.forward(currWordsOneHot, neighborsOneHot)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"\nLoss >> {loss:.4f}")

    saveEmbs(model.W1.weight.data.T)
