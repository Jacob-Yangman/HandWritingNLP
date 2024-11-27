"""
encoding: utf-8
@Author: Jacob Y
@Date  : 11/24/2024
@Desc  : 负采样1.3
单轮训练时间预估：2~3m

优化日志：不再获取全局triples，而是按照每个词采集正负样本，构建二元组，二元组中直接使用词的index

编码细节：
1. 对每个词使用二元组构建特征和标签，二元组中直接使用词的index
2. 取中心词index对应的W1中的行
3. 取样本词index对应的W2中的列

负采样策略：随机从正样本和全体样本的补集中抽取负样本

"""
import os
import jieba
import random
random.seed(1123)
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
            if sample and i > sample:
                break
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

    def forward(self, currWordIdx, correspondIdx, label):
        """
        :param currWordIdx: (1,)
        :param correspondIdx: (1,)
        :return: loss
        """
        global lr
        # self.x, self.label = x, label
        self.label = label
        self.currWordIdx = currWordIdx
        self.correspondIdx = correspondIdx

        self.hiddenLayer = self.W1[currWordIdx, None]  # (1, embLen)
        predict = self.hiddenLayer @ (self.W2[:, correspondIdx, None])  # (1, 1)
        self.proba = self.sigmoid(predict)  # (1, 1)

        loss = -np.mean(self.label * np.log(self.proba) +
                        (1 - self.label) * np.log(1 - self.proba))
        return loss

    def backward(self):
        G = deltaPred = self.proba - self.label     # (1, 1)
        deltaW2 = self.hiddenLayer.T @ G  # (embLen, 1)
        deltaHidden = G @ self.W2[:, self.correspondIdx, None].T  # (1, embLen)
        deltaW1 = deltaHidden  # (1, embLen)

        self.W1[self.currWordIdx, None] -= lr * deltaW1
        self.W2[:, self.correspondIdx, None] -= lr * deltaW2


def getNeighbors(currLine, currID):
    global nGrams
    leftN = currLine[max(0, currID - nGrams): currID]
    rightN = currLine[currID + 1: currID + nGrams + 1]
    return leftN + rightN


def getCurrPairs(currId, lineWords):
    global negNum, words2Idx
    pairs = list()
    currWord = lineWords[currId]


    # neighbors = getNeighbors(lineWords, currId)
    neighbors = lineWords[max(0, currId - nGrams): currId] + lineWords[currId + 1: currId + nGrams + 1]

    for nei in neighbors:
        pairs.append((words2Idx[nei], 1))

    negativeRange = list(set(words2Idx) ^ set([currWord] + neighbors))

    negativeSamples = random.sample(negativeRange, negNum)

    for neg in negativeSamples:
        pairs.append((words2Idx[neg], 0))


    return pairs




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
    allWords = parseData("data/数学原始数据.csv")

    # words2idx, idx2words
    words2Idx, idx2Words = getWords2Idx(allWords)

    # hyper params
    lr = 0.1
    epoch = 5

    step = 1
    nGrams = 5
    negNum = 10

    wordsNum = len(words2Idx)
    embLen = 200


    model = MyModel()

    for e in range(epoch):
        print("\n" + f"Epoch No.{e + 1}".center(60, "-"))

        for currLineWords in tqdm(allWords, desc="Training..."):
            # for currWordIdx, currWord in enumerate(currLineWords):
            for currId in range(0, len(currLineWords), step):
                currWordIdx = words2Idx[currLineWords[currId]]
                currWordPairs = getCurrPairs(currId, currLineWords)

                for pair in currWordPairs:
                    loss = model.forward(currWordIdx, *pair)
                    model.backward()
        print(f"\nLoss >> {loss}")
    saveEmbs()


