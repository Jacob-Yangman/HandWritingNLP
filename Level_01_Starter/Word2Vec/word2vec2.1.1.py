# coding=utf-8
"""
@Author: Jacob Y
@Date  : 11/23/2024
@Desc  : 负采样1.1.1

优化日志：摒弃OneHot编码，直接传入词的index
在1.1的基础上构建Dataset、DataLoader，批量抽取triples
使用PyTorch封装模型

编码细节：
1. 使用三元组构建特征和标签
2. 传入每个三元组中词语对应的词表index
3. 取中心词index对应的W1中的行
4. 取样本词index对应的W2中的列

负采样策略：随机从正样本和全体样本的补集中抽取负样本

"""

import os
import jieba
import random
random.seed(1124)
import numpy as np
import torch.cuda
from tqdm import tqdm
import pickle
from datetime import datetime
import logging
logging.getLogger('jieba').setLevel(logging.ERROR)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

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


# def mysigmoid(x):
#     x = np.clip(x, -20, 20)
#     return 1 / (1 + np.exp(-x))

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


class WordLevelTripletDataset(Dataset):
    def __init__(self, allWords):
        self.allWords = allWords
        # self.triples = WordLevelTripletDataset.getTriples(self.allWords)

    @staticmethod
    def getNeighbors(currLine, currID):
        global nGrams
        leftN = currLine[max(0, currID - nGrams): currID]
        rightN = currLine[currID + 1: currID + nGrams + 1]
        return leftN + rightN

    def generateTriples(self, lineWords):
        """
                动态生成一个句子的三元组
                """
        global step, negNum, words2Idx

        triples = []
        for currId in range(0, len(lineWords), step):
            currWord = lineWords[currId]

            # 获取正样本
            neighbors = self.getNeighbors(lineWords, currId)
            for nei in neighbors:
                triples.append((currWord, nei, 1.))  # 正样本标签1

            # 随机负采样
            negativeRange = list(set(words2Idx) - set([currWord] + neighbors))
            negativeSamples = random.sample(negativeRange, negNum)
            for neg in negativeSamples:
                triples.append((currWord, neg, 0.))  # 负样本标签0

        return triples

    def __len__(self):
        """
        直接返回词总数，因为每个词是三元组的起点
        """
        return sum(len(line) for line in self.allWords)

    def __getitem__(self, item):
        """
        动态按索引生成样本，节省内存
        """
        # 将一个一维的全局索引（item）映射到二维结构（self.allWords中某一行的某一列）
        lineId, wordId = 0, item
        for lineWords in self.allWords:
            if wordId < len(lineWords):
                break
            wordId -= len(lineWords)
            lineId += 1

        # 动态生成指定句子的三元组
        triples = self.generateTriples(self.allWords[lineId])
        currTriple = triples[wordId % len(triples)]

        idx1 = words2Idx[currTriple[0]]
        idx2 = words2Idx[currTriple[1]]
        label = torch.tensor(currTriple[-1], dtype=torch.float32)

        return idx1, idx2, label


class IndexableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        """
        自定义支持索引的线性层
        """
        super(IndexableLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)

    def forward(self, input_idx, dim):
        """
        支持通过索引访问权重
        :param input_idx: 索引 (torch.LongTensor)，例如 (batch_size,)
        :param dim: 在 weight 的哪个维度上索引 (0 或 1)
        :return: 索引得到的张量，与梯度计算兼容
        """
        if dim == 0:
            # 索引第 0 维（行）
            output = self.weight[input_idx]
        elif dim == 1:
            # 索引第 1 维（列）
            output = self.weight[:, input_idx]
        else:
            raise ValueError("dim must be 0 (row) or 1 (column)")
        return output




class MyModel(nn.Module):
    global wordsNum, embLen

    def __init__(self):
        super().__init__()

        self.W1 = IndexableLinear(wordsNum, embLen)  # 自定义线性层
        self.W2 = IndexableLinear(embLen, wordsNum)
        self.sigmoid = nn.Sigmoid()
        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, currWordIdx, correspondIdx, label):
        """
        :param currWordIdx: (1,)
        :param correspondIdx: (1,)
        :return: loss
        """

        # 提取 hiddenLayer，shape: (1, embLen)
        hiddenLayer = self.W1(currWordIdx, dim=0)  # 按行索引

        # 提取 W2 的对应列，shape: (1, embLen)
        W2_column = self.W2(correspondIdx, dim=1)  # 按列索引

        # 预测值
        predict = hiddenLayer @ W2_column  # 内积
        proba = self.sigmoid(predict)  # 概率化

        loss = self.lossFn(proba, label)
        return loss


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
    lr = 0.1
    epoch = 3
    batch_size = 20

    step = 2
    nGrams = 6
    negNum = 12

    wordsNum = len(words2Idx)
    embLen = 200

    myDataset = WordLevelTripletDataset(allWords)
    myDataloader = DataLoader(myDataset, batch_size=batch_size, shuffle=False)

    # 测试加载一个批次
    for idx1, idx2, label in myDataloader:
        print(f"中心词索引: {idx1}, 上下文/负样本索引: {idx2}, 标签: {label}")
        break


    # model, optimizer
    model = MyModel().to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        print("\n" + f"Epoch No.{e + 1}".center(60, "-"))
        for currWordIdx, correspondWordIdx, label in tqdm(myDataloader, desc="Training..."):
            currWordIdx, correspondWordIdx, label = currWordIdx.to(device), correspondWordIdx.to(device), label.to(
                device)

            loss = model.forward(currWordIdx, correspondWordIdx, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        print(f"\nLoss >> {loss:.4f}")
    saveEmbs(model.W1.cpu().weight.data.T)

