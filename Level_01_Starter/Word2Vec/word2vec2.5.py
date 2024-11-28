"""
encoding: utf-8
@Author: Jacob Y
@Date  : 11/24/2024
@Desc  : 负采样1.5  基于Dataset和DataLoader实现批量运算

优化日志：提供采用基于词频的负采样策略，提高模型对高频词关联度的捕获敏感度

编码细节：
1. 采样时选取词频的P(w)^(3/4)
2. 取中心词index对应的W1中的行
3. 取样本词index对应的W2中的列

负采样策略：随机从正样本和全体样本的补集中抽取负样本

"""
import os
import pickle
import random
random.seed(1126)
from datetime import datetime
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
torch.manual_seed(1126)

import jieba
import logging
logging.getLogger('jieba').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

OutPutDir = r"./output"
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
        self.negSampleStrategy = "freq"
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
        sampleRange = set(self.words2Idx.values()) ^ set([centerID] + neighborsID)

        negativesID = random.sample(sampleRange, negNum)
        negSamples = [(centerID, negID, 0) for negID in negativesID]
        return negSamples

    def negFreqSample(self, centerID, neighborsID, negNum):
        """
        基于词频概率分布采集负样本，平衡高低频词的采样概率
        """
        adjusted_probs = {}
        for word, freq in self.wordsFreqs.items():
            adjusted_probs[word] = freq ** (3 / 4)

        # 归一化调整后的概率
        total_adjusted_prob = sum(adjusted_probs.values())
        normalized_probs = {word: prob / total_adjusted_prob for word, prob in adjusted_probs.items()}

        sampleRange = {word for word in normalized_probs.keys() if word not in ([centerID] + neighborsID)}
        negPreSample = random.choices(list(sampleRange), weights=[normalized_probs[w] for w in sampleRange], k=negNum)


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

                if self.negSampleStrategy == "freq":
                    negaSamples = self.negFreqSample(centerID, neighborsID, negNum)
                else:
                    negaSamples = self.negRandomSample(centerID, neighborsID, negNum)
                triples.extend(negaSamples)
        return triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, item):
        currTriple = self.triples[item]

        return currTriple


class Model(nn.Module):
    def __init__(self, wordsNum, embeddingDim):
        super().__init__()
        self.W1 = nn.Linear(wordsNum, embeddingDim)
        self.W2 = nn.Linear(embeddingDim, wordsNum)

    def sigmoid(self, data):
        data = torch.clip(data, -20, 20)
        return 1 / (1 + torch.exp(-data))

    def forward(self, centerID, otherID, label):
        mat1 = self.W1.weight.T[centerID]  # (bs, embDim)
        mat2 = self.W2.weight.T[:, otherID].T  # (bs, embDim)

        dotProduct = mat1 * mat2

        pred = torch.mean(dotProduct, dim=-1)  # (bs, )
        pred -= torch.max(dotProduct, dim=-1).values     # 防止溢出

        p = self.sigmoid(pred)  # (bs, )

        loss = -torch.mean(label * torch.log(p) +
                        (1 - label) * torch.log(1 - p))

        return loss



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
            pickle.dump((model.W1.weight.data.T.cpu().detach().numpy(),
                         words2Idx), file)
        return filePath


if __name__ == '__main__':
    dataPath = r"./data/数学原始数据.csv"
    stopWordsPath = r"./data/hit_stopwords.txt"

    epoch = 10
    batch_size = 64

    dataset = LocalDataset(dataPath, sample=None, stopWordsPath=stopWordsPath, step=2, nGram=10, negNum=30)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddingDim = 100
    wordsNum = len(dataset.words2Idx)

    lr = 0.001
    model = Model(wordsNum, embeddingDim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        print("\n" + f"Epoch >> {e + 1}".center(80, "-"))
        for centerWord, otherWord, label in tqdm(dataloader, desc="Training..."):
            # centerWord, otherWord, label = centerWord.numpy(), otherWord.numpy(), label.numpy()
            centerWord, otherWord, label = centerWord.to(device), otherWord.to(device), label.to(device)
            loss = model.forward(centerWord, otherWord, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"\nLoss >> {loss}\t\tlr >> {lr}\t\tdevice >> {loss.device}")

        # 学习率策略
        if loss < 0.1 and e % 3 == 0:
            lr *= 0.3
            continue
        if loss < 0.5 and e % 2 == 0:
            lr *= 0.9

    savedPath = model.saveEmbs(dataset.words2Idx)
    print(f"Model has been saved at {savedPath}")
