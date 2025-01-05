# coding=utf-8
"""
@Author: Jacob Y
@Date  : 12/16/2024
@Desc  : BERT主要框架，复现NSP和MLM两个子任务的主要思路
编码细节：为避免预训练语料中包含[PAD],[UNK],[CLS],[SEP],[MASK]，
        在预处理时将其更换为<<PAD>>,<<UNK>>,<<CLS>>,<<SEP>>,<<MASK>>
"""
import random
import sys
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


def readData(filePath):
    allData = list()
    with open(filePath, encoding="utf-8") as f:
        content = f.read().splitlines()
    return content


def tokenize(data):
    words2Idx = {"<<PAD>>": 0}
    words2Idx.update({f"unused{i-1}": i for i in range(1, 100)})
    words2Idx.update({"<<UNK>>": 100, "<<CLS>>": 101, "<<SEP>>": 102, "<<MASK>>": 103})
    # words2Idx.update({f"unused{i}": i+5 for i in range(99, 994)})

    for line in data:
        lineCut = line.split("<<SEP>>")
        for sentence in lineCut:
            for w in sentence:
                words2Idx[w] = words2Idx.get(w, len(words2Idx))

    return words2Idx


def makeTriples(data):
    triples = list()
    for line in data:
        sentences = line.split("<<SEP>>")

        for si in range(len(sentences) - 1):
            if not sentences[si]:
                continue
            # context
            triples.append((sentences[si], sentences[si+1], 1))
            # Non-context
            negi = sentences[:max(0, si-1)] + sentences[min(si+2, len(sentences)):]

            triples.extend([(sentences[si], s, 0) for s in negi])

    return triples

class MyDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, item):
        currTriple = self.triples[item]

        # 截断和填充
        s1Idx = [words2IDx.get(c, words2IDx["<<UNK>>"]) for c in currTriple[0]][:maxLen]
        s1Idx += [words2IDx["<<PAD>>"]] * (maxLen - len(s1Idx))
        s2Idx = [words2IDx.get(c, words2IDx["<<UNK>>"]) for c in currTriple[1]][:maxLen]
        s2Idx += [words2IDx["<<PAD>>"]] * (maxLen - len(s2Idx))
        # 编码
        inputTrueDataIdx= ([words2IDx["<<CLS>>"]] + s1Idx +
                       [words2IDx["<<SEP>>"]] + s2Idx +
                       [words2IDx["<<SEP>>"]])

        # MLM Data (mask-padding idx)
        maskedDataIdx = inputTrueDataIdx.copy()


        # 选取15%的位置填充MASK
        maskNum = int(len(inputTrueDataIdx) * 0.15)
        maskLocalIdx = sorted(random.sample([i for i in range(len(inputTrueDataIdx))], maskNum + 3))
        maskLocalIdx = [i for i in maskLocalIdx if inputTrueDataIdx[i] not in (100, 101, 102)][:maskNum]

        for m in maskLocalIdx:
            maskedDataIdx[m] = words2IDx["<<MASK>>"]


        # MLM Label  (除MASK位置外，都用-100填充，因为CrossEntropyLoss中ignore_index默认为-100)
        unmaskLabel = list()
        for i in range(len(inputTrueDataIdx)):
            if i in maskLocalIdx:
                unmaskLabel.append(inputTrueDataIdx[i])
            else:
                unmaskLabel.append(-100)

        # (NSP Data, NSP Label), (MLM Data, MLM Label)
        return torch.tensor(inputTrueDataIdx), torch.tensor(currTriple[2]), torch.tensor(maskedDataIdx), torch.tensor(unmaskLabel)



class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(wordsNum, embDim)
        self.positionalEmb = nn.Embedding(wordsNum, embDim)
        self.encoder = nn.Linear(embDim, hiddenDim)

    def genEmbedding(self, inputIdx):
        # Segment Embedding
        wordsEmb = self.embedding(inputIdx)
        segmentEmb = torch.zeros_like(wordsEmb)
        positionalEmb = self.positionalEmb(inputIdx)

        sepIdx = inputIdx[-1].tolist().index(words2IDx["<<SEP>>"])
        segmentEmb[:, sepIdx + 1:] = 1
        emb = wordsEmb + segmentEmb + positionalEmb
        return emb


    def forward(self, emb):
        return self.encoder(emb)



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = nn.Parameter((torch.rand(1)))
        self.bert = BERT()
        self.clsNLP = nn.Linear(hiddenDim, 2)
        self.clsMLM = nn.Linear(hiddenDim, wordsNum)
        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, batchDataNSP, batchLabelNSP, batchDataMLM, batchLabelMLM):
        embNSP = self.bert.genEmbedding(batchDataNSP)
        embMLM = self.bert.genEmbedding(batchDataMLM)

        x1, x2 = self.bert(embNSP), self.bert(embMLM)
        pred1, pred2 = self.clsNLP(x1), self.clsMLM(x2)

        pred1 = torch.mean(pred1, dim=1)

        loss1 = self.lossFn(pred1, batchLabelNSP)
        loss2 = self.lossFn(pred2.reshape(-1, pred2.shape[-1]), batchLabelMLM.reshape(-1))
        # print(f"Loss1 >> {loss1.item():.3f}\t\tLoss2 >> {loss2.item():.3f}")


        """trick: 为不同loss指定可学习权重"""
        # 防止学习过程中self.rate超出(0, 1)范围
        currRate = torch.sigmoid(self.rate) if self.rate < 0 or self.rate > 1 else self.rate
        loss = loss1 * currRate + loss2 * (1 - currRate)
        return loss


if __name__ == '__main__':
    lr = 0.001
    epoch = 20
    batch_size = 4
    maxLen = 30
    embDim = 1024
    hiddenDim = 128

    allData = readData("../data/train4Bert.txt")
    words2IDx = tokenize(allData)
    wordsNum = len(words2IDx)

    triples = makeTriples(allData)

    trainDataset = MyDataset(triples)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for e in range(epoch):
        model.train()
        for batchDataNSP, batchLabelNSP, batchDataMLM, batchLabelMLM in tqdm(trainLoader, file=sys.stdout):
            loss = model.forward(batchDataNSP, batchLabelNSP, batchDataMLM, batchLabelMLM)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Loss >> {loss.item():.3f}", flush=True)