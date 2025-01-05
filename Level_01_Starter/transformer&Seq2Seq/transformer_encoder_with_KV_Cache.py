# coding=utf-8
"""
@Author: Jacob Y
@Date  : 12/09/2024
@Desc  : 
"""
import random
import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

def readData(filePath, sample=None):
    allTexts, allLabels = list(), list()
    with open(filePath, encoding="utf-8") as f:
        content = f.read().splitlines()[:sample] if sample else f.read().splitlines()
    for line in content:
        try:
            text, label = line.split("\t")
            label = int(label)
        except:
            continue
        allTexts.append(text)
        allLabels.append(label)
    return allTexts, allLabels


def getWords2Idx(contents):
    words2Idx = dict((("<PAD>", 0), ("<UNK>", 1)))
    for line in contents:
        for w in line:
            words2Idx[w] = words2Idx.get(w, len(words2Idx))

    return words2Idx, list(words2Idx)


class MyDataset(Dataset):
    def __init__(self, allTexts, allLabels):
        self.allTexts = allTexts
        self.allLabels = allLabels

    def __len__(self):
        return sum([True for _ in self.allTexts])


    def __getitem__(self, item):
        currText = self.allTexts[item]
        currLabel = self.allLabels[item]

        textIdx = [words2Idx.get(w, words2Idx["<UNK>"]) for w in currText]

        return textIdx, currLabel, len(textIdx)

    def collateFn(self, batchData):
        batchIdx, batchLabels, batchLength = zip(*batchData)
        batchMaxLen = max(batchLength)
        newBatchIdx = list()
        # Padding
        for b in batchIdx:
            b += [words2Idx["<PAD>"]] * (batchMaxLen - len(b))
            newBatchIdx.append(b)

        return torch.tensor(newBatchIdx), torch.tensor(batchLabels)


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
    def __init__(self, seqLen:int, embDim:int):
        super().__init__()
        self.peMatrix = torch.zeros(seqLen, embDim, device=device)
        posMatrix = torch.tensor(
            [[pos / (10000 ** (2*i/embDim)) for i in range(embDim//2)] for pos in range(seqLen)]
        )

        self.peMatrix[:, 0::2] = torch.sin(posMatrix)
        self.peMatrix[:, 1::2] = torch.cos(posMatrix)

    def forward(self, x):
        _, maxLen, _ = x.size()
        return self.peMatrix[:maxLen]


class MultiHeadAttention(nn.Module):
    def __init__(self, embDim, hiddenDim, headNum):
        super().__init__()
        self.headNum = headNum
        self.d_k = torch.tensor(embDim // headNum)
        assert embDim % headNum == 0, "划分多头错误"
        self.wQ = nn.Linear(hiddenDim, hiddenDim, device=device)
        self.wK = nn.Linear(hiddenDim, hiddenDim, device=device)
        self.wV = nn.Linear(hiddenDim, hiddenDim, device=device)

    def forward(self, x):
        # e.g. x -> (4, 30, 1024)  headNum->4
        oriShape = x.size()
        query = self.wQ(x)
        key = self.wK(x)
        value = self.wV(x)

        # (4, 4, 30, 256)
        query = query.reshape(batch_size, x.shape[1], self.headNum, -1).transpose(1, 2)
        key = key.reshape(batch_size, x.shape[1], self.headNum, -1).transpose(1, 2)
        value = value.reshape(batch_size, x.shape[1], self.headNum, -1).transpose(1, 2)

        # (4, 4, 30, 30)
        preAtt = query @ key.transpose(2, 3) / torch.sqrt(self.d_k)
        attScores = torch.softmax(preAtt, dim=-1)
        x = attScores @ value   # (4, 4, 30, 256)

        x = x.transpose(1, 2).reshape(oriShape)

        return x


class AddNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(hiddenDim, device=device)


    def forward(self, x):
        return self.norm(x)



class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.mtx = nn.Linear(hiddenDim, hiddenDim, device=device)

    def forward(self, x):
        return self.mtx(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiHeadAtt = MultiHeadAttention(embDim, hiddenDim, headNum)
        self.add_norm = AddNorm()
        self.feedForward = FeedForward()

    def forward(self, x):
        x = x + self.multiHeadAtt(x)
        x = self.add_norm(x)
        x = x + self.feedForward(x)
        x = self.add_norm(x)
        return x

class MyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = PositionalEncoding(MaxSeqLen, embDim)
        self.blocks = [Block() for _ in range(nBlocks)]


    def forward(self, x):
        posEmb = self.pe(x).to(device)
        x = x.to(device)
        x += posEmb
        for block in self.blocks:
            x = block.forward(x).to(device)
        return x


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(wordsNum, embDim)
        self.transformer = MyTransformer()

        self.cls = nn.Linear(embDim, classes)
        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        x = self.embedding(x)
        x = self.transformer(x)
        pred = self.cls(x)
        pred = torch.mean(pred, dim=1)

        if label is not None:
            return self.lossFn(pred, label)
        return torch.argmax(pred, dim=-1)




if __name__ == '__main__':
    classes = 10
    lr = 0.01
    epoch = 20
    batch_size = 64
    embDim = 1024
    hiddenDim = 1024
    nBlocks = 2
    MaxSeqLen = 60
    headNum = 4

    trainTexts, trainLabels = readData(r"../data/THUCNews/data/train.txt",
                                       sample=2000)
    validateTexts, validateLabels = readData(r"../data/THUCNews/data/dev.txt")

    assert len(trainTexts) == len(trainLabels)
    assert len(validateTexts) == len(validateLabels)

    words2Idx, idx2Words = getWords2Idx(trainTexts)
    wordsNum = len(words2Idx)

    trainDataset = MyDataset(trainTexts, trainLabels)
    validDataset = MyDataset(validateTexts, validateLabels)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size,
                             shuffle=False,
                             collate_fn=trainDataset.collateFn)
    validateLoader = DataLoader(validDataset, batch_size=batch_size,
                             shuffle=False,
                             collate_fn=validDataset.collateFn)

    model = MyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        COLOURS = list({'BLACK': '\x1b[30m', 'RED': '\x1b[31m', 'GREEN': '\x1b[32m',
                        'YELLOW': '\x1b[33m', 'BLUE': '\x1b[34m', 'MAGENTA': '\x1b[35m',
                        'CYAN': '\x1b[36m', 'WHITE': '\x1b[37m'})
        model.train()
        for batchIdx, batchLabel in tqdm(trainLoader, file=sys.stdout,
                                         total=len(trainLoader), colour=random.choice(COLOURS),
                                         desc="Training...", smoothing=0.01
                                         ):
            batchIdx, batchLabel = batchIdx.to(device), batchLabel.to(device)
            loss = model.forward(batchIdx, batchLabel)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # tqdm.write(f"Loss >> {loss:.5f}")
        print(f"Loss >> {loss:.5f}", flush=True)
        # sys.stdout.flush()  # 手动刷新缓冲区

        model.eval()
        bingo = 0
        for batchIdx, batchLabel in tqdm(validateLoader, file=sys.stdout,
                                         total=len(validateLoader), colour=random.choice(COLOURS),
                                         desc="Validating...", smoothing=0.01
                                         ):
            batchIdx, batchLabel = batchIdx.to(device), batchLabel.to(device)
            infer = model.forward(batchIdx)
            bingo += sum(infer == batchLabel)

        acc = bingo / len(validDataset)
        # tqdm.write(f"Acc >> {acc:.3%}")
        print(f"Acc >> {acc:.3%}", flush=True)
        # sys.stdout.flush()  # 手动刷新缓冲区




