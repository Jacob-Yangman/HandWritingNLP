# coding=utf-8
"""
@Author: Jacob Y
@Date  : 12/04/2024
@Desc  : 
"""
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
torch.seed(1204)

class LocalDataset(Dataset):
    def __init__(self, dataPath, sample=None, isTrain=False):
        self.allTexts, self.allLabels = self.readData(dataPath, sample)
        assert len(self.allTexts) == len(self.allLabels)
        if isTrain:
            self.words2Idx, self.idx2Words = self.genWordsMapping()

    def readData(self, path, sample=None):
        allTexts, allLabels = [], []
        with open(path, encoding="utf-8") as f:
            if sample:
                content = f.read().splitlines()[:sample]
            else:
                content = f.read().splitlines()
        for line in content:
            lineCut = line.split("\t")
            try:
                text, label = lineCut
                label = int(label)
            except:
                continue
            allTexts.append(text)
            allLabels.append(label)
        return allTexts, allLabels

    def genWordsMapping(self):
        words2Idx = dict(PAD=0, UNK=1)
        for line in self.allTexts:
            for ch in line:
                words2Idx[ch] = words2Idx.get(ch, len(words2Idx))
        return words2Idx, list(words2Idx)

    def collateFn(self, batchData):
        batchTextCodings, batchLabels, lengths = zip(*batchData)
        batchMaxLen = max(lengths)

        # 填充
        for textidxs in batchTextCodings:
            textidxs += [self.words2Idx["PAD"]] * (batchMaxLen - len(textidxs))

        return torch.tensor(batchTextCodings), torch.tensor(batchLabels)

    def __getitem__(self, item):
        currText = self.allTexts[item]
        currLabel = self.allLabels[item]

        # 编码
        txtCoding = [self.words2Idx.get(ch, self.words2Idx["UNK"]) for ch in currText]

        return txtCoding, currLabel, len(txtCoding)

    def __len__(self):
        return len(self.allTexts)


class MyRNN(nn.Module):
    def __init__(self, embDim, hiddenDim, batch_first=False, bias=True):
        super().__init__()
        self.batch_first = batch_first
        self.W = nn.Linear(embDim, hiddenDim)
        self.U = nn.Linear(hiddenDim, hiddenDim)
        self.V = nn.Linear(hiddenDim, embDim)
        self.activation = nn.Tanh()
        self.bias = nn.Parameter(torch.normal(0, 1, size=(1, hiddenDim))) if bias else None

    def forward(self, x, hx=None):
        if self.batch_first:
            resultEmbs = torch.zeros_like(x)
        else:
            resultEmbs = torch.zeros_like(x.transpose(0, 1))


        if not hx: hx = torch.zeros_like(x[:, 0, None])
        for i in range(x.shape[1]):

            hx = self.activation(self.U(hx) +
                                          self.W(x[:, i, None]) + (self.bias if self.bias is not None else 0))

            if self.batch_first:
                resultEmbs[:, i, None, :] = hx
            else:
                resultEmbs[i, None, :, :] = hx.transpose(0, 1)

        return (resultEmbs, hx) if self.batch_first else (resultEmbs, hx.transpose(0, 1))

class MyModel(nn.Module):
    def __init__(self, inFeatures, embDim, outFeatures):
        super().__init__()
        self.embd = nn.Embedding(inFeatures, embDim)
        # self.rnn = nn.RNN(embDim, 1024, batch_first=True)
        self.rnn = MyRNN(embDim, hiddenDim=hiddenDim, batch_first=True, bias=True)
        self.cls = nn.Linear(1024, outFeatures)
        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, x, label):
        emb = self.embd(x)
        x1, x2 = self.rnn(emb)
        pred = self.cls(x1)
        predMean = torch.mean(pred, dim=1).squeeze()
        if label is not None:
            return self.lossFn(predMean, label)
        return torch.argmax(pred, dim=-1)


def train(lr, epoch, batch_size):
    trainDataPath = r"../data/THUCNews/data/train.txt"
    devDataPath = r"../data/THUCNews/data/dev.txt"
    trainDataset = LocalDataset(trainDataPath, sample=1000, isTrain=True)
    devDataset = LocalDataset(devDataPath)
    devDataset.words2Idx, devDataset.idx2Words = trainDataset.words2Idx, trainDataset.idx2Words
    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=trainDataset.collateFn)
    devDataloader = DataLoader(devDataset, batch_size=batch_size, shuffle=False)

    wordsNum = len(trainDataset.words2Idx)
    embDim = 1024
    classes = 10


    model = MyModel(wordsNum, embDim, classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for e in range(epoch):
        for batchTexts, batchLabels in tqdm(trainDataloader):
            loss = model.forward(batchTexts, batchLabels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print(f"\nLoss >> {loss:.3f}")


if __name__ == '__main__':
    lr = 0.01
    epoch = 10
    batch_size = 8
    hiddenDim = 1024
    train(lr, epoch, batch_size)
    # fire.Fire(train)