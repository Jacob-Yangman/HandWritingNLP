# coding=utf-8
"""
@Author: Jacob Y
@Date  : 12/3/2024
@Desc  : 手写最朴素的RNN层
"""
import fire
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
torch.manual_seed(1203)
from tqdm import tqdm


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


class MyRNN:
    """
    batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
    """
    def __init__(self, inFeatures, outFeatures, batch_first=False):
        super().__init__()
        self.rnnMtx = nn.Linear(inFeatures, outFeatures)
        self.batch_first = batch_first



    def forward(self, x):
        """
        x > (batch, seq, features)
        """
        lastSum = 0
        if self.batch_first:
            resultEmbs = torch.zeros_like(x)
        else:
            resultEmbs = torch.zeros_like(x.transpose(0, 1))

        for i in range(x.shape[1]):
            lastSum = self.rnnMtx(lastSum + x[:, i, None, :])
            if self.batch_first:
                resultEmbs[:, i, None, :] = lastSum
            else:
                resultEmbs[i, None, :, :] = lastSum.transpose(0, 1)
        return (resultEmbs, lastSum) if self.batch_first else (resultEmbs, lastSum.transpose(0, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MyModel(nn.Module):
    def __init__(self, inFeatures, embDim, outFeatures):
        super().__init__()
        self.embd = nn.Embedding(inFeatures, embDim)
        # self.rnn = nn.RNN(embDim, 1024, batch_first=True)
        self.rnn = MyRNN(embDim, 1024, batch_first=True)
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
    train(lr, epoch, batch_size)
    # fire.Fire(train)