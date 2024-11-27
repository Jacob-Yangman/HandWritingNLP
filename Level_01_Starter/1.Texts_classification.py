"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/11/01
    @Desc  : 文本十五分类
"""
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import jieba
import logging

logging.getLogger('jieba').setLevel(logging.ERROR)

device = "cuda"
torch.manual_seed(1120)


def softmax(data):
    ex = torch.exp(data)
    exSum = torch.sum(ex, dim=-1)
    return ex / exSum


def readData(filepath, sample=None):
    with open(filepath, encoding='utf-8') as f:
        if sample:
            return f.read().splitlines()[:sample]
        return f.read().splitlines()


def processData(data, isTrain=False, trainClasses=None):
    texts = list()
    labels = list()

    for line in data:
        try:
            lineCut = line.split("_!_")
            text = lineCut[-2]
            if not text:
                raise Exception
            # jieba cut
            lineWordsCut = jieba.lcut(text)
            label = int(lineCut[1])
            if not isTrain:
                label = trainClasses[label]
        except:
            continue
        texts.append(lineWordsCut)
        labels.append(label)

    if isTrain:
        labelsMapping = {l: i for i, l in enumerate(set(labels))}
        labels = [labelsMapping[l] for l in labels]
        return texts, labels, labelsMapping
    return texts, labels


def makeWords2Index(train_texts):
    words2idx = dict(PAD=0, UNK=1)
    for line in train_texts:
        for word in line:
            words2idx[word] = words2idx.get(word, len(words2idx))
    return words2idx, list(words2idx)


def makeWords2Emb(length):
    global embLen
    return torch.normal(0, 1, size=(length, embLen))


def collateFunc(batchData):
    global words2idx, words2emb
    batch_texts, batch_labels = zip(*batchData)
    currBatchMaxLen = max([len(text) for text in batch_texts])  # 取最大长度

    batchEmbs = list()

    for text in batch_texts:
        # Encoding
        textIdx = [words2idx.get(ch, words2idx["UNK"]) for ch in text]

        textIdx += [words2idx["PAD"]] * (currBatchMaxLen - len(textIdx))  # 填充

        # Embedding
        textEmb = [words2emb[i] for i in textIdx]
        # Stack
        textEmb = torch.stack(textEmb)
        # Pooling
        # sentenceEmb = torch.mean(textEmb, dim=0)

        batchEmbs.append(textEmb)

    batchEmbs = torch.stack(batchEmbs)
    batchLabels = torch.tensor(batch_labels)

    return batchEmbs, batchLabels


class MyDataset(Dataset):

    def __init__(self, allTexts, allLabels, isTrain=False):
        self.allTexts = allTexts
        self.allLabels = allLabels
        self.isTrain = isTrain

    def __len__(self):
        return sum([True for _ in self.allTexts])

    def __getitem__(self, item):
        global classes
        currText = self.allTexts[item]
        currlabel = self.allLabels[item]

        # TrainLabelOneHotEncoding
        if self.isTrain:
            currLabelOneHot = np.zeros(classes, )
            currLabelOneHot[currlabel] = 1.

            return currText, currLabelOneHot
        return currText, currlabel


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(embLen, hiddenNum)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(hiddenNum, classes)
        # self.lossFn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, label=None):
        x = self.linear(x)
        x = self.act(x)
        pred = self.classifier(x)
        pred = torch.mean(pred, dim=1)
        p = self.softmax(pred)
        if label is not None:
            # return self.lossFn(pred, label)
            return - torch.mean(label * torch.log(p))
        return torch.argmax(p, dim=1)


def train(model, train_dataloader, optimizer):
    model.train()
    for batch_texts, batch_labels in tqdm(trainDataloader, desc="Training..."):
        batch_texts = batch_texts.to(device)
        batch_labels = batch_labels.to(device)
        loss = model.forward(batch_texts, batch_labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    print(f"\n{loss}")


def validate(model, validateDataloader, validateDataset):
    model.eval()
    bingo = 0
    for batch_texts, batch_labels in tqdm(validateDataloader, desc="Validating..."):
        batch_texts = batch_texts.to(device)
        batch_labels = batch_labels.to(device)
        pred = model.forward(batch_texts)
        bingo += int(sum(pred == batch_labels))
    acc = bingo / len(validateDataset)
    print(f"\nAcc >> {acc:.3%}")


if __name__ == '__main__':
    train_data = readData('./data/toutiao_cat_data.txt', sample=50000)
    validate_data = readData('./data/test_data.txt')

    train_texts, train_labels, trainClasses = processData(train_data, isTrain=True)
    validate_texts, validate_labels = processData(validate_data, trainClasses=trainClasses)

    assert len(train_texts) == len(train_labels)
    assert len(validate_texts) == len(validate_labels)

    # 超参
    classes = len(trainClasses)
    embLen = 128
    hiddenNum = 1024
    # maxLen = 40

    lr = 0.001
    epoch = 10
    batch_size = 20

    # 编码
    words2idx, idx2words = makeWords2Index(train_texts)

    # global emb [Matrix]
    words2emb = makeWords2Emb(len(words2idx))

    model = MyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainDataset = MyDataset(train_texts, train_labels, isTrain=True)
    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, collate_fn=collateFunc)

    validateDataset = MyDataset(validate_texts, validate_labels)
    validateDataloader = DataLoader(validateDataset, batch_size=8, shuffle=False, collate_fn=collateFunc)

    for e in range(epoch):
        train(model, trainDataloader, optimizer)
        validate(model, validateDataloader, validateDataset)
