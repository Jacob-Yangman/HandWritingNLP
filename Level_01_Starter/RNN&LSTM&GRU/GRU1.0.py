# coding=utf-8
"""
@Author: Jacob Y
@Date  : 12/08/2024
@Desc  : 
"""
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
torch.manual_seed(1208)
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


class MyLSTM(nn.Module):
    def __init__(self, inputSize, hiddenDim, batch_first=False, bias=True):
        super().__init__()
        self.batch_first = batch_first
        self.W1 = nn.Linear(inputSize, hiddenDim)
        self.W2 = nn.Linear(inputSize, hiddenDim)
        self.W3 = nn.Linear(inputSize, hiddenDim)
        self.W4 = nn.Linear(hiddenDim, inputSize)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.bias = nn.Parameter(torch.normal(0, 1, size=(1, hiddenDim))) if bias else 0

    def forward(self, x, hx=None):
        if self.batch_first:
            resultEmbs = torch.zeros_like(x)
        else:
            resultEmbs = torch.zeros_like(x.transpose(0, 1))

        if self.batch_first:
            hiddenUp, hiddenDown = hx if hx is not None else (torch.zeros_like(x[:, 0, None]),
                                            torch.zeros_like(x[:, 0, None]))

            for t in range(x.shape[1]):
                xt = x[:, t, None, :]
                sum_ = hiddenDown + xt
                forgetGate = self.sigmoid(self.W1(sum_) + self.bias)
                inputGate = (self.sigmoid(self.W2(sum_) + self.bias)
                             * self.tanh(self.W3(sum_) + self.bias))

                cellUpdate = hiddenUp * forgetGate + inputGate

                hiddenUp = cellUpdate

                outputGate = self.sigmoid(self.W4(sum_) + self.bias) * self.tanh(cellUpdate)

                hiddenDown = outputGate.clone()

                resultEmbs[:, t, None, :] = hiddenDown
        return resultEmbs, (hiddenUp, hiddenDown)


class MyGRUCell(nn.Module):
    def __init__(self, inputSize, hiddenDim, batch_first=False, bias=True):
        super().__init__()
        # 定义更新门的参数
        self.Wz = nn.Linear(inputSize, hiddenDim)
        self.Uz = nn.Linear(hiddenDim, hiddenDim)

        # 定义重置门的参数
        self.Wr = nn.Linear(inputSize, hiddenDim)
        self.Ur = nn.Linear(hiddenDim, hiddenDim)

        # 定义候选隐状态的参数
        self.Wh = nn.Linear(inputSize, hiddenDim)
        self.Uh = nn.Linear(hiddenDim, hiddenDim)

    def forward(self, x, hx=None):
        # 更新门 z_t
        z_t = torch.sigmoid(self.Wz(x) + self.Uz(hx))

        # 重置门 r_t
        r_t = torch.sigmoid(self.Wr(x) + self.Ur(hx))

        # 候选隐状态 \tilde{h}_t
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(r_t * hx))

        # 最终隐状态 h_t
        h_t = (1 - z_t) * hx + z_t * h_tilde
        return h_t


        return resultEmbs, hx


class MyGRU(nn.Module):
    def __init__(self, inputSize, hiddenDim, num_layers=1, batch_first=False):
        super(MyGRU, self).__init__()
        self.hiddenDim = hiddenDim
        self.num_layers = num_layers
        self.batch_first = batch_first

        # 多层 GRU
        self.cells = nn.ModuleList([
            MyGRUCell(inputSize if i == 0 else hiddenDim, hiddenDim)
            for i in range(num_layers)
        ])

    def forward(self, x, h0=None):
        # 处理 batch_first 情况
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hiddenDim, device=x.device)

        outputs = []
        hx = h0
        for t in range(seq_len):
            x_t = x[t]
            h_t_list = []
            for layer in range(self.num_layers):
                h_t = self.cells[layer](x_t, hx[layer])
                h_t_list.append(h_t)
                x_t = h_t
            hx = torch.stack(h_t_list)
            outputs.append(h_t)

        outputs = torch.stack(outputs)  # 转换为 (seq_len, batch_size, hiddenDim)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, hx



class MyModel(nn.Module):
    def __init__(self, inFeatures, embDim, outFeatures):
        super().__init__()
        self.embd = nn.Embedding(inFeatures, embDim)
        # self.lstm = MyLSTM(embDim, 1024, batch_first=True)
        self.gru = MyGRU(embDim, 1024, num_layers=2)
        self.cls = nn.Linear(1024, outFeatures)
        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, x, label):
        emb = self.embd(x)
        x1, x2 = self.gru(emb)
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