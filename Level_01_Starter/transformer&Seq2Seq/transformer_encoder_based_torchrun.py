# coding=utf-8
"""
@Author: Jacob Y
@Date  : 12/15/2024
@Desc  : 
"""
import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import time
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


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
        self.d_k = torch.tensor(embDim // headNum, device=device)
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
        posEmb = self.pe(x)
        x += posEmb
        for block in self.blocks:
            x = block.forward(x)
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
    """初始化分布式环境"""
    dist.init_process_group(backend='nccl')  # 使用 NCCL 后端 (推荐用于多GPU)
    rank = dist.get_rank()  # 当前进程的 rank
    world_size = dist.get_world_size()  # 总进程数
    torch.cuda.set_device(rank)  # 将每个进程绑定到不同的 GPU
    device = torch.device(f'cuda:{rank}')


    classes = 10
    lr = 0.007
    epoch = 20
    batch_size = 64
    embDim = 1024
    hiddenDim = 1024
    nBlocks = 2
    MaxSeqLen = 60
    headNum = 4

    trainTexts, trainLabels = readData(r"./data/Thuc_train.txt",
                                       sample=2000)
    validateTexts, validateLabels = readData(r"./data/Thuc_dev.txt")

    assert len(trainTexts) == len(trainLabels)
    assert len(validateTexts) == len(validateLabels)

    words2Idx, idx2Words = getWords2Idx(trainTexts)
    wordsNum = len(words2Idx)

    trainDataset = MyDataset(trainTexts, trainLabels)
    validDataset = MyDataset(validateTexts, validateLabels)

    """使用 torch.utils.data.DistributedSampler 对数据集进行分布式采样"""
    trainSampler = DistributedSampler(trainDataset, num_replicas=world_size, rank=rank)
    validSampler = DistributedSampler(validDataset, num_replicas=world_size, rank=rank)

    trainLoader = DataLoader(
        trainDataset, batch_size=batch_size, sampler=trainSampler, collate_fn=trainDataset.collateFn
    )
    validateLoader = DataLoader(
        validDataset, batch_size=batch_size, sampler=validSampler, collate_fn=validDataset.collateFn
    )



    """包裹模型为DDP"""
    model = MyModel().to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        model.train()
        for batchIdx, batchLabel in tqdm(trainLoader, file=sys.stdout):
            batchIdx, batchLabel = batchIdx.to(device), batchLabel.to(device)
            loss = model.forward(batchIdx, batchLabel)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # tqdm.write(f"Loss >> {loss:.5f}")
        print(f"Loss >> {loss:.5f}", flush=True)
        # sys.stdout.flush()  # 手动刷新缓冲区


        """汇总验证指标"""
        model.eval()
        # bingo = 0
        bingo = torch.tensor(0, device=device)
        for batchIdx, batchLabel in tqdm(validateLoader, file=sys.stdout):
            batchIdx, batchLabel = batchIdx.to(device), batchLabel.to(device)
            infer = model.forward(batchIdx)
            bingo += torch.sum(infer == batchLabel)

        # 汇总所有GPU的bingo
        dist.all_reduce(bingo, op=dist.ReduceOp.SUM)
        acc = bingo.item() / len(validDataset)

        if rank == 0:  # 仅主进程打印结果
            print(f"Acc >> {acc:.3%}", flush=True)

        # acc = bingo / len(validDataset)
        # # tqdm.write(f"Acc >> {acc:.3%}")
        # print(f"Acc >> {acc:.3%}", flush=True)
        # sys.stdout.flush()  # 手动刷新缓冲区




