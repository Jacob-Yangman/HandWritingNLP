"""
encoding: utf-8
    @Author: Jacob
    @Date  : 2024/11/14
    @Desc  : 
"""
import os
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pickle
import logging
from datetime import datetime

current_time = datetime.now().strftime("%m-%d-%H-%M-%S")

loggingPath = './logs/'
modelPath = './models'

# 确保路径存在
os.makedirs(loggingPath, exist_ok=True)
os.makedirs(modelPath, exist_ok=True)

logging.basicConfig(filename=os.path.join(loggingPath, f'training_{current_time}.log'), level=logging.INFO,
                    format='%(message)s')

random.seed(1113)
torch.manual_seed(1113)

device = "cuda" if torch.cuda.is_available() else "cpu"


def readData(filepath):
    with open(filepath, encoding="utf-8") as f:
        return f.read().splitlines()


def parseData(contents, sample=None):
    label_mapping = {
        100: 0, 101: 1, 102: 2, 103: 3, 104: 4,
        106: 5, 107: 6, 108: 7, 109: 8, 110: 9, 112: 10, 113: 11, 114: 12, 115: 13, 116: 14,
    }

    all_texts = list()
    all_labels = list()

    random.shuffle(contents)
    if sample:
        contents = contents[:sample]
    for line in tqdm(contents, desc="Parsing data..."):
        line_cut = line.split("_!_")
        try:
            newLabel = label_mapping.get(int(line_cut[1]))
            newTitle = line_cut[-2] if len(line_cut) == 5 else line_cut[-1]
            # print(f"{newLabel} -- {newTitle}")
        except:
            continue
        all_texts.append(newTitle)
        all_labels.append(newLabel)
    return all_texts, all_labels


def getChar2Vec(contents):
    global emb_len
    word2vec = dict(UNK=torch.normal(0, 1, size=(emb_len,)))
    for line in contents:
        for ch in line:
            word2vec[ch] = word2vec.get(ch, torch.normal(0, 1, size=(emb_len,)))
    return word2vec


class MyDataset(Dataset):

    def __init__(self, all_texts, all_labels):
        self.all_texts = all_texts
        self.all_labels = all_labels

    def __len__(self):
        return sum([True for _ in self.all_texts])

    def __getitem__(self, item):
        global word2vec
        currText = self.all_texts[item]
        currLabel = self.all_labels[item]

        wordEmb = [word2vec.get(c, word2vec["UNK"]) for c in currText]

        wordEmb = torch.stack(wordEmb)

        sentenceEmb = torch.mean(wordEmb, dim=0)
        # print(sentenceEmb.shape)

        return sentenceEmb, currLabel


class Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, 784),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(0.5),
        )
        # 分类器
        self.classifier = nn.Linear(256, out_dim)
        # 损失函数
        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        x = self.features(x)
        x = self.classifier(x)

        if label is not None:
            loss = self.lossFn(x, label)
            return loss
        else:
            pred_idx = torch.argmax(x, dim=-1)
            return pred_idx


def train(model, train_dataloader, optimizer):
    model.train()
    for batch_idx, (train_batch_texts, train_batch_labels) in tqdm(enumerate(train_dataloader), desc="Training..."):
        train_batch_texts = train_batch_texts.to(device)
        train_batch_labels = train_batch_labels.to(device)

        loss = model.forward(train_batch_texts, train_batch_labels)
        loss.backward()

        optimizer.step()
        # if batch_idx % zero_step == 0:
        #     optimizer.zero_grad()
        optimizer.zero_grad()
    print(f"\nLoss >> {loss.item():.4f}")

    return loss


def validate(model, test_dataset, test_dataloader):
    model.eval()
    with torch.no_grad():
        bingo = 0
        for test_batch_texts, test_batch_labels in tqdm(test_dataloader, desc="Validating..."):
            test_batch_texts = test_batch_texts.to(device)
            test_batch_labels = test_batch_labels.to(device)
            pred_idx = model.forward(test_batch_texts)

            bingo += sum(pred_idx == test_batch_labels).item()

        acc = bingo / len(test_dataset)
    print(f"\nAccuracy >> {acc:.3%}\n")

    return acc


if __name__ == '__main__':
    # 读取数据
    contents = readData("./data/toutiao_cat_data.txt")
    all_texts, all_labels = parseData(contents)

    # 超参
    classes = 15
    epoch = 200
    lr = 0.001
    batch_size = 64
    emb_len = 2048
    zero_step = 3

    # 记录超参
    logging.info("Hyperparameters:")
    logging.info(
        f"""* lr: {lr}
* epoch: {epoch}
* batch_size: {batch_size}
* embedding_length: {emb_len}
        """
    )

    # 划分训练集和验证集
    test_split_rate = 0.1

    train_texts, train_labels = (all_texts[:int(len(all_texts) * (1 - test_split_rate))],
                                 all_labels[:int(len(all_labels) * (1 - test_split_rate))])

    test_texts, test_labels = (all_texts[-int(len(all_texts) * test_split_rate):],
                               all_labels[-int(len(all_labels) * test_split_rate):])

    # 构建词向量字典
    word2vec = getChar2Vec(train_texts)

    # dataset
    train_dataset = MyDataset(train_texts, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDataset(test_texts, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False)

    # 定义模型
    model = Model(emb_len, classes).to(device)

    # 记录模型
    logging.info(model.__repr__())

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 学习率策略
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.5, end_factor=1.0, total_iters=20)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.08, patience=3, threshold=1e-8, cooldown=0)

    # 训练循环
    for e in range(epoch):
        currLR = optimizer.param_groups[0]['lr']
        print(f"\n********** Epoch No.{e + 1} **********  LR > {currLR}")
        logging.info(f'Epoch {e + 1}')
        # 记录学习率
        # logging.info(f'Epoch {e + 1} - Learning Rate: {currLR:.5f}')
        # 训练
        loss = train(model, train_dataloader, optimizer)
        # logging.info(f'Epoch {e + 1} - Training Loss: {loss.item():.4f}')
        # 验证
        acc = validate(model, test_dataset, test_dataloader)
        # logging.info(f'Epoch {e + 1} - Validation Accuracy: {acc:.3%}')
        logging.info(f'- Learning Rate: {currLR:.5f}\tTraining Loss: {loss.item():.4f}\tValidation Accuracy: {acc:.3%}')
        logging.info("-" * 60)

        scheduler.step(acc)
        # if acc > 0.73 and e % 3 == 0:
        #     optimizer.param_groups[0]['lr'] *= 0.8
        # elif acc > 0.71 and e % 2 == 0:
        #     optimizer.param_groups[0]['lr'] *= 0.9

        # 保存模型
        if acc > 0.85:
            with open(os.path.join(modelPath, fr"{int(acc * 1e5)}.pkl"), "wb") as f:
                pickle.dump(model, f)
