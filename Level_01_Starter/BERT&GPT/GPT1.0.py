# coding=utf-8
"""
@Author: Jacob Y
@Date  : 12/25/2024
@Desc  : 
"""

import torch
from torch.utils.data import DataLoader, Dataset
import random
import time




def readData(filePath):
    with open(filePath, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    words2Idx = {"PAD": 0, "UNK": 1, "MASK": 2, "EOS": 3}
    result = []

    for data in all_data:
        if len(data) < 10:
            continue
        # data1,data2 = data.split("\n")
        result.append(data)



        for w in data:
            words2Idx[w] = words2Idx.get(w, len(words2Idx))
        # for w in data2:
        #     if w not in words2Idx:
        #         words2Idx[w] = len(words2Idx)

    return result, words2Idx


class MyDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        input_data = data

        input_data_idx = [words2Idx.get(i, 1) for i in input_data][:max_len - 1] + [
            words2Idx["EOS"]]  # 转 idx ， 截断 ， 加上 EOS 结束符
        input_data_idx = input_data_idx + [0] * (max_len + 1 - len(input_data_idx))  # 填充

        label_data_idx = input_data_idx[1:]
        input_data_idx = input_data_idx[:-1]

        return torch.tensor(input_data_idx), torch.tensor(label_data_idx)


class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position_emb = torch.nn.Embedding(max_len, hidden_num)
        self.token_emb = torch.nn.Embedding(len(words2Idx), hidden_num)

    def forward(self, batch_input_idx):
        token_emb = self.token_emb(batch_input_idx)
        position_emb = self.position_emb(torch.tensor([i for i in range(batch_input_idx.shape[1])]))

        return token_emb + position_emb


class Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_num)

    def forward(self, x):
        return self.norm(x)


class Feed_Forward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(hidden_num, hidden_num)
        self.act = torch.nn.SiLU()
        self.w2 = torch.nn.Linear(hidden_num, hidden_num)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class Multi_Head_Att(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Q = torch.nn.Linear(hidden_num, hidden_num)
        self.K = torch.nn.Linear(hidden_num, hidden_num)
        self.V = torch.nn.Linear(hidden_num, hidden_num)

    def forward(self, x, pad_mask):  # 1 * 32 * 768
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        weight = q @ k.transpose(1, 2)

        # weight.masked_fill_(,-1e10)

        look_ahead_mask = torch.triu(torch.ones(size=(*pad_mask.shape, pad_mask.shape[-1])), diagonal=1)
        attention_mask = pad_mask.unsqueeze(-1).expand(-1, -1, pad_mask.shape[-1]).transpose(1, 2)

        mask = (look_ahead_mask + attention_mask) > 0

        weight.masked_fill_(mask, -1e10)

        score = torch.softmax(weight / torch.sqrt(torch.tensor(hidden_num)), dim=-1)  # batch * seq_len * seq_len
        result = score @ v

        return result


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_att_mask = Multi_Head_Att()
        self.norm1 = Norm()
        self.multi_att = Multi_Head_Att()
        self.norm2 = Norm()
        self.feed_forward = Feed_Forward()
        self.norm3 = Norm()

    def forward(self, x, pad_mask):
        output = self.multi_att_mask.forward(x, pad_mask)
        x = self.norm1(output + x)

        output = self.multi_att.forward(x, pad_mask)
        x = self.norm2.forward(output + x)

        output = self.feed_forward(x)
        x = self.norm3.forward(output + x)

        return x


class GPT_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = Embedding()
        self.blocks = torch.nn.ModuleList([Block() for i in range(layer_num)])

    def forward(self, batch_input_idx):
        pad_mask = (batch_input_idx == 0)

        input_emb = self.emb.forward(batch_input_idx)

        x = input_emb
        for block in self.blocks:
            x = block.forward(x, pad_mask)
        return x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt = GPT_model()
        # self.linear = torch.nn.Module([i for i in range(100000)])
        self.cls = torch.nn.Linear(hidden_num, len(words2Idx), bias=False)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, batch_input_idx, batch_label_idx=None):

        gpt_output = self.gpt.forward(batch_input_idx)

        predict = self.cls(gpt_output)

        if batch_label_idx is not None:
            loss = self.loss(predict.reshape(-1, predict.shape[-1]), batch_label_idx.reshape(-1))

            return loss
        else:
            return predict


def generate_greedy_search(prompt):
    input_idx = [words2Idx.get(i, 1) for i in prompt]

    result = ""
    i = 0
    while True:
        if i >= max_len:
            break
        input_idx_tensor = torch.tensor([input_idx])

        pre = model.forward(input_idx_tensor)
        pre_idx = int(torch.argmax(pre[0][-1]))

        if pre_idx == 3:
            break
        new_word = index_2_word[pre_idx]
        result += new_word

        input_idx = input_idx + [pre_idx]

        i += 1

    print(result)


def generate_random_search(prompt, tok_n=5):
    input_idx = [words2Idx.get(i, 1) for i in prompt]

    # result = ""
    i = 0
    while True:
        if i > max_len:
            break
        input_idx_tensor = torch.tensor([input_idx])

        pre = model.forward(input_idx_tensor)
        pre_idxs = torch.argsort(pre[0][-1]).tolist()[::-1][:tok_n]

        pre_idx = random.choice(pre_idxs)

        if pre_idx == 3:
            break
        new_word = index_2_word[pre_idx]
        # result += new_word
        print(new_word, end="")

        input_idx = input_idx + [pre_idx]

        time.sleep(random.random() * 0.1)

        i += 1

    print("")


def generate_rate_search(prompt, tok_n=5):
    input_idx = [words2Idx.get(i, 1) for i in prompt]

    # result = ""
    i = 0
    while True:
        if i > max_len:
            break

        input_idx_tensor = torch.tensor([input_idx])

        pre = model.forward(input_idx_tensor)
        pre_idxs = torch.argsort(pre[0][-1])[-tok_n:]
        pre_rate = torch.softmax(pre[0][-1][pre_idxs] / 10, dim=-1)

        pre_idx = random.choices(pre_idxs.tolist(), pre_rate.tolist())[0]

        if pre_idx == 3:
            break
        new_word = index_2_word[pre_idx]

        print(new_word, end="")

        input_idx = input_idx + [pre_idx]

        time.sleep(random.random() * 0.1)

        i += 1
    print("")


# if __name__ == "__main__":
#     datas, words2Idx = read_data_keyan()
#     index_2_word = list(words2Idx)
#
#     max_len = 64
#     batch_size = 2
#     epoch = 10
#     layer_num = 2
#     hidden_num = 768
#     lr = 0.0005
#
#     train_dataset = MyDataset(datas)
#     train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
#
#     model = Model()
#     opt = torch.optim.Adam(model.parameters(), lr)
#
#     for e in range(epoch):
#         for batch_idx, (batch_input_idx, batch_label_idx) in enumerate(train_dataloader):
#
#             loss = model.forward(batch_input_idx, batch_label_idx)
#             loss.backward()
#             opt.step()
#             opt.zero_grad()
#
#             if batch_idx % 10 == 0:
#                 print(loss.item())
#
#     while True:
#         input_x = input("请输入：")
#         generate_greedy_search(input_x)

if __name__ == '__main__':
    filePath = r"../data/假如给我三天光明.txt"
    contentLst, words2Idx = readData(filePath)
    ...