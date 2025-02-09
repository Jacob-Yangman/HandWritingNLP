# coding=utf-8
"""
@Author: Jacob Y
@Date  : 1/5/2025
@Desc  : 使用tqdm随机彩色进度条，并防止和print标准输出流（stdout）混合使用导致显示错乱
"""
import random
import sys
import time

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MyDataset(Dataset):
    pass


if __name__ == '__main__':
    epoch = 10
    dataset = MyDataset()
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    for e in range(epoch):
            COLOURS = list({'BLACK': '\x1b[30m', 'RED': '\x1b[31m', 'GREEN': '\x1b[32m',
                       'YELLOW': '\x1b[33m', 'BLUE': '\x1b[34m', 'MAGENTA': '\x1b[35m',
                       'CYAN': '\x1b[36m', 'WHITE': '\x1b[37m'})
            for n,(batch_emb,batch_label) in tqdm(enumerate(train_dataloader,start=1),
                                                  total=len(train_dataloader),
                                                  colour=random.choice(COLOURS),
                                                  desc="训练中",
                                                  smoothing=0.01,
                                                  file=sys.stdout):
                ...


                time.sleep(0.1)
                print(f"Processing epoch {e}", flush=True)
