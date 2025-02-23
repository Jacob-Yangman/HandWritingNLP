# coding=utf-8
"""
@Author: Jacob Y
@Date  : 1/14/2025
@Desc  : 
"""
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer

def readData(path):
    with open(path, encoding="utf-8") as f:
        allData = f.read().split("\n\n")


class MyDataset(Dataset):
    def __init__(self):
        pass


    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


if __name__ == '__main__':
    readData("data/trainNER.txt")
