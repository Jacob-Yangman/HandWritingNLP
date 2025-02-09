# coding=utf-8
"""
@Author: Jacob Y
@Date  : 1/9/2025
@Desc  : 
"""
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import logging

import fire

# 配置日志记录
logging.basicConfig(
    filename='training_summary.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='a'  # 确保以追加模式打开
)


class LocalDataset(Dataset):
    ...


class LocalModel(nn.Module):
    ...


# 定义训练函数
def train(max_epoch, batch_size, lr):
    # 记录当前参数
    logging.info(f"Training started with max_epoch={max_epoch}, batch_size={batch_size}, lr={lr}")

    # 模拟训练过程
    print(f"Training with max_epoch={max_epoch}, batch_size={batch_size}, lr={lr}")
    for epoch in range(max_epoch):
        # 记录每个 epoch 的信息
        logging.info(f"Epoch {epoch + 1}/{max_epoch} completed with current_lr={lr}")


# 使用 fire 启动命令行接口
if __name__ == '__main__':
    fire.Fire(train)