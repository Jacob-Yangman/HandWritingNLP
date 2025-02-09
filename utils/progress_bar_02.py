# coding=utf-8
"""
@Author: Jacob Y
@Date  : 1/7/2025
@Desc  : 使用rich进度条
"""
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from rich.progress import Progress


# 简单的文本分类数据集示例
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# 假设已经将文本转换为数字表示，且标签为0或1（简单二分类）
texts = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7]]
labels = [0, 1, 0, 1]

dataset = TextDataset(texts, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# 定义简单的文本分类模型
class TextClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# 模型初始化
input_dim = 3  # 输入的维度，这里假设每个文本有3个数字
output_dim = 2  # 输出的维度（分类数），假设是二分类
model = TextClassifier(input_dim, output_dim)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 训练函数，带有rich进度条
def train(model, dataloader, criterion, optimizer, epochs=5):
    model.train()

    with Progress() as progress:
        # 创建一个进度条，设定总步骤数
        task = progress.add_task("[cyan]Training...", total=len(dataloader) * epochs)

        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs.float())  # 假设输入是float类型
                loss = criterion(outputs, targets)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 更新进度条
                progress.update(task, advance=1)
                if batch_idx % 10 == 0:  # 每10个batch输出一次损失
                    print(
                        f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")


# 开始训练
train(model, dataloader, criterion, optimizer, epochs=5)
