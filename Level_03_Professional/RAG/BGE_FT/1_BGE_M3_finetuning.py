# coding=utf-8
"""
@Author: Jacob Y
@Date  : 2/10/2025
@Desc  : 
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("../models/bge-m3")
model = AutoModel.from_pretrained("../models/bge-m3")


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['text'], item['label']


# 数据预处理函数
def collate_fn(batch):
    texts, labels = zip(*batch)
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    return encoded, torch.tensor(labels)


# 训练函数
def train(model, train_loader, optimizer, scheduler, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)

            # 这里使用输出的最后一层隐藏状态的平均值
            embeddings = outputs.last_hidden_state.mean(dim=1)

            # 根据任务类型定义损失函数（这里以对比学习为例）
            loss = torch.nn.functional.cosine_embedding_loss(
                embeddings[::2],
                embeddings[1::2],
                labels[::2]
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")


# 主函数
def main():

    model.to(device)

    # 加载数据
    train_dataset = CustomDataset("../data/train_data.json")
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 优化器设置
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 学习率调度器
    num_training_steps = len(train_loader) * 3  # 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    train(model, train_loader, optimizer, scheduler, device)

    # 保存模型
    model.save_pretrained("./fine_tuned_bge_m3")
    tokenizer.save_pretrained("./fine_tuned_bge_m3")


if __name__ == "__main__":
    main()