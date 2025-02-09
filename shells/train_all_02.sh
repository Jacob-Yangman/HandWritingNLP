#!/bin/bash

# 设定参数组合
batch_sizes=(64 128 256)
lrs=(0.001 0.01 0.1)
lr2s=(0.000 0.0001 0.001)

# 创建日志文件目录
log_dir="./logs"
mkdir -p "$log_dir"

# 循环遍历每个参数组合
for batch_size in "${batch_sizes[@]}"; do
  for lr in "${lrs[@]}"; do
    for lr2 in "${lr2s[@]}"; do
      # 构建日志文件名
      log_file="$log_dir/train_batch${batch_size}_lr${lr}_lr2${lr2}.log"

      # 运行训练脚本，并将输出写入日志文件
      echo "Running training with batch_size=${batch_size}, lr=${lr}, lr2=${lr2}..."
      python main.py main --max_epoch=6 --batch_size=${batch_size} --lr=${lr} --lr2=${lr2} > "$log_file" 2>&1

      # 输出日志文件路径
      echo "Training complete. Logs saved to: $log_file"
    done
  done
done
