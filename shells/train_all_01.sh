#!/bin/bash

# 调用训练脚本时记录参数
python train.py --max_epoch=6 --batch_size=64 --lr=0.00
python train.py --max_epoch=6 --batch_size=128 --lr=0.001
python train.py --max_epoch=6 --batch_size=256 --lr=0.001
python train.py --max_epoch=6 --batch_size=64 --lr=0.01
python train.py --max_epoch=6 --batch_size=64 --lr=0.1
python train.py --max_epoch=6 --batch_size=64 --lr=0.001
