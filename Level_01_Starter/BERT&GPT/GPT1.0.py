# coding=utf-8
"""
@Author: Jacob Y
@Date  : 1/5/2025
@Desc  : 
"""
import json


def readData(path):
    queries, responses = [], []
    with open(path, encoding="utf-8") as f:
        content = json.load(f)

    for data in content:
        queries.append(data['instruction'])
        responses.append(data['output'])
    return queries, responses


if __name__ == '__main__':

    filePath = "../data/identity_DIY.json"
    queries, responses = readData(filePath)
    ...