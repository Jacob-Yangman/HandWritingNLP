# coding=utf-8
"""
@Author: Jacob Y
@Date  : 11/23/2024
@Desc  : 加载词向量和词表，根据余弦相似度测试训练效果
"""
import os
from sklearn.metrics.pairwise import cosine_similarity
import pickle

LoadingDir = r"./output"


def loadWords2Idx_Embs(filePath):
    with open(filePath, "rb") as file:
        return pickle.load(file)


def calSimilarity(word01, word02) -> float:
    global word2Idx
    word01Idx = word2Idx.get(word01)
    word02Idx = word2Idx.get(word02)
    if word01Idx is not None and word2Idx is not None:
        word01Emb = W1[word01Idx].reshape(1, -1)
        word02Emb = W1[word02Idx].reshape(1, -1)
        return cosine_similarity(word01Emb, word02Emb).item()


def getMostSimilarWords(currWord, n) -> list[str]:
    global word2Idx
    if currWord not in word2Idx:
        print(f"'{currWord}'不存在!")
        return
    simLst = list()

    for word in word2Idx:
        simLst.append((word, calSimilarity(word, currWord)))

    simSorted = sorted(simLst, key=lambda x: x[1], reverse=True)

    return simSorted[1:n+1]

    # return [simSorted[i] for i in range(1, n+1)]






if __name__ == '__main__':
    pklFiles = [f for f in os.listdir(LoadingDir) if f.endswith('.pkl')]

    # 获取最新的文件
    latestFile = max(pklFiles, key=lambda x: os.path.getmtime(os.path.join(LoadingDir, x)))

    filePath = os.path.join(LoadingDir, latestFile)
    W1, word2Idx = loadWords2Idx_Embs(filePath)
    print(f"已加载词向量 >> {filePath}")

    # while True:
    #     word01 = input("词汇1 >> ")
    #     word2 = input("词汇2 >> ")
    #     if all((word01, word2)):
    #         print(calSimilarity(word01, word2))
    #     else:
    #         break

    n = 10
    while 1:
        print("-" * 80)
        word = input("词汇 >> ")
        if not word:
            break
        result = getMostSimilarWords(word, n)
        if not result:
            continue
        for tup in result:
            print(f"{tup[0]} - {tup[1]:.3f}")



