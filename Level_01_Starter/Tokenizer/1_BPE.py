# coding=utf-8
"""
@Author: Jacob Y
@Date  : 12/17/2024
@Desc  :
BPE的最大化目标是选取单词内相邻单元对的频数
1. 分词器会为语料库中的字符建立一个基础词表
2. 执行字词合并，字词合并即每次选取出现频率最高的字词单元对，形成新字词单元，然后将其加入词表中
"""

from collections import Counter, defaultdict
from transformers import AutoTokenizer


class BPE:
    def __init__(self, corpus, vocab_size):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base_uncased")
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = {}

    def train(self):
        # 计算语料中每个单词的频率
        for text in self.corpus:
            pre_tokenizer = self.tokenizer.backend_tokenizer.pre_tokenizer
            words_with_offsets = pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] += 1

        # 计算语料中所有字符的基础词汇
        alphabet = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)

        alphabet.sort()

        # 在词表开始处添加特殊标记</w>
        vocab = ["</w>"] + alphabet.copy()

        # 在训练前将每个单词分隔成单个字符
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

        # 迭代地合并最频繁的对，直到达到词表的大小
        while len(vocab) < self.vocab_size:

            # 计算每对的频率
            pair_freqs = self.compute_pair_freqs()

            # 找到最频繁的对
            best_pair = ""
            max_freq = None

            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            # 合并最频繁的对
            self.splits = self.merge_pair(*best_pair)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])

        return self.merges

    def compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b):
        """合并给定的对"""
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0

            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits

    def tokenize(self, text):
        """
        用训练好的BPE分词器对给定的文本分词（包括预分词、分隔和合并）
        """
        pre_tokenizer = self.tokenizer._tokenizer.pre_tokenizer
        pre_tokenize_result = pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits_text = [[l for l in word] for word in pre_tokenized_text]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits_text):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits_text[idx] = split
        result = sum(splits_text, [])
        return result
