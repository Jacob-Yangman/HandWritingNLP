# coding=utf-8
"""
@Author: Jacob Y
@Date  : 12/17/2024
@Desc  :
1. 准备足够大的训练语料库，确定期望的字词词表大小
2. 将单词拆分成最小单元，作为初始词表
3. 根据不同的分词策略，以不同的最大化目标合并字词形成新的字词单元
4. 重复执行单元合并，直至达到设定的字词词表大小

字词合并时最大化的目标是本次合并相邻字词能否使得整个语料库的似然值升高,每次合并需要保证当前的合并操作对句子概率值的提升最大，直到字词的词表达到预先设定的大小
"""

def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return list()
    tokens = text.split()
    return tokens



class WordPieceTokenizer:
    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        将一段文本分词成词片段
        使用一个贪婪的最长匹配优先算法来执行，
        并使用给定词汇的分词

        参数：text，传入的文本，需要经由Basic tokenizer处理变为单个标记或空格分隔的标记

        返回：list
        """

        output_tokens = list()
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = list()
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                        end -= 1
                    if cur_substr is None:
                        is_bad = True
                        break
                    sub_tokens.append(cur_substr)
                    start = end

                if is_bad:
                    output_tokens.append(self.unk_token)
                else:
                    output_tokens.extend(sub_tokens)
            return output_tokens


