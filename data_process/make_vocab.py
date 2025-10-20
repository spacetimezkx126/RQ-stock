import json
import os
import pickle as pkl
from collections import Counter

path = "/home/zhaokx/formal_from_autodl/finance_stock_datasets/stocknet/score"
def build_vocab(corpus, tokenizer, min_freq=3):
    # 使用Counter来统计词频
    word_count = Counter()
    for data in corpus:
        for word in tokenizer(data):
            word_count[word] += 1
    # 筛选出词频大于等于min_freq的词
    return {word: count for word, count in word_count.items() if count >= min_freq}

files = os.listdir(path)
contentall = []
for folder in files:
    files_json_all = os.listdir(os.path.join(path,folder))
    for file in files_json_all:
        with open(os.path.join(path, folder, file), 'r') as f:
            data = json.load(f)
            all_news = [b["News_content"] for b in data["Each_item"]]
            for news in all_news:
                # print(news)
                contentall.append(news)
                # contentall += news.split(" ")


UNK, PAD = '<unk>', '<pad>'

# 使用 Counter 对语料库中的单词进行统计
vocab_dict = build_vocab(contentall, tokenizer=lambda x: [y for y in x.split(" ")])

# 获取符合词频要求的词汇，并按词频降序排列
vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)

vocab = {word: idx for idx, (word, _) in enumerate(vocab_list)}

vocab[UNK] = len(vocab)
vocab[PAD] = len(vocab) + 1

print(f"Vocabulary size: {len(vocab)}")  # 应该是 4764 (包含 <UNK> 和 <PAD>)

# 保存词汇表到文件
print(vocab)
pkl.dump(vocab, open("dict_tweet.pkl", 'wb'))