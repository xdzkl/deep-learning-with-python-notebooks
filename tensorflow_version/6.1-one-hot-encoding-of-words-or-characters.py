#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
print(tf.__version__)
# 2.0.0-alpha0

import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 构建数据中所有标记的索引
token_index = {}
for sample in samples:
#    利用split方法对样本进行分词，在实际应用中，还需要从样本中去掉标点和特殊字符
    for word in sample.split():
        if word not in token_index:
#            为每个唯一单词指定一个唯一索引，注意，没有为索引标号0指定单词
            token_index[word] = len(token_index) + 1

# 对文本进行分词，只考虑每个样本前max_lenght个单词
max_length = 10

# 将结果保存在results中
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.


import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# 所有可打印的ascii字符
characters = string.printable  
token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample[:max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1.



from tensorflow.keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 创建一个分词器，设置为只考虑前1000个最常见的分词
tokenizer = Tokenizer(num_words=1000)
#构建单词索引
tokenizer.fit_on_texts(samples)

# 将字符串转换为整数索引组成的列表
sequences = tokenizer.texts_to_sequences(samples)

# 也可以直接得到one-hot二进制表示，这个分词器也支持除one-hot编码外的其他向量化模式
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# This is how you can recover the word index that was computed
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))



samples = ['The cat sat on the mat.', 'The dog ate my homework.']

dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
# 将单词散列为0-1000范围内的一个随机整数索引
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.

