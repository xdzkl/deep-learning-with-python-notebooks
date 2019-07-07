#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
tf.__version__
# '2.0.0-alpha0'


# 本节使用路透社数据集，它包含许多短新闻机器对应的主题，由路透社在1986年发布，
# 它是一个简单的，广泛使用的文本分类数据集，它包含46个不同的主题，某些主题的样本更多
# 但训练集中国每个主题都至少有10个样本
from tensorflow.keras.datasets import reuters

# 加载数据集，num_words意味着只保留训练集中最常出现的10000的单词，不经常出现的单词被抛弃，最终所有评论的维度保持相同，
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# train_data的大小是(8982,)，test_data的大小是(8982,)
# test_labels的大小是(2246,)，test_labels的大小是(2246,)
len(train_data)
# 8982
len(test_data)
#2246
train_data[10][:10]
#[1, 245, 273, 207, 156, 53, 74, 160, 26, 14]

# 获得reuters中，单词和数字的对应表，形如下面：
# {':6709,at:20054}
word_index = reuters.get_word_index()
# 将单词和数字的对应表的键值反转，并最终保存为字典，结果形如下面：
# {1:'the',2:'of',···}
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 这里含义是找出train_data[0]中数字列表，然后从reverse_word_index中找出对应的value
# 并使用空格连接起来
# 字典中的get方法语法是dict.get(key,default=None),这里'?'就是默认值
# 这里-3的含义是，因为0，1，2，是为padding(填充)，start of sequence（序列开始），unknown(未知词)分别保留的索引。
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

decoded_newswire
#? ? ? said as a result of its december acquisition of space co it expects earnings per share in 1987 of 1 15 to 1 30 dlrs per share up from 70 cts in 1986 the company

train_labels[10]
# 3

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # 创建一个形状为（len(sequences)，dimesion）的矩阵
    results = np.zeros((len(sequences), dimension))
    #     进行one-hot编码
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# shape是（25000，10000），将训练数据向量化
x_train = vectorize_sequences(train_data)
# shape是（25000，10000），将训练数据向量化
x_test = vectorize_sequences(test_data)

# 进行One-hot编码
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


one_hot_train_labels = to_one_hot(train_labels)

one_hot_test_labels = to_one_hot(test_labels)


from tensorflow.python.keras.utils.np_utils import to_categorical

# 将整型标签转换为onehot编码
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


# 导入模型层
from tensorflow.keras import models
# 导入层
from tensorflow.keras import layers

# 建立一个序贯模型，是多个网络层的线性堆叠，也就是一条路走到黑，
#详细信息见：https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
model = models.Sequential()
# 输入维度（10000，）输出维度（64，）激活函数是relu
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# 输入维度(64,)，输出维度(64,)，激活函数是relu
model.add(layers.Dense(64, activation='relu'))
# 输入维度是(64,),输出维度(46,),激活函数是softmax
model.add(layers.Dense(46, activation='softmax'))
model.summary()
#Model: "sequential_4"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_12 (Dense)             (None, 64)                640064    
#_________________________________________________________________
#dense_13 (Dense)             (None, 64)                4160      
#_________________________________________________________________
#dense_14 (Dense)             (None, 46)                2990      
#=================================================================
#Total params: 647,214
#Trainable params: 647,214
#Non-trainable params: 0
#_________________________________________________________________

# compile的功能是编译模型，对学习过程进行配置，optimizer是优化器，
# loss是损失函数,这里是分类交叉熵，metrics是指标列表
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 将原始训练数据留出1000个样本作为验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 使用512个样本组成的小批量，将模型训练20个轮次，监控留出的10000个样本上的损失和精度，可以通过将验证数据传入validation_data参数来完成
# 调用fit方法会返回一个History对象，这个对象有一个成员history，它是一个字典，包含训练过程中的所有数据
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# clf的含义是清除图像
plt.clf()   # clear figure

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 建立一个序贯模型，是多个网络层的线性堆叠，也就是一条路走到黑，
#详细信息见：https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
model = models.Sequential()
# 输入维度（10000，）输出维度（16，）激活函数是relu
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# 输入维度(16,)，输出维度(16,)，激活函数是relu
model.add(layers.Dense(64, activation='relu'))
# 输入维度是(16,),输出维度(1,),激活函数是softmax
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))
# 在测试模式下返回模型的误差值和评估标准值
results = model.evaluate(x_test, one_hot_test_labels)

import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels)
# 0.1861086375779163

# 预测
predictions = model.predict(x_test)
#(46,)
predictions[0].shape
#0.99999994
np.sum(predictions[0])
#3
np.argmax(predictions[0])

# 将标签转换为整数张量
y_train = np.array(train_labels)
y_test = np.array(test_labels)


model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))
#精度下降了约8%
