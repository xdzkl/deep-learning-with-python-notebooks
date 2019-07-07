#!/usr/bin/env python
# coding: utf-8

# 防止过拟合的方法是：
# 获取更多的训练数据，减小网络容量，添加权重正则化，添加dropout

import tensorflow as tf
tf.__version__
# '2.0.0-alpha0'

# 导入imdb数据集，imdb数据集有5万条来自网络电影数据库的评论，电影评论转换成了一系列数字，每个数字代表字典汇总的一个单词，下载后放到~/.keras/datasets/目录下，即可正常运行。)中找到下载，下载后放到~/.keras/datasets/目录下，即可正常运行。
from tensorflow.keras.datasets import imdb
import numpy as np

# 加载数据集，num_words意味着只保留训练集中最常出现的10000的单词，不经常出现的单词被抛弃，最终所有评论的维度保持相同，变量train_data,test_data是电影评论的列表，每条评论由数字(对应单词在词典中出现的位置下标)列表组成。train_labels,test_labels是0,1列表，0负面评论，1表示正面评论。
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# train_data的大小是（25000，0），test_data的大小是（250000，）
# test_labels的大小是（25000，0），test_labels的大小是（25000，）
train_data[0][:10]
#[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65]
train_labels[:10]
# array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0], dtype=int64)

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

# 将结构数据转换为ndarray,np.array（默认情况下）将会copy该对象，而 np.asarray除非必要，否则不会copy该对象。
# astype的含义是将数据类型转换为float32
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# 导入模型层
from tensorflow.keras import models
# 导入层
from tensorflow.keras import layers

# 建立一个序贯模型，是多个网络层的线性堆叠，也就是一条路走到黑，
#详细信息见：https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
original_model = models.Sequential()
# 输入维度（10000，）输出维度（16，）激活函数是relu
original_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# 输入维度(16,)，输出维度(16,)，激活函数是relu
original_model.add(layers.Dense(16, activation='relu'))
# 输入维度是(16,),输出维度(1,),激活函数是sigmoid
original_model.add(layers.Dense(1, activation='sigmoid'))
original_model.summary()
#Model: "sequential_2"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_6 (Dense)              (None, 16)                160016    
#_________________________________________________________________
#dense_7 (Dense)              (None, 16)                272       
#_________________________________________________________________
#dense_8 (Dense)              (None, 1)                 17        
#=================================================================
#Total params: 160,305
#Trainable params: 160,305
#Non-trainable params: 0
#_________________________________________________________________

# compile的功能是编译模型，对学习过程进行配置，optimizer是优化器，
# loss是损失函数，metrics是指标列表
original_model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['acc'])


#构建小的模型
smaller_model = models.Sequential()
smaller_model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
smaller_model.add(layers.Dense(4, activation='relu'))
smaller_model.add(layers.Dense(1, activation='sigmoid'))

smaller_model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])

# 使用512个样本组成的小批量，将模型训练20个轮次，监控留出的10000个样本上的损失和精度，可以通过将验证数据传入validation_data参数来完成
# 调用fit方法会返回一个History对象，这个对象有一个成员history，它是一个字典，包含训练过程中的所有数据
original_hist = original_model.fit(x_train, y_train,
                                   epochs=20,
                                   batch_size=512,
                                   validation_data=(x_test, y_test))

# 使用512个样本组成的小批量，将模型训练20个轮次，监控留出的10000个样本上的损失和精度，可以通过将验证数据传入validation_data参数来完成
# 调用fit方法会返回一个History对象，这个对象有一个成员history，它是一个字典，包含训练过程中的所有数据
smaller_model_hist = smaller_model.fit(x_train, y_train,
                                       epochs=20,
                                       batch_size=512,
                                       validation_data=(x_test, y_test))


epochs = range(1, 21)
original_val_loss = original_hist.history['val_loss']
smaller_model_val_loss = smaller_model_hist.history['val_loss']



import matplotlib.pyplot as plt

# b+ is for "blue cross"
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
# "bo" is for "blue dot"
plt.plot(epochs, smaller_model_val_loss, 'bo', label='Smaller model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()

# 建立一个更大的模型
bigger_model = models.Sequential()
bigger_model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
bigger_model.add(layers.Dense(512, activation='relu'))
bigger_model.add(layers.Dense(1, activation='sigmoid'))
bigger_model.summary()
# compile的功能是编译模型，对学习过程进行配置，optimizer是优化器，
# loss是损失函数,这里是分类交叉熵，metrics是指标列表
bigger_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['acc'])

# 使用512个样本组成的小批量，将模型训练20个轮次，监控留出的10000个样本上的损失和精度，可以通过将验证数据传入validation_data参数来完成
# 调用fit方法会返回一个History对象，这个对象有一个成员history，它是一个字典，包含训练过程中的所有数据
bigger_model_hist = bigger_model.fit(x_train, y_train,
                                     epochs=20,
                                     batch_size=512,
                                     validation_data=(x_test, y_test))

bigger_model_val_loss = bigger_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_val_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.show()

original_train_loss = original_hist.history['loss']
bigger_model_train_loss = bigger_model_hist.history['loss']

plt.plot(epochs, original_train_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_train_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.legend()
plt.show()

from tensorflow.keras import regularizers


l2_model = models.Sequential()
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(10000,)))
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
l2_model.add(layers.Dense(1, activation='sigmoid'))

# compile的功能是编译模型，对学习过程进行配置，optimizer是优化器，
# loss是损失函数,这里是分类交叉熵，metrics是指标列表
l2_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['acc'])

# 使用512个样本组成的小批量，将模型训练20个轮次，监控留出的10000个样本上的损失和精度，可以通过将验证数据传入validation_data参数来完成
# 调用fit方法会返回一个History对象，这个对象有一个成员history，它是一个字典，包含训练过程中的所有数据
l2_model_hist = l2_model.fit(x_train, y_train,
                             epochs=20,
                             batch_size=512,
                             validation_data=(x_test, y_test))


l2_model_val_loss = l2_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, l2_model_val_loss, 'bo', label='L2-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.show()


from tensorflow.keras import regularizers

# 做L1正则化,绝对值，数值是权重
regularizers.l1(0.001)

#同时做L1和L2正则化，平方
regularizers.l1_l2(l1=0.001, l2=0.001)

# 建立一个序贯模型，是多个网络层的线性堆叠，也就是一条路走到黑，
#详细信息见：https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
dpt_model = models.Sequential()
dpt_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# dropout比例是被设置为0的特征所占的比例，通常在0.2~0.5范围内，测试时，没有单元被舍弃，
# 而改成的输出值需要按dropout比率缩小，因为这时比训练时有更多的单元被激活，需要加以平衡。
# 另外一种实现方式是两个运算都在训练时进行，而在测试时保持不变，训练时对激活矩阵使用dropout,并在
# 训练时成比例增大，测试时激活矩阵保持不变
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(16, activation='relu'))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(1, activation='sigmoid'))

# compile的功能是编译模型，对学习过程进行配置，optimizer是优化器，
# loss是损失函数,这里是分类交叉熵，metrics是指标列表
dpt_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])


# 使用512个样本组成的小批量，将模型训练20个轮次，监控留出的10000个样本上的损失和精度，可以通过将验证数据传入validation_data参数来完成
# 调用fit方法会返回一个History对象，这个对象有一个成员history，它是一个字典，包含训练过程中的所有数据
dpt_model_hist = dpt_model.fit(x_train, y_train,
                               epochs=20,
                               batch_size=512,
                               validation_data=(x_test, y_test))

dpt_model_val_loss = dpt_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, dpt_model_val_loss, 'bo', label='Dropout-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.show()

