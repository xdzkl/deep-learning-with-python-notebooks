# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:05:37 2019

@author: Administrator
"""
import tensorflow as tf
print(tf.__version__)
# 2.0.0-alpha0

from tensorflow.keras.datasets import mnist


# mnist数据集是手写数据集，主要是对手写的数字图片进行识别，图片的大小是28*28
(train_image,train_labels),(test_images,test_labels) = mnist.load_data()
print(train_image.shape)
# (60000, 28, 28)
print(len(train_image))
# 60000
train_labels[:10]
# array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4], dtype=uint8)

print(test_images.shape)
#(10000, 28, 28)
print(len(test_labels))
#10000
test_labels[:10]
#array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9], dtype=uint8)

# model是一个模型
from tensorflow.keras import models
# layers是一个层，可以这么理解，多个层构成了一个模型，或者说一个神经网络
from tensorflow.keras import layers

# 建立一个序贯模型，是多个网络层的线性堆叠，也就是一条路走到黑，
#详细信息见：https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
network = models.Sequential()
#Dense是常用的全连接，activation是逐元素计算的激活函数，512的参数名是util，代表的是该层的输出维度，input_shape是输入维度，28*28=784
network.add(layers.Dense(512,activation='relu',input_shape = (28*28,)))
# 再添加一层，是全连接层，输出维度是10，激活函数是softmax
# 详情查看连接https://keras-cn.readthedocs.io/en/latest/layers/core_layer/#dense
network.add(layers.Dense(10,activation='softmax'))

# compile是功能是编译模型，对学习过程进行配置，optimizer是优化器，loss是损失函数，metrics是指标列表
network.compile(optimizer='rmsprop',
                loss = 'categorical_crossentropy',
                metrics =['accuracy'])

# 训练集和测试的类型是numpy.ndarray,
#可以更改大小, (60000, 28, 28)->(60000,784)
train_image = train_image.reshape((60000,28*28))
# 原来的大小是0-255，现在改为0-1,原来的数据类型是uint8,现在改为float32
# print(train_image.dtype)
# uint8
train_image = train_image.astype("float32")/255

##可以更改大小, (10000, 28, 28)->(10000,784)
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

from tensorflow.keras.utils import to_categorical
#to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
train_labels = to_categorical(train_labels)
# (10000,)->(10000, 10)
test_labels = to_categorical(test_labels)

# 将模型训练epochs轮，batch_size，指定进行梯度下降时，每个batch包含的样本数
network.fit(train_image,train_labels,epochs=5,batch_size = 128)
# evaluate计算在某些输入数据上，模型的误差，
test_loss,test_acc = network.evaluate(test_images,test_labels)
print('test_acc:',test_acc)
#test_acc: 0.9806
#打印模型的概览
print(network.summary())
#Model: "sequential_1"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_2 (Dense)              (None, 512)            401920  =(784*512+512)  
#________________________________________________(784*512是权重，再加512是偏置)
#dense_3 (Dense)              (None, 10)                5130=512*10+10   
#=================================================================
#Total params: 407,050 = 401920+5130
#Trainable params: 407,050
#Non-trainable params: 0
#_________________________________________________________________