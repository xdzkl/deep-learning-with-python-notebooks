#!/usr/bin/env python
# coding: utf-8

# 卷积神经网络convnet

import tensorflow as tf
tf.__version__
# '2.0.0-alpha0'

# 导入模型层
from tensorflow.keras import models
# 导入层
from tensorflow.keras import layers

# 每个conv2d和maxpooling2D层的输出都是一个形状为（height,width,channels)的3D向量
# 建立一个序贯模型，是多个网络层的线性堆叠，也就是一条路走到黑，
#详细信息见：https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
model = models.Sequential()
# 添加一个二维卷积层，32代表卷积核的数量，卷积核大小是3*3，激活函数是relu,输入维度是28*28*1
# 输出维度是26，26，32
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 添加一个最大池化层，池化核大小是2*2
# 输人维度是26，26，32，# 输出维度是13，13，32
model.add(layers.MaxPooling2D((2, 2)))

# 添加一个二维卷积层，32代表卷积核的数量，卷积核大小是3*3，激活函数是relu,输入维度是32*32*1
# 输入维度是13，13，32，输出维度是11，11，64
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 添加一个最大池化层，池化核大小是2*2
# 输入维度是11，11，64，输出维度是5，5，64
model.add(layers.MaxPooling2D((2, 2)))

# 添加一个二维卷积层，32代表卷积核的数量，卷积核大小是3*3，激活函数是relu,输入维度是32*32*1
# 输入维度是5，5，64，输出维度是3，3，64
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

# 将3D输出展平为1D
model.add(layers.Flatten())
# 添加全连接层，输出维度是64，激活函数是relu,输入维度是
model.add(layers.Dense(64, activation='relu'))
# 添加全连接层，输出维度是10，激活函数是softmax，输入维度是64
model.add(layers.Dense(10, activation='softmax'))

model.summary()


from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)


test_loss, test_acc = model.evaluate(test_images, test_labels)


test_acc


