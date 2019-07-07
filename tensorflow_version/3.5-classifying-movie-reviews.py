# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:52:14 2019

@author: Administrator
"""
# 导入imdb数据集，imdb数据集有5万条来自网络电影数据库的评论，电影评论转换成了一系列数字，每个数字代表字典汇总的一个单词，下载后放到~/.keras/datasets/目录下，即可正常运行。)中找到下载，下载后放到~/.keras/datasets/目录下，即可正常运行。
from tensorflow.keras.datasets import imdb

# 加载数据集，num_words意味着只保留训练集中最常出现的10000的单词，不经常出现的单词被抛弃，最终所有评论的维度保持相同，变量train_data,test_data是电影评论的列表，每条评论由数字(对应单词在词典中出现的位置下标)列表组成。train_labels,test_labels是0,1列表，0负面评论，1表示正面评论。
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
# train_data的大小是（25000，0），test_data的大小是（250000，）
# test_labels的大小是（25000，0），test_labels的大小是（25000，）
train_data[0][:10]
#[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65]
train_labels[:10]
# array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0], dtype=int64)

max([max(sequence) for sequence in train_data])
#9999，这里是和num_words相对应

# 获得imdb中，单词和数字的对应表，形如下面：
# {a:68893,own:70879}
word_index = imdb.get_word_index()

# 将单词和数字的对应表的键值反转，并最终保存为字典，结果形如下面：
# {34071:'fawn',52006:'tsukino',···}
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

# 这里含义是找出train_data[0]中数字列表，然后从reverse_word_index中找出对应的value
# 并使用空格连接起来
# 字典中的get方法语法是dict.get(key,default=None),这里'?'就是默认值
# 这里-3的含义是，因为0，1，2，是为padding(填充)，start of sequence（序列开始），unknown(未知词)分别保留的索引。
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

decoded_review
# 形如下面
#? was not for it's self joke professional disappointment see already pretending their staged a every so found of his movies 

import numpy as np

def vectorize_sequence (sequences,dimension = 10000):
# 创建一个形状为（len(sequences)，dimesion）的矩阵
    results = np.zeros((len(sequences),dimension))
#     进行one-hot编码
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

# shape是（25000，10000），将训练数据向量化
x_train = vectorize_sequence(train_data)
# shape是(25000,10000)
x_test = vectorize_sequence(test_data)

x_train[0]
#array([0., 1., 1., ..., 0., 0., 0.])

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
model = models.Sequential()
# 输入维度（10000，）输出维度（16，）激活函数是relu
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
# 输入维度(16,)，输出维度(16,)，激活函数是relu
model.add(layers.Dense(16,activation='relu'))
# 输入维度是(16,),输出维度(1,),激活函数是sigmoid
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
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
model.compile(optimizer="rmsprop",
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 到入优化器类型
from tensorflow.keras import optimizers
# 使用RMSprop激活器，学习补偿是0.001
model.compile(optimizer = optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 导入损失函数类和指标类
from tensorflow.keras import losses
from tensorflow.keras import metrics

# loss使用二元交叉熵，但对于输出概率值的模型，交叉熵往往是最好的选择，用于衡量概率分布之间的距离
# 在这个例子中，就是真实分布和预测值之间的余力
model.compile(optimizer = optimizers.RMSprop(lr=0.001),
              loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])

# 将原始训练数据留出1000个样本作为验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 使用512个样本组成的小批量，将模型训练20个轮次，监控留出的10000个样本上的损失和精度，可以通过将验证数据传入validation_data参数来完成
# 调用fit方法会返回一个History对象，这个对象有一个成员history，它是一个字典，包含训练过程中的所有数据
history = model.fit(partial_x_train,partial_y_train,
                    epochs =20,batch_size = 512,validation_data=(x_val,y_val))
history_dict = history.history
history_dict.keys()
#dict_keys(['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])
#每个key包含一个列表，列表中有20个元素

import matplotlib.pyplot as plt

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
# 添加标签
plt.legend()
plt.show()

# clf的含义是清除图像
plt.clf()
acc_value = history_dict['binary_accuracy']
val_acc_value = history_dict['val_binary_accuracy']

plt.plot(epochs,acc,'bo',label='training acc')
plt.plot(epochs,val_acc,'b',label='validation acc')
plt.title('trainging and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=4,batch_size = 512)
# 在测试模式下返回模型的误差值和评估标准值
result = model.evaluate(x_test,y_test)
result
model.predict(x_test)