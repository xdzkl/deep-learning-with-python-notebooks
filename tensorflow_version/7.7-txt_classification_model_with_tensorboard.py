# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:50:52 2019

@author: Administrator
"""

'''
代码清单7-7 使用了Tensorboard的文本分类模型
'''
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

max_feature = 2000
max_len = 500

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words = max_feature)
x_train = sequence.pad_sequences(x_train,maxlen = max_len)
x_test = sequence.pad_sequences(x_test,maxlen = max_len)

model = models.Sequential()
model.add(layers.Embedding(max_feature,128,input_length=max_len,name='embed'))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# 创建保存日志的文件夹
callbacks = [keras.callbacks.TensorBoard(log_dir = 'F:\zkl_repository\log',
#                                          每一轮之后记录激活直方图
                                         histogram_freq =1,
#                                          每一轮之后记录嵌入数据
                                         embeddings_freq =1,)]

hitory = model.fit(x_train,y_train,epochs =20,batch_size =128,
                   validation_split=0.2,
                   callbacks =callbacks)

# 启动tensorboard
# tensorboard --logdi PATH