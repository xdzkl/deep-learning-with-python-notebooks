#!/usr/bin/env python
# coding: utf-8


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




from keras import models
from keras import layers

original_model = models.Sequential()
original_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
original_model.add(layers.Dense(16, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))

original_model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['acc'])



smaller_model = models.Sequential()
smaller_model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
smaller_model.add(layers.Dense(4, activation='relu'))
smaller_model.add(layers.Dense(1, activation='sigmoid'))

smaller_model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])


# 



original_hist = original_model.fit(x_train, y_train,
                                   epochs=20,
                                   batch_size=512,
                                   validation_data=(x_test, y_test))


# In[7]:


smaller_model_hist = smaller_model.fit(x_train, y_train,
                                       epochs=20,
                                       batch_size=512,
                                       validation_data=(x_test, y_test))


# In[8]:


epochs = range(1, 21)
original_val_loss = original_hist.history['val_loss']
smaller_model_val_loss = smaller_model_hist.history['val_loss']


# In[9]:


import matplotlib.pyplot as plt

# b+ is for "blue cross"
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
# "bo" is for "blue dot"
plt.plot(epochs, smaller_model_val_loss, 'bo', label='Smaller model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# 
# As you can see, the smaller network starts overfitting later than the reference one (after 6 epochs rather than 4) and its performance 
# degrades much more slowly once it starts overfitting.
# 
# Now, for kicks, let's add to this benchmark a network that has much more capacity, far more than the problem would warrant:

# In[11]:


bigger_model = models.Sequential()
bigger_model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
bigger_model.add(layers.Dense(512, activation='relu'))
bigger_model.add(layers.Dense(1, activation='sigmoid'))

bigger_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['acc'])


# In[12]:


bigger_model_hist = bigger_model.fit(x_train, y_train,
                                     epochs=20,
                                     batch_size=512,
                                     validation_data=(x_test, y_test))


# Here's how the bigger network fares compared to the reference one. The dots are the validation loss values of the bigger network, and the 
# crosses are the initial network.

# In[26]:


bigger_model_val_loss = bigger_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_val_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# 
# The bigger network starts overfitting almost right away, after just one epoch, and overfits much more severely. Its validation loss is also 
# more noisy.
# 
# Meanwhile, here are the training losses for our two networks:

# In[28]:


original_train_loss = original_hist.history['loss']
bigger_model_train_loss = bigger_model_hist.history['loss']

plt.plot(epochs, original_train_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_train_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.legend()

plt.show()


# As you can see, the bigger network gets its training loss near zero very quickly. The more capacity the network has, the quicker it will be 
# able to model the training data (resulting in a low training loss), but the more susceptible it is to overfitting (resulting in a large 
# difference between the training and validation loss).

# ## Adding weight regularization
# 
# 
# You may be familiar with _Occam's Razor_ principle: given two explanations for something, the explanation most likely to be correct is the 
# "simplest" one, the one that makes the least amount of assumptions. This also applies to the models learned by neural networks: given some 
# training data and a network architecture, there are multiple sets of weights values (multiple _models_) that could explain the data, and 
# simpler models are less likely to overfit than complex ones.
# 
# A "simple model" in this context is a model where the distribution of parameter values has less entropy (or a model with fewer 
# parameters altogether, as we saw in the section above). Thus a common way to mitigate overfitting is to put constraints on the complexity 
# of a network by forcing its weights to only take small values, which makes the distribution of weight values more "regular". This is called 
# "weight regularization", and it is done by adding to the loss function of the network a _cost_ associated with having large weights. This 
# cost comes in two flavors:
# 
# * L1 regularization, where the cost added is proportional to the _absolute value of the weights coefficients_ (i.e. to what is called the 
# "L1 norm" of the weights).
# * L2 regularization, where the cost added is proportional to the _square of the value of the weights coefficients_ (i.e. to what is called 
# the "L2 norm" of the weights). L2 regularization is also called _weight decay_ in the context of neural networks. Don't let the different 
# name confuse you: weight decay is mathematically the exact same as L2 regularization.
# 
# In Keras, weight regularization is added by passing _weight regularizer instances_ to layers as keyword arguments. Let's add L2 weight 
# regularization to our movie review classification network:

# In[17]:


from keras import regularizers

l2_model = models.Sequential()
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(10000,)))
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
l2_model.add(layers.Dense(1, activation='sigmoid'))


# In[18]:


l2_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['acc'])


# `l2(0.001)` means that every coefficient in the weight matrix of the layer will add `0.001 * weight_coefficient_value` to the total loss of 
# the network. Note that because this penalty is _only added at training time_, the loss for this network will be much higher at training 
# than at test time.
# 
# Here's the impact of our L2 regularization penalty:

# In[19]:


l2_model_hist = l2_model.fit(x_train, y_train,
                             epochs=20,
                             batch_size=512,
                             validation_data=(x_test, y_test))


# In[30]:


l2_model_val_loss = l2_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, l2_model_val_loss, 'bo', label='L2-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# 
# 
# As you can see, the model with L2 regularization (dots) has become much more resistant to overfitting than the reference model (crosses), 
# even though both models have the same number of parameters.
# 
# As alternatives to L2 regularization, you could use one of the following Keras weight regularizers:

# In[ ]:


from keras import regularizers

# L1 regularization
regularizers.l1(0.001)

# L1 and L2 regularization at the same time
regularizers.l1_l2(l1=0.001, l2=0.001)


# ## Adding dropout
# 
# 
# Dropout is one of the most effective and most commonly used regularization techniques for neural networks, developed by Hinton and his 
# students at the University of Toronto. Dropout, applied to a layer, consists of randomly "dropping out" (i.e. setting to zero) a number of 
# output features of the layer during training. Let's say a given layer would normally have returned a vector `[0.2, 0.5, 1.3, 0.8, 1.1]` for a 
# given input sample during training; after applying dropout, this vector will have a few zero entries distributed at random, e.g. `[0, 0.5, 
# 1.3, 0, 1.1]`. The "dropout rate" is the fraction of the features that are being zeroed-out; it is usually set between 0.2 and 0.5. At test 
# time, no units are dropped out, and instead the layer's output values are scaled down by a factor equal to the dropout rate, so as to 
# balance for the fact that more units are active than at training time.
# 
# Consider a Numpy matrix containing the output of a layer, `layer_output`, of shape `(batch_size, features)`. At training time, we would be 
# zero-ing out at random a fraction of the values in the matrix:

# In[ ]:


# At training time: we drop out 50% of the units in the output
layer_output *= np.randint(0, high=2, size=layer_output.shape)


# 
# At test time, we would be scaling the output down by the dropout rate. Here we scale by 0.5 (because we were previous dropping half the 
# units):

# In[ ]:


# At test time:
layer_output *= 0.5


# 
# Note that this process can be implemented by doing both operations at training time and leaving the output unchanged at test time, which is 
# often the way it is implemented in practice:

# In[ ]:


# At training time:
layer_output *= np.randint(0, high=2, size=layer_output.shape)
# Note that we are scaling *up* rather scaling *down* in this case
layer_output /= 0.5


# 
# This technique may seem strange and arbitrary. Why would this help reduce overfitting? Geoff Hinton has said that he was inspired, among 
# other things, by a fraud prevention mechanism used by banks -- in his own words: _"I went to my bank. The tellers kept changing and I asked 
# one of them why. He said he didn’t know but they got moved around a lot. I figured it must be because it would require cooperation 
# between employees to successfully defraud the bank. This made me realize that randomly removing a different subset of neurons on each 
# example would prevent conspiracies and thus reduce overfitting"_.
# 
# The core idea is that introducing noise in the output values of a layer can break up happenstance patterns that are not significant (what 
# Hinton refers to as "conspiracies"), which the network would start memorizing if no noise was present. 
# 
# In Keras you can introduce dropout in a network via the `Dropout` layer, which gets applied to the output of layer right before it, e.g.:

# In[ ]:


model.add(layers.Dropout(0.5))


# Let's add two `Dropout` layers in our IMDB network to see how well they do at reducing overfitting:

# In[22]:


dpt_model = models.Sequential()
dpt_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(16, activation='relu'))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(1, activation='sigmoid'))

dpt_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])


# In[23]:


dpt_model_hist = dpt_model.fit(x_train, y_train,
                               epochs=20,
                               batch_size=512,
                               validation_data=(x_test, y_test))


# Let's plot the results:

# In[32]:


dpt_model_val_loss = dpt_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, dpt_model_val_loss, 'bo', label='Dropout-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# 
# Again, a clear improvement over the reference network.
# 
# To recap: here the most common ways to prevent overfitting in neural networks:
# 
# * Getting more training data.
# * Reducing the capacity of the network.
# * Adding weight regularization.
# * Adding dropout.
