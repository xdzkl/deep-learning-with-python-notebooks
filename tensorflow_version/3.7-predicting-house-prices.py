#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
tf.__version__
# '2.0.0-alpha0'


# 波士顿房价数据
from tensorflow.keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()
# train_data的大小是(404, 13)，test_data的大小是(404,)
# test_labels的大小是(102, 13)，test_labels的大小是(102,)
train_targets[:10]
#array([15.2, 42.3, 50. , 21.1, 17.7, 18.5, 11.3, 15.6, 15.6, 14.4])


# 进行均值方差归一化
mean = train_data.mean(axis=0)
#'array([-1.01541438e-16,  1.09923072e-17,  1.80933376e-15, -7.80453809e-17,
#       -5.25047552e-15,  6.43187374e-15,  2.98441140e-16,  4.94653823e-16,
#        1.12671149e-17, -1.05526149e-16,  2.36614908e-14,  5.96710525e-15,
#        6.13920356e-16])
train_data -= mean
std = train_data.std(axis=0)
# array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
train_data /= std

test_data -= mean
test_data /= std



# 导入模型层
from tensorflow.keras import models
# 导入层
from tensorflow.keras import layers

def build_model():
# 建立一个序贯模型，是多个网络层的线性堆叠，也就是一条路走到黑，
#详细信息见：https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
    model = models.Sequential()
    # 输入维度（13，）输出维度（64，）激活函数是relu
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    # 输入维度（64，）输出维度（64，）激活函数是relu
    model.add(layers.Dense(64, activation='relu'))
    # 输入维度（64，）输出维度（1，）
    model.add(layers.Dense(1))
    # compile的功能是编译模型，对学习过程进行配置，optimizer是优化器，
# loss是损失函数,这里是mse，metrics是指标列表,这里是mae
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model





import numpy as np

k = 4
# len(train_data)是404，num_val_sample是101
num_val_samples = len(train_data) // k
# num_epochs是轮次
num_epochs = 100
# 所有的分数
all_scores = []
for i in range(k):
#     第i折
    print('processing fold #', i)
#    准备验证数据，第k个分区的数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 准备训练数据，其他所有分区的数据
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # 构建模型，已经编译过的
    model = build_model()
    # 训练模型（静默模式，verbose=0）
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
#    在验证数据上评估模型
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

all_scores
#[2.2945063, 2.591721, 2.8060346, 2.2932646]

np.mean(all_scores)
#2.4963818

from tensorflow.keras import backend as K
# 销毁当前的TF图并创建一个新图，有助于避免旧模型/涂层混乱
K.clear_session()

# 设置500个轮次
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # 准备验证数据，第k个分区的数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

#    ·准备训练数据，其他所有分区的数据
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # 构建模型，已经编译过的
    model = build_model()
    # 训练模型（静默模式，verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
#    计算每个轮次中所有折MAE的平均值
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)



average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
#Out[111]: 
#[4.3805985,
# 3.3160563,
# 2.902602,
# 2.7533863,
# 2.7148235,
# 2.79872,
# 2.80879,
# 2.6642942,
# 2.5138829,
# 2.6127791
# ···]
len(average_mae_history)
#Out[112]: 500



import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()



def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

# 删除前10个数据点，将每个数据点替换为前面数据点的指数移动平均值，以得到光滑的曲线
smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# 一个全新的编译好的模型
model = build_model()
# 在所有训练数据上训练模型
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

test_mae_score
#18.600066241096048 预测的房价和实际价格相差约186000美元