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
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    # Evaluate the model on the validation data
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)



all_scores



np.mean(all_scores)





from keras import backend as K

# Some memory clean-up
K.clear_session()



num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)


# We can then compute the average of the per-epoch MAE scores for all folds:

# In[23]:


average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


# Let's plot this:

# In[26]:


import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# 
# It may be a bit hard to see the plot due to scaling issues and relatively high variance. Let's:
# 
# * Omit the first 10 data points, which are on a different scale from the rest of the curve.
# * Replace each point with an exponential moving average of the previous points, to obtain a smooth curve.

# In[38]:


def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# 
# According to this plot, it seems that validation MAE stops improving significantly after 80 epochs. Past that point, we start overfitting.
# 
# Once we are done tuning other parameters of our model (besides the number of epochs, we could also adjust the size of the hidden layers), we 
# can train a final "production" model on all of the training data, with the best parameters, then look at its performance on the test data:

# In[28]:


# Get a fresh, compiled model.
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)


# In[29]:


test_mae_score


# We are still off by about \$2,550.

# ## Wrapping up
# 
# 
# Here's what you should take away from this example:
# 
# * Regression is done using different loss functions from classification; Mean Squared Error (MSE) is a commonly used loss function for 
# regression.
# * Similarly, evaluation metrics to be used for regression differ from those used for classification; naturally the concept of "accuracy" 
# does not apply for regression. A common regression metric is Mean Absolute Error (MAE).
# * When features in the input data have values in different ranges, each feature should be scaled independently as a preprocessing step.
# * When there is little data available, using K-Fold validation is a great way to reliably evaluate a model.
# * When little training data is available, it is preferable to use a small network with very few hidden layers (typically only one or two), 
# in order to avoid severe overfitting.
# 
# This example concludes our series of three introductory practical examples. You are now able to handle common types of problems with vector data input:
# 
# * Binary (2-class) classification.
# * Multi-class, single-label classification.
# * Scalar regression.
# 
# In the next chapter, you will acquire a more formal understanding of some of the concepts you have encountered in these first examples, 
# such as data preprocessing, model evaluation, and overfitting.
