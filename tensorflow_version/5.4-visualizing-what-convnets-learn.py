#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
print(tf.__version__)
# 2.0.0-alpha0

from tensorflow.keras.models import load_model

# 导入模型
model = load_model('./cats_and_dogs_small_2.h5')
# 作为提醒
model.summary() 
#Model: "sequential_3"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_11 (Conv2D)           (None, 148, 148, 32)      896       
#_________________________________________________________________
#max_pooling2d_10 (MaxPooling (None, 74, 74, 32)        0         
#_________________________________________________________________
#conv2d_12 (Conv2D)           (None, 72, 72, 64)        18496     
#_________________________________________________________________
#max_pooling2d_11 (MaxPooling (None, 36, 36, 64)        0         
#_________________________________________________________________
#conv2d_13 (Conv2D)           (None, 34, 34, 128)       73856     
#_________________________________________________________________
#max_pooling2d_12 (MaxPooling (None, 17, 17, 128)       0         
#_________________________________________________________________
#conv2d_14 (Conv2D)           (None, 15, 15, 128)       147584    
#_________________________________________________________________
#max_pooling2d_13 (MaxPooling (None, 7, 7, 128)         0         
#_________________________________________________________________
#flatten_3 (Flatten)          (None, 6272)              0         
#_________________________________________________________________
#dropout (Dropout)            (None, 6272)              0         
#_________________________________________________________________
#dense_6 (Dense)              (None, 512)               3211776   
#_________________________________________________________________
#dense_7 (Dense)              (None, 1)                 513       
#=================================================================
#Total params: 3,453,121
#Trainable params: 3,453,121
#Non-trainable params: 0
#_________________________________________________________________


img_path = 'F:\zkl_repository\pic\\all_pic\cat.1700.jpg'

# 将图像预处理为一个4D张量
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
# 处理成一个4D张量
img_tensor = np.expand_dims(img_tensor, axis=0)
# 请记住，训练模型的输入数据都用这种方法预处理
img_tensor /= 255.

# 其形状为（1，150，150，3）
print(img_tensor.shape)


import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()


from tensorflow.keras import models

# 提取前8层的输出
layer_outputs = [layer.output for layer in model.layers[:8]]
# 创建一个模型，给定模型的输入，可以返回这些输出
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# 返回8个numpy数组组成的列表，每个层激活对应一个numpy数组
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
# (1, 148, 148, 32)


# 将第3个通道可视化
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()
# 将第30个通道可视化
plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
plt.show()

from tensorflow import keras

# 层的名称，这样可以将这些名称画到图中
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# 显示特征图
for layer_name, layer_activation in zip(layer_names, activations):
    # 特征图中特征的个数
    n_features = layer_activation.shape[-1]

    # 特征图的形状为（1，size,seize,n_features)
    size = layer_activation.shape[1]

    # 在这个矩阵中将激活通道平铺
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 将每个过滤器平铺到一个大的水平网格中
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # 对特征进行后处理，使其看起来更美观
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # 显示网格
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()


from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
import tensorflow as tf

model = VGG16(weights='imagenet',
              include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# 这个函数不知道应该改成什么，现在报错
#with tf.GradientTape() as g:
#    g.watch(model.input)
#    loss = K.mean(layer_output[:, :, :, filter_index])
grads = K.gradients(loss, model.input)
#
## We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
#grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#
#
iterate = K.function([model.input], [loss, grads])

# Let's test it:
import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

# 从一张带有噪声的灰度图像开始
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

# 运行40次，梯度上升
step = 1. 
for i in range(40):
    #计算损失值和梯度值
    loss_value, grads_value = iterate([input_img_data])
    # 沿着让损失最大化的方向调节输入图像
    input_img_data += grads_value * step

def deprocess_image(x):
    # 对张量做标准化，使其均值为0，标准差为0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # 将x裁剪到[0,1]之间
    x += 0.5
    x = np.clip(x, 0, 1)

    # 转换成RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):
#     构建一个损失函数，将该层的第n个过滤器的激活最大化
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # 计算这个损失相对于输入图像的梯度
    grads = K.gradients(loss, model.input)[0]

    # 标准化技巧：将梯度标准化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)



plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()


for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:
    size = 64
    margin = 5

    # This a empty (black) image where we will store our results.
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(8):  # iterate over the rows of our results grid
        for j in range(8):  # iterate over the columns of our results grid
            # Generate the pattern for filter `i + (j * 8)` in `layer_name`
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # Display the results grid
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.show()


from tensorflow.keras.applications.vgg16 import VGG16

K.clear_session()

# 注意，网络中包括了密集连接诶分类器，在前面的例子中，我们都舍弃了这个分类器
model = VGG16(weights='imagenet')

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# 目标图片路径
img_path = 'F:\zkl_repository\deep-learning-with-python-notebooks\creative_commons_elephant.jpg'

# 大小为2248224的python图像库图像
img = image.load_img(img_path, target_size=(224, 224))

#  形状为224*224*3的float32格式的numpy数组
x = image.img_to_array(img)

#添加一个维度，将数组转换成（1，224，224，3）
x = np.expand_dims(x, axis=0)

#对批量进行预处理（按通道进行颜色标准化）
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])


np.argmax(preds[0])

# 预测向量中非洲象的元素
african_elephant_output = model.output[:, 386]

# block5_conv3层的输出特征图，它是VGG16的最后一个卷积层
last_conv_layer = model.get_layer('block5_conv3')

# 非洲象类别相对于block5_conv3的输出特征图的梯度
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

#形状为(512,)的向量，每个元素是特定特征图通道的梯度平均大小
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)


heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()


import cv2

# We use cv2 to load the original image
img = cv2.imread(img_path)

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img

# Save the image to disk
cv2.imwrite('/Users/fchollet/Downloads/elephant_cam.jpg', superimposed_img)
