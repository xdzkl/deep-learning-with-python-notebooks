#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
print(tf.__version__)
# 2.0.0-alpha0

import os, shutil

# 原始数据集解压目录的路径
original_dataset_dir = 'F:\zkl_repository\pic\\all_pic'

# 保存较小数据集的目录
base_dir = 'F:\zkl_repository\small_pics'
try:
    os.mkdir(base_dir)
except(FileExistsError):
    print('文件夹已经创建')

# 分别对应划分后的训练，验证和测试目录
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

#猫的训练图像目录
train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)

#狗的训练图像目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)

#猫的验证图像目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir):
    os.mkdir(validation_cats_dir)

#狗的验证图像目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)

#猫的测试图像目录
test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)

#狗的测试图像目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)

#shutil.copyfile(src, dst):将名为src的文件的内容（无元数据）复制到名为dst的文件中 。 dst必须是完整的目标文件名
# 将1000张猫的图像复制到train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将500张猫的图像复制到validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# 将500张猫的图像复制到test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# 将1000张狗的图像复制到train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# 将500张狗的图像复制到validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# 将500张狗的图像复制到test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


print('total training cat images:', len(os.listdir(train_cats_dir)))
#total training cat images: 1000

print('total training dog images:', len(os.listdir(train_dogs_dir)))
#total training dog images: 1000

print('total validation cat images:', len(os.listdir(validation_cats_dir)))
#total validation cat images: 500

print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
#total validation dog images: 500

print('total test cat images:', len(os.listdir(test_cats_dir)))
#total test cat images: 500

print('total test dog images:', len(os.listdir(test_dogs_dir)))
#total test dog images: 500


# model是一个模型
from tensorflow.keras import models
# layers是一个层，可以这么理解，多个层构成了一个模型，或者说一个神经网络
from tensorflow.keras import layers

# 建立一个序贯模型，是多个网络层的线性堆叠，也就是一条路走到黑，
#详细信息见：https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
model = models.Sequential()

# 添加一个二维卷积层，32代表卷积核的数量，卷积核大小是3*3，激活函数是relu,输入维度是150*150*3
# 输出维度是148*148*32
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))

# 添加一个最大池化层，池化核大小是2*2
# 输人维度是148*148*32，# 输出维度是74*74，*32
model.add(layers.MaxPooling2D((2, 2)))

# 添加一个二维卷积层，64代表卷积核的数量，卷积核大小是3*3，激活函数是relu,输入维度是32*32*1
# 输入维度是74*74*32，输出维度是72*72*64
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 添加一个最大池化层，池化核大小是2*2
# 输入维度是72*72*64，输出维度是36*36*64
model.add(layers.MaxPooling2D((2, 2)))

# 添加一个二维卷积层，128代表卷积核的数量，卷积核大小是3*3，激活函数是relu,输入维度是32*32*1
# 输入维度是36*36*64，输出维度是34*34*128
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# 添加一个最大池化层，池化核大小是2*2
# 输入维度是34*34*128，输出维度是17*17*128
model.add(layers.MaxPooling2D((2, 2)))

# 添加一个二维卷积层，128代表卷积核的数量，卷积核大小是3*3，激活函数是relu,输入维度是32*32*1
# 输入维度是17*17*128，输出维度是15*15*128
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# 添加一个最大池化层，池化核大小是2*2
# 输入维度是15*15*128，输出维度是7*7*128
model.add(layers.MaxPooling2D((2, 2)))

# 将3D输出展平为1D，7*7*128=6727
model.add(layers.Flatten())

# 添加全连接层，输出维度是512，激活函数是relu,输入维度是
model.add(layers.Dense(512, activation='relu'))

# 添加全连接层，输出维度是1，激活函数是softmax，输入维度是512
model.add(layers.Dense(1, activation='sigmoid'))


model.summary()
#Model: "sequential_2"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_7 (Conv2D)            (None, 148, 148, 32)      896       
#_________________________________________________________________
#max_pooling2d_6 (MaxPooling2 (None, 74, 74, 32)        0         
#_________________________________________________________________
#conv2d_8 (Conv2D)            (None, 72, 72, 64)        18496     
#_________________________________________________________________
#max_pooling2d_7 (MaxPooling2 (None, 36, 36, 64)        0         
#_________________________________________________________________
#conv2d_9 (Conv2D)            (None, 34, 34, 128)       73856     
#_________________________________________________________________
#max_pooling2d_8 (MaxPooling2 (None, 17, 17, 128)       0         
#_________________________________________________________________
#conv2d_10 (Conv2D)           (None, 15, 15, 128)       147584    
#_________________________________________________________________
#max_pooling2d_9 (MaxPooling2 (None, 7, 7, 128)         0         
#_________________________________________________________________
#flatten_2 (Flatten)          (None, 6272)              0         
#_________________________________________________________________
#dense_4 (Dense)              (None, 512)               3211776   
#_________________________________________________________________
#dense_5 (Dense)              (None, 1)                 513       
#=================================================================
#Total params: 3,453,121
#Trainable params: 3,453,121
#Non-trainable params: 0
#_________________________________________________________________

from tensorflow.keras import optimizers

# compile的功能是编译模型，对学习过程进行配置，optimizer是优化器，
# loss是损失函数，metrics是指标列表
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# 数据增强是从现有的训练样本中生成更多的训练数据
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 将所有的图像乘以1/255缩放
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # 将所有图像的大小调整为150*150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')




for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


history = model.fit_generator(
      train_generator,
#      从生成器中抽取steps_per_epoch个批量后，拟合过程进入下一轮次
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

# 保存模型
model.save('cats_and_dogs_small_1.h5')



import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


datagen = ImageDataGenerator(
#        角度值
      rotation_range=40,
#      图像在水平方向上平移的范围，相对于总宽度的比例
      width_shift_range=0.2,
#      图像在垂直方向上平移的范围，相对于总高度的比例
      height_shift_range=0.2,
#      随机错切变换的角度
      shear_range=0.2,
#       图形随机缩放的范围
      zoom_range=0.2,
#       随机将一半图像水平翻转
      horizontal_flip=True,
#      用于填充新创建像素的方法
      fill_mode='nearest')


# 图像预处理工具模块
from tensorflow.keras.preprocessing import image
# 获得文件名
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# 选择一张图片进行增强
img_path = fnames[3]

# 读取图像并调整大小
img = image.load_img(img_path, target_size=(150, 150))

# 将其转换为（150，150，3）的numpy数组
x = image.img_to_array(img)

# 将其转换为(1,150,150,3)的数组
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# 不能增强验证数据
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)


model.save('cats_and_dogs_small_2.h5')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


