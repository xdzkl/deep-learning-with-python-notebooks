#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
print(tf.__version__)
# 2.0.0-alpha0

from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
conv_base.summary()
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#input_1 (InputLayer)         [(None, 150, 150, 3)]     0         
#_________________________________________________________________
#block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
#_________________________________________________________________
#block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
#_________________________________________________________________
#block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
#_________________________________________________________________
#block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
#_________________________________________________________________
#block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
#_________________________________________________________________
#block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
#_________________________________________________________________
#block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
#_________________________________________________________________
#block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
#_________________________________________________________________
#block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
#_________________________________________________________________
#block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
#_________________________________________________________________
#block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
#_________________________________________________________________
#block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
#_________________________________________________________________
#block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
#_________________________________________________________________
#block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
#_________________________________________________________________
#block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
#_________________________________________________________________
#block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
#_________________________________________________________________
#block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
#_________________________________________________________________
#block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
#=================================================================
#Total params: 14,714,688
#Trainable params: 14,714,688
#Non-trainable params: 0
#_________________________________________________________________


import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = 'F:\zkl_repository\small_pics'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)


train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))


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

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
#Model: "sequential_2"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#vgg16 (Model)                (None, 4, 4, 512)         14714688  
#_________________________________________________________________
#flatten (Flatten)            (None, 8192)              0         
#_________________________________________________________________
#dense_4 (Dense)              (None, 256)               2097408   
#_________________________________________________________________
#dense_5 (Dense)              (None, 1)                 257       
#=================================================================
#Total params: 16,812,353
#Trainable params: 16,812,353
#Non-trainable params: 0
#_________________________________________________________________

print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))

# trainable设置的目的是冻结网络，不再进行训练
conv_base.trainable = False


print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# 注意，不能增强验证数据
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
#         目标目录
        train_dir,
#       将所有图像的大小调整为150*150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)

model.save('cats_and_dogs_small_3.h5')


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

conv_base.summary()
#Model: "vgg16"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#input_1 (InputLayer)         [(None, 150, 150, 3)]     0         
#_________________________________________________________________
#block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
#_________________________________________________________________
#block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
#_________________________________________________________________
#block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
#_________________________________________________________________
#block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
#_________________________________________________________________
#block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
#_________________________________________________________________
#block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
#_________________________________________________________________
#block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
#_________________________________________________________________
#block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
#_________________________________________________________________
#block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
#_________________________________________________________________
#block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
#_________________________________________________________________
#block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
#_________________________________________________________________
#block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
#_________________________________________________________________
#block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
#_________________________________________________________________
#block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
#_________________________________________________________________
#block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
#_________________________________________________________________
#block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
#_________________________________________________________________
#block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
#_________________________________________________________________
#block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
#=================================================================
#Total params: 14,714,688
#Trainable params: 0
#Non-trainable params: 14,714,688
#_________________________________________________________________


# 重新训练网络
# 目的是模型微调，与特征提取互为补充，微调是指将其顶部几层"解冻"，并将这解冻的几层
#和新增部分（本文是全练级分类器）联合训练，之所以叫微调，是因为它只是略微调整了所复用
#模型中更加抽象的部分
conv_base.trainable = True

# 只将第卷积块5解冻，进行微调
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False



model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)

# 保存模型
model.save('cats_and_dogs_small_4.h5')


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



def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)

