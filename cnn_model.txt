# -*- coding: utf-8 -*-
"""CNN_Model.ipynb

Original file is located at
    https://colab.research.google.com/drive/1K4P31Ibbz-ctVR3hTDBpW-Vh0UwYBBEt
"""
# %tensorflow_version 2.x
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, plot_model

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('Train Images: {}'.format(train_images.shape))
print('Train Labels: {}'.format(train_labels.shape))
print('Test Images: {}'.format(test_images.shape))
print('Test Labels: {}'.format(test_labels.shape))

train_images = train_images.reshape(train_images.shape[0],train_images.shape[1],train_images.shape[2], 1)
test_images = test_images.reshape(test_images.shape[0],test_images.shape[1],test_images.shape[2], 1)

train_images  = train_images / 255.0
test_images = test_images / 255.0

val_images = test_images[:5000]
val_labels = test_labels[:5000]

test_images = test_images[5000:]
test_labels = test_labels[5000:]

l = keras.layers
model = keras.Sequential([
                          l.Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
                          l.MaxPool2D(pool_size=(2,2), padding = 'same'),
                          l.Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform'),
                          l.MaxPool2D(pool_size=(2,2), padding = 'same'),
                          l.Flatten(),
                          l.Dense(1000, activation='relu', kernel_initializer='he_uniform'),
                          l.Dropout(0.5),
                          l.Dense(500, activation='relu', kernel_initializer='he_uniform'),
                          l.Dropout(0.5),
                          l.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

callback = keras.callbacks.ModelCheckpoint('Model.h5', verbose=1, save_best_only=True)

model.fit(train_images, train_labels, epochs=30, verbose=1, callbacks=[callback], validation_data=(val_images, val_labels))

model.evaluate(test_images, test_labels)

model.save('CNN_Model_99.6%.h5')

json = model.to_json()
with open('CNN_Model_99.6%.json', 'w') as json_f:
  json_f.write(json)

yaml = model.to_yaml()
with open('CNN_Model_99.6%.yaml', 'w') as yaml_f:
  yaml_f.write(yaml)

plot_model(model, 'CNN_Model_99.6%.png')

