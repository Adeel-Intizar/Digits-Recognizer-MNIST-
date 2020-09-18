# -*- coding: utf-8 -*-
"""Digit Recognizer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FK_Ceh-XqQmDEoHo1BSlIOEWhlc1tZIw
"""
# Commented out IPython magic to ensure Python compatibility.


# %tensorflow_version 2.x

from tensorflow.keras.datasets import mnist
from tensorflow import keras
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

train_images = train_images.reshape(train_images.shape[0],784)
test_images = test_images.reshape(test_images.shape[0], 784)

model = keras.Sequential([
                          keras.layers.Dense(1000, activation='relu', kernel_initializer='he_uniform', input_dim = 784),
                          keras.layers.Dropout(0.4),
                          keras.layers.Dense(500, activation='relu', kernel_initializer='he_uniform'),
                          keras.layers.Dropout(0.4),
                          keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
                          keras.layers.Dropout(0.4),
                          keras.layers.Dense(10, activation='softmax')
])

model.summary()

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

model.fit(train_images, train_labels, epochs=150, validation_data= (test_images, test_labels), callbacks=[callback])

loss, acc = model.evaluate(train_images, train_labels, verbose=0)
print('Training Loss: {:.4} /t Training Accuracy {:.4}'.format(loss, acc))
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print('Testing Loss: {:.4} /t Testing Accuracy {:.4}'.format(loss, acc))

json_model = model.to_json()

yaml_model = model.to_yaml()

model.save('Model.h5')
