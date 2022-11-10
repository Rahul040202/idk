import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale
print("Feature matrix:", x_train.shape)
print("Target matrix:", x_test.shape)
print("Feature matrix:", y_train.shape)
print("Target matrix:", y_test.shape)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=SGD(learning_rate=0.01),
loss='sparse_categorical_crossentropy',
metrics = ['sparse_categorical_accuracy'])
history = model.fit(x_train, y_train,
epochs=100,
batch_size=2000,
validation_split=0.3)
result = model.evaluate(x_test, y_test)
print("Loss: {}, Sparse-categorical-accuracy: {}".format(result[0], result[1]))
