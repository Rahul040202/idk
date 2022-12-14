import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(100)
x_data = np.linspace(0, 100, 100)
y_data = np.linspace(0, 100, 100)
x_data += np.random.uniform(-5, 5, 100)
y_data += np.random.uniform(-5, 5, 100)
plt.scatter(x_data, y_data)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=[1])) # adding a single unit
model.compile(loss='mean_squared_error',
optimizer=tf.keras.optimizers.Adamax(0.01),
metrics='mean_squared_error')
model.summary()
history = model.fit(x_data, y_data, epochs=100)
plt.scatter(x_data, y_data)
plt.plot(x_data, y_data_est, color='r')
