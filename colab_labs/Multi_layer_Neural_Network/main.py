import numpy as np
import tensorflow as tf
from tensorflow import keras

xs:np.array = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys:np.array = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

my_layer_1 = keras.layers.Dense(units=2, input_shape=[1])
my_layer_2 = keras.layers.Dense(units=1)
model = tf.keras.Sequential([my_layer_1, my_layer_2])
model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=500)

print(model.predict(np.array([10.0])))  
print(my_layer_1.get_weights())
print(my_layer_2.get_weights())