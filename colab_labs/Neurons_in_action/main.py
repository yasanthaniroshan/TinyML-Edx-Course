import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE=0.09

TRUE_w = 2.0
TRUE_b = -1.0

class GenericModel(object):
    W:float = None
    b:float = None

    def __init__(self):
        self.W = tf.Variable(10.0)
        self.b = tf.Variable(10.0)

    def __call__(self,inputs:np.array):
        return self.W*inputs + self.b
    
def loss_function(predicted_y: tf.Tensor, target_y: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(predicted_y - target_y))


def train(model:GenericModel, inputs:np.array, outputs:np.array, learning_rate:float)->float:
    with tf.GradientTape() as t:
        current_loss = loss_function(model(inputs),outputs)
        dW, db = t.gradient(current_loss, [model.W, model.b])
        model.W.assign_sub(learning_rate * dW)
        model.b.assign_sub(learning_rate * db)
        return current_loss


xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]


model = GenericModel()


list_w, list_b = [], []
epochs = range(200)
losses = []

for epoch in epochs:
  list_w.append(model.W.numpy())
  list_b.append(model.b.numpy())
  current_loss = train(model, xs, ys, learning_rate=LEARNING_RATE)
  losses.append(current_loss)
  print('Epoch %2d: w=%1.2f b=%1.2f, loss=%2.5f' % (epoch, list_w[-1], list_b[-1], current_loss))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title('W values')
ax1.plot(epochs, list_w, 'r',epochs, [TRUE_w] * len(epochs), 'b')
ax2.set_title('b values')
ax2.plot(epochs, list_b, 'r',epochs, [TRUE_b] * len(epochs), 'b')
ax3.set_title('Loss')
ax3.plot(epochs, losses, 'r')
plt.show()