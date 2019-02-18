import numpy as np

from chainer0 import Variable
from chainer0.functions import tanh, matmul, log, sum


def sigmoid(x):
    return 0.5 * (tanh(x) + 1)

def logistic_predictions(weights, inputs):
    score = matmul(inputs, weights)
    return sigmoid(score)

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -sum(log(label_probabilities))

# Build a toy dataset.
inputs = Variable(np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]]))
targets = Variable(np.array([[True], [True], [False], [True]]))
weights = Variable(np.array([[0.0], [0.0], [0.0]]))


# Define a function that returns gradients of training loss using Autograd.

print("Initial loss:", training_loss(weights).data)
assert training_loss(weights).data == 2.772588722239781

for i in range(100):
    weights.cleargrad()
    loss = training_loss(weights)
    loss.backward()
    #weights -= weights.grad * 0.01
    weights.data -= weights.grad * 0.01

print("Trained loss:", training_loss(weights).data)
assert training_loss(weights).data == 0.38900754315581143

'''
training_gradient_fun = grad(training_loss)

# Optimize weights using gradient descent.

print("Initial loss:", training_loss(weights))
for i in range(100):
    weights -= training_gradient_fun(weights) * 0.01

print("Trained loss:", training_loss(weights))
'''
