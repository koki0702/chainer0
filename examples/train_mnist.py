
#!/usr/bin/env python
"""Fully-connected neural network example using MNIST dataset
This code is a custom loop version of train_mnist.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
import numpy as np
import chainer0
from chainer0.datasets import mnist
import chainer0.links as L
import chainer0.functions as F
from chainer0 import Variable
#from chainer0 import serializers


# Network definition
class MLP(chainer0.Chain):

    def __init__(self, n_units, n_out):
        super().__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def l2_norm(params):
    t = 0
    for p in params:
        p = F.flatten(p)
        t += F.matmul(p, p)
    return t

def main():
    model = L.Classifier(MLP(100, 10))
    optimizer = chainer0.optimizers.SGD()  # chainer0.optimizers.Adam()
    optimizer.setup(model)
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    """
    if args.resume:
        # Resume from a snapshot
        serializers.load_npz('{}/mlp.model'.format(args.resume), model)
        serializers.load_npz('{}/mlp.state'.format(args.resume), optimizer)
    """

    max_epoch = 10
    batch_size = 256
    batch_num = x_train.shape[0]
    max_iter = batch_num // batch_size
    L2_reg = 0.001

    print("     Epoch     |    Train accuracy  |       Test accuracy  ")

    for e in range(max_epoch):
        for i in range(max_iter):
            model.cleargrads()
            x = x_train[(i * batch_size):((i + 1) * batch_size)]
            t = t_train[(i * batch_size):((i + 1) * batch_size)]
            x, t = Variable(x), Variable(np.array(t, "i"))
            loss = model(x, t)
            loss += L2_reg * l2_norm(model.params())
            loss.backward()
            optimizer.update()

        model(x_train, t_train)
        acc_train = model.accuracy
        model(x_test, t_test)
        acc_test = model.accuracy
        print("{:15}|{:20}|{:20}".format(e, acc_train.data, acc_test.data))


if __name__ == '__main__':
    main()