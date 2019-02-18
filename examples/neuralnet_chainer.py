import numpy as np
from chainer0 import Function, Variable, Chain
from chainer0.links import Linear
from chainer0.optimizers import SGD
from chainer0.functions import sigmoid, mean_squared_error


data = Variable(np.array([[0,0],[0,1],[1,0],[1,1]]))
target = Variable(np.array([[0],[1],[0],[1]]))

model = Chain(
    f1=Linear(2, 3, nobias=True),
    f2=Linear(3,1, nobias=True),
)


np.random.seed(0)

model.f1.W.data = np.random.rand(2,3)
model.f2.W.data = np.random.rand(3,1)
corrects = [1.5526762252937214,0.7752226848043366,0.4495819126794703,0.30774152115761005,0.2444660944197518,0.21570282077417782,0.20233092035441014,0.19588246846140353,0.19256481794850783,0.1906683350559659]


def forward(x):
    h = sigmoid(model.f1(x))
    return model.f2(h)

optimizer = SGD(lr=0.1)
optimizer.setup(model)


for i in range(10):
    y = forward(data)
    loss = mean_squared_error(y, target)
    model.cleargrads()
    loss.backward()
    optimizer.update()
    print(loss.data)

    assert loss.data == corrects[i]

'''


ws = list()
ws.append(Variable(np.random.rand(2,3)))
ws.append(Variable(np.random.rand(3,1)))

for i in range(10):
    t = matmul(data, ws[0])
    pred = matmul(t, ws[1])
    diff = (pred - target)
    loss = sum(diff * diff)
    loss.backward()

    for w in ws:
        w.data -= w.grad * 0.1
        w.cleargrad()
    print(loss)
'''