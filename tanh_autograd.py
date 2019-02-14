from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad

#def tanh(x):
#    return np.tanh(x)#(1.0 - np.exp(-x))  / (1.0 + np.exp(-x))

def tanh(x):
    return (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))

x = np.linspace(-7, 7, 200)

plt.plot(x, tanh(x),
         x, egrad(tanh)(x))                                     # first derivative
#         x, egrad(egrad(tanh))(x),                              # second derivative
#         x, egrad(egrad(egrad(tanh)))(x),                       # third derivative
#         x, egrad(egrad(egrad(egrad(tanh))))(x),                # fourth derivative
#         x, egrad(egrad(egrad(egrad(egrad(tanh)))))(x),         # fifth derivative
#         x, egrad(egrad(egrad(egrad(egrad(egrad(tanh))))))(x))  # sixth derivative



plt.show()