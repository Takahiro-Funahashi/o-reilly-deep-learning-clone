# coding: utf-8
import numpy as np
import sys

sys.path.append('./common/')


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        from functions import softmax, cross_entropy_error
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == '__main__':
    from gradient import numerical_gradient

    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])

    net = simpleNet()

    def f(w): return net.loss(x, t)

    dW = numerical_gradient(f, net.W)

    print(dW)
