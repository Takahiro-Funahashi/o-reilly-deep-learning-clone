# coding: utf-8
import numpy as np
import sys

sys.path.append('./ch05/')
sys.path.append('./dataset/')

if __name__ == '__main__':
    from two_layer_net import TwoLayerNet
    from mnist import load_mnist

    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ":" + str(diff))
