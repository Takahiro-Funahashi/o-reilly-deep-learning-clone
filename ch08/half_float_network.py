# coding: utf-8
import numpy as np
import sys

sys.path.append('./ch08/')
sys.path.append('./dataset/')

if __name__ == '__main__':
    from deep_convnet import DeepConvNet
    from mnist import load_mnist

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

    network = DeepConvNet()
    network.load_params("deep_convnet_params.pkl")

    sampled = 10000  # 高速化のため
    x_test = x_test[:sampled]
    t_test = t_test[:sampled]

    print("caluculate accuracy (float64) ... ")
    print(network.accuracy(x_test, t_test))

    # float16に型変換
    x_test = x_test.astype(np.float16)
    for param in network.params.values():
        param[...] = param.astype(np.float16)

    print("caluculate accuracy (float16) ... ")
    print(network.accuracy(x_test, t_test))
