# coding: utf-8
from PIL import Image
import numpy as np
import sys

sys.path.append('./dataset/')


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


if __name__ == '__main__':

    from mnist import load_mnist

    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=False)

    img = x_train[0]
    label = t_train[0]
    print(label)  # 5

    print(img.shape)  # (784,)
    img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
    print(img.shape)  # (28, 28)

    img_show(img)
