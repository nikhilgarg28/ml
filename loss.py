import numpy as np


class Function(object):
    def get_loss(self, p, y):
        raise NotImplementedError()

    def get_gradient(self, p, y):
        raise NotImplementedError()

    def get_f0(self, ys):
        raise NotImplementedError()


class L2(Function):
    def get_loss(self, P, Y):
        return np.mean((P - Y) ** 2)

    def get_gradient(self, P, Y):
        return 2 * (P - Y)

    def get_f0(self, Y):
        return np.mean(Y)
