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


class Log(Function):
    def get_loss(self, P, Y):
        return - Y * np.log(P) - (1 - Y) * np.log(1 - P)

    def get_gradient(self, P, Y):
        Z = P * (1 - P)
        Z = np.where(np.abs(Z) < 1e-9, 1e-9, Z)
        return (P - Y) / Z

    def get_f0(self, Y):
        return np.mean(Y)
