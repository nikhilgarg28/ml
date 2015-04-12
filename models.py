import loss
import numpy as np

class Model(object):
    pass


class LinearRegressor(Model):
    def __init__(self, learn_rate, max_iter, loss_fun):
        self._learn_rate = learn_rate
        self._max_iter = max_iter
        self._loss_fun = loss_fun
        self._W = None

    def fit(self, X, Y):
        X = self._pre_process_input(X)
        N, F = X.shape
        loss_fun = self._loss_fun

        # initial guess of our hyperplane
        W = [1] * F

        for _ in range(self._max_iter):
            # single iteration of gradient descent
            G = [0] * F
            for X, y in zip(X, Y):
                p = self._predict(W, X)
                g = loss_fun.get_gradient(p, y)
                G += g * X

            # and we update our best model by moving opposite of gradient
            W = W - self._learn_rate * G

        self._W = W
        return self

    def predict(self, X):
        X = self._process(X)
        return self._predict(self._W, X)

    def _pre_process_input(self, X):
        N, F = X.shape
        X_ = np.zeros(N, F + 1)
        X_[:, :-1] = X
        return X_

    def _predict(self, W, X):
        return np.dot(W, X)
