from . import loss

import numpy as np

class Model(object):
    pass


class LinearRegression(Model):
    def __init__(self, learn_rate, max_iter, loss_fun):
        self._learn_rate = learn_rate
        self._max_iter = max_iter
        self._loss_fun = loss_fun

        # we keep W to be 1d array
        self._W = None

    def fit(self, X, Y, verbose=False):
        X = self._pre_process_input(X)
        N, F = X.shape
        loss_fun = self._loss_fun

        # initial guess of our hyperplane
        W = np.ones(F)

        for i in range(self._max_iter):
            # single iteration of gradient descent
            if verbose:
                print('-' * 30, i, '-' * 30)
            P = self._predict(W, X)

            avg_loss = loss_fun.get_loss(P, Y)
            if verbose:
                print('Avg loss after %d iterations: %.3f' % (i, avg_loss))

            # gs is N sized 1d array containing loss fn gradients at each point
            gs = loss_fun.get_gradient(P, Y)
            G = np.dot(X.T, gs) / N

            # and we update our best model by moving opposite of gradient
            W = W - self._learn_rate * G

        self._W = W
        if verbose:
                print('-' * 30, 'DONE', '-' * 30)
                print('Learnt linear model:', self._W)
                P = self._predict(self._W, X)
                avg_loss = loss_fun.get_loss(P, Y)
                print('Avg loss of the learnt model: %.3f' % avg_loss)
        return self

    def predict(self, X):
        X = self._pre_process_input(X)
        return self._predict(self._W, X)

    def _pre_process_input(self, X):
        N, F = X.shape
        X_ = np.ones((N, F + 1))
        X_[:, :-1] = X
        return X_

    def _predict(self, W, X):
        return np.dot(X, W)
