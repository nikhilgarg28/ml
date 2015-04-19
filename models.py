from . import loss

import numpy as np

class Model(object):
    def fit(self, X, Y, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, X, *args, **kwargs):
        raise NotImplementedError()


class _LinearRegression(Model):
    def __init__(self, learn_rate, max_iter, loss_fun, normalize=True):
        """Linear Regression model based on gradient descent.

        Parameters
        ----------
        learn_rate : float
            The rate at which model learns from gradient descent.

        max_iter: int
            The number of gradient descent iterations to try.

        loss_fun : loss function object
            An object that implements get_loss, get_gradient and get_f0
            functions.

        normalize : boolean, optional, default True
            If features should be centered and normalized before regression.

        """
        self._learn_rate = learn_rate
        self._max_iter = max_iter
        self._loss_fun = loss_fun
        self._normalize = normalize

        # feature centering parameters
        self._means = None
        self._scales = None

        # W represents the learnt model, we keep it 1D array
        self._W = None

        # We also store all the models we learn along the way
        # useful in plotting train/test errors as function of # of iterations
        self._models = []

    def fit(self, X, Y, verbose=False):
        self._compute_center_params(X)
        X = self._pre_process_input(X)
        N, F = X.shape

        # initial guess of our hyperplane
        W = np.ones(F)
        self._models.append(W)

        for i in range(self._max_iter):
            # single iteration of gradient descent
            if verbose:
                print('-' * 30, i, '-' * 30)
            P = self._predict(W, X)

            if verbose:
                avg_loss = self._get_loss(P, Y)
                print('Avg loss after %d iterations: %.3f' % (i, avg_loss))

            # gs is N sized 1d array containing loss fn gradients at each point
            gs = self._get_gradient(P, Y)
            G = np.dot(X.T, gs) / N

            # and we update our best model by moving opposite of gradient
            W = W - self._learn_rate * G
            self._models.append(W)

        self._W = W
        if verbose:
                print('-' * 30, 'DONE', '-' * 30)
                print('Learnt linear model:', self._W)
                P = self._predict(self._W, X)
                avg_loss = self._get_loss(P, Y)
                print('Avg loss of the learnt model: %.3f' % avg_loss)
        return self

    def _get_gradient(self, P, Y):
        # for numeric stability reasons, we don't always use loss_fun's own
        # gradient.
        return P - Y

    def _get_loss(self, P, Y):
        return self._loss_fun.get_loss(P, Y)

    def predict(self, X, model_iter=None):
        """Predicts the regression output for X.

        Parameters
        ----------
        X : numpy array
        model_iter : int, optional, default None
            If provided, the model corresponding to the given iteration of the
            learning algorithm is used to predict.

        """
        X = self._pre_process_input(X)
        if model_iter is None:
            W = self._W
        else:
            W = self._models[model_iter]

        return self._predict(W, X)

    def _pre_process_input(self, X):
        if self._normalize:
            X = (X - self._means) / self._scales
        N, F = X.shape
        X_ = np.ones((N, F + 1))
        X_[:, :-1] = X
        return X_

    def _compute_center_params(self, X):
        if self._normalize:
            self._means = np.mean(X, axis=0)
            self._scales = np.std(X, axis=0)

    def _predict(self, W, X):
        return np.dot(X, W)


class LinearRegression(_LinearRegression):
    def __init__(self, learn_rate, max_iter, normalize=True):
        loss_fun = loss.L2()
        super().__init__(learn_rate, max_iter, loss_fun, normalize=normalize)


class LogisticRegression(_LinearRegression):
    def __init__(self, learn_rate, max_iter, normalize=True):
        loss_fun = loss.Log()
        super().__init__(learn_rate, max_iter, loss_fun, normalize=normalize)
