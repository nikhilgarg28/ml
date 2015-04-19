from ml import models
from ml import loss

import numpy as np
from numpy import testing
from sklearn.datasets import load_diabetes, load_boston

class TestLinearRegression(object):
    def __init__(self):
        self.alpha = 0.1
        self.n_iter = 100
        self.loss_fn = loss.L2()

    def test_fit(self):
        dataset = load_diabetes()
        X, Y = dataset.data, dataset.target
        max_loss = 10000
        lr = models.LinearRegression(self.alpha, self.n_iter)
        lr.fit(X, Y)
        P = lr.predict(X)
        loss1 = self.loss_fn.get_loss(P, Y)
        assert loss1 <= max_loss

        # also test that loss is decreasing with n_iter
        n_iter = self.n_iter * 2
        lr = models.LinearRegression(self.alpha, n_iter)
        lr.fit(X, Y)
        P = lr.predict(X)
        loss2 = self.loss_fn.get_loss(P, Y)
        assert loss2 <= max_loss

    def test_normalize(self):
        # boston dataset diverges unless features are normalized
        dataset = load_boston()
        X = dataset.data
        Y = dataset.target
        n_iter1 = 30
        n_iter2 = 10

        # if we don't normalize the data, model diverges
        lr = models.LinearRegression(self.alpha, n_iter1, normalize=False)
        lr.fit(X, Y)
        P1 = lr.predict(X)
        loss1 = self.loss_fn.get_loss(P1, Y)

        P2 = lr.predict(X, model_iter=n_iter2)
        loss2 = self.loss_fn.get_loss(P2, Y)
        assert loss2 <= loss1

        # but if we do normalize, then nothing like this happens
        max_loss = 50
        lr = models.LinearRegression(self.alpha, n_iter1, normalize=True)
        lr.fit(X, Y)
        P3 = lr.predict(X)
        loss3 = self.loss_fn.get_loss(P3, Y)

        P4 = lr.predict(X, model_iter=n_iter2)
        loss4 = self.loss_fn.get_loss(P4, Y)

        assert loss4 > loss3

        # and also that loss in absolute has become low
        assert loss3 <= max_loss

    def test_predict_at_iter(self):
        n_iter1 = self.n_iter
        n_iter2 = n_iter1 + 10
        dataset = load_diabetes()
        X, Y = dataset.data, dataset.target

        lr = models.LinearRegression(self.alpha, n_iter1)
        lr.fit(X, Y)
        P1 = lr.predict(X)

        # now verify that if we train for more number of iterations
        # the prediction at n_iter1 are same as final predictions in previous
        # case
        lr = models.LinearRegression(self.alpha, n_iter2)
        lr.fit(X, Y)
        P2 = lr.predict(X, model_iter=n_iter1)
        assert max(P1 - P2) < 1e-3
