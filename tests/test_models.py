from ml import models
from ml import loss

import numpy as np
from numpy import testing
from sklearn.datasets import load_diabetes

class TestLinearRegression(object):
    def __init__(self):
        dataset = load_diabetes()
        self.X = dataset.data
        self.Y = dataset.target
        self.max_loss = 10000

    def test_fit(self):
        alpha = 0.1
        n_iter = 100
        loss_fn = loss.L2()
        lr = models.LinearRegression(alpha, n_iter, loss_fn)
        lr.fit(self.X, self.Y)
        P = lr.predict(self.X)
        loss1 = loss_fn.get_loss(P, self.Y)
        assert loss1 <= self.max_loss

        # also test that loss is decreasing with n_iter
        n_iter *= 5
        lr = models.LinearRegression(alpha, n_iter, loss_fn)
        lr.fit(self.X, self.Y)
        P = lr.predict(self.X)
        loss2 = loss_fn.get_loss(P, self.Y)
        assert loss2 <= loss1
