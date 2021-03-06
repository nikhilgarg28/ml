from ml import loss

from nose.tools import *

import numpy as np
from numpy import testing


class TestL2(object):
    def __init__(self):
        self.l2 = loss.L2()
        self.a = np.ones(3)
        self.b = self.a + 1

    def test_loss(self):
        # zero loss for perfect prediction, should work on both arrays & scalars
        eq_(self.l2.get_loss(self.a, self.a), 0)
        eq_(self.l2.get_loss(13.1, 13.1), 0)

        # correct l2 loss otherwise
        eq_(self.l2.get_loss(3, 0), 9)
        eq_(self.l2.get_loss(0, 3), 9)
        eq_(self.l2.get_loss(self.a, self.b), 1)
        eq_(self.l2.get_loss(self.b, self.a), 1)

        eq_(self.l2.get_loss(5, 1), 16)
        eq_(self.l2.get_loss(5.7, 1.7), 16)
        eq_(self.l2.get_loss(5.7, 1), 4.7 ** 2)

    def test_gradient(self):
        # zero gradient at minima
        eq_(self.l2.get_gradient(1, 1), 0)
        eq_(self.l2.get_gradient(9, 9), 0)

        # works for arrays too, returns array of same size
        testing.assert_array_almost_equal(
                self.l2.get_gradient(self.a, self.a),
                np.zeros(3)
        )

        # correct gradient otherwise
        eq_(self.l2.get_gradient(9, 0), 18)
        eq_(self.l2.get_gradient(0, 9), -18)
        testing.assert_array_almost_equal(
                self.l2.get_gradient(self.a, self.b),
                np.array([-2, -2, -2])
        )
        testing.assert_array_almost_equal(
                self.l2.get_gradient(self.b, self.a),
                np.array([2, 2, 2])
        )

    def test_f0(self):
        Y = np.array([1, 1, 1])
        eq_(self.l2.get_f0(Y), 1)
        Y = np.array([1, 2, 3, 4])
        eq_(self.l2.get_f0(Y), 2.5)


class TestLog(object):
    def __init__(self):
        self.lf = loss.Log()
        self.a = np.array([0.1, 0.2, 0.3])
        self.b = np.array([0.3, 0.2, 0.1])

    def test_loss(self):
        eq_(self.lf.get_loss(0.3, 0), -np.log(0.7))
        eq_(self.lf.get_loss(0.3, 1), -np.log(0.3))
        assert abs(self.lf.get_loss(self.a, self.b) - 1.70633504237) < 1e-6

    def test_gradient(self):
        # zero gradient at minima
        eq_(self.lf.get_gradient(0.5, 0.5), 0)

        # works for arrays too, returns array of same size
        testing.assert_array_almost_equal(
                self.lf.get_gradient(self.a, self.a),
                np.zeros(3)
        )

        # correct gradient otherwise
        assert abs(self.lf.get_gradient(0.3, 0) - 1.0 / 0.7) < 1e-6
        assert abs(self.lf.get_gradient(0.3, 1) + 1.0 / 0.3) < 1e-6

        testing.assert_array_almost_equal(
                self.lf.get_gradient(self.b, self.a),
                np.array([ 0.95238095, 0, -2.22222222])
        )

    def test_f0(self):
        Y = np.array([1, 1, 1])
        eq_(self.lf.get_f0(Y), 1)

        Y = np.array([1, 2, 3, 4])
        eq_(self.lf.get_f0(Y), 2.5)


class TestZeroOne(object):
    def __init__(self):
        self.lf = loss.ZeroOne()
        self.a = np.array([0, 1, 2])
        self.b = np.array([2, 1, 0])

    def test_loss(self):
        eq_(self.lf.get_loss(1, 1), 0)
        eq_(self.lf.get_loss(0, 1), 1)
        eq_(self.lf.get_loss(0, 2), 1)

        eq_(self.lf.get_loss(self.a, self.b), 2)

    @raises(loss.UndefinedError)
    def test_gradient(self):
        self.lf.get_gradient(1, 1)


    def test_f0(self):
        eq_(self.lf.get_f0(np.array([0, 0, 1])), 0)
        eq_(self.lf.get_f0(np.array([5, 5, 5])), 5)
