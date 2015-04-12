from ml import loss

import numpy as np


class TestL2(object):
    def __init__(self):
        self.l2 = loss.L2()

    def test_loss(self):
        # zero loss for perfect prediction
        assert self.l2.get_loss(1, 1) == 0
        assert self.l2.get_loss(13.1, 13.1) == 0

        # correct l2 loss otherwise
        assert self.l2.get_loss(3, 0) == 9
        assert self.l2.get_loss(0, 3) == 9

        assert self.l2.get_loss(5, 1) == 16
        assert self.l2.get_loss(5.7, 1.7) == 16
        assert self.l2.get_loss(5.7, 1) == 4.7 ** 2

    def test_gradient(self):
        # zero gradient at minima
        assert self.l2.get_gradient(1, 1) == 0
        assert self.l2.get_gradient(9, 9) == 0

        # correct gradient otherwise
        assert self.l2.get_gradient(9, 0) == 18
        assert self.l2.get_gradient(0, 9) == -18

    def test_f0(self):
        Y = np.array([1, 1, 1])
        assert self.l2.get_f0(Y) == 1


        Y = np.array([1, 2, 3, 4])
        assert self.l2.get_f0(Y) == 2.5
