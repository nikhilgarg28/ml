from . import loss

import numpy as np
import random

class Model(object):
    def __init__(self, normalize=True, intercept=False):
        # feature centering parameters
        self._means = None
        self._scales = None
        self._normalize = normalize
        self._intercept = intercept

    def fit(self, X, Y, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, X, *args, **kwargs):
        raise NotImplementedError()

    def _compute_center_params(self, X):
        if self._normalize:
            self._means = np.mean(X, axis=0)
            self._scales = np.std(X, axis=0)

    def _pre_process_input(self, X):
        if self._normalize:
            X = (X - self._means) / self._scales
        if self._intercept:
            N, F = X.shape
            X_ = np.ones((N, F + 1))
            X_[:, :-1] = X
            X = X_
        return X


class _LinearRegression(Model):
    def __init__(self, learn_rate, max_iter, loss_fun, normalize=True,
            intercept=True):
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


        # W represents the learnt model, we keep it 1D array
        self._W = None

        # We also store all the models we learn along the way
        # useful in plotting train/test errors as function of # of iterations
        self._models = []

        super().__init__(normalize=normalize, intercept=intercept)

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

    def _predict(self, W, X):
        return np.dot(X, W)


class LinearRegression(_LinearRegression):
    def __init__(self, learn_rate, max_iter, normalize=True):
        loss_fun = loss.L2()
        super().__init__(learn_rate, max_iter, loss_fun, normalize=normalize)


class LogisticRegression(_LinearRegression):
    def __init__(self, learn_rate, max_iter, normalize=True, intercept=True):
        loss_fun = loss.Log()
        super().__init__(learn_rate, max_iter, loss_fun, normalize=normalize,
                intercept=intercept)

    def _predict(self, W, X):
        d = super()._predict(W, X)
        return self._sigmoid(d)

    def _sigmoid(self, d):
        return (1. / (1 + np.exp(-d)))

    def predict(self, X, model_iter=None):
        d = super().predict(X, model_iter=model_iter)
        # binarize the output
        return np.where(d <= 0.5, 0, 1)


# TODO: what happens with categoy features?
# TODO: use a tighter bound for num_nodes
class DecisionTreeClassifier(Model):
    def __init__(self, criterion_fun, max_depth=None, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion_fun = criterion_fun

    def fit(self, X, Y, verbose=False):
        N, F = X.shape
        # max number of nodes in tree with upto N leaves
        self._num_nodes = 4 * N
        self._data = [None] * self._num_nodes
        self._splits = [None] * self._num_nodes
        self._values = [None] * self._num_nodes

        self._data[1] = np.arange(N)
        self._split(X, Y, 1, 1, verbose=verbose)

    def _split(self, X, Y, index, depth, verbose=False):
        if verbose:
            print('Splitting at index: %d' % index)

        rows = self._data[index]
        num_samples = len(rows)
        if num_samples <= self.min_samples_leaf:
            self._values[index] = self._region_value(Y[rows])
            return

        if self.max_depth is not None and depth > self.max_depth:
            self._values[index] = self._region_value(Y[rows])
            return

        # we now try to split the node at given index
        N, F = X.shape
        best_feature, best_threshold, best_gain = None, None, 0

        if self.max_features is None:
            max_features = F
        else:
            max_features = self.max_features
        candidate_features = np.random.choice(np.arange(F), max_features,
                replace=False)
        for feature in candidate_features:
            if verbose:
                print('Trying to split on feature: %d' % feature)
            points = X[rows, feature]
            points = np.sort(points)
            for i in range(len(points)):
                w = np.random.rand()
                if i < len(points) - 1:
                    threshold = w * points[i] + (1-w) * points[i+1]
                else:
                    threshold = points[i] + w * (points[i] - points[i-1])
                if verbose:
                    print('Trying threshold:', threshold)
                split_gain = self._eval_split(X, Y, rows, feature, threshold)
                if verbose:
                    print('Gain is: ', split_gain)

                if best_gain is None or split_gain > best_gain:
                    best_feature = feature
                    best_threshold = threshold
                    best_gain = split_gain

        if best_feature is not None:
            # some valid split is found
            self._splits[index] = (best_feature, best_threshold)

            split_condition = X[rows, best_feature] < best_threshold
            left, right = 2*index, 2*index + 1
            self._data[left] = rows[split_condition]
            self._split(X, Y, left, depth + 1, verbose=verbose)

            self._data[right] = rows[~split_condition]
            self._split(X, Y, right, depth + 1, verbose=verbose)
        else:
            self._values[index] = self._region_value(Y[rows])

    def _eval_split(self, X, Y, rows, feature, threshold):
        condition = X[rows, feature] < threshold
        left = rows[condition]
        right = rows[~condition]

        return self.criterion_fun.get_gain(Y[left], Y[right])

    def _region_value(self, Y):
        return self.criterion_fun.get_value(Y)
        # return the most frequent class for classifier
        # TODO: bincount only works for non-negative integers, generalize
        counts = np.bincount(Y)
        return np.argmax(counts)

    def predict(self, X):
        return self._predict(X, 1)

    def _predict(self, X, index):
        N, _ = X.shape
        if self._values[index] is None:
            feature, threshold = self._splits[index]
            split = X[:, feature] <= threshold
            left, right = 2*index, 2*index + 1
            X_left = X[split]
            Y_left = self._predict(X_left, left)

            X_right = X[~split]
            Y_right = self._predict(X_right, right)

            Y = np.empty(N)
            Y[split] = Y_left
            Y[~split] = Y_right
            return Y
        else:
            return np.repeat(self._values[index], N)


class CriterionFunction(object):
    def get_gain(self, Y1, Y2):
        Y = np.append(Y1, Y2)
        counts1 = np.bincount(Y1)
        counts2 = np.bincount(Y2)
        counts = np.bincount(Y)

        left_info = self._info(counts1)
        right_info = self._info(counts2)
        old_info = self._info(counts)

        l1 = np.sum(counts1)
        l2 = np.sum(counts2)
        r = float(l1) / (l1 + l2)

        new_info = r*left_info + (1-r)*right_info
        return old_info - new_info

    def get_value(self, Y):
        counts = np.bincount(Y)
        return np.argmax(counts)

    def _info(self, counts):
        counts = counts [ counts > 0 ]
        probs = counts / np.sum(counts)
        return  -np.sum(probs * np.log2(probs))
