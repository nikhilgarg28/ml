import matplotlib.pyplot as plt
import numpy as np


def plot_errors(train_errors, test_errors=None):
    """Plots train (& optionally test) errors as function of # of iterations."""
    n_iter = len(train_errors)
    X = np.arange(1, n_iter + 1)
    plt.plot(X, train_errors, marker='o', linestyle='-', color='r',
            label='Train')
    if test_errors is not None:
        plt.plot(X, test_errors, marker='^', linestyle='--', color='b',
                label='Test')

    plt.xlabel('# of iterations')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
