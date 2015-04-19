import itertools
import matplotlib.pyplot as plt
import numpy as np


def learning_curve(train_errors, loss_fn_name=None, test_errors=None):
    """Plots train (& optionally test) errors as function of # of iterations."""
    n_iter = len(train_errors)
    X = np.arange(1, n_iter + 1)
    plt.plot(X, train_errors, marker='o', linestyle='-', color='r',
            label='Train')
    if test_errors is not None:
        plt.plot(X, test_errors, marker='^', linestyle='--', color='b',
                label='Test')

    plt.xlabel('# of iterations')
    ylabel = 'Loss'
    if loss_fn_name is not None:
        ylabel += ': %s' % loss_fn_name
    plt.ylabel(ylabel)
    plt.title('Learning Curve')
    plt.legend()
    plt.show()


def scatterplot_matrix(data, names, classes=None, W=None):
    """Plots a scatterplot matrix of subplots.

    Each row of "data" is plotted against other rows, resulting in a nrows by
    nrows grid of subplots with the diagonal subplots labeled with "names".

    Parameters
    ----------
    data: 2D numpy array (n_points x n_variables)
        Feature data.
    names: string list (n_variables)
        List of names of all the features.
    classes: list of ints (n_points)
        List of identifiers that uniquely identiy each point as belonging to a
        particular class. Useful for classification problems.
    W: 1D numpy array or size n_variables or n_variables + 1
        Hyperplane in the feature space which should also be plotted in each sub plot.
        If it is of dimension n_variables, intercept is assumed to be zero.
        If it is of dimension n_variables + 1, last entry is assumed to be the intercept.

    """
    n_points, n_vars = data.shape
    data_ = data.T
    fig, axes = plt.subplots(nrows=n_vars, ncols=n_vars, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')


    # Choose the right colors for each point
    # if classes are provided, we try to paint points of each class with
    # different color
    colors = 'rbgycmkw'

    if classes is not None:
        color = {}
        all_classes = sorted(list(set(classes)))
        for class_ in classes:
            idx = all_classes.index(class_) % len(colors)
            color[class_] = colors[idx]

        c = [color[class_] for class_ in classes]

    else:
        c = 'black'

    # make sure W if present is n_points x (n_vars + 1)
    if W is not None:
        assert n_vars <= len(W) <= n_vars + 1
        if len(W) == n_vars:
            W_ = np.ones(n_vars + 1)
            W_[:-1] = W
            W = _W

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].scatter(data_[x], data_[y], c=c)

            # if
            if W is not None:
                x_is_zero = abs(W[x]) <1e-9
                y_is_zero = abs(W[y]) <1e-9
                if x_is_zero and y_is_zero:
                    continue
                if not x_is_zero:
                    line_y =  (-W[-1] - W[x] * data_[x]) / W[y]
                    axes[x, y].plot(data_[x], line_y, color='black', linestyle='--')
                else:
                    line_x =  (-W[-1] - W[y] * data_[y]) / W[x]
                    axes[x, y].plot(data_[y], line_x, color='black', linestyle='--')

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(n_vars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    plt.show()
