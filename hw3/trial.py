import numba as nb


@nb.jit()
def trial_precompiled(order, X, w, y):
    """
    :param order: list of indices
    :param X: list of dict
    :param w: dict
    :param y: list of scaling factors
    :return: no
    """
    for idx in order:
        x, l = X[idx], y[idx]
        for k, v in x.items():
            w[k] = w.get(k, 0) + v * l


def trial_python(order, X, w, y):
    for idx in order:
        x, l = X[idx], y[idx]
        for k, v in x.items():
            w[k] = w.get(k, 0) + v * l

#  TODO: use %timeit to test the performance of
#   these two functions and find there is no performance
#   gap between, the numba jit version is even slower.
