from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from ridge_regression import RidgeRegression, plot_confusion_matrix
from setup_problem import load_problem


def coefs_retrival(l2reg: float):
    ridge = RidgeRegression(l2reg)
    x_train, y_train, _, _, _, coefs_true, featurizer = load_problem('../data/hw2/lasso_data.pickle')
    X_train = featurizer(x_train)
    ridge.fit(X_train, y_train)
    return ridge.w_, coefs_true


def coefs_sparsify(epsilon: float, w: np.ndarray, coefs_true):
    # x_train, y_train, x_test, y_test, target_func, coefs_true, featurizer = load_problem('lasso_data.pickle')
    # X_train, X_test = map(featurizer, (x_train, x_test))
    # grid, results = do_grid_search_ridge(X_train, y_train, X_test, y_test)
    coefs_pred = deepcopy(w)
    mask = coefs_pred < epsilon
    coefs_pred[mask] = 0
    coefs_pred[~mask] = 1
    coefs_pred = coefs_pred.astype(np.int8)
    coefs_true[coefs_true != 0] = 1
    coefs_true = coefs_true.astype(np.int8)
    cnf_matrix = confusion_matrix(coefs_true, coefs_pred)
    plot_confusion_matrix(cnf_matrix, f'Confusion Matrix for $\epsilon={epsilon}$',
                          ['Zero', 'Non-zero'])
    plt.show()


if __name__ == '__main__':
    epsilons = [10**p for p in range(-6, 1)]
    w, coefs_true = coefs_retrival(0.01)
    for epsilon in epsilons:
        coefs_sparsify(epsilon, w, coefs_true)
