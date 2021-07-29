from functools import partial
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from setup_problem import load_problem
from ridge_regression import plot_prediction_functions, compare_parameter_vectors


@nb.jit(nopython=True)
def shooting_single_pass(order, w, l1reg, X, y):
    for idx in order:
        x_idx = np.ascontiguousarray(X[:, idx])
        a = 2 * np.sum(x_idx**2)
        if a == 0:
            w[idx] = 0
        else:
            c = 2 * (np.sum(x_idx * y) - np.sum(w * (x_idx@X))) + w[idx] * a
            w[idx] = np.sign(c) * max(.0, np.abs(c)-l1reg) / a


def shooting_algo(l1reg, X: np.ndarray, y, tolerance, w=None, mode='cyclic', init='zero', max_pass=1000):
    assert mode in ('cyclic', 'random')
    assert init in ('zero', 'ridge')
    assert max_pass > 0
    num_instances, num_features = X.shape

    if w is None:
        if init == 'zero':
            w = np.zeros(num_features)
        else:
            w = np.linalg.inv(X.T@X + l1reg*np.identity(num_features)) @ X.T @ y
    w = w.astype(np.float64)
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    for i in range(max_pass):
        loss_bf = loss_with_reg(X, y, w)
        order = np.arange(num_features)
        if mode == 'random':
            np.random.shuffle(order)
        shooting_single_pass(order, w, l1reg, X, y)
        loss_af = loss_with_reg(X, y, w)
        if np.abs(loss_af-loss_bf) < tolerance:
            break

    return w, i+1


def loss_no_reg(X, y, w):
    err = X@w - y
    return np.einsum('i,i', err, err)


def loss_with_reg(X, y, w):
    err = X@w - y
    loss = np.einsum('i,i', err, err) + np.sum(np.abs(w))
    return loss


def test_shooting(l1reg, tolerance, max_pass=1000):
    x_train, y_train, x_test, y_test, target_fn, coefs_true, featurizer = load_problem('../data/hw2/lasso_data.pickle')
    X_train, X_test = map(featurizer, (x_train, x_test))
    w_cy, times_cy = shooting_algo(l1reg, X_train, y_train, tolerance, mode='cyclic', init='ridge', max_pass=max_pass)
    w_rd, times_rd = shooting_algo(l1reg, X_train, y_train, tolerance, mode='random', init='ridge', max_pass=max_pass)
    w_zero, times_zero = shooting_algo(l1reg, X_train, y_train, tolerance, init='zero', max_pass=max_pass)
    # TODO: Performance test and solution comparison.

    no_reg_loss = partial(loss_no_reg, X=X_test, y=y_test)
    l_cy = no_reg_loss(w=w_cy)
    l_rd = no_reg_loss(w=w_rd)
    l_zero = no_reg_loss(w=w_zero)
    return (w_cy, w_rd, w_zero), (times_cy, times_rd, times_zero), \
           (l_cy, l_rd, l_zero)


def select_reg(l1set: list):
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurizer = load_problem('../data/hw2/lasso_data.pickle')
    X_train, X_val = map(featurizer, (x_train, x_val))
    losses = []
    paras = []
    for l1reg in l1set:
        w, _ = shooting_algo(l1reg, X_train, y_train, tolerance=1e-8, mode='random', init='ridge', max_pass=1000)
        loss = loss_no_reg(X_val, y_val, w)
        losses.append(loss)
        paras.append(w)
    idx = losses.index(min(losses))
    return losses, paras, idx


def homotopy_opt():
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurizer = load_problem('../data/hw2/lasso_data.pickle')
    X_train, X_val = map(featurizer, (x_train, x_val))
    num_val = X_val.shape[0]
    max_l1reg = 2*np.max(np.abs(y_train@X_train))
    power = np.arange(30)
    l1set = max_l1reg*(0.8**power)
    losses = []
    w, _ = shooting_algo(l1set[0], X_train, y_train, 1e-8, mode='random', init='ridge')
    loss = loss_no_reg(X_val, y_val, w) / num_val
    losses.append(loss)
    for l1reg in l1set[1:]:
        w, _=  shooting_algo(l1reg, X_train, y_train, 1e-8, w, mode='random')
        loss = loss_no_reg(X_val, y_val, w) / num_val
        losses.append(loss)
    return losses, l1set


def plot_lasso(best_reg, best_w):
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurizer = load_problem('../data/hw2/lasso_data.pickle')
    pred_fns = []
    x = np.sort(np.concatenate([np.arange(0, 1, .001), x_train]))
    X = featurizer(x)
    name = "Target Parameter Values (i.e. Bayes Optimal)"
    pred_fns.append({"name": name, "coefs": coefs_true, "preds": target_fn(x)})
    y = X@best_w
    name = f'Best Lasso Predictor with L1Reg={best_reg}'
    pred_fns.append({'name': name, 'coefs': best_w, 'preds': y})
    f = plot_prediction_functions(x, pred_fns, x_train, y_train)
    f.show()
    f = compare_parameter_vectors(pred_fns)
    f.show()


if __name__ == '__main__':
    # w, times, loss = test_shooting(5e-1, 1e-8, 1000)

    # l1set = [2**i for i in range(-5, 6)]
    # losses, w, idx = select_reg(l1set)
    # plot_lasso(l1set[idx], w[idx])

    losses, l1set = homotopy_opt()
    plt.plot(l1set, losses)
    plt.xlabel('L1Reg')
    plt.ylabel('Average Validation Loss')
    plt.xscale('symlog')
    plt.show()
