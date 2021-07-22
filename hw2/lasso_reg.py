import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from setup_problem import load_problem
from ridge_regression import plot_prediction_functions, compare_parameter_vectors


class Lasso:
    def __init__(self, l1reg, w=None):
        self.l1reg = l1reg
        if w is not None:
            self.w = w

    def fit(self):
        pass

    def predict(self, X):
        try:
            getattr(self, 'w')
        except AttributeError:
            raise RuntimeError('Train the model before prediction')
        y_pred = X@self.w
        return y_pred

    def evaluation(self, X, y):
        y_pred = self.predict(X)
        loss = self.loss_no_reg(y_pred, y) / len(y)
        return loss

    def eval_with_reg(self, X, y):
        y_pred = self.predict(X)
        loss = self.loss_with_reg(y_pred, y)
        return loss

    def loss_no_reg(self, y_pred, y_true):
        try:
            getattr(self, 'w')
        except AttributeError:
            raise RuntimeError('Initialize the model before scoring')
        err = y_pred - y_true
        return np.dot(err, err)

    def loss_with_reg(self, y_pred, y_true):
        loss = self.loss_no_reg(y_pred, y_true) + np.sum(np.abs(self.w))
        return loss


def fit_sht_algo(lasso, X: np.ndarray, y, tol, mode='cyclic', init_para=True, init='zero', max_pass=1000):
    assert mode in ('cyclic', 'random')
    assert init in ('zero', 'ridge')
    assert max_pass > 0 and isinstance(max_pass, int)
    num_instances, num_features = X.shape

    if init_para or not hasattr(lasso, 'w'):
        if init == 'zero':
            lasso.w = np.zeros(num_features)
        else:
            lasso.w = np.linalg.inv(X.T @ X + lasso.l1reg * np.identity(num_features)) @ X.T @ y

    lasso.w = lasso.w.astype(np.float64)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    loss_bf = lasso.eval_with_reg(X, y)

    for i in range(max_pass):
        order = np.arange(num_features)
        if mode == 'random':
            np.random.shuffle(order)
        shooting_single_pass(lasso.w, lasso.l1reg, order, X, y)
        loss_af = lasso.eval_with_reg(X, y)
        if np.abs(loss_af - loss_bf) < tol:
            break
        else:
            loss_bf = loss_af

    return i + 1


@nb.jit(nopython=True)
def shooting_single_pass(w, l1reg, order, X, y):
    for idx in order:
        x_idx = np.ascontiguousarray(X[:, idx])
        a = 2 * np.sum(x_idx ** 2)
        if a == 0:
            w[idx] = 0
        else:
            c = 2 * (np.sum(x_idx * y) - np.sum(w * (x_idx @ X))) + w[idx] * a
            w[idx] = np.sign(c) * max(.0, np.abs(c) - l1reg) / a


def test_shooting(l1reg, tol, max_pass=1000):
    x_train, y_train, x_test, y_test, target_fn, coefs_true, featurizer = load_problem('../data/hw2/lasso_data.pickle')
    X_train, X_test = map(featurizer, (x_train, x_test))
    lasso_cy, lasso_rd, lasso_zero = Lasso(l1reg), Lasso(l1reg), Lasso(l1reg)
    _, times_cy = fit_sht_algo(lasso_cy, X_train, y_train, tol, mode='cyclic', init='ridge', max_pass=max_pass)
    _, times_rd = fit_sht_algo(lasso_rd, X_train, y_train, tol, mode='random', init='ridge', max_pass=max_pass)
    _, times_zero = fit_sht_algo(lasso_zero, X_train, y_train, tol, init='zero', max_pass=max_pass)
    # TODO: Performance test and solution comparison.

    l_cy = lasso_cy.evaluation(X_test, y_test)
    l_rd = lasso_rd.evaluation(X_test, y_test)
    l_zero = lasso_zero.evaluation(X_test, y_test)
    return (lasso_cy, lasso_rd, lasso_zero), (times_cy, times_rd, times_zero), \
           (l_cy, l_rd, l_zero)


def homotopy_opt():
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurizer = load_problem('../data/hw2/lasso_data.pickle')
    X_train, X_val = map(featurizer, (x_train, x_val))
    max_l1reg = 2*np.max(np.abs(y_train@X_train))
    power = np.arange(30)
    l1set = max_l1reg*(0.8**power)

    losses = []
    paras = []
    lasso = Lasso(l1set[0])
    _ = fit_sht_algo(lasso, X_train, y_train, 1e-8, mode='random', init='ridge')
    losses.append(lasso.evaluation(X_val, y_val))
    paras.append(lasso)
    for l1reg in l1set[1:]:
        lasso.l1reg = l1reg
        _ = fit_sht_algo(lasso, X_train, y_train, 1e-8, init_para=False, mode='random')
        losses.append(lasso.evaluation(X_val, y_val))
        paras.append(lasso)

    return losses, l1set, paras


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
    losses, l1set, paras = homotopy_opt()

    # Function curve plot and bar chart
    idx = losses.index(min(losses))
    w, l1reg = paras[idx], l1set[idx]
    plot_lasso(l1reg, w)

    # Average validation loss versus l1reg parameters
    plt.plot(l1set, losses)
    plt.xlabel('L1Reg')
    plt.ylabel('Average Validation Loss')
    plt.xscale('symlog')
    plt.show()
