import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import StandardScaler
from functools import partial


def f_objective(theta, X, y, l2_param=1):
    """
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    """
    no_reg_loss = evaluate_no_reg(theta, X, y)
    reg_loss = no_reg_loss + l2_param*np.dot(theta, theta)
    return reg_loss


def evaluate_no_reg(theta, X, y):
    num_ins = len(X)
    loss_prob = np.logaddexp(0, y * (X @ theta))
    no_reg_loss = np.sum(loss_prob) / num_ins
    return no_reg_loss


def fit_logistic_reg(X, y, objective_function, l2_param=1):
    """
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter

    Returns:
        optimal_theta: 1D numpy array of size num_features
    """
    num_feat = X.shape[1]
    theta_init = np.zeros(num_feat)
    opt_result = opt.minimize(objective_function, theta_init, args=(X, y, l2_param))
    return opt_result


def data_preprocess(X_train, X_test, y_train, y_test):
    num_train_ins = len(X_train)
    num_test_ins = len(X_test)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.concatenate([X_train, np.ones([num_train_ins, 1])], axis=1)
    X_test = np.concatenate([X_test, np.ones([num_test_ins, 1])], axis=1)
    zero_mask = y_train == 0
    y_train[zero_mask] = -1
    zero_mask = y_test == 0
    y_test[zero_mask] = -1
    return X_train, X_test, y_train, y_test


def reg_para_tuning(param_set, X_train, X_test, y_train, y_test):
    loss_set = []
    for l2para in param_set:
        opt_result = fit_logistic_reg(X_train, y_train, f_objective, l2para)
        loss = evaluate_no_reg(opt_result.x, X_test, y_test)
        loss_set.append(loss)
    param_loss_pair = list(zip(loss_set, param_set))
    return min(param_loss_pair)


def calibration_analysis(theta, X, y):
    pass


def sgd_opt(theta_init, X, y, tolerance):
    pass


if __name__ == '__main__':
    path = ['../data/hw5/X_train.txt', '../data/hw5/X_val.txt',
            '../data/hw5/y_train.txt', '../data/hw5/y_val.txt']
    load_func = partial(np.loadtxt, delimiter=',')
    X_train, X_val, y_train, y_val = map(load_func, path)
    X_train, X_val, y_train, y_val = data_preprocess(X_train, X_val, y_train, y_val)
    param_set = [2 ** i for i in range(-8, 2)]
    loss, l2para = reg_para_tuning(param_set, X_train, X_val, y_train, y_val)
