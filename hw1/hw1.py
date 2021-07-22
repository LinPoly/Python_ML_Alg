import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


### Assignment Owner: Pengyun Lin
### Date: 2021.5.2


#######################################
### Feature normalization
def feature_normalization(train: np.ndarray, test: np.ndarray) -> (np.ndarray, np.ndarray):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test - test set, a 2D numpy array of size (num_instances, num_features)

    Returns:
        train_normalized - training set after normalization
        test_normalized - test set after normalization
    """
    # TODO
    max_list = np.max(train, axis=0)
    min_list = np.min(train, axis=0)
    con_checker = (max_list!=min_list)
    new_train = train[:, con_checker]
    new_test = test[:, con_checker]
    new_min = min_list[con_checker]
    new_train -= new_min
    new_test -= new_min
    new_max = max_list[con_checker] - new_min
    normalized_train = new_train/new_max
    normalized_test = new_test/new_max
    return normalized_train, normalized_test

#######################################
### The square loss function
def compute_square_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """
    Given a set of X, y, theta, compute the average square loss for predicting y with X*theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the average square loss, scalar
    """
    # TODO
    m = X.shape[0]
    assert m != 0
    err = X@theta - y
    loss = np.einsum('i, i', err, err) / m
    return loss


#######################################
### The gradient of the square loss function
def compute_square_loss_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the average square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    # TODO
    m = X.shape[0]
    grad = 2*X.T@(X@theta - y)/m
    return grad


#######################################
### Gradient checker
# Getting the gradient calculation correct is often the trickiest part
# of any gradient-based optimization algorithm. Fortunately, it's very
# easy to check that the gradient calculation is correct using the
# definition of gradient.
# See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X: np.ndarray, y: np.ndarray, theta: np.ndarray, epsilon=0.01, tolerance=1e-4) -> bool:
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1))

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicating whether the gradient is correct or not
    """
    true_grad= compute_square_loss_gradient(X, y, theta) #The true gradient
    num_features = theta.shape[0]
    # approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    # TODO
    num_instances = X.shape[0]
    theta_incr = theta + epsilon*np.eye(num_features)
    theta_decr = theta - epsilon*np.eye(num_features)
    err_incr = theta_incr @ X.T - y
    err_decr = theta_decr @ X.T - y
    # following Js are not divided by num_instances
    # but approx_grad is. this can reduce computation times
    J_incr = np.einsum('ij,ij->i', err_incr, err_incr)
    J_decr = np.einsum('ij,ij->i', err_decr, err_decr)
    approx_grad = (J_incr - J_decr) / (2*num_instances*epsilon)
    diff = approx_grad - true_grad
    error = np.einsum('i, i', diff, diff)
    if error > tolerance:
        return False
    else:
        return True


#######################################
### Generic gradient checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. 
    And check whether gradient_func(X, y, theta) returned the true 
    gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    # TODO
    # use numba's guvectorize() to vectorize objective_func


#######################################
### Batch gradient descent
def batch_grad_descent(X, y, alpha=0.1, num_step=1000, grad_check=False):
    """
    In this question you will implement batch gradient descent to
    minimize the average square loss objective.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step (num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array, (num_step+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step+1, num_features)) # Initialize theta_hist
    loss_hist = np.zeros(num_step+1) # Initialize loss_hist
    theta = np.zeros(num_features) # Initialize theta
    # TODO
    loss_hist[0] = compute_square_loss(X, y, theta)

    for i in range(num_step):
        grad = compute_square_loss_gradient(X, y, theta)
        # of_pos = np.argwhere(grad==np.inf)
        # if of_pos.shape[0] != 0:
        #     print(f'Gradient diverges in {i+1}th iteration, step size is {alpha}')
        #     theta_hist = theta_hist[: i + 1]
        #     loss_hist = loss_hist[: i + 1]
        #     break
        temp_theta = theta - alpha*grad
        mask = np.isinf(temp_theta)
        of_pos = np.argwhere(mask)
        if of_pos.shape[0] != 0:
            print(f'Parameters diverge in {i+1}th iteration, step size is {alpha}')
            theta_hist = theta_hist[: i + 1]
            loss_hist = loss_hist[: i + 1]
            break
        theta = temp_theta
        theta_hist[i+1] = theta
        loss = compute_square_loss(X, y, theta)
        loss_hist[i+1] = loss

    return theta_hist, loss_hist


#######################################
### Backtracking line search
# Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
# TODO
def backtracking_line_search(X: np.ndarray, y: np.ndarray, theta, loss_func, grad_func,
                             num_backtracking, c, tau, ori_step_size) -> (np.ndarray, bool):
    step_size = ori_step_size
    grad = grad_func(X, y, theta)
    D = c*np.einsum('i, i', grad, grad)
    flag = True

    for i in range(num_backtracking):
        iter_theta = theta - step_size*grad
        ori_loss = loss_func(X, y, theta)
        loss = loss_func(X, y, iter_theta)
        diff = ori_loss - loss
        exp_diff = step_size*D
        if diff > exp_diff:
            return iter_theta, flag
        else:
            step_size *= tau

    flag = False
    return theta, flag

#######################################
### The gradient of regularized batch gradient descent
def compute_regularized_square_loss_gradient(X: np.ndarray, y: np.ndarray,
                                             theta, lambda_reg) -> (np.ndarray, bool):
    """
    Compute the gradient of L2-regularized average square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    # TODO
    num_instances = X.shape[0]
    of_flag = False
    grad = 2*(X.T@(X@theta - y)/num_instances + lambda_reg*theta)
    pos = np.argwhere(grad == np.inf)
    if pos.shape[0] != 0:
        of_flag = True
    return grad, of_flag


#######################################
### Regularized batch gradient descent
def regularized_grad_descent(X, y, alpha=0.05, lambda_reg=10**-2, num_step=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        num_step - number of steps to run
    
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step (num_step+1) is theta_hist[-1]
        loss hist - the history of average square loss function without the regularization term, 1D numpy array.
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    # TODO
    loss_hist[0] = compute_square_loss(X, y, theta)

    for i in range(num_step):
        loss = compute_square_loss(X, y, theta)
        loss_hist[i + 1] = loss
        grad, of_flag = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        if of_flag:
            print(f'Overflow occurs in {i}th iteration, step size is {alpha}')
            break
        theta -= alpha * grad
        theta_hist[i + 1] = theta

    return theta_hist, loss_hist


#######################################
### Stochastic gradient descent
def stochastic_grad_descent(X, y, mode, ori_stepsize=0.01, lambda_reg=10 ** -2,
                            num_epoch=1000) -> (np.ndarray, np.ndarray):
    """
    In this question you will implement stochastic gradient descent with regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float, step size in gradient descent
                NOTE: In SGD, it's not a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every step is the float.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t).
                if alpha == "1/t", alpha = 1/t.
        lambda_reg - the regularization coefficient
        num_epoch - number of epochs to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_epoch, num_instances, num_features)
                     for instance, theta in epoch 0 should be theta_hist[0], theta in epoch (num_epoch) is theta_hist[-1]
        loss hist - the history of loss function vector, 2D numpy array of size (num_epoch, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) # Initialize theta

    theta_hist = np.zeros((num_epoch, num_instances, num_features)) # Initialize theta_hist
    loss_hist = np.zeros((num_epoch, num_instances)) # Initialize loss_hist
    # TODO
    assert (mode in ['1/sqrt(t)', '1/t'] or isinstance(mode, float))
    order = np.arange(num_instances)
    times = np.arange(num_instances) + 1
    if mode == '1/sqrt(t)':
        steps = ori_stepsize / np.sqrt(times)  # try different constant besides 1.
    elif mode == '1/t':
        steps = ori_stepsize / times
    else:
        steps = mode * np.ones(num_instances)
    # TODO: use numba to accelerate this double-layer loop

    for epoch in range(num_epoch):
        np.random.shuffle(order)
        # TODO: maybe add divergence checking functionality

        for idx in range(num_instances):
            grad = 2*((np.dot(X[idx], theta) - y[idx])*X[idx] + lambda_reg*theta)
            theta -= steps[idx]*grad
            err = X @ theta - y
            loss = np.einsum('i, i', err, err) / num_instances
            theta_hist[epoch, idx] = theta
            loss_hist[epoch, idx] = loss

    return theta_hist, loss_hist


def main():
    # Loading the dataset
    # print('loading the dataset')

    df = pd.read_csv('../data/hw1/hw1_data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    # print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    # print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term
    # TODO
    theta_hist, loss_hist = stochastic_grad_descent(X_train, y_train, '1/sqrt(t)', 0.05, num_epoch=100)
    test_loss = compute_square_loss(X_test, y_test, theta_hist[-1, -1])
    return theta_hist, loss_hist, test_loss


if __name__ == "__main__":
    # alphas = np.asarray([0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5])
    # # the last three parameters will lead to divergence of BGD.
    # losses = np.zeros([7], dtype=np.float64)
    # for i in range(7):
    #     _, __, test_loss = main(alphas[i])
    #     losses[i] = test_loss
    # plt.plot(alphas, losses, marker='o')
    # plt.savefig('./step_size.png')
    # plt.show()
    theta_hist, loss_hist, test_loss = main()