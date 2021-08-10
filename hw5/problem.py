import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import support_code


def likelihood_func(w, X, y, likelihood_var):
    '''
    Implement likelihood_func. This function returns the data likelihood
    given f(y | X; w) ~ Normal(y; Xw, likelihood_var).

    Args:
        w: Weights
        X: Training design matrix with first col all ones
        y: Training response vector
        likelihood_var: likelihood variance

    Returns:
        likelihood: Data likelihood (float)
    '''

    likelihood = multivariate_normal.pdf(y.flatten(), (X@w).flatten(), likelihood_var)
    return likelihood


def get_posterior_params(X, y, prior: dict, likelihood_var=0.2 ** 2):
    """
    Implement get_posterior_params. This function returns the posterior
    mean vector \mu_p and posterior covariance matrix \Sigma_p for
    Bayesian regression (normal likelihood and prior).

    Note support_code.make_plots takes this completed function as an argument.

    Args:
        X: Training design matrix with first col all ones
        y: Training response vector
        prior: Prior parameters; dict with 'mean' (prior mean)
               and 'var' (prior covariance)
        likelihood_var: likelihood variance- default (0.2**2) per the lecture slides

    Returns:
        post_mean: Posterior mean
        post_var: Posterior var
    """

    prior_mean, prior_precision = prior['mean'], np.linalg.inv(prior['var'])
    trans_product = X.T @ X
    post_mean = np.linalg.inv(trans_product + likelihood_var * prior_precision) @ X.T @ y
    post_var = np.linalg.inv(trans_product/likelihood_var + prior_precision)
    return post_mean, post_var


def get_predictive_params(x_new, post_mean, post_var, likelihood_var=0.2 ** 2):
    """
    Implement get_predictive_params. This function returns the predictive
    distribution parameters (mean and variance) given the posterior mean
    and covariance matrix (returned from get_posterior_params) and the
    likelihood variance (default value from lecture).

    Args:
        x_new: New observation
        post_mean, post_var: Returned from get_posterior_params
        likelihood_var: likelihood variance (0.2**2) per the lecture slides

    Returns:
        - pred_mean: Mean of predictive distribution
        - pred_var: Variance of predictive distribution
    """

    pred_mean = np.dot(post_mean.T, x_new)
    pred_var = x_new.T @ post_var @ x_new + likelihood_var
    return pred_mean, pred_var


if __name__ == '__main__':

    '''
    If your implementations are correct, running
        python problem.py
    inside the Bayesian Regression directory will, for each sigma in sigmas_to-test generates plots
    '''

    np.random.seed(46134)
    actual_weights = np.asarray([0.3, 0.5])
    data_size = 40
    noise = {"mean": 0, "var": 0.2 ** 2}
    likelihood_var = noise["var"]
    x_train, y_train = support_code.generate_data(data_size, noise, actual_weights)

    # Question (b)
    sigmas_to_test = [1/2, 1/(2**5), 1/(2**10)]
    for sigma_squared in sigmas_to_test:
        prior = {"mean": np.asarray([0, 0]),
                 "var": np.identity(2) * sigma_squared}

        support_code.make_plots(actual_weights,
                                x_train,
                                y_train,
                                likelihood_var,
                                prior,
                                likelihood_func,
                                get_posterior_params,
                                get_predictive_params)
