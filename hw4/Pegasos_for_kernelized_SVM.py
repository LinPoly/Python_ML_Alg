from functools import partial
import numpy as np
import scipy.spatial as sp
import sklearn as skl

#  Simply take a look of the data distribution by running the code in ipynb for hw4.
#  We can see that negative labeled data cluster in the center while positive data are around.


def load_svm_data(training_path: str, test_path: str):
    training_data, test_data = np.loadtxt(training_path), np.loadtxt(test_path)
    x_train, y_train = training_data[:, 0:2], training_data[:, 2]
    x_test, y_test = test_data[:, 0:2], test_data[:, 2]
    for y in (y_train, y_test):
        mask = y >= 0
        y[mask] = 1
        y[~mask] = -1
    return x_train, x_test, y_train, y_test


def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)


def polynomial_kernel(X1, X2, degree, offset):
    inpro_mat = np.dot(X1, X2.T)
    poly_mat = (inpro_mat + offset) ** degree
    return poly_mat


def RBF_kernel(X1, X2, sigma):
    sqdist_mat = sp.distance.cdist(X1, X2, 'sqeuclidean')
    rbf_mat = np.exp(-sqdist_mat / (2 * sigma**2))
    return rbf_mat


def kernelized_pegasos(svm, max_epoch=1000, tolerance=1e-6):
    order = np.arange(svm.num_proto)
    loss_init = svm.evaluate_reg_loss()
    t = 1

    for i in range(max_epoch):
        np.random.shuffle(order)
        for j in order:
            factor = svm.l2reg*t
            if svm.y[j]*np.dot(svm.alpha, svm.gram_mat[j]) / factor < 1:
                svm.alpha[j] += svm.y[j]
            loss_update = svm.evaluate_reg_loss(factor)

            if np.abs(loss_update-loss_init) < tolerance:
                return svm.alpha / factor
            else:
                loss_init = loss_update
            t = t+1

    return svm.alpha / (factor-svm.l2reg)


class Kernelized_SVM:
    def __init__(self, l2reg, kernel, alpha=None, X=None):
        self.l2reg = l2reg
        self.kernel = kernel

        if X is not None:
            self.prototypes = X
            self.num_proto = len(self.prototypes)
            self.alpha = alpha

    def get_score(self, X):
        if (not hasattr(self, 'prototypes')) or (not hasattr(self, 'alpha')):
            raise RuntimeError('Train or initialize the model primarily.')
        kernel_mat = self.kernel(X, self.prototypes)
        return np.dot(kernel_mat, self.alpha)

    def predict(self, X):
        score = self.get_score(X)
        prediction = np.ones(score.shape)
        prediction[score < 0] = -1
        return prediction

    def error_rate(self, X, y):
        y_pre = self.predict(X)
        result = y_pre * y
        return 1 - np.sum(result > 0)/len(X)

    def evaluate_reg_loss(self, factor=1):
        #  The parameter factor is used for implementing the faster version of Pegasos.
        if (not hasattr(self, 'prototypes')) or (not hasattr(self, 'y')):
            raise RuntimeError('Train or initialize the model before evaluation.')

        if not hasattr(self, 'gram_mat'):
            self.gram_mat = self.kernel(self.prototypes, self.prototypes)

        hinge_vector = np.maximum(0, factor - self.alpha@self.gram_mat*self.y)
        reg_loss = self.l2reg * self.alpha@self.gram_mat@self.alpha / (2*factor**2) + \
                   np.sum(hinge_vector) / (self.num_proto*factor)
        return reg_loss

    def evaluate_loss(self, X, y):
        if not hasattr(self, 'prototypes'):
            raise RuntimeError('Train or initialize the model before evaluation.')

        kernel_mat = self.kernel(self.prototypes, X)
        hinge_vector = np.maximum(0, 1 - self.alpha@kernel_mat*y)
        loss = np.sum(hinge_vector) / len(X)
        return loss

    def fit(self, y, X=None):
        if X is not None:
            self.prototypes = X
            self.num_proto = len(self.prototypes)
            self.gram_mat = self.kernel(self.prototypes, self.prototypes)

        if not hasattr(self, 'prototypes'):
            raise RuntimeError('No prototype points for the kernel machine.')

        if not hasattr(self, 'alpha'):
            self.alpha = np.zeros(self.num_proto)

        if not hasattr(self, 'gram_mat'):
            self.gram_mat = self.kernel(self.prototypes, self.prototypes)

        self.y = y
        self.alpha = kernelized_pegasos(self)


if __name__ == '__main__':
    training_path = '../data/hw4/svm-train.txt'
    test_path = '../data/hw4/svm-test.txt'
    x_train, x_test, y_train, y_test = load_svm_data(training_path, test_path)
    kernel = partial(RBF_kernel, sigma=0.02)
    svm = Kernelized_SVM(l2reg=0.01, kernel=kernel)
    svm.fit(y_train, x_train)
    error_rate = svm.error_rate(x_test, y_test)
