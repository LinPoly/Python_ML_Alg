from collections import Counter
from random import shuffle
import pickle
import numpy as np
from util import dotProduct, increment


class SVM:
    def __init__(self, l2reg, w=None):
        self.l2reg = l2reg
        if w is not None:
            self.w = w

    def predict(self, x):
        if hasattr(self, 'w') is False:
            raise RuntimeError('Initialize or train the model before prediction')
        else:
            score = dotProduct(self.w, x)
            if score > 0:
                return 1
            else:
                return -1

    def eval_margin(self, x, y):
        if hasattr(self, 'w') is False:
            raise RuntimeError('Initialize or train the model before prediction')
        margin = y*dotProduct(self.w, x)
        return margin

    def fit(self, X, y, version, epoch=1000, tol=1e-8):
        assert version in ('v1', 'v2')
        if version == 'v1':
            w, t = Pegasos_opt_v1(self, X, y, epoch=epoch, tol=tol)
        else:
            w, t = Pegasos_opt_v2(self, X, y, epoch=epoch, tol=tol)
        return w, t

    def evaluate(self, X, y, factor=1):
        if hasattr(self, 'w') is False:
            raise RuntimeError('Initial or train the model before prediction')
        num_ins = len(X)
        loss_no_reg = 0
        for x, l in zip(X, y):
            loss_no_reg += max(0, 1/factor - dotProduct(self.w, x)*l)
        loss = factor*loss_no_reg/num_ins + factor**2*self.l2reg*sum([i**2 for i in self.w.values()])/2
        return loss


def gene_word_bags(wordlist):
    return dict(Counter(wordlist))


def data_process(X):
    y = []
    X_prcd = []
    for x in X:
        X_prcd.append(gene_word_bags(x[:-1]))
        y.append(x[-1])
    return X_prcd[:1500], y[:1500], X_prcd[1500:], y[1500:]


def error_percent(svm, X, y):
    num_ins = len(X)
    num_err = 0
    for x, label in zip(X, y):
        score = svm.predict(x)
        if label*score < 0:
            num_err += 1
    return num_err/num_ins


def Pegasos_opt_v1(svm, X, y, epoch=1000, tol=1e-8):
    assert isinstance(epoch, int) and epoch > 0
    if hasattr(svm, 'w') is False:
        svm.w = {}
    num_ins = len(X)
    order = list(range(num_ins))
    loss_init = svm.evaluate(X, y)
    t = 1

    for i in range(epoch):
        shuffle(order)
        # this is a slower implementation of Pegasos
        # because of the frequent update of dict
        for idx in order:
            x, l = X[idx], y[idx]
            increment(svm.w, -1/t, svm.w)
            margin = svm.eval_margin(x, l)
            if margin < 1:
                increment(svm.w, l/(t*svm.l2reg), x)
            t += 1
        loss_upd = svm.evaluate(X, y)
        if abs(loss_upd-loss_init) < tol:
            break
        else:
            loss_init = loss_upd

    return svm.w, t


def Pegasos_opt_v2(svm, X, y, epoch=1000, tol=1e-8):
    assert isinstance(epoch, int) and epoch > 0
    if hasattr(svm, 'w') is False:
        svm.w = {}
    num_ins = len(X)
    order = list(range(num_ins))
    factor, t = 1, 2
    loss_init = svm.evaluate(X, y)

    for i in range(epoch):
        shuffle(order)
        for idx in order:
            x, l = X[idx], y[idx]

            # If margin >= 1, w is simply multiplied
            # by 1-1/t, so just update the factor until
            # the margin < 1. Update w this moment by
            # multiplying it by the factor and plus a scaled x.

            margin = svm.eval_margin(x, l)
            if factor*margin < 1:
                factor *= 1 - 1 / t
                increment(svm.w, l / (t * svm.l2reg * factor), x)
            else:
                factor *= 1 - 1/t
            t += 1
        loss_upd = svm.evaluate(X, y, factor)
        if abs(loss_upd - loss_init) < tol:
            break
        else:
            loss_init = loss_upd

    for k in svm.w.keys():
        svm.w[k] *= factor

    return svm.w, t


def hyper_tuning(l2set, X_train, y_train, X_val, y_val):
    para_set = []
    err_set = []
    svm = SVM(0)
    for l2reg in l2set:
        svm.l2reg = l2reg
        svm.fit(X_train, y_train, 'v2')
        err = error_percent(svm, X_val, y_val)
        err_set.append(err)
        para_set.append(svm.w)
    idx = err_set.index(min(err_set))
    return idx, err_set, para_set[idx], l2set[idx]


def confidence_analysis(svm, X, y, chunk_num):
    margin_pred_pairs = [(svm.eval_margin(x, l), svm.predict(x)*l) for x, l in zip(X, y)]
    margin_pred_pairs.sort()
    num_ins = len(y)
    chunk_size = int(num_ins/chunk_num)
    error_collections = []
    for i in range(chunk_num - 1):
        error_rate = [pair[1] for pair in margin_pred_pairs[chunk_size*i:chunk_size*(i+1)]].count(-1)/chunk_size
        error_collections.append(error_rate)
    error_rate = [pair[1] for pair in
                  margin_pred_pairs[chunk_size*(chunk_num-1):]].count(-1)/(num_ins-(chunk_num-1)*chunk_size)
    error_collections.append(error_rate)
    return [pair[0] for pair in margin_pred_pairs], error_collections


def analysis_misclassification(svm, X, y, idx):
    # words like 'and', 'the', 'i', 'see' which should be neural contribute most to
    # misclassification. Margin is expected to be positive but weight*times*label is
    # negative for these words.
    triplets = [(svm.eval_margin(x, l), x, l) for x, l in zip(X, y)]
    triplets.sort()
    label = triplets[idx][2]
    ins = triplets[idx][1]
    margin = triplets[idx][0]
    word_weights = [(k, svm.w.get(k, 0)*label) for k, v in ins.items()]
    word_weights.sort()
    return word_weights, margin


if __name__ == '__main__':
    filepath = '../data/hw3/review.pickle'
    with open(filepath, 'rb') as f:
        X = pickle.load(f)
    X_train, y_train, X_val, y_val = data_process(X)

    # svm = SVM(1e-3)
    # w, t = svm.fit(X_train, y_train, 'v2')
    # error_rate = error_percent(svm, X_val, y_val)

    # l2set = [10**i for i in range(-6, 2)]
    # idx, errs, _, _ = hyper_tuning(l2set, X_train, y_train, X_val, y_val)

    # l2set = 0.09*(2**np.arange(11)-1)/1023+0.01  # The orders of magnitude should be 1e-2.
    # idx, errs, _, _ =  hyper_tuning(l2set, X_train, y_train, X_val, y_val)

    svm = SVM(0.05)
    svm.fit(X_train, y_train, 'v2')
    margins, error_percent = confidence_analysis(svm, X_val, y_val, 10)
    # all the mis-classified instances is in the first two chunks
