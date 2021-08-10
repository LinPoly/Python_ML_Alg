"""I wrote this program since I suppose that the setting and structure in the provided
skeleton code are not reasonable, at least it is not proper to pre-specify num_class
and num_out_feat at the same time."""

import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt


def target_margin(y1, y2):
    pass

def feature_mapping(X, n_class):
    pass


def subgradient_descent():
    pass


class MulticlassSVM:
    pass

if __name__ == "__main__":
    centers = np.array([(-3, 1), (0, 2), (3, 1)])
    X, y = skl.datasets.make_blobs(n_samples=300, cluster_std=.25, centers=centers)
