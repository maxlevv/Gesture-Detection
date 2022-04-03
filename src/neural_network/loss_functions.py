import numpy as np
from activation_functions import softmax


def cross_entropy(h: np.array, y: np.array):
    h = np.clip(h, 0.000000001, 0.99999999)
    return np.mean(- y * np.log(h) - (1 - y) * np.log(1 - h))


def d_cross_entropy(h: np.array, y: np.array):
    h = np.clip(h, 0.000000001, 0.99999999)
    return - y / h + (1-y) / (1-h)


def categorical_cross_entropy(h: np.array, y_one_hot: np.array):
    h = np.clip(h, a_min=0.000000001, a_max=None)

    return - (y_one_hot * np.log(h)).sum(axis=1).mean(axis=0)


def d_categorical_cross_entropy_with_softmax(z: np.array, y_one_hot: np.array):
    # z (n x b); y_one_hot (b x n)
    # res (b x n)
    return softmax(z).T - y_one_hot


def mse(h: np.array, y: np.array):
    return np.mean(np.power(h - y, 2), axis=0)


def d_mse(h: np.array, y: np.array):
    return np.mean(2 * (h - y), axis=0)
