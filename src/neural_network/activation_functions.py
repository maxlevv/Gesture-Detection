import numpy as np


def softmax(z: np.array):
    # z (n x d)
    m = z.max(axis=0)
    trans_z = z - m[np.newaxis, :]
    e = np.exp(trans_z)  # (n x d)
    return e / e.sum(axis=0).reshape(1, -1)


def sigmoid(z: np.array):
    res = np.zeros_like(z)
    res[z > 0] = 1 / (1 + np.e**(-z[z > 0]))
    res[z <= 0] = np.exp(z[z <= 0]) / (1 + np.exp(z[z <= 0]))
    return res


def sigmoid_d(z: np.array):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.max(np.concatenate([z[:, :, np.newaxis], np.zeros((z.shape[0], z.shape[1], 1))], axis=2), axis=2)


def relu_d(z):
    d = np.zeros_like(z)
    d[np.where(z >= 0)] = 1
    return d


def leaky_relu(z):
    res = np.zeros_like(z)
    res[np.where(z >= 0)] = z[np.where(z >= 0)]
    res[np.where(z < 0)] = 0.01 * z[np.where(z < 0)]
    return res


def leaky_relu_d(z):
    d = np.ones_like(z)
    d[np.where(z < 0)] = 0.01
    return d


def test_relu():
    X = np.array([[1, 2, 3], [0.4, 0.1, -1], [-0.1, -2, 1]])
    res = relu(X)
    d = relu_d(X)
    res2 = leaky_relu(X)
    d2 = leaky_relu_d(X)


if __name__ == '__main__':
    test_relu()
