import numpy as np

def softmax(z):
        # z (n x d)
        e = np.exp(z) # (n x d)
        return e / e.sum(axis=0).reshape(1, -1)  # axis verändert von 1 auf 0

sigmoid = lambda z: 1 / (1 + np.e**-z)
sigmoid_d = lambda z: 1 / (1 + np.e**-z) * (1 - 1 / (1 + np.e**-z))