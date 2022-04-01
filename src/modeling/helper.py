import numpy as np 

def softmax2one_hot(h: np.array):
    h_onehot = np.zeros_like(h)
    h_onehot[np.arange(len(h)), h.argmax(axis=1)] = 1
    return h_onehot

def one_hot_encoding(y: np.array):
    y_g = np.zeros((len(y), 10))
    for i in range(len(y)):
        j = y[i]
        y_g[i, j] = 1
    return y_g
