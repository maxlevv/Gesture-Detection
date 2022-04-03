import numpy as np
from helper import softmax2one_hot

def precision(confusion_matrix: np.array, attribute: int):
    # attribute as int according to the class preprocessing_functions.Labels
    if confusion_matrix[attribute, :].sum() == 0:
        return 0
    return confusion_matrix[attribute, attribute] / confusion_matrix[attribute, :].sum()


def recall(confusion_matrix: np.array, attribute: int):
    if confusion_matrix[:, attribute].sum() == 0:
        return 0
    return confusion_matrix[attribute, attribute] / confusion_matrix[:, attribute].sum()


def f1_score(confusion_matrix: np.array, attribute: int):
    prec = precision(confusion_matrix, attribute)
    rec = recall(confusion_matrix, attribute)
    if (prec == 0) or (rec == 0):
        return 0
    score = 2 * (prec * rec) / (prec + rec)
    return score


def calc_confusion_matrix(h: np.array, y: np.array):
    if np.shape(h) != np.shape(y):
        raise Exception(
            "ground truth vector y and hypothesis h are not the same size")

    n = y.shape[1]
    matrix = np.zeros(shape=(n, n), dtype='int')

    # for each row set the maximal value to one and the rest to zero
    # instead of h.round()
    h_onehot = softmax2one_hot(h)

    for i in range(n):
        for j in range(n):
            counter = ((h_onehot[:, i] == 1) & ((y[:, j]) == 1)).sum()
            matrix[i, j] = counter

    return matrix