import numpy as np
np.seterr(all="ignore")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy(h, y):
    return (np.round(h) == y).all(axis=1).sum() / y.shape[0]


def calc_confusion_matrix(h, y):
    n = y.shape[1]
    matrix = np.zeros(shape=(n, n), dtype='int')
    for i in range(n):
        for j in range(n):
            counter = ((h.round()[:,i] == 1) & (y[:,j]) == 1).sum()
            matrix[i,j] = counter

    return matrix


def print_confusion_matrix(h, y):
    matrix = calc_confusion_matrix(h, y)   # Oder übergeben als Parameter

    fig, ax = plt.subplots(figsize=(10, 8))
    values = np.array([])
    sns.heatmap(matrix, annot=matrix , fmt="", ax=ax)
    ax.set_xlabel("ground truth")
    ax.set_ylabel("predicted")
    ax.set_title("confusion matrix")
    fig.show()


def precision(confusion_matrix, attribute: int):
    return confusion_matrix[attribute, attribute] / confusion_matrix[attribute,:].sum()


def recall(confusion_matrix, attribute: int):
    return confusion_matrix[attribute, attribute] / confusion_matrix[:,attribute].sum()


def f1_score(confusion_matrix, attribute: int):
    prec = precision(confusion_matrix, attribute)
    rec = recall(confusion_matrix, attribute)
    score = 2 * (prec * rec) / (prec + rec)
    return score


h = np.array([[0.9, 0.1, 0],
              [0, 1, 0],
              [0.7, 0.2, 0.1],
              [1, 0, 0]])

y = np.array([[1, 0, 0],
              [0, 0, 1],
              [1, 0, 0],
              [0, 1, 0]])


X = calc_confusion_matrix(h, y)
print(precision(X, 0))
print(recall(X, 0))
print(f1_score(X, 0))
print_confusion_matrix(h, y)