import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def accuracy(h: np.array, y: np.array):
    return (np.round(h) == y).all(axis=1).sum() / y.shape[0]


def calc_confusion_matrix(h: np.array, y: np.array):
    if np.shape(h) != np.shape(y):
        raise Exception("ground truth vector y and hypothesis h are not the same size")

    n = y.shape[1]
    matrix = np.zeros(shape=(n, n), dtype='int')

    # for each row set the maximal value to one and the rest to zero
    # instead of h.round()
    h_onehot = np.zeros_like(h)
    h_onehot[np.arange(len(h)), h.argmax(axis=1)] = 1

    for i in range(n):
        for j in range(n):
            counter = ((h_onehot[:,i] == 1) & (y[:,j]) == 1).sum()
            matrix[i,j] = counter

    return matrix


def print_confusion_matrix(confusion_matrix: np.array):

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(confusion_matrix, annot=confusion_matrix , fmt="", ax=ax)
    ax.set_xlabel("ground truth")
    ax.set_ylabel("predicted")
    ax.set_title("confusion matrix")
    fig.show()

# attribute Auswahl als Integer entsprechend der Nummerierung der Label in der Klasse preprocessing_functions.Labels
def precision(confusion_matrix: np.array, attribute: int):
    return confusion_matrix[attribute, attribute] / confusion_matrix[attribute,:].sum()


def recall(confusion_matrix: np.array, attribute: int):
    return confusion_matrix[attribute, attribute] / confusion_matrix[:,attribute].sum()


def f1_score(confusion_matrix: np.array, attribute: int):
    prec = precision(confusion_matrix, attribute)
    rec = recall(confusion_matrix, attribute)
    score = 2 * (prec * rec) / (prec + rec)
    return score

