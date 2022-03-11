import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.preprocessing_functions import Labels


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

def calc_metrics(h: np.array, y: np.array):

    # accuracy

    conf_matrix = calc_confusion_matrix(h, y)
    print_confusion_matrix(conf_matrix)
    f1_scores = []
    precisions = []
    recalls = []
    for label in Labels:
        f1_scores.append(f1_score(conf_matrix, label.value))
        precisions.append(precision(conf_matrix, label.value))
        recalls.append(recall(conf_matrix, label.value))

    df = pd.DataFrame(columns=Labels.get_label_list(), index=['f1_score', 'precision', 'recall'])
    df.loc['f1_score', :] = f1_scores
    df.loc['precision', :] = precisions
    df.loc['recall', :] = recalls

    print(df)


