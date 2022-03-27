import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.preprocessing_functions import LabelsMandatory, LabelsOptional
from modeling.helper import softmax2one_hot


def accuracy(h: np.array, y: np.array):
    return (softmax2one_hot(h) == y).all(axis=1).sum() / y.shape[0]


def generate_confusion_plot(h: np.array, y: np.array, ax: plt.axes = None, title: str = None):
    if y.shape[1] == 4:
        Labels = LabelsMandatory
    elif y.shape[1] == 11:
        Labels = LabelsOptional
    else:
        raise Exception("number of Labels is not 4 (mandatory) or 11 (optional)")

    conf = calc_confusion_matrix(h, y)
    plot_confusion_matrix(conf, ax=ax, title=title, Labels_Mandatory_Optional=Labels)
    


def calc_confusion_matrix(h: np.array, y: np.array):
    if np.shape(h) != np.shape(y):
        raise Exception("ground truth vector y and hypothesis h are not the same size")

    n = y.shape[1]
    matrix = np.zeros(shape=(n, n), dtype='int')

    # for each row set the maximal value to one and the rest to zero
    # instead of h.round()
    h_onehot = softmax2one_hot(h)

    for i in range(n):
        for j in range(n):
            counter = ((h_onehot[:,i] == 1) & ((y[:,j]) == 1)).sum()
            matrix[i,j] = counter

    return matrix


def plot_confusion_matrix(confusion_matrix: np.array, Labels_Mandatory_Optional , fig: plt.figure = None, ax: plt.axes = None, title: str = None, verbose: bool = False) -> plt.figure:
    # fig situation hier is bisschen weird nicht wundern 

    if ax is None: 
        fig, ax = plt.subplots(figsize=(10, 8))

    Labels = Labels_Mandatory_Optional
    confusion_df = pd.DataFrame(data=confusion_matrix, columns=Labels.get_label_list(), index=Labels.get_label_list())

    sns.heatmap(confusion_df, annot=confusion_matrix , fmt="", ax=ax)
    ax.set_xlabel("ground truth")
    ax.set_ylabel("predicted")
    if not title:
        ax.set_title("confusion matrix")
    else:
        ax.set_title(title)

    if verbose: fig.show()

    

# attribute Auswahl als Integer entsprechend der Nummerierung der Label in der Klasse preprocessing_functions.Labels
def precision(confusion_matrix: np.array, attribute: int):
    if confusion_matrix[attribute,:].sum() == 0:
        return 0
    return confusion_matrix[attribute, attribute] / confusion_matrix[attribute,:].sum()


def recall(confusion_matrix: np.array, attribute: int):
    if confusion_matrix[:,attribute].sum() == 0:
        return 0
    return confusion_matrix[attribute, attribute] / confusion_matrix[:,attribute].sum()


def f1_score(confusion_matrix: np.array, attribute: int):
    prec = precision(confusion_matrix, attribute)
    rec = recall(confusion_matrix, attribute)
    if (prec == 0) or (rec == 0):
        return 0
    score = 2 * (prec * rec) / (prec + rec)
    return score


def calc_metrics(h: np.array, y: np.array):

    if y.shape[1] == 4:
        Labels = LabelsMandatory
    elif y.shape[1] == 11:
        Labels = LabelsOptional
    else:
        raise Exception("number of Labels is not 4 (mandatory) or 11 (optional)")

    # accuracy
    conf_matrix = calc_confusion_matrix(h, y)
    plot_confusion_matrix(conf_matrix, verbose=False, Labels_Mandatory_Optional=Labels)
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


