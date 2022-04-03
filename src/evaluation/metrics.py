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
        raise Exception(
            "number of Labels is not 4 (mandatory) or 11 (optional)")

    conf = calc_confusion_matrix(h, y)
    plot_confusion_matrix(conf, ax=ax, title=title,
                          Labels_Mandatory_Optional=Labels)





def plot_confusion_matrix(confusion_matrix: np.array, Labels_Mandatory_Optional, fig: plt.figure = None, 
                          ax: plt.axes = None, title: str = None, verbose: bool = False) -> plt.figure:
    # fig situation hier is bisschen weird nicht wundern

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    Labels = Labels_Mandatory_Optional
    confusion_df = pd.DataFrame(
        data=confusion_matrix, columns=Labels.get_label_list(), index=Labels.get_label_list())

    sns.heatmap(confusion_df, annot=confusion_matrix, fmt="", ax=ax)
    ax.set_xlabel("ground truth")
    ax.set_ylabel("predicted")
    if not title:
        ax.set_title("confusion matrix")
    else:
        ax.set_title(title)

    if verbose:
        fig.show()





def calc_metrics(h: np.array, y: np.array):

    if y.shape[1] == 4:
        Labels = LabelsMandatory
    elif y.shape[1] == 11:
        Labels = LabelsOptional
    else:
        raise Exception(
            "number of Labels is not 4 (mandatory) or 11 (optional)")

    # accuracy
    conf_matrix = calc_confusion_matrix(h, y)
    plot_confusion_matrix(conf_matrix, verbose=False,
                          Labels_Mandatory_Optional=Labels)
    f1_scores = []
    precisions = []
    recalls = []
    for label in Labels:
        f1_scores.append(f1_score(conf_matrix, label.value))
        precisions.append(precision(conf_matrix, label.value))
        recalls.append(recall(conf_matrix, label.value))

    df = pd.DataFrame(columns=Labels.get_label_list(), index=[
                      'f1_score', 'precision', 'recall'])
    df.loc['f1_score', :] = f1_scores
    df.loc['precision', :] = precisions
    df.loc['recall', :] = recalls

    print(df)
