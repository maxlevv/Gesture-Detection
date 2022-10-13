import sys
sys.path.append('neural_net_pack')
sys.path.append('../../neural_net_pack')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.preprocessing_functions import LabelsMandatory, LabelsOptional
from helper import softmax2one_hot
from nn_metrics import calc_confusion_matrix, f1_score, precision, recall



def accuracy(h: np.array, y: np.array):
    return (softmax2one_hot(h) == y).all(axis=1).sum() / y.shape[0]


def noramlize_confusion_matrix(conf_matrix: np.array):
    return np.round((conf_matrix / conf_matrix.sum(axis=0) )* 100, 2) 


def generate_confusion_plot(h: np.array, y: np.array, ax: plt.axes = None, title: str = None):
    if y.shape[1] == 4:
        Labels = LabelsMandatory
    elif y.shape[1] == 11:
        Labels = LabelsOptional
    else:
        raise Exception(
            "number of Labels is not 4 (mandatory) or 11 (optional)")

    conf = calc_confusion_matrix(h, y)
    conf = noramlize_confusion_matrix(conf)
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

    if len(Labels_Mandatory_Optional) == 11: # large net
        sns.set(font_scale=1.3)
        sns.heatmap(confusion_df, annot=confusion_matrix, fmt="", ax=ax)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 18, rotation=25)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 18)
        ax.set_xlabel("ground truth", fontsize=30, labelpad=15)
        ax.collections[0].colorbar.ax.tick_params(labelsize=23)
        
    else:
        sns.set(font_scale=2.)
        sns.heatmap(confusion_df, annot=confusion_matrix, fmt="", ax=ax)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 25)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 25)
        ax.set_xlabel("ground truth", fontsize=30, labelpad=30)

    
    ax.set_ylabel("predicted", fontsize=30, labelpad=30)

    if not title:
        ax.set_title("confusion matrix")
    else:
        ax.set_title(title, fontsize=40, pad=35)
        
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
    plot_confusion_matrix(conf_matrix, verbose=True,
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
