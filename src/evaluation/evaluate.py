from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import TYPE_CHECKING, List
from evaluation.metrics import generate_confusion_plot
from preprocessing.preprocessing_functions import Labels
if TYPE_CHECKING:
    from modeling.neural_network import FCNN


def generate_acc_plot(neural_net: FCNN, ax: plt.axes):
    ax.plot(neural_net.acc_hist, label='train_accuracy')
    ax.plot(neural_net.val_acc_hist, label='val_accuracy')
    ax.set_xlabel('iterations')
    ax.set_ylabel('acc')
    ax.set_title('accuracy')
    ax.legend(loc='lower right')

def generate_loss_plot(neural_net: FCNN, ax: plt.axes):
    ax.plot(neural_net.loss_hist, label='loss')
    ax.set_xlabel('iterations')
    ax.set_ylabel('loss')
    ax.set_title('loss')
    ax.legend(loc='upper right')

def generate_f1_score_plot(neural_net: FCNN, ax: plt.axes, mode: str):
    # num_classes = neural_net.layer_list[-1]  wäre auch möglich

    if mode == 'train':
        f1_np = np.array(neural_net.f1_score_hist)
        for label in Labels:
            ax.plot(f1_np[:, label.value], label=str(label.name))
        ax.set_title('train_f1_score')
    elif mode == 'val':
        f1_np = np.array(neural_net.f1_score_val_hist)
        for label in Labels:
            ax.plot(f1_np[:, label.value], label=str(label.name))
        ax.set_title('val_f1_score')
    
    ax.set_xlabel('iteration')
    ax.set_ylabel('f1_score')
    ax.legend(loc='lower right')
    



def evaluate_neural_net(neural_net: FCNN, X_train, y_train, X_val, y_val, save_plot_path:Path = None):
    fig, axs = generate_evaluation_plot(neural_net, X_train, y_train, X_val, y_val, save_plot_path)


def generate_evaluation_plot(neural_net: FCNN, X_train, y_train, X_val, y_val, save_plot_path:Path = None):
    neural_net.clear_data_specific_parameters()
    neural_net.forward_prop(X_train)
    h_train = neural_net.O[-1].T
    neural_net.clear_data_specific_parameters()
    neural_net.forward_prop(X_val)
    h_val = neural_net.O[-1].T

    fig, axs = plt.subplots(2, 4, figsize=(60, 40))  # TODO: figsize doesnt change anything if changed, fig is humongous
    generate_confusion_plot(h_train, y_train, ax=axs[0, 0], title='confusion_matrix_train')
    generate_confusion_plot(h_val, y_val, ax=axs[1, 0], title='confusion_matrix_val')

    generate_loss_plot(neural_net, ax=axs[1, 1])

    generate_f1_score_plot(neural_net, ax=axs[0, 2], mode='train')
    generate_f1_score_plot(neural_net, ax=axs[1, 2], mode='val')

    generate_acc_plot(neural_net, ax=axs[0, 1])

    if not save_plot_path == None:
        fig.savefig(save_plot_path)
    else:
        fig.show()

    return fig, axs



