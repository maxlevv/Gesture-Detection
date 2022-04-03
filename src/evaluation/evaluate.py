from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from evaluation.metrics import generate_confusion_plot
from preprocessing.preprocessing_functions import LabelsMandatory, LabelsOptional
from modeling.grid_search import generate_dataset
from preprocessing.pca import generate_pca_dataset
import json
from math import factorial

from modeling.neural_network import FCNN


def generate_acc_plot(neural_net: FCNN, ax: plt.axes):
    ax.plot(neural_net.acc_hist, label='train_accuracy')
    ax.plot(neural_net.val_acc_hist, label='val_accuracy')
    ax.set_xlabel('epochs', fontsize=20)
    ax.set_ylabel('acc', fontsize=20)
    ax.set_title('accuracy', fontsize=20)
    ax.legend(loc='lower right', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid()

    ax.set_facecolor('lavender')


def generate_loss_plot(neural_net: FCNN, ax: plt.axes):
    ax.plot(neural_net.loss_hist, label='loss')
    ax.set_xlabel('epochs', fontsize=20)
    ax.set_ylabel('loss', fontsize=20)
    ax.set_title('loss', fontsize=20)
    ax.legend(loc='upper right', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid()

    ax.set_facecolor('lavender')


def generate_f1_score_plot(neural_net: FCNN, ax: plt.axes, mode: str, Labels_Mandatory_Optional):
    # num_classes = neural_net.layer_list[-1]  wäre auch möglich

    only_plot_larger_value = 0

    Labels = Labels_Mandatory_Optional
    if mode == 'train':
        f1_np = np.array(neural_net.f1_score_hist)
        for label in Labels:
            ax.plot(np.arange(len(f1_np[:, label.value]))[f1_np[:, label.value] > only_plot_larger_value],
                    f1_np[:, label.value][f1_np[:, label.value] > only_plot_larger_value], label=str(label.name))
        ax.set_title('train_f1_score')
    elif mode == 'val':
        f1_np = np.array(neural_net.f1_score_val_hist)
        for label in Labels:
            ax.plot(np.arange(len(f1_np[:, label.value]))[f1_np[:, label.value] > only_plot_larger_value],
                    f1_np[:, label.value][f1_np[:, label.value] > only_plot_larger_value], label=str(label.name))
        ax.set_title('val_f1_score', fontsize=20)

    ax.set_xlabel('epochs', fontsize=20)
    ax.set_ylabel('f1_score', fontsize=20)
    ax.legend(loc='lower right', fontsize=15)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid()

    ax.set_facecolor('lavender')


def generate_mean_f1_score_plot(neural_net: FCNN, ax: plt.axes):
    # num_classes = neural_net.layer_list[-1]  wäre auch möglich
    only_plot_larger_value = 0

    f1_train_np = np.array(neural_net.f1_score_hist)
    f1_train_mean = np.mean(f1_train_np, axis=1)
    ax.plot(np.arange(len(f1_train_mean))[f1_train_mean > only_plot_larger_value],
            f1_train_mean[f1_train_mean > only_plot_larger_value], label='mean f1 train')

    f1_val_np = np.array(neural_net.f1_score_val_hist)
    f1_val_mean = np.mean(f1_val_np, axis=1)
    ax.plot(np.arange(len(f1_val_mean))[f1_val_mean > only_plot_larger_value],
            f1_val_mean[f1_val_mean > only_plot_larger_value], label='mean f1 val')

    ax.set_title('mean f1_score', fontsize=20)

    ax.set_xlabel('epochs', fontsize=20)
    ax.set_ylabel('f1_score', fontsize=20)
    ax.legend(loc='lower right', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid()

    ax.set_facecolor('lavender')


def evaluate_neural_net(neural_net: FCNN, X_train, y_train, X_val, y_val, save_plot_path: Path = None):
    fig, axs = generate_evaluation_plot(
        neural_net, X_train, y_train, X_val, y_val, save_plot_path)


def generate_evaluation_plot(neural_net: FCNN, X_train, y_train, X_val, y_val, save_plot_path: Path = None):
    neural_net.clear_data_specific_parameters()
    neural_net.forward_prop(X_train)
    h_train = neural_net.O[-1].T
    neural_net.clear_data_specific_parameters()
    neural_net.forward_prop(X_val)
    h_val = neural_net.O[-1].T

    if y_train.shape[1] == 4:
        Labels = LabelsMandatory
    elif y_train.shape[1] == 11:
        Labels = LabelsOptional
    else:
        raise Exception(
            "number of Labels is not 4 (mandatory) or 11 (optional)")

    # TODO: figsize doesnt change anything if changed, fig is humongous
    fig, axs = plt.subplots(2, 4, figsize=(60, 40))
    generate_confusion_plot(
        h_train, y_train, ax=axs[0, 0], title='confusion_matrix_train')
    generate_confusion_plot(
        h_val, y_val, ax=axs[1, 0], title='confusion_matrix_val')

    generate_loss_plot(neural_net, ax=axs[1, 1])

    generate_f1_score_plot(
        neural_net, ax=axs[0, 2], mode='train', Labels_Mandatory_Optional=Labels)
    generate_f1_score_plot(
        neural_net, ax=axs[1, 2], mode='val', Labels_Mandatory_Optional=Labels)

    generate_acc_plot(neural_net, ax=axs[0, 1])

    generate_mean_f1_score_plot(neural_net, ax=axs[0, 3])

    rect = fig.patch
    rect.set_facecolor('whitesmoke')

    if not save_plot_path == None:
        fig.savefig(save_plot_path, facecolor=fig.get_facecolor(),
                    edgecolor='none')
    else:
        fig.show()

    plt.close(fig)

    return fig, axs


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.     
    The Savitzky-Golay filter removes high frequency noise from data.     
    It has the advantage of preserving the original shape and     
    features of the signal better than other types of filtering     approaches, 
    such as moving averages techniques.     
    Parameters     ----------     y : array_like, shape (N,)
         the values of the time history of the signal.
     window_size : int
         the length of the window. Must be an odd integer number.
     order : int
         the order of the polynomial used in the filtering.
         Must be less then `window_size` - 1.
     deriv: int
         the order of the derivative to compute (default = 0 means only smoothing)
     Returns
     -------
     ys : ndarray, shape (N)
         the smoothed signal (or it's n-th derivative).
     Notes
     -----
     The Savitzky-Golay is a type of low-pass filter, particularly
     suited for smoothing noisy data. The main idea behind this
     approach is to make for each point a least-square fit with a
     polynomial of high order over a odd-sized window centered at
     the point.
     Examples
     --------
     t = np.linspace(-4, 4, 500)
     y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
     ysg = savitzky_golay(y, window_size=31, order=4)
     import matplotlib.pyplot as plt
     plt.plot(t, y, label='Noisy signal')
     plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
     plt.plot(t, ysg, 'r', label='Filtered signal')
     plt.legend()
     plt.show()
     References
     ----------
     .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
     .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
     """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range]
               for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def generate_mean_f1_overview_plot(run_folder_paths: List[Path], preproc_params_list: List):
    if len(run_folder_paths) != len(preproc_params_list):
        raise RuntimeError('lists specifed to not match')

    counter = 0
    table_dict = {'id': [], 'lr': [], 'epochs': [], 
            'batch_size': [], 'architecture': [], 'act_func': [], 
            'window_size': [], 'pattern': []}
    f1_mean_list = []
    for run_folder_path, preproc_params in zip(run_folder_paths, preproc_params_list):
        run_folder_f1_mean_list = []
        for meta_json in run_folder_path.glob('**/*_meta.json'):
            with open(meta_json, 'r') as meta_json_file:
                meta_data_dict = json.load(meta_json_file)
                f1_hists_np = np.array(meta_data_dict['f1_score_val_hist'])
                mean_f1 = np.mean(f1_hists_np, axis=1)
                run_folder_f1_mean_list.append(mean_f1)

                table_dict['id'].append(counter)
                table_dict['lr'].append(meta_data_dict['lr'])
                table_dict['epochs'].append(meta_data_dict['epochs'])
                table_dict['batch_size'].append(meta_data_dict['batch_size'])
                table_dict['architecture'].append(
                    meta_data_dict['architecture'])
                table_dict['act_func'].append(
                    meta_data_dict['activation_functions'][0])
                table_dict['window_size'].append(preproc_params['window_size'])
                table_dict['pattern'].append(preproc_params['pattern'])

                print('counter:', counter, 'name',
                      meta_json.parent.parent.name)

                counter += 1

        f1_mean_list += run_folder_f1_mean_list

    df = pd.DataFrame(table_dict)

    fig, axs = plt.subplots(1, 2, figsize=(30, 20))

    NUM_COLORS = len(f1_mean_list)
    cm = plt.get_cmap('gist_rainbow')

    for i, f1_mean in enumerate(f1_mean_list):

        yhat = savitzky_golay(f1_mean[f1_mean > 0.8], 81, 3)
        # this would be without smoothening
        # lines = axs[0].plot(np.arange(len(f1_mean))[f1_mean > 0.8], f1_mean[f1_mean > 0.8], label=i)
        lines = axs[0].plot(np.arange(len(f1_mean))[
                            f1_mean > 0.8], yhat, label=i)
        lines[0].set_color(cm(i/NUM_COLORS))

    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('f1_val_score')
    axs[0].legend(loc='lower right')
    axs[0].grid()

    fig_background_color = 'skyblue'
    fig_border = 'steelblue'

    # Pop the headers from the data array
    column_headers = df.columns

    # Get some lists of color specs for row and column headers
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
  
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=df.values,
                          rowLoc='right',
                          colColours=ccolors,
                          colLabels=column_headers,
                          loc='center')
    
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 2)
    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    axs[1] = ax

    fig.savefig(run_folder_paths[0] / 'overview_plot.png')


def evaluate_runs(runs_folder_path: Path):
    # train_folder_path = Path(r'../../data\preprocessed_frames\new_window=10,cumsum=all\train')
    # val_folder_path = Path(r'../../data\preprocessed_frames\new_window=10,cumsum=all\validation')

    # X_train, y_train, scaler = generate_dataset(train_folder_path, select_mandatory_label=False)
    # X_val, y_val = generate_dataset(val_folder_path, scaler, select_mandatory_label=False)

    train_folder_path = Path(
        r'C:\Users\Jochen\Jonas\ML\ml_dev_repo\data\preprocessed_frames\new_window=10,cumsum=all\train\mandatory_data')
    val_folder_path = Path(
        r'C:\Users\Jochen\Jonas\ML\ml_dev_repo\data\preprocessed_frames\new_window=10,cumsum=all\validation\mandatory_data')

    # X_train, y_train, scaler = generate_dataset(train_folder_path, select_mandatory_label=False)
    # X_val, y_val = generate_dataset(val_folder_path, scaler, select_mandatory_label=False)

    X_train, y_train, scaler, pca = generate_pca_dataset(
        train_folder_path, select_mandatory_label=True, keep_percentage=99)
    X_val, y_val = generate_pca_dataset(
        val_folder_path, scaler, select_mandatory_label=True, pca=pca)

    pca.save(r'C:\Users\Jochen\Jonas\ML\ml_dev_repo\data\preprocessed_frames\new_window=10,cumsum=all\pca_mandatory.json')

    # random_search_multipro(X_train, y_train, X_val, y_val, scaler, Path(r'..\..\saved_runs\jonas_final_gross_2'),
    #     author='Jonas', description='window10_all, ohne Nina, second big run')

    # train_folder_path = Path(r'../../data\preprocessed_frames\new_window=10,cumsum=all\train')
    # val_folder_path = Path(r'../../data\preprocessed_frames\new_window=10,cumsum=all\validation')

    # X_train, y_train, scaler, pca = generate_pca_dataset(train_folder_path, select_mandatory_label=False, keep_percentage=99)
    # X_val, y_val = generate_pca_dataset(val_folder_path, scaler, select_mandatory_label=False, pca=pca)

    # for meta_json_path in runs_folder_path.glob(r'**/*_meta.json'):
    #     neural_net = FCNN.load_run(meta_json_path.parent)
    #     save_path = meta_json_path.parent / \
    #         (str(meta_json_path.parent.parent.name) + "_eval_plot.png")
    #     generate_evaluation_plot(
    #         neural_net, X_train, y_train, X_val, y_val, save_plot_path=save_path)
    #     print('saving', save_path)


if __name__ == '__main__':
    evaluate_runs(Path(r'C:\Users\Jochen\Jonas\ML\ml_dev_repo\saved_runs\jonas_klein_pca'))
    # generate_mean_f1_overview_plot(
    #     run_folder_paths=[
    #         Path(r'C:\Users\Jochen\Jonas\ML\ml_dev_repo\saved_runs\jonas_klein_pca')],
    #     preproc_params_list=[{'window_size': 10, 'pattern': 'every'}]
    # )
