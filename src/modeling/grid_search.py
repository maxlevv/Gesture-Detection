import numpy as np
import pandas as pd
from pathlib import Path
from neural_network import FCNN
#from preprocessing.preprocessing_functions import Labels
from preprocessing.preprocessing_functions import LabelsMandatory
from preprocessing.preprocessing_functions import LabelsOptional
from feature_scaling import StandardScaler
from evaluation.evaluate import evaluate_neural_net
import matplotlib.pyplot as plt

from process_videos.helpers.colors import bcolors


def generate_dataset(preproc_folder_path: Path, scaler: StandardScaler = None, select_mandatory_label: bool = True):
    df = None
    for preproc_csv_file_path in preproc_folder_path.glob('**/*_preproc.csv'):
        next_df = pd.read_csv(preproc_csv_file_path, sep=' *,', engine='python')
        if df is None:
            df = next_df
        else:
            df = pd.concat([df, next_df], axis=0)

    if select_mandatory_label == True:
        Labels = LabelsMandatory
        y = df[Labels.get_column_names()].to_numpy()
    else:
        Labels = LabelsOptional
        y = df[Labels.get_column_names()].to_numpy()

    df = df.drop(LabelsOptional.get_column_names(), axis=1)
    X = df.to_numpy()
    X = X[:, 1:]

    ###################################################################################################################################################
    # TODO: this needs removing
    print(f'{bcolors.FAIL}ONLY TRAINING WITH 10000 samples! Change grid_search.py - generate_dataset(){bcolors.ENDC}')
    X = X[:10000, :]
    y = y[:10000, :]
    ###################################################################################################################################################

    del df

    if scaler == None:
        # standardize columns for training data
        new_scaler = StandardScaler()
        new_scaler.fit(X)
        X = new_scaler.transform(X)
        return X, y, new_scaler

    else:
        # validation/test data
        X = scaler.transform(X)
        return X, y


def grid_search(X_train, y_train, X_val, y_val, scaler):

    # define grid
    activation_list = ['sigmoid', 'relu', 'leaky_relu']
    epoch_list = [10]
    bsize_list = [300]
    lr_list = [0.001, 0.005, 0.01]
    wdecay_list = [0, 0.00001, 0.001]

    f1_train = []
    f1_val = []
    x_axis = []

    for activation_function in activation_list:
        
        for epochs in epoch_list:
            for batch_size in bsize_list:
                for weight_decay in wdecay_list:
                    for lr in lr_list:
                        # initialize the network
                        neural_net = FCNN(
                            input_size=X_train.shape[1],
                            layer_list=[40, 40, 30, 20, 10, y_train.shape[1]],
                            bias_list=[1, 1, 1, 1, 1, 1],
                            activation_funcs=[activation_function] * 5 + ['softmax'],
                            loss_func='categorical_cross_entropy',
                            scaler=scaler
                        )

                        neural_net.clear_attributes()

                        neural_net.init_weights()
                        neural_net.fit(X_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size,
                                       optimizer='adam', weight_decay=weight_decay, X_val=X_val, Y_g_val=y_val)

                        save_folder_path = neural_net.save_run(save_runs_folder_path=Path(r'../../saved_runs\jonas_3_grid_gross'),
                                            run_group_name=f'{activation_function},ep={epochs},bs={batch_size},lr={lr},wd={weight_decay}',
                                            author='Jonas', data_file_name='', lr=lr, batch_size=batch_size, epochs=epochs,
                                            num_samples=X_train.shape[0], description='erster Grid Search vamos')

                        neural_net.evaluate_model(X_train, y_train, X_val, y_val, save_folder_path / 'metrics_plot.png')

                        x_axis.append([activation_function, epochs, batch_size, lr, weight_decay])
                        # f1_train.append(min(neural_net.f1_score_hist[-1]))
                        # f1_val.append(min(neural_net.f1_score_val_hist[-1]))

                        del neural_net

    fig, ax = plt.subplots()
    ax.scatter(list(range(len(f1_train))), f1_train, label='f1_train')
    ax.scatter(list(range(len(f1_val))), f1_val, label='f1_val')
    ax.set_ylabel('f1')
    ax.set_title('f1 score')
    ax.set_xticks([])
    ax.table(cellText=list(map(list, zip(*x_axis))),
             rowLabels=['activation function', 'epochs', 'batch size', 'lr', 'weight decay'],
             loc='bottom')
    plt.subplots_adjust(left=0.3, bottom=0.2)

    plt.show()
    #while True:
    #    import time
    #    time.sleep(1)
    fig.savefig('grid_search_plot.png')


if __name__ == '__main__':
    train_folder_path = Path(r'../../data\preprocessed_frames\final\train')
    val_folder_path = Path(r'../../data\preprocessed_frames\final\validation')

    X_train, y_train, scaler = generate_dataset(train_folder_path, select_mandatory_label=False)
    X_val, y_val = generate_dataset(val_folder_path, scaler, select_mandatory_label=False)



    grid_search(X_train, y_train, X_val, y_val, scaler)

