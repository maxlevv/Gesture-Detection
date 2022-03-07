import sys
import numpy as np
import pandas as pd
from pathlib import Path
from neural_network import FCNN
sys.path.append(r"C:\Users\Max\PycharmProjects\ml_dev_repo\src")
from src.preprocessing.preprocessing_functions import Labels
from feature_scaling import StandardScaler
from src.evaluation import metrics


def do_train_run(preproc_folder_path: Path):
    # function for loading the data for training, standardization and start training

    df = None
    for preproc_csv_file_path in preproc_folder_path.glob('**/*_preproc.csv'):
        next_df = pd.read_csv(preproc_csv_file_path, sep=' *,', engine='python')
        if df is None:
            df = next_df
        else:
            df = pd.concat([df, next_df], axis=0)
    
    y = df[Labels.get_column_names()].to_numpy()
    X = df.drop(Labels.get_column_names(), axis=1).to_numpy()
    print(X.shape)
    print(y.shape)
    del df

    # seperate train and validation data
    train_percentage = 0.8
    val_percentage = 0.2

    shuffled_indices = np.random.choice(X.shape[0], X.shape[0], replace=False)
    split_index = int(len(shuffled_indices) * train_percentage)

    X_train = X[shuffled_indices[:split_index]]
    y_train = y[shuffled_indices[:split_index]]

    X_val = X[shuffled_indices[split_index + 1:]]
    y_val = y[shuffled_indices[split_index + 1:]]

    # standardize columns
    scaler = StandardScaler()
    scaler.fit(X_train) 
    X_train = scaler.transform(X_train)

    # initialize the network
    neural_net = FCNN(
        input_size = X_train.shape[1],
        layer_list = [40, 40, 30, 20, 10, 4],
        bias_list = [1, 1, 1, 1, 1, 1],
        activation_funcs = ['sigmoid'] * 5 + ['softmax'],
        loss_func = 'categorical_cross_entropy',
        scaler = scaler
    )
    neural_net.init_weights()

    # start training
    neural_net.fit(X_train, y_train, lr=0.001, epochs=500, batch_size=20, lambd=0.0001)
    print(neural_net.loss_hist[-1])

    #neural_net.init_weights()
    #neural_net.fit(X_train, y_train, lr=0.001, epochs=100, batch_size=20, lambd=0)

    neural_net.plot_stats()
    print(neural_net.loss_hist[-1])

    o_train, z = neural_net.forward_it(X_train)
    h_train = np.asarray(o_train[-1].T)

    o_val, z = neural_net.forward_it(X_val)
    h_val = np.asarray(o_val[-1].T)

    matrix_train = metrics.calc_confusion_matrix(h_train, y_train)
    matrix_val = metrics.calc_confusion_matrix(h_val, y_val)

    metrics.print_confusion_matrix(matrix_train)
    metrics.print_confusion_matrix(matrix_val)
    print("done")


if __name__ == '__main__':
    do_train_run(Path(r'../../data/preprocessed_frames/train_run_max_2'))
