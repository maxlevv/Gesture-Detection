import sys
import numpy as np
import pandas as pd
from pathlib import Path
from neural_network import FCNN
# sys.path.append("..")
from preprocessing.preprocessing_functions import Labels
from feature_scaling import StandardScaler



def do_train_run(preproc_folder_path: Path):
    # function for loading the data for training, standardization and start training

    df = None
    for preproc_csv_file_path in preproc_folder_path.glob('**/*_preproc.csv'):
        next_df = pd.read_csv(preproc_csv_file_path, sep=' *,', engine='python')
        if df is None:
            df = next_df
        else:
            pd.concat([df, next_df], axis=0)
    
    y = df[Labels.get_column_names()].to_numpy()
    X = df.drop(Labels.get_column_names(), axis=1).to_numpy()

    del df

    # seperate train and validation data
    train_percentage = 0.8
    val_percentage = 0.2

    shuffled_indices = np.random.choice(X.shape[0], X.shape[0], replace=False)
    split_index = int(len(shuffled_indices) * train_percentage)

    X_train = X[shuffled_indices[:split_index]]
    y_train = y[shuffled_indices[:split_index]]

    X_val = X[shuffled_indices[split_index:]]
    y_val = y[shuffled_indices[split_index:]]

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
    neural_net.fit(X_train, y_train, lr=0.001, epochs=100, batch_size=20)

    print("done")


if __name__ == '__main__':
    do_train_run(Path(r'../../data/preprocessed_frames/scaled_to_torso'))