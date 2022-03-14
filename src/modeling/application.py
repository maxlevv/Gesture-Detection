import sys
import numpy as np
import pandas as pd
from pathlib import Path
from neural_network import FCNN
# sys.path.append("..")
from preprocessing.preprocessing_functions import Labels
from feature_scaling import StandardScaler

np.random.seed(0)

def generate_dataset(preproc_folder_path: Path):
    df = None
    for preproc_csv_file_path in preproc_folder_path.glob('**/*_preproc.csv'):
        next_df = pd.read_csv(preproc_csv_file_path, sep=' *,', engine='python')
        if df is None:
            df = next_df
        else:
            df = pd.concat([df, next_df], axis=0)
    
    y = df[Labels.get_column_names()].to_numpy()
    X = df.drop(Labels.get_column_names(), axis=1).to_numpy()
    X = X[:, 1:]

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
    X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val, scaler


def do_train_run(preproc_folder_path: Path):
    # function for loading the data for training, standardization and start training

    X_train, y_train, X_val, y_val, scaler = generate_dataset(preproc_folder_path)

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
    lr = 0.001
    epochs = 100
    batch_size = 50
    neural_net.fit(X_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size, optimizer='adam', weight_decay=0.0001, X_val=X_val, Y_g_val=y_val)

    neural_net.save_run(Path(r'../../saved_runs'), 'first_run_max', author='Max', data_file_name='scaled_angle', \
        lr=lr, batch_size=batch_size, epochs=epochs, num_samples=X_train.shape[0], \
        description="just a test")

    neural_net.evaluate_model(X_train, y_train, X_val, y_val)

    print("done")


def load_training_run_and_evaluate(run_folder_path: Path, preproc_folder_path: Path):
    neural_net = FCNN.load_run(run_folder_path)

    X_train, y_train, X_val, y_val, _ = generate_dataset(preproc_folder_path)

    # neural_net.calc_metrics(X_train, y_train)
    # neural_net.calc_metrics(X_val, y_val)

    neural_net.evaluate_model(X_train, y_train, X_val, y_val)

    print("done")


if __name__ == '__main__':
    do_train_run(Path(r'../../data/preprocessed_frames/scaled_angle'))
    #load_training_run_and_evaluate(Path(r'../../saved_runs\first_run_max\2022-03-12_0_73-40-40-30-20-10-4'), \
    #     Path(r'../../data/preprocessed_frames/scaled_angle'))
