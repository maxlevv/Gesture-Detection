import sys
sys.path.append('neural_net_pack')
sys.path.append('../../neural_net_pack')
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from evaluation.metrics import f1_score
from feature_scaling import StandardScaler
from neural_network import FCNN
from preprocessing.preprocessing_functions import scale_to_body_size_and_dist_to_camera


def preprocess(df):
    preprocessed: pd.DataFrame = df.loc[:, 'left_eye_outer_x'] - df.loc[:, 'right_eye_outer_x']
    # preprocessed = pd.DataFrame()
    # preprocessed['ear_dist'] = df.loc[:, 'left_ear_x'] - df.loc[:, 'right_ear_x']
    # preprocessed['eye distance'] = df.loc[:, 'left_eye_outer_x'] - df.loc[:, 'right_eye_outer_x']
    # preprocessed['eye ear distance left'] = df.loc[:, 'left_ear_x'] - df.loc[:, 'left_eye_outer_x']
    # preprocessed['eye ear distance right'] = df.loc[:, 'right_ear_x'] - df.loc[:, 'right_eye_outer_x']
    # print(preprocessed)
    # print('df type', type(df))
    eye_distance = preprocessed.to_numpy()[:, np.newaxis]
    return scale_to_body_size_and_dist_to_camera(eye_distance, df)


def train(X, Y_g, lr, epochs, batch_size, X_val, Y_g_val):
    # my_net = FCNN(0, [1], [1], ['softmax'], loss_func='categorical_cross_entropy',
    my_net = FCNN(1, [1], [1], ['sigmoid'], loss_func='cross_entropy',
                  scaler=StandardScaler())
    my_net.scaler.fit(X)
    X_scaled = my_net.scaler.transform(X)
    X_val_scaled = my_net.scaler.transform(X_val)
    my_net.init_weights()

    my_net.clear_data_specific_parameters()
    my_net.fit(X_scaled, Y_g, lr=lr, epochs=epochs, batch_size=batch_size, optimizer='adam', X_val=X_val_scaled,
               Y_g_val=Y_g_val)
    my_net.clear_data_specific_parameters()

    return my_net


def predict(df, net):
    X = preprocess(df)
    X_scaled = net.scaler.transform(X)
    net.forward_prop(X_scaled)
    prediction = np.array(net.O[-1].T)
    return prediction


def round_prediction(h):
    return h.round().astype(int)


def calc_acc(h, y):
    predicted = round_prediction(h)
    return np.sum(predicted == y) / y.shape[0]


def plot(y_label, title, ar, dir, file_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(np.arange(1, len(ar) + 1, 1, dtype=int), ar)
    fig.savefig(dir / file_name)


def calc_confusion_matrix_binary(h: np.array, y: np.array):
    if np.shape(h) != np.shape(y):
        raise Exception("ground truth vector y and hypothesis h are not the same size")

    n = 2
    matrix = np.zeros(shape=(n, n), dtype='int')

    prediction = round_prediction(h)

    for i in range(y.shape[0]):
        matrix[prediction[i, 0], y[i, 0]] = matrix[prediction[i, 0], y[i, 0]] + 1

    return matrix


def plot_confusion_matrix_binary(confusion_matrix, dir, file_name):
    fig, ax = plt.subplots(figsize=(10, 8))

    labels = ['no_look', 'idle']
    confusion_df = pd.DataFrame(data=confusion_matrix, columns=labels, index=labels)

    sns.heatmap(confusion_df, annot=confusion_matrix, fmt="", ax=ax)
    ax.set_xlabel("ground truth")
    ax.set_ylabel("predicted")
    ax.set_title("confusion matrix")
    fig.savefig(dir / file_name)


def predict_labels(df, net):
    h = predict(df, net)
    prediction = pd.DataFrame(round_prediction(h)).loc[:, 0]
    labels = prediction.map(inverse_label_encodings)
    return labels


label_encodings = {
    'no_look': 1,
    'idle': 0
}
inverse_label_encodings = {
    1: 'no_look',
    0: 'idle'
}

if __name__ =='__main__':
    df = pd.read_csv(
        '../../data/labeled_frames/ready_to_train_look/train/03-18_jonas_look_train_labeled.csv')
    X = preprocess(df)
    Y_g = df.loc[:, 'ground_truth'].map(label_encodings).to_numpy()[:, np.newaxis]

    df_val = pd.read_csv(
        '../../data/labeled_frames/ready_to_train_look/val/03-18_jonas_look_val_labeled.csv')
    X_val = preprocess(df_val)
    Y_g_val = df_val.loc[:, 'ground_truth'].map(label_encodings).to_numpy()[:, np.newaxis]

    lr = 0.03
    batch_size = 64
    epochs = 100
    save_path = Path('../../saved_runs/no_look')

    # net = train(X, Y_g, lr, epochs, batch_size, X_val, Y_g_val)
    #
    # save_folder_path = net.save_run(save_path,
    #                                 'first_run_nina_no_look', author='Nina', data_file_name='test',
    #                                 lr=lr, batch_size=batch_size, epochs=epochs, num_samples=X.shape[0],
    #                                 description="test")



    no_look_path = save_path / 'first_run_nina_no_look'
    for dir in next(os.walk(no_look_path))[1]:
        dir = Path(no_look_path, dir)

        load_path = dir
        net = FCNN.load_run(load_path)

        plot(ar=net.loss_hist, dir=dir, y_label="Loss", file_name='learning_curve.png', title="Learning Curve")

        h_val = predict(df_val, net)
        confusion_matrix = calc_confusion_matrix_binary(h_val, Y_g_val)
        plot_confusion_matrix_binary(confusion_matrix, dir, 'Confusion_matrix.png')
        acc = calc_acc(h_val, Y_g_val)
        print(acc)
        f1 = f1_score(confusion_matrix, 0)
        print(f1)

        with open(dir / f'accuracy: {acc}', 'w') as _:
            pass
        with open(dir / f'f1: {f1}', 'w') as _:
            pass
