import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from evaluation.metrics import calc_confusion_matrix
from feature_scaling import StandardScaler
from gradient_checking import check_gradient_of_neural_net
from helper import softmax2one_hot
from neural_network import FCNN
from preprocessing.preprocessing_functions import scale_to_body_size_and_dist_to_camera


def preprocess(df):
    preprocessed: pd.DataFrame = df.loc[:, 'left_eye_outer_x'] - df.loc[:, 'right_eye_outer_x']
    eye_distance = preprocessed.to_numpy()[:, np.newaxis]
    return scale_to_body_size_and_dist_to_camera(eye_distance, df)


def train(X, Y_g, lr, epochs, batch_size, X_val, Y_g_val):
    # my_net = FCNN(0, [1], [1], ['softmax'], loss_func='categorical_cross_entropy',
    # my_net = FCNN(1, [1], [1], ['sigmoid'], loss_func='cross_entropy',
    my_net = FCNN(1, [1], [1], ['softmax'], loss_func='categorical_cross_entropy',
                  scaler=StandardScaler())
    my_net.scaler.fit(X)
    X_scaled = my_net.scaler.transform(X)
    X_val_scaled = my_net.scaler.transform(X_val)
    my_net.init_weights()

    my_net.clear_data_specific_parameters()
    my_net.fit(X_scaled, Y_g, lr=lr, epochs=epochs, batch_size=batch_size, optimizer='adam', X_val=X_val_scaled, Y_g_val=Y_g_val)
    my_net.clear_data_specific_parameters()

    return my_net


def predict(df, net):
    X = preprocess(df)
    X_scaled = net.scaler.transform(X)
    net.forward_prop(X_scaled)
    prediction = softmax2one_hot(net.O[-1].T)[:, 0][:, np.newaxis].astype(int)
    return prediction


def plot(y_label, title, ar, dir, file_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(np.arange(1, len(ar) + 1, 1, dtype=int), ar)
    # plt.show()
    fig.savefig(dir / file_name)


def calc_confusion_matrix_binary(h: np.array, y: np.array):
    if np.shape(h) != np.shape(y):
        raise Exception("ground truth vector y and hypothesis h are not the same size")

    n = 2
    matrix = np.zeros(shape=(n, n), dtype='int')

    # for each row set the maximal value to one and the rest to zero
    # instead of h.round()
    h_onehot = softmax2one_hot(h)

    for i in range(y.shape[0]):
        matrix[h_onehot[i, 0], y[i, 0]] = matrix[h_onehot[i, 0], y[i, 0]] + 1

    return matrix


def plot_confusion_matrix_binary(confusion_matrix, dir, file_name):
    fig, ax = plt.subplots(figsize=(10, 8))

    labels = ['no_look', 'idle']
    confusion_df = pd.DataFrame(data=confusion_matrix, columns=labels, index=labels)

    sns.heatmap(confusion_df, annot=confusion_matrix , fmt="", ax=ax)
    ax.set_xlabel("ground truth")
    ax.set_ylabel("predicted")
    ax.set_title("confusion matrix")
    # plt.show()
    fig.savefig(dir / file_name)


label_encodings = {
    'no_look': 1,
    'idle': 0
}
label_encodings_inverse = {
    1: 'no_look',
    0: 'idle'
}


df = pd.read_csv(
    '../../data/labeled_frames/ready_to_train_look/train/03-18_jonas_look_train_labeled.csv')
X = preprocess(df)
Y_g = df.loc[:, 'ground_truth'].map(label_encodings).to_numpy()[:, np.newaxis]

df_val = pd.read_csv(
    '../../data/labeled_frames/ready_to_train_look/val/03-18_jonas_look_val_labeled.csv')
X_val = preprocess(df_val)
Y_g_val = df_val.loc[:, 'ground_truth'].map(label_encodings).to_numpy()[:, np.newaxis]

lr = 1
batch_size = 64
epochs = 10

net = train(X, Y_g, lr, epochs, batch_size, X_val, Y_g_val)
save_path = Path('../../saved_runs/no_look')

save_folder_path = net.save_run(save_path,
             'first_run_nina_no_look', author='Nina', data_file_name='test',
             lr=lr, batch_size=batch_size, epochs=epochs, num_samples=X.shape[0],
             description="test")

# load_path = Path('../../saved_runs/no_look/first_run_nina_no_look/first_run_nina_no_look/2022-03-31_19_1-1')
# net = FCNN.load_run(load_path)
# predict(df, net)


no_look_path = save_path / 'first_run_nina_no_look'
for dir in next(os.walk(no_look_path))[1]:
    dir = Path(no_look_path, dir)
    plot(ar=net.val_acc_hist, dir=dir, y_label="Accuracy", file_name='validation_accuracy.png', title="Validation Accuracy")
    plot(ar=net.loss_hist, dir=dir, y_label="Loss", file_name='learning_curve.png', title="Learning Curve")
    plot(ar=net.f1_score_val_hist, dir=dir, y_label="F1 Score", file_name='f1_score.png', title="F1 Score")

    h_val = predict(df_val, net)
    confusion_matrix  = calc_confusion_matrix_binary(h_val, Y_g_val)
    plot_confusion_matrix_binary(confusion_matrix, dir, 'Confusion_matrix.png')