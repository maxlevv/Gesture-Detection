from pathlib import Path

import numpy as np
import pandas as pd

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
    print(my_net.scaler.to_dict())

    return my_net


def predict(df, net):
    X = preprocess(df)
    X_scaled = net.scaler.transform(X)
    net.forward_prop(X_scaled)
    prediction = pd.Series(softmax2one_hot(net.O[-1].T)[:, 0]).map(label_encodings_inverse)
    print(prediction)
    return prediction


label_encodings = {
    'no_look': 1,
    'idle': 0
}
label_encodings_inverse = {
    1: 'no_look',
    0: 'idle'
}


df = pd.read_csv(
    '/home/nina/Documents/4_Machine_Learning_for_User_Interfaces/software projects/ml_dev_repo/data/labeled_frames/ready_to_train_look/train/03-18_jonas_look_train_labeled.csv')
X = preprocess(df)
Y_g = df.loc[:, 'ground_truth'].map(label_encodings).to_numpy()[:, np.newaxis]

df_val = pd.read_csv(
    '/home/nina/Documents/4_Machine_Learning_for_User_Interfaces/software projects/ml_dev_repo/data/labeled_frames/ready_to_train_look/val/03-18_jonas_look_val_labeled.csv')
X_val = preprocess(df_val)
Y_g_val = df_val.loc[:, 'ground_truth'].map(label_encodings).to_numpy()[:, np.newaxis]

lr = 0.01
batch_size = 64
epochs = 2

net = train(X, Y_g, lr, epochs, batch_size, X_val, Y_g_val)
save_path = Path('/home/nina/Documents/4_Machine_Learning_for_User_Interfaces/software projects/ml_dev_repo/saved_runs/no_look/first_run_nina_no_look')
net.save_run(save_path,
             'first_run_nina_no_look', author='Nina', data_file_name='test',
             lr=lr, batch_size=batch_size, epochs=epochs, num_samples=X.shape[0],
             description="test")

# load_path = Path('/home/nina/Documents/4_Machine_Learning_for_User_Interfaces/software projects/ml_dev_repo/saved_runs/no_look/first_run_nina_no_look/2022-03-30_1_1-1-1')
# net = FCNN.load_run(load_path)
predict(df, net)
