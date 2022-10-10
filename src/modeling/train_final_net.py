import sys, os
sys.path.append('neural_net_pack')
sys.path.append('../../neural_net_pack')
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
from pathlib import Path
from grid_search import generate_dataset
from neural_network import FCNN
from evaluation.evaluate import evaluate_neural_net



def train_small_net(save_run_folder_path: Path, author: str, description: str):

    train_folder_path = Path(r'data\preprocessed_frames\SS22\window=10,cumsum=all_original\train\mandatory_data')
    val_folder_path = Path(r'data\preprocessed_frames\SS22\window=10,cumsum=all_original\validation\mandatory_data')

    X_train, y_train, scaler = generate_dataset(train_folder_path, select_mandatory_label=True)
    X_val, y_val = generate_dataset(val_folder_path, scaler, select_mandatory_label=True)
    
    epochs = 600
    activation_function = 'relu'
    batch_size = 512
    lr = 0.000875
    weight_decay = 0
    architecture = [30, 30, 15, 4]

    neural_net = FCNN(
        input_size=X_train.shape[1],
        layer_list=architecture,
        bias_list=[1] * len(architecture),
        activation_funcs=[activation_function] * (len(architecture) - 1) + ['softmax'],
        loss_func='categorical_cross_entropy',
        scaler=scaler,
        evaluate_model_func=evaluate_neural_net
    )

    neural_net.clear_attributes()

    neural_net.init_weights()

    neural_net.fit(X_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size,
                    optimizer='adam', weight_decay=weight_decay, X_val=X_val, Y_g_val=y_val, gradient_label_weighting=True)

    save_folder_path = neural_net.save_run(save_runs_folder_path=save_run_folder_path,
                                            run_group_name=f'{activation_function},ep={epochs},bs={batch_size},lr={lr},wd={weight_decay}',
                                            author=author, data_file_name='', lr=lr, batch_size=batch_size,
                                            epochs=epochs,
                                            num_samples=X_train.shape[0], description=description)

    neural_net.evaluate_model(X_train, y_train, X_val, y_val, save_folder_path / f'{activation_function}, \
                              ep={epochs},bs={batch_size},lr={lr},wd={weight_decay}_ metrics_plot.png')


def train_large_net(save_run_folder_path: Path, author: str, description: str):

    train_folder_path = Path(r'data\preprocessed_frames\SS22\window=10,cumsum=all_original\train')
    val_folder_path = Path(r'data\preprocessed_frames\SS22\window=10,cumsum=all_original\validation')

    X_train, y_train, scaler = generate_dataset(train_folder_path, select_mandatory_label=False)
    X_val, y_val = generate_dataset(val_folder_path, scaler, select_mandatory_label=False)
    
    epochs = 700
    activation_function = 'relu'
    batch_size = 512
    lr = 0.000875
    weight_decay = 0
    architecture = [40, 40, 30, 20, 11]

    neural_net = FCNN(
        input_size=X_train.shape[1],
        layer_list=architecture,
        bias_list=[1] * len(architecture),
        activation_funcs=[activation_function] * (len(architecture) - 1) + ['softmax'],
        loss_func='categorical_cross_entropy',
        scaler=scaler,
        evaluate_model_func=evaluate_neural_net
    )

    neural_net.clear_attributes()

    neural_net.init_weights()

    neural_net.fit(X_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size,
                    optimizer='adam', weight_decay=weight_decay, X_val=X_val, Y_g_val=y_val, gradient_label_weighting=True)

    save_folder_path = neural_net.save_run(save_runs_folder_path=save_run_folder_path,
                                            run_group_name=f'{activation_function},ep={epochs},bs={batch_size},lr={lr},wd={weight_decay}',
                                            author=author, data_file_name='', lr=lr, batch_size=batch_size,
                                            epochs=epochs,
                                            num_samples=X_train.shape[0], description=description)

    neural_net.evaluate_model(X_train, y_train, X_val, y_val, save_folder_path / f'{activation_function}, \
                              ep={epochs},bs={batch_size},lr={lr},wd={weight_decay}_ metrics_plot.png')


if __name__ == '__main__':
    train_small_net(Path(r'saved_runs\SS22\kleines_net_original'), author='Jonas', description='small net like last semester')
    # train_large_net(Path(r'saved_runs\SS22\grosses_net_original'), author='Jonas', description='large net like last semester')