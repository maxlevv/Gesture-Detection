from platform import architecture
import numpy as np
import pandas as pd
import os
import random
from pathlib import Path
from neural_network import FCNN
import matplotlib.pyplot as plt
from grid_search import generate_dataset
import multiprocessing

random.seed(0)

def inner(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description):

    random.seed() # we want different random values in each process

    # define range of parameters
    lr = (5 * np.power(10, random.uniform(-4, -3))).round(6)
    weight_decay = (random.choices([0, np.power(10, random.uniform(-3, -2)).round(6)], [0.25, 0.75]))[0]
    batch_size = random.choice([64, 128, 256, 512])
    epochs = 10
    activation_functions = ['sigmoid', 'relu', 'leaky_relu']
    activation_function = random.choice(activation_functions)

    architecture = [40, 40, 30, 20, 10, y_train.shape[1]]

    neural_net = FCNN(
        input_size=X_train.shape[1],
        layer_list=architecture,
        bias_list=[1] * len(architecture),
        activation_funcs=[activation_function] * (len(architecture) - 1) + ['softmax'],
        loss_func='categorical_cross_entropy',
        scaler=scaler
    )

    neural_net.clear_attributes()

    neural_net.init_weights()
    neural_net.fit(X_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size,
                    optimizer='adam', weight_decay=weight_decay, X_val=X_val, Y_g_val=y_val)

    save_folder_path = neural_net.save_run(save_runs_folder_path=save_runs_folder_path,
                                            run_group_name=f'{activation_function},ep={epochs},bs={batch_size},lr={lr},wd={weight_decay}',
                                            author=author, data_file_name='', lr=lr, batch_size=batch_size,
                                            epochs=epochs,
                                            num_samples=X_train.shape[0], description=description)

    neural_net.evaluate_model(X_train, y_train, X_val, y_val, save_folder_path / 'metrics_plot.png')



    return [activation_function, epochs, batch_size, lr, weight_decay], min(neural_net.f1_score_hist[-1]), min(neural_net.f1_score_val_hist[-1])


def random_search_multipro(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description):
    num_iterations = 30
    num_simultaneous_processes = 6

    with multiprocessing.Pool(num_simultaneous_processes) as pool:
        res = pool.starmap(inner, [(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description)] * num_iterations )

    # print('here')   
    res = sorted(res)

    x_axis, f1_train, f1_val = tuple(map(list, zip(*res)))


    # df minimal f1
    x_axis_df = pd.DataFrame(x_axis, columns= ['activation function', 'epochs', 'batch size', 'lr', 'weight decay'])
    f1_train_df = pd.DataFrame(f1_train, columns=['min_f1_train'])
    f1_val_df = pd.DataFrame(f1_val, columns=['min_f1_val'])
    df = pd.concat(objs=[x_axis_df, f1_train_df, f1_val_df], axis=1)
    df.to_csv(save_runs_folder_path / 'min_f1_score.csv')

    # fig minimal f1
    fig, ax = plt.subplots(figsize=(60,40))

    x = np.arange(len(f1_train))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(x - width/2, f1_train, width, label='f1_train')
    rects2 = ax.bar(x + width/2, f1_val, width, label='f1_val')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    # ax.scatter(list(range(len(f1_train))), f1_train, label='f1_train', s=200)
    # ax.scatter(list(range(len(f1_val))), f1_val, label='f1_val', s=200)
    ax.set_ylabel('f1')
    ax.set_title('f1 score')
    ax.set_xticks([])
    ax.table(cellText=list(map(list, zip(*x_axis))),
             rowLabels=['activation function', 'epochs', 'batch size', 'lr', 'weight decay'],
             loc='bottom')
    # ax.subplots_adjust(left=0.3, bottom=0.2)

    # plt.show()
    fig.savefig(save_runs_folder_path / 'min_f1_score.png')



def random_search(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path):
    f1_train = []
    f1_val = []
    x_axis = []

    i = 1
    while i <= 10:
        i += 1

        # define range of parameters
        lr = (5 * np.power(10, random.uniform(-4, -3))).round(6)
        weight_decay = (random.choices([0, np.power(10, random.uniform(-3, -2)).round(6)], [0.25, 0.75]))[0]
        batch_size = random.choice([64, 128, 256, 512])
        epochs = 30
        activation_functions = ['sigmoid', 'relu', 'leaky_relu']
        activation_function = random.choice(activation_functions)

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

        save_folder_path = neural_net.save_run(save_runs_folder_path=save_runs_folder_path,
                                               run_group_name=f'{activation_function},ep={epochs},bs={batch_size},lr={lr},wd={weight_decay}',
                                               author='Jonas', data_file_name='', lr=lr, batch_size=batch_size,
                                               epochs=epochs,
                                               num_samples=X_train.shape[0], description='erster Grid Search vamos')

        neural_net.evaluate_model(X_train, y_train, X_val, y_val, save_folder_path / 'metrics_plot.png')

        x_axis.append([activation_function, epochs, batch_size, lr, weight_decay])
        f1_train.append(min(neural_net.f1_score_hist[-1]))
        f1_val.append(min(neural_net.f1_score_val_hist[-1]))

        del neural_net

    # df minimal f1
    x_axis_df = pd.DataFrame(x_axis, columns= ['activation function', 'epochs', 'batch size', 'lr', 'weight decay'])
    f1_train_df = pd.DataFrame(f1_train, columns=['min_f1_train'])
    f1_val_df = pd.DataFrame(f1_val, columns=['min_f1_val'])
    df = pd.concat(objs=[x_axis_df, f1_train_df, f1_val_df], axis=1)
    df.to_csv('min_f1_score.csv')

    # fig minimal f1
    fig, ax = plt.subplots(figsize=(60,40))
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
    fig.savefig('min_f1_score.png')


if __name__ == '__main__':
    train_folder_path = Path(r'../../data\preprocessed_frames\window=10,cumsum=all\train')
    val_folder_path = Path(r'../../data\preprocessed_frames\window=10,cumsum=all\validation')

    X_train, y_train, scaler = generate_dataset(train_folder_path, select_mandatory_label=False)
    X_val, y_val = generate_dataset(val_folder_path, scaler, select_mandatory_label=False)

    random_search_multipro(X_train, y_train, X_val, y_val, scaler, Path(r'..\..\saved_runs\jonas_random_1\arch1_ep=80\cumsum_all'),
        author='Jonas', description='ohne nina daten')

    train_folder_path = Path(r'../../data\preprocessed_frames\window=10,cumsum=every_second\train')
    val_folder_path = Path(r'../../data\preprocessed_frames\window=10,cumsum=every_second\validation')

    X_train, y_train, scaler = generate_dataset(train_folder_path, select_mandatory_label=False)
    X_val, y_val = generate_dataset(val_folder_path, scaler, select_mandatory_label=False)

    random_search_multipro(X_train, y_train, X_val, y_val, scaler, Path(r'..\..\saved_runs\jonas_random_1\arch1_ep=80\cumsum_every_second'),
        author='Jonas', description='ohne nina daten')
