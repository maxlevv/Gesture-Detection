from platform import architecture
import numpy as np
import pandas as pd
import os
import random
from pathlib import Path
from neural_network import FCNN
import matplotlib.pyplot as plt
from grid_search import generate_dataset
from evaluation.evaluate import evaluate_neural_net
import multiprocessing

random.seed(0)

def inner(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description, counter):

    random.seed() # we want different random values in each process

    # define range of parameters
    # lr = (np.power(10, random.uniform(-4, -3))).round(6)
    # weight_decay = (random.choices([0, np.power(10, random.uniform(-3, -2)).round(6)], [0.25, 0.75]))[0]
    # batch_size = random.choice([64, 128, 256, 512])

    # if counter == 0:
    #     lr = 0.001
    # if counter == 1:
    #     lr = 0.1
    # weight_decay = 0

    # batch_sizes = [10000, 50000, 5000]
    # batch_size = batch_sizes[counter]
    # batch_size = X_train.shape[0]
    


    epochs = 700
    activation_functions = ['relu', 'leaky_relu']
    activation_function = random.choice(activation_functions)
    activation_function = 'relu'

    # architecture = [40, 40, 20, 20, y_train.shape[1]]

    run_tuples = [(0.000398,    32,     [30, 30, 30, y_train.shape[1]],     0),
                  (0.000398,     4096,  [30, 30, 30, y_train.shape[1]]     , 0),
                  (0.000398,     512,   [30, 30, y_train.shape[1]]      , 0),
                  (0.00015,     512,    [30, 30, 30, y_train.shape[1]]      , 0),
                  (0.000398,     512,   [30, 30, 30, y_train.shape[1]]      , 0),
                  (0.000875,     512,   [30, 30, 30, y_train.shape[1]]      , 0),
                  (0.00015,     512,    [30, 30, 30, y_train.shape[1]]      , 0.0945),
                  (0.000398,     512,   [30, 30, 30, y_train.shape[1]]      , 0.0945),
                  (0.000875,     512,   [30, 30, 30, y_train.shape[1]]      , 0.0945),
                  (0.00015,     512,    [30, 30, 30, y_train.shape[1]]      , 0.001456),
                  (0.000398,     512,   [30, 30, 30, y_train.shape[1]]      , 0.001456),
                  (0.000875,     512,   [30, 30, 30, y_train.shape[1]]      , 0.001456),
                  (0.00015,     512,    [40, 40, 30, 20, y_train.shape[1]]      , 0),
                  (0.000398,     512,   [40, 40, 30, 20, y_train.shape[1]]      , 0),
                  (0.000875,     512,   [40, 40, 30, 20, y_train.shape[1]]      , 0),
                  (0.00015,     512,    [40, 40, 30, 20, y_train.shape[1]]      , 0.0945),
                  (0.000398,     512,   [40, 40, 30, 20, y_train.shape[1]]      , 0.0945),
                  (0.000875,     512,   [40, 40, 30, 20, y_train.shape[1]]      , 0.0945),
                  (0.00015,     512,    [40, 40, 30, 20, y_train.shape[1]]      , 0.001456),
                  (0.000398,     512,   [40, 40, 30, 20, y_train.shape[1]]      , 0.001456),
                  (0.000875,     512,   [40, 40, 30, 20, y_train.shape[1]]      , 0.001456)
                ]
    lr, batch_size, architecture, weight_decay = run_tuples[counter]


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

    save_folder_path = neural_net.save_run(save_runs_folder_path=save_runs_folder_path,
                                            run_group_name=f'{activation_function},ep={epochs},bs={batch_size},lr={lr},wd={weight_decay}',
                                            author=author, data_file_name='', lr=lr, batch_size=batch_size,
                                            epochs=epochs,
                                            num_samples=X_train.shape[0], description=description)

    neural_net.evaluate_model(X_train, y_train, X_val, y_val, save_folder_path / f'{activation_function},ep={epochs},bs={batch_size},lr={lr},wd={weight_decay}_ metrics_plot.png')


    return [activation_function, epochs, batch_size, lr, weight_decay], np.array(neural_net.f1_score_hist[-7:]).mean(), np.array(neural_net.f1_score_val_hist[-7:]).mean()
    # return [activation_function, epochs, batch_size, lr, weight_decay], min(neural_net.f1_score_hist[-1]), min(neural_net.f1_score_val_hist[-1])


def random_search_multipro(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description):
    # num_iterations = 3
    num_simultaneous_processes = 7

    with multiprocessing.Pool(num_simultaneous_processes) as pool:
        # res = pool.starmap(inner, [(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description)] * num_iterations )
        res = pool.starmap(inner, [(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description, x) for x in range(21)]) # + # )  
            # [(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description, 1)]) #  + 
        #  [(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description, 2)])

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



def random_search(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description):
    num_iterations = 3
    num_simultaneous_processes = 6

    counter = 1
    random.seed() # we want different random values in each process

    # define range of parameters
    lr = (np.power(10, random.uniform(-4, -3))).round(6)
    weight_decay = (random.choices([0, np.power(10, random.uniform(-3, -2)).round(6)], [0.25, 0.75]))[0]
    batch_size = random.choice([64, 128, 256, 512])

    lr = 0.001
    weight_decay = 0

    batch_sizes = [10000, 50000, 5000]
    batch_size = batch_sizes[counter]

    epochs = 2
    activation_functions = ['relu', 'leaky_relu']
    activation_function = random.choice(activation_functions)

    architecture = [40, 40, 20, 20, y_train.shape[1]]

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

    save_folder_path = neural_net.save_run(save_runs_folder_path=save_runs_folder_path,
                                            run_group_name=f'{activation_function},ep={epochs},bs={batch_size},lr={lr},wd={weight_decay}',
                                            author=author, data_file_name='', lr=lr, batch_size=batch_size,
                                            epochs=epochs,
                                            num_samples=X_train.shape[0], description=description)

    neural_net.evaluate_model(X_train, y_train, X_val, y_val, save_folder_path / f'{activation_function},ep={epochs},bs={batch_size},lr={lr},wd={weight_decay}_ metrics_plot.png')


    x_axis, f1_train, f1_val = ( [activation_function, epochs, batch_size, lr, weight_decay], np.array(neural_net.f1_score_hist[-7:]).mean(), np.array(neural_net.f1_score_val_hist[-7:]).mean() )

    # with multiprocessing.Pool(num_simultaneous_processes) as pool:
    #     # res = pool.starmap(inner, [(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description)] * num_iterations )
    #     res = pool.starmap(inner, [(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description, 0)] +
    #      [(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description, 1)] + 
    #      [(X_train, y_train, X_val, y_val, scaler, save_runs_folder_path, author, description, 2)])

    # print('here')   
    # res = sorted(res)

    # x_axis, f1_train, f1_val = tuple(map(list, zip(*res)))


    # df minimal f1
    x_axis_df = pd.DataFrame([x_axis], columns= ['activation function', 'epochs', 'batch size', 'lr', 'weight decay'])
    f1_train_df = pd.DataFrame([f1_train], columns=['min_f1_train'])
    f1_val_df = pd.DataFrame([f1_val], columns=['min_f1_val'])
    df = pd.concat(objs=[x_axis_df, f1_train_df, f1_val_df], axis=1)
    df.to_csv(save_runs_folder_path / 'min_f1_score.csv')

    # fig minimal f1
    fig, ax = plt.subplots(figsize=(60,40))

    x = np.arange(len([f1_train]))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(x - width/2, [f1_train], width, label='f1_train')
    rects2 = ax.bar(x + width/2, [f1_val], width, label='f1_val')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    # ax.scatter(list(range(len(f1_train))), f1_train, label='f1_train', s=200)
    # ax.scatter(list(range(len(f1_val))), f1_val, label='f1_val', s=200)
    ax.set_ylabel('f1')
    ax.set_title('f1 score')
    ax.set_xticks([])
    ax.table(cellText=list(map(list, zip(*[x_axis]))),
             rowLabels=['activation function', 'epochs', 'batch size', 'lr', 'weight decay'],
             loc='bottom')
    # ax.subplots_adjust(left=0.3, bottom=0.2)

    # plt.show()
    fig.savefig(save_runs_folder_path / 'min_f1_score.png')



if __name__ == '__main__':
    train_folder_path = Path(r'../../data\preprocessed_frames\new_window=10,cumsum=all\train')
    val_folder_path = Path(r'../../data\preprocessed_frames\new_window=10,cumsum=all\validation')

    X_train, y_train, scaler = generate_dataset(train_folder_path, select_mandatory_label=False)
    X_val, y_val = generate_dataset(val_folder_path, scaler, select_mandatory_label=False)

    random_search_multipro(X_train, y_train, X_val, y_val, scaler, Path(r'..\..\saved_runs\jonas_final_gross'),
        author='Jonas', description='window10_all, ohne Nina')

    # train_folder_path = Path(r'../../data\preprocessed_frames\window=8,cumsum=every_second\train')
    # val_folder_path = Path(r'../../data\preprocessed_frames\window=8,cumsum=every_second\validation')

    # X_train, y_train, scaler = generate_dataset(train_folder_path, select_mandatory_label=True)
    # X_val, y_val = generate_dataset(val_folder_path, scaler, select_mandatory_label=True)

    # random_search_multipro(X_train, y_train, X_val, y_val, scaler, Path(r'..\..\saved_runs\jonas_random_2\small,arch1_ep=80,win=8,cumsum=every_second'),
    #     author='Jonas', description='alle daten')
