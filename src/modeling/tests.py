import numpy as np
from gradient_checking import check_gradient_of_neural_net
from pathlib import Path
from neural_network import FCNN
from save_and_load import save_run, load_run
from feature_scaling import StandardScaler
from metaData import MetaData

def test_neural_net():
    my_net = FCNN(4, [3, 2, 2], [1, 1, 1], ['relu']
                    * 2 + ['softmax'], loss_func='categorical_cross_entropy', scaler=StandardScaler())
    my_net.init_weights()
    
    X = np.array([[0.1, 0.2, 0.9, 0.9],
                  [0.2, 0.1, 0.8, 0.9],
                  [0.3, 0.2, 0.7, 0.9],
                  [0.7, 0.8, 0.1, 0.2],
                  [0.9, 0.7, 0.2, 0.3],
                  [0.8, 0.8, 0.1, 0.2],
                  ])
    
    Y_g = np.array([[0, 1],
                    [0, 1],
                    [0, 1],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    ], dtype=int)

    
    my_net.clear_data_specific_parameters()
    my_net.fit(X, Y_g, lr=0.01, epochs=800, batch_size=10, optimizer='adam')
    my_net.clear_data_specific_parameters()
    
    # my_net.calc_stats(X, Y_g)
    # my_net.plot_stats().show()
    
    grads_ok = check_gradient_of_neural_net(my_net, np.array([[0.1, 0.2, 0.9, 0.9]]), np.array([[0, 1]]))
    print("Grads_ok:", grads_ok)
    
    return my_net


def test_save_run():
    my_net = FCNN(4, [3, 2, 2], [1, 1, 1], ['sigmoid']
                    * 2 + ['softmax'], loss_func='categorical_cross_entropy', scaler=StandardScaler())
    my_net.init_weights()
    # my_net.scaler.fit(np.array([[1, 1], [2, 2]]))
    my_net.acc_hist.append(0.1)
    # meta_data = MetaData(my_net, 'Jonas', 'test_data.file', 0.001, 5, 100, 1000, description='just a test run')
    # save_meta_json_and_weights(my_net, meta_data, Path(r'src\test_folder'), "test_runs2")
    # W, meta_data_dict = load_run(Path(r'src\test_folder\test_runs2\2022-02-02_2_4-3-2-2'))

    save_run(Path(r'saved_runs'), 'test_runs1', my_net, 'Jonas', 'test_data.file', 0.001, 5, 100, 1000, description='just a test run')

def test_FCNN_save_run():
    my_net = FCNN(4, [3, 2, 2], [1, 1, 1], ['sigmoid']
                    * 2 + ['softmax'], loss_func='categorical_cross_entropy', scaler=StandardScaler())
    my_net.init_weights()
    # my_net.scaler.fit(np.array([[1, 1], [2, 2]]))
    my_net.acc_hist.append(0.2)
    # meta_data = MetaData(my_net, 'Jonas', 'test_data.file', 0.001, 5, 100, 1000, description='just a test run')
    # save_meta_json_and_weights(my_net, meta_data, Path(r'src\test_folder'), "test_runs2")
    # W, meta_data_dict = load_run(Path(r'src\test_folder\test_runs2\2022-02-02_2_4-3-2-2'))
    my_net.save_run(Path(r'saved_runs'), 'test_runs1', 'Jonas', 'test_data.file', 0.001, 5, 100, 1000, description='just a test run')
    
def test_FCNN_load_run():
    loaded_net = FCNN.load_run(Path(r'saved_runs\test_runs1\2022-02-06_0_4-3-2-2'))
    print('debug')


def test_construct_MetaData_instance_from_dict():
    W, meta_data = load_run(Path(r'saved_runs\test_runs1\2022-02-06_0_4-3-2-2'))
    print('debug')


if __name__ == '__main__':
    test_neural_net()
    # test_save_run()
    # test_construct_MetaData_instance_from_dict()
    # test_FCNN_save_run()
    # test_FCNN_load_run()
    print('done')