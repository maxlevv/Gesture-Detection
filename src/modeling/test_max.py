import pandas as pd
import sys
from neural_network import FCNN
from feature_scaling import StandardScaler
sys.path.append(r"C:\Users\Max\PycharmProjects\ml_dev_repo\src")
from src.preprocessing import preprocessing_functions as pf
from src.evaluation import metrics


FILE_PATH = "../../data/preprocessed_frames/02-25_max_rotate_right_2022-02-28_18-52-43_preproc.csv"
frames_all = pd.read_csv(FILE_PATH)

X, y = pf.preprocessing_difference(frames, number_timestamps=5, number_shifts=5)

def test_neural_net(data, ground_truth):

    my_net = FCNN(4, [6, 4, 1], [1, 1 ,1], ["sigmoid", "sigmoid", "sigmoid"],
                loss_func="cross_entropy", scaler=StandardScaler())

    my_net.init_weights()
    my_net.check_and_correct_shapes(data, ground_truth)
    my_net.check_forward_prop_requirements(X)
    my_net.fit(data, ground_truth, 0.01, 10000, batch_size=data.shape[0])
    my_net.plot_stats()

    return my_net

#test_net = test_neural_net(X, y)
print(test_net.loss_hist[-1])


metrics.accuracy()