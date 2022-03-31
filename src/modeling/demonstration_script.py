from mnist import MNIST
from mnist_helper import mnist_downloader
from neural_network import FCNN
from feature_scaling import StandardScaler
from evaluation.metrics import calc_confusion_matrix
import matplotlib.pyplot as plt
from helper import one_hot_encoding


""" Demo script to show the basic usage of the framework implemented in the class FCNN using the mnist_helper data set"""

download_folder = "../../data/mnist_helper"
#mnist_downloader.download_and_unzip(download_folder)

mndata = MNIST('../../data/mnist', return_type="numpy")

images_train, labels_train = mndata.load_training()
images_validation, labels_validation = mndata.load_testing()

""" Perform one hot encoding """
y_g = one_hot_encoding(labels_train)
y_g_val = one_hot_encoding(labels_validation)

""" Scale the input data """
scaler = StandardScaler()
scaler.fit(images_train)
X_train = scaler.transform(images_train)
X_val = scaler.transform(images_validation)

""" Create FCNN object """
neural_net = FCNN(
        input_size=X_train.shape[1],
        layer_list=[30, y_g.shape[1]],
        bias_list=[1, 1],
        activation_funcs=['sigmoid'] * 1 + ['softmax'],
        loss_func='categorical_cross_entropy',
        scaler=scaler
    )

""" Train the model """
neural_net.init_weights()
neural_net.fit(X_train, y_g, lr=0.1, epochs=15, batch_size=32, optimizer='sgd',
               X_val=X_val, Y_g_val=y_g_val)

""" Get loss and accuracy from the training data which gets calculated in every epoch """
loss_hist = neural_net.loss_hist
acc_hist = neural_net.acc_hist

""" Since we also passed the validation data to the fit method we can also take a look at the accuracy for this data 
    The FCNN object also keeps track of the f1 scores for multiclassification tasks, but we won't consider them here """
val_acc_hist = neural_net.val_acc_hist

""" Visualize the histories """
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].plot(loss_hist)
axes[0].set_title("loss history train data")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("train loss")
axes[1].plot(acc_hist)
axes[1].set_title("accuracy history train data")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("train accuracy")
axes[2].plot(val_acc_hist)
axes[2].set_title("accuracy history validation data")
axes[2].set_xlabel("epoch")
axes[2].set_ylabel("validation accuracy")
fig.show()

""" Ultimately we calculate the confusion matrices for the train and validation data using the dedicated function from src.evaluation.metrics 
    Here we just print them out, the labeling would range from top to bottom and left to right starting with a 0 up to 9 in counting order """
neural_net.forward_prop(X_train)
h_train = neural_net.O[-1].T
conf_matrix_train = calc_confusion_matrix(h_train, y_g)
print(conf_matrix_train)

neural_net.forward_prop(X_val)
h_val = neural_net.O[-1].T
conf_matrix_val = calc_confusion_matrix(h_val, y_g_val)
print(conf_matrix_val)
