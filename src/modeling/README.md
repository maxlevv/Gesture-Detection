## Introduction to Neural Net Framework

In the following, the neural net framework is roughly described. Further information can be found in the documentation `Neural-Net-Docu.md`.

The main functionalities of this neural network framework are built on the 'FCNN' class 
(fully connected neural network) and its methods. This class provides the core functionality for working with FCNNs. 

The class allows for constructing FCNNS of arbitrary (of course memory-bounded) numbers of layers and neurons per layer, respectively. The activation functions sigmoid, relu, leaky relu and softmax are implemented, and the loss functions mse, cross entropy and categorical cross entropy. The FCNN can be saved to and loaded from disc, and forward and backward propagation are implemented. 
