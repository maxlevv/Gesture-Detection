## Documentation for neural_network.py

The main functionalities of this neural network framework are built on the 'FCNN' class 
(fully connected neural network) and its methods. 



### Creating a neural_network.FCNN object

neural_net.FCNN(input_layer, layer_list, bias_list, activation_funcs, loss_func, lr, scaler, 
loss_hist, acc_hist, val_acc_hist, f1_score_hist, f1_score_val_hist)


**Parameters**:
- input_layer: *int*  
    - integer determining the number of input neurons in the network (without bias)
- layer_list: *list*
    - list of integers that determine the number of neurons in each layer (without bias),  
  the last integer defines the output layer size
    - length of *list* is the number of hidden layers plus one output layer
- bias_list: *list*
    - list consisting of *ones* and *zeros* to determine for each layer (except output layer)  
  to either use (=1) or not use (=0) a bias neuron for the layer (order according to layer order)
    - length has to be equal to length of *layer_list*
- activation_funcs: *list*, *{'sigmoid', 'relu', 'leaky_relu', 'softmax'}*
  - list consisting of strings to determine for each layer which activation function to use
  - *'softmax'* is only to be used for the last layer
  - length has to be equal to length of *layer_list*
- loss_func: *string*, *{'mse', 'cross_entropy', 'categorical_cross_entropy'}*
  - loss function
- lr: *float, optional*
    - the learning rate used in gradient descent
- scaler: *class object, optional*
  - instance of a class with methods fit(), transform() etc.
- loss_hist: *list, optional*
- acc_hist: *list, optional*
- val_acc_hist: *list, optional*
- f1_score_hist: *list, optional*
- f1_score_val_hist : *list, optional*  

**Initialising methods:**  
These methods are called every time a new FCNN object is created.
- FCNN.check_for_init_inconsistencies(activation_funcs, loss_func)
    - checks for invalid combinations of parameter choices
- FCNN._init_activation_funcs(activation_funcs)
    - initialises activations functions and their derivatives 
- FCNN._init_loss_func(loss_func)
    - initialises loss function and its derivative


### Training a model
Once an object is created and the data is prep

FCNN.init_weights(W)  
Initialises the weight matrices between each layer randomly (if W is not passed), according to the layers, bias and activation functions used.
- W: *list, optional*
    - list of weight matrices, size according to layers
    - mainly used when loading an already trained network  
  

Note that the following FCNN.fit method **does not scale** the data that is passed to it.  
Therefore, the scaling process has to be performed prior to using this method and outside the neural net framework.  

FCNN.fit(X, Y_g, lr, epochs, batch_size, optimizer, weight_decay, lambd, X_val, Y_g_val)  
Trains the model with the given data and parameters.  

- X: *np.array*
    - input data 
    - each row corresponds to one sample and each column corresponds to one feature
    - X.shape[1] must coincide with the length of the input layer
- Y_g: *np.array*
   - ground truth data 
   - each row must consist of the ground truth value (potentially one hot encoded) for the corresponding input sample
   - Y_g.shape[1] must coincide with the length of the output layer
- lr: *float*
   - learning rate used for weight updating
- epochs: *int*
   - number of epochs 
- batch_size: *int*
   - size of the mini batch that the weight updating is performed on
- optimizer: *str, optional, {'adam', 'sgd'}*
   - default is 'adam'
   - choose between the adam optimizer and classic stochastic gradient descent
- weight_decay: *float, optional*
   - default is 0
   - regularization parameter used together with the adam optimizer
- lambd: *float, optional* 
   - default is 0
   - L2 regularization parameter used with sgd optimizer
- X_val: *np.array, optional*
   - default is None
   - validation input data for tracking metrics such as loss, accuracy etc. in each epoch
   - X_val.shape[1] must coincide with X.shape[1]
- Y_g_val: *np.array, optional*
   - default is None
   - validation ground truth data for tracking metrics such as loss, accuracy etc. in each epoch
   - Y_g_val.shape[1] must coincide with Y_g.shape[1]  

After setting all parameters accordingly, the FCNN.fit method itself once per epoch calls two other methods. 
Those are FCNN.train and FCNN.track_epoch.  
The train method assembles the batches randomly and then performs forward propagation, backward propagation and weight updating for each batch.  
The method track_epoch tracks the metrics for both the training and validation data (if passed to FCNN.fit).

FCNN.train(X, Y_g, batch_size, optimizer)  

- X: *np.array*
  - input data 
- Y: *np.array*
  - ground truth data
- batch_size: *int*
- optimizer: *string, optional*
  - default is adam  

FCNN.track_epoch(X, Y_g, X_val, Y_g_val)  
Saves the loss, accuracy, and f1 scores in self.loss_hist, self.acc_hist and self.f1_score_hist.
- X: *np.array*
- Y_g: *np.array*
- X_val: *np.array, optional*
- Y_g_val: *np.array, optional*

FCNN.forward_prop(X)  
- Calls the methods self.check_forward_prop_requirements and self.forward_it,  
the return of self.forward_it is saved in self.O and self.Z

FCNN.check_forward_prop_requirements(X)  
- Checks if the weights in self.W are initialized and fit the architecture  
and if the data input has the correct shape

FCNN.forward_it(X)
- Forward propagates the input data through the network and returns two lists  
one containing all values for every layer before the activation function is applied (Z_list)  
the other one containing all values for every layer after the activation function is applied (O_list)

FCNN.backprop(Y)  
- performs backpropagation 


FCNN.update_weights(optimizer, batch_size)  
- Updates the weights that are stored in self.W with the computed error stored in self.dW and all other 
parameters such as self.lr and optionally the regularization parameter self.lambd or self.weight_decay


### 

### Saving and loading a model




