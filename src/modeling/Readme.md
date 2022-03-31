## Documentation for neural_network.py

The main functionalities of this neural network framework are built on the 'FCNN' class 
(fully connected neural network) and its methods. 



### Creating a neural_network.FCNN object

neural_net.FCNN(input_layer, layer_list, bias_list, activation_funcs, loss_func, lr, scaler, 
loss_hist, acc_hist, val_acc_hist, f1_score_hist, f1_score_val_hist, calc_metrics_func, evaluate_model_func)


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
  - can be passed to save the values of the scaling process but won't be applied within the class
- loss_hist: *list, optional*
  - used when loading an existing network
- acc_hist: *list, optional*
  - used when loading an existing network
- val_acc_hist: *list, optional*
  - used when loading an existing network
- f1_score_hist: *list, optional*
  - used when loading an existing network
- f1_score_val_hist : *list, optional*  
  - used when loading an existing network
- calc_metrics_func: *Callable*
  - call a function to calculate metrics -> call via FCNN.calc_metrics
- evaluate_model_func: *Callable*
  - call a function to evaluate the model -> call via FCNN.evaluate_model

**Initialising methods:**  
These methods are called every time a new FCNN object is created.
- FCNN.check_for_init_inconsistencies(activation_funcs, loss_func)
    - checks for invalid combinations of parameter choices
- FCNN._init_activation_funcs(activation_funcs)
    - initialises activations functions and their derivatives 
- FCNN._init_loss_func(loss_func)
    - initialises loss function and its derivative


### Training a model
To train a model firstly the weights need to be initialized and then the fit method can be called by passing all  
desired parameters to it.

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
- Create batch and call the methods forward_prop, backprop and update_weights

FCNN.track_epoch(X, Y_g, X_val(optional), Y_g_val(optional))  
- Saves the current loss, accuracy, and f1 scores in self.loss_hist, self.acc_hist and self.f1_score_hist  
  (self.val_acc_hist, self.f1_score_val_hist respectively).

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
- Performs backpropagation by iterating over every layer and computing the error term dW for each layer.
In each iteration the error gets multiplied with the activated output self.O of the current layer to then obtain the  
gradient belonging to this layer as a mean taken over all samples considered in the batch. The gradient information  
is then stored in self.dW, a list containing the gradients for each layer.

FCNN.update_weights(optimizer, batch_size)  
- Updates the weights that are stored in self.W with the computed gradient stored in self.dW and all other 
parameters such as self.lr and optionally the regularization parameter self.lambd or self.weight_decay

FCNN.calc_loss(Y_g)
- Calculates the current loss with the loss function stored in self.loss_func

FCNN.calc_stats(Y, Y_g, Y_val(optional), Y_g_val(optional)):
- Only used for multiclass classification so Y_g has to be binary. Calculates the current accuracy and f1 score

FCNN.apply_label_weighting(gradient_batch_tensor, Y_g, balanc_index):
- gradient_batch_tensor: *np.array*
  - Gradient vector in a batch where the weighting is applied
- Y_g: *np.array*
  - The ground truth vector containing one hot encoded binary data
- balanc_index: *int*
  - The integer position in Y_g of the label that the weighting should be applied to

FCNN.calc_gradient_label_weights(Y_g, balanc_index)
- Calculates the weights depending on the number of occurrences of the label in Y_g determined  
integerwise by balanc_index compared to the total number of labels

FCNN.check_and_correct_shapes(X, Y_g):
- Called in FCNN.fit to check the shapes of the input data and to potentially correct the shapes

FCNN.clear_attributes()
- Resets all attributes, not used within the class but when performing multiple train runs 

FCNN.clear_data_specific_parameters()
- Resets data specific parameters, used every time a new mini batch gets propagated through the network

FCNN._init_adam_parameters()
- Initializes the adam parameters to fixed values when the adam optimizer is selected in the FCNN.fit

### Saving and loading a model
To save or load a model pass a Path variable that contains the directory for the meta data to be saved at or loaded from.

FCNN.save_run(save_runs_folder_path, run_group_name:str, author:str, 
                 data_file_name:str, lr:float, batch_size:int, epochs:int, num_samples:int, 
                 description:str=None, name:str=None)
- save_runs_folder_path: *Path*
  - the directory path where the model should be saved at  

- run_group_name, author, data_file_name, lr, batch_size, epochs, num_samples, description(optional), name(optional): *str*
  - setting some attributes to identify a run  

FCNN.load_run(from_folder_path)   
Classmethod returning a FCNN object  
- from_folder_path: *Path*
  - the directory the model is loaded from




