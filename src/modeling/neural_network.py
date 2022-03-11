import numpy as np
from typing import List, Dict, Tuple, Callable
import matplotlib.pyplot as plt
from pathlib import Path
from gradient_checking import check_gradient, check_gradient_of_neural_net
from tqdm import tqdm
from loss_functions import cross_entropy, d_cross_entropy, categorical_cross_entropy, d_categorical_cross_entropy_with_softmax
from activation_functions import softmax, sigmoid, sigmoid_d, relu, relu_d, leaky_relu, leaky_relu_d
from feature_scaling import StandardScaler
from save_and_load import save_run, load_run
from evaluation.metrics import calc_metrics, accuracy, f1_score, calc_confusion_matrix
from evaluation.evaluate import evaluate_neural_net 
from helper import softmax2one_hot



class FCNN:

    def __init__(self, input_size: int, layer_list: List[float], bias_list: List[int], activation_funcs: List, loss_func: str, 
                       lr=None, scaler=None, loss_hist=[], acc_hist=[], val_acc_hist=[], f1_score_hist=[], f1_score_val_hist=[]) -> None:
        
        self.n = [([input_size] + layer_list)[i] + bias for i, bias in enumerate(bias_list + [0])]
        self.bias_list = bias_list
        self.layer_list = layer_list
        self.input_size = input_size

        self.W = list()     # weights marixes
        self.dW = list()    # derivative of the whole problem with respect to W structured like W
        # outputs of each layer (not sure if this will be used)
        self.O = None       # list of outputs of each layer, set in forward prop and used in backprop
        self.Z = None       # matrix products of each layer without activation function set in forward prop used for backprop
        self.loss = None    # loss for specific data
        self.lr = lr        # learning rate
        self.activation_funcs = list()      # activation function of each layer
        self.d_activation_funcs = list()    # derivative of activation function of each layer
        self.activation_func_string_list = activation_funcs # is needed for checking backprop if softmax is last act func
        self.loss_func = None
        self.d_loss_func = None     # derivative of loss function
        self.loss_func_str = loss_func
        # self.layer_output_funcs = []    # functions to calcualte the output of each layer
        self.scaler = scaler     # instance of a class with methods fit(), transform() like in notebook 5
        self.weight_decay = None
        self.lambd = None

        self.adam_moment1 = None
        self.adam_moment2 = None
        self.adam_iteration_counter = None
        self.adam_beta1 = None
        self.adam_beta2 = None
        self.adam_eps = None

        self.loss_hist = loss_hist # save losses through a training run
        self.acc_hist = acc_hist  # save accuracy through a training run
        self.val_acc_hist = val_acc_hist    # validation accuracy computed by stats
        self.f1_score_hist = f1_score_hist
        self.f1_score_val_hist = f1_score_val_hist

        self._activation_func_dict = {
            'sigmoid': sigmoid,
            'sigmoid_d': sigmoid_d,
            'softmax' : softmax,
            'relu': relu,
            'relu_d': relu_d,
            'leaky_relu': leaky_relu,
            'leaky_relu_d': leaky_relu_d,
        }

        self._loss_func_dict = {
            'cross_entropy': cross_entropy,
            'cross_entropy_d': d_cross_entropy,
            'categorical_cross_entropy': categorical_cross_entropy,
        }

        # check for inconsistencies
        self.check_for_init_inconsistencies(activation_funcs, loss_func)

        self._init_activation_funcs(activation_funcs)
        self._init_loss_func(loss_func)


    def check_for_init_inconsistencies(self, activation_funcs:List[str], loss_func:str):
        if not len(activation_funcs) == len(self.layer_list):
            raise Exception(f'Not enough or too many activation functions specified! ' + \
                f'{len(activation_funcs)} given, {len(self.layer_list)} required.')
        if not len(self.bias_list) == len(self.layer_list):
            raise Exception(f'Wrong bias list size! {len(self.bias_list)} given, {len(self.layer_list)} expected')
        if (activation_funcs[-1] == 'softmax') ^ (loss_func == 'categorical_cross_entropy'):
            raise Exception(f'Softmax and Categorical crossentropy only work together')
        if self.layer_list[-1] > 1 and not loss_func == 'categorical_cross_entropy':
            raise Exception(f'Multiple output neurons can only be handeled by categorical_cross_entropy loss')

    def _init_activation_funcs(self, activation_funcs:List[str]):
        for act_string in activation_funcs:
            self.activation_funcs.append(
                self._activation_func_dict[act_string])
            if not act_string == 'softmax':
                self.d_activation_funcs.append(
                    self._activation_func_dict[act_string + '_d'])

    def _init_loss_func(self, loss_func_str: str):
        self.loss_func = self._loss_func_dict[loss_func_str]
        if not loss_func_str == 'categorical_cross_entropy':
            self.d_loss_func = self._loss_func_dict[loss_func_str + '_d']
    
    def _init_adam_parameters(self):
        if len(self.W) == 0:
            raise RuntimeError('Weights are not initialized!')
        
        self.adam_iteration_counter = 0
        self.adam_moment1 = [np.zeros_like(W) for W in self.W]
        self.adam_moment2 = [np.zeros_like(W) for W in self.W]
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8
    

    def init_weights(self, W:np.array=None):

        if W:
            self.W = W
        else:
            np.random.seed(0)
            curr_layer_size = self.input_size
            self.W = []
            for next_layer_size, bias in zip(self.layer_list, self.bias_list):
                # have the weights distributed from -0.5 to 0.5
                W = np.random.rand(next_layer_size, curr_layer_size + bias) - 0.5
                self.W.append(W)
                curr_layer_size = next_layer_size

    def clear_data_specific_parameters(self):
        """
        reset the parameters which are used in each iteration 
        """
        self.dW = []   
        self.O = None
        self.Z = None       
        self.loss = None


    def check_and_correct_shapes(self, X:np.array, Y_g:np.array):
        self.check_forward_prop_requirements(X)
        if len(Y_g.shape) == 1:
            Y_g = Y_g.reshape(-1, 1)
        if not Y_g.shape[1] == self.layer_list[-1]:
            raise Exception(f'Wrong Y_g shape: Y_g {Y_g.shape}, last layer ouput {self.layer_list[-1]}')
        if not X.shape[0] == Y_g.shape[0]:
            raise Exception(f'X shape does not match Y_g shape! X {X.shape[0]}, Y_g {Y_g.shape[0]}')
        return Y_g


    def check_forward_prop_requirements(self, X:np.array):
        """
        check the shapes of the input
        """
        if not self.W:
            raise Exception('W needs to be initialized first')
        if not self.input_size == X.shape[1]:
            raise Exception('Input shape is not correct')
        for i, W in enumerate(self.W):
            if not (W.shape[1] == self.n[i] and W.shape[0] == self.layer_list[i]):
                raise Exception(f'W[{i}] does not match the specified architechture')
        

    def forward_it(self, X:np.array):
        """
        evtl kann mann hier mit dynamic initialization und exec() befehl iterativ einen String mit python code aufbauen sodass man 
        wie eigentlich gewünscht die auswertung ohne schleife in einer großen verketteten function durchführen kann

        diese implementierung wie jetzt ist wahrscheinlich besser wenn man die zwischenergebnisse speichern will, weil man sie noch für 
        die backprop braucht
        """
        o_list = []     # list of outputs of the layers
        z_list = []     # list of the matrix product without activation function
        o = None
        for layer in range(len(self.W)):
            if layer == 0:
                o = X.T     # (n_0 x d)
                if self.bias_list[0] == 1:
                    o = np.r_[np.ones(o.shape[1]).reshape(1, -1), o]    
                o_list.append(o)
            
            z = self.W[layer] @ o   # (n_(layer+1) x d)
            o = self.activation_funcs[layer](z)

            if not layer+1 == len(self.W):
                if self.bias_list[layer+1] == 1:
                    o = np.r_[np.ones(o.shape[1]).reshape(1, -1), o]
            
            o_list.append(o)
            z_list.append(z)
        return o_list, z_list


    def forward_prop(self, X:np.array):
        
        self.check_forward_prop_requirements(X)

        # return self.forward_rek(X, layer=len(self.size['layer_list'])) old
        self.O, self.Z = self.forward_it(X)

    def calc_loss(self, Y_g):
        if self.lambd is None:
            self.loss = self.loss_func(self.O[-1].T, Y_g)
        else:
            sum = 0
            for matrix in self.W:
                sum = sum + np.sum(np.square(matrix))
            self.loss = self.loss_func(self.O[-1].T, Y_g) + (self.lambd / (2 * np.shape(self.O[-1])[1]) * sum)


    def backprop(self, Y_g:np.array):
        """
        y_g: y groud truth
        """
        if len(Y_g.shape) == 1:
            Y_g = Y_g.reshape(-1, 1)
        y = self.O[-1].T
        if not Y_g.shape == y.shape:
            raise Exception(f'Y shapes differ: Y_g {Y_g.shape}, Y {y.shape}')

        dW = None    # save in here what can be reused for the next gradient level

        for i in range(len(self.W)-1, -1, -1):  # loop for deriving each dW derivative

            if i == len(self.W)-1:
                # hier bias[i] entscheident, for this derivative whether there is a bias in the previous layer or not does 
                # not change the computations and everything works itself out

                # for categorical_cross_entropy and softmax a predefined derivative is used
                # in othercases the derivative of the loss and the last layer is computed individually
                if self.activation_func_string_list[-1] == 'softmax':
                    # in this case loss is also categorical crossentropy (this has been checked in init) 
                    dW = d_categorical_cross_entropy_with_softmax(self.Z[i], Y_g)
                else:
                    d_J = self.d_loss_func(y, Y_g)  # (batchsize b x output size)
                    dW = d_J    # save in here what can be reused for the next gradient level

                    # (b x n_(i+1))  (Z[i]: (n_(i+1) x b))
                    dW = dW * self.d_activation_funcs[i](self.Z[i].T)

                self.dW.insert(0,
                               np.mean(     # changed from mean
                                   dW.T[:, np.newaxis, :] * self.O[i][np.newaxis, :, :],
                                   axis=2))  # (n_(i+1) x n_i x b)
            else:
                if self.bias_list[i+1]:
                    # removing the first column of the weight matrix is due to bias
                    dW = dW @ self.W[i+1][:, 1:] * self.d_activation_funcs[i](self.Z[i].T)
                else:
                    # (b x n_(i+1))  (Z[i]: (n_(i+1) x b))
                    dW = dW @ self.W[i+1] * self.d_activation_funcs[i](self.Z[i].T)

                # performing the last muliplication with the O values and summing (was averaging in a earlier verison) over the gradients 
                # in the batch
                self.dW.insert(0,
                               np.mean(     # changed from mean
                                   dW.T[:, np.newaxis, :] * self.O[i][np.newaxis, :, :],
                                   axis=2))


    def update_weights(self, optimizer, batch_size):

        for i in range(len(self.W)):
            if optimizer == 'sgd':
                if self.lambd == 0:
                    self.W[i] = self.W[i] - self.lr * self.dW[i]
                else:
                    self.W[i] = self.W[i] - self.lr * (self.dW[i] + self.lambd / batch_size * self.W[i])

            elif optimizer == 'adam':
                # src: https://arxiv.org/pdf/1412.6980.pdf

                self.adam_iteration_counter += 1

                g = self.dW[i]

                self.adam_moment1[i] = self.adam_beta1 * self.adam_moment1[i] + (1 - self.adam_beta1) * g
                self.adam_moment2[i] = self.adam_beta2 * self.adam_moment2[i] + (1 - self.adam_beta2) * np.power(g, 2)

                m_hat = np.divide(self.adam_moment1[i], 1 - np.power(self.adam_beta1, self.adam_iteration_counter))
                v_hat = np.divide(self.adam_moment2[i], 1 - np.power(self.adam_beta2, self.adam_iteration_counter))

                if self.weight_decay == 0:
                    self.W[i] = self.W[i] - self.lr * np.divide(m_hat, np.sqrt(v_hat) + self.adam_eps)
                else:
                    # src: https://openreview.net/pdf?id=rk6qdGgCZ

                    self.W[i] = self.W[i] - self.lr * (np.divide(m_hat, np.sqrt(v_hat) + self.adam_eps) + self.weight_decay * self.W[i])

            else:
                raise RuntimeError(f'Optimizer was not specified correctly: {optimizer}')

    
    def train(self, X:np.array, Y_g:np.array, batch_size:int, optimizer: str = 'adam', X_val: np.array = None, Y_g_val: np.array = None):
        # TODO: I dont know if it is necessary to shuffle new in every epoch or if it can be done once for every epoch
        shuffled_indices = np.random.choice(X.shape[0], X.shape[0], replace=False)
        remaining_indices = shuffled_indices.copy()

        if len(Y_g.shape) == 1:
            Y_g = Y_g.reshape(-1, 1)

        while len(remaining_indices > 0):

            # get the indices for the next batch with batch_size or just the last few in the last batch
            end_batch_index = np.min([len(remaining_indices), batch_size])
            batch_indices = remaining_indices[:end_batch_index]
            remaining_indices = remaining_indices[end_batch_index:]

            self.clear_data_specific_parameters()
            self.forward_prop(X[batch_indices, :])
            # self.calc_loss(Y_g[batch_indices])
            self.backprop(Y_g[batch_indices])
            self.update_weights(optimizer, end_batch_index)

    def track_epoch(self, X:np.array, Y_g:np.array, X_val: np.array = None, Y_g_val: np.array = None):
        # calc loss over whole data
        self.clear_data_specific_parameters()
        self.forward_prop(X)
        self.calc_loss(Y_g)
        self.loss_hist.append(self.loss)
        Y = self.O[-1].T
        if not X_val is None: 
            self.clear_data_specific_parameters()
            self.forward_prop(X_val)
            self.calc_loss(Y_g_val)
            Y_val = self.O[-1].T
            acc, val_acc, f1_scores, f1_scores_val = self.calc_stats(Y, Y_g, Y_val, Y_g_val)

            self.val_acc_hist.append(val_acc)
            self.f1_score_val_hist.append(f1_scores_val)
        else:
            acc, f1_scores = self.calc_stats(Y, Y_g)

        self.acc_hist.append(acc)
        self.f1_score_hist.append(f1_scores)


    def fit(self, X:np.array, Y_g:np.array, lr:float, epochs:int, batch_size:int, optimizer: str = 'adam',  X_val: np.array = None, Y_g_val: np.array = None):
        Y_g = self.check_and_correct_shapes(X, Y_g)
        self.lr = lr

        # scaling the data with the specified scaler instance
        # TODO: does y_d need to be scaled here?
        
        # this sould be done elsewhere, as only train data goes into fit()
        # self.scaler.fit(X)
        # X = self.scaler.transform(X)

        if optimizer == 'sgd':
            if weight_decay != 0:
                raise Exception(f'sgd and weight decay dont go together')
            else:
                self.lambd = lambd

        if optimizer == 'adam':
            if lambd != 0:
                raise Exception(f'adam and L2 regularization dont go together')
            else:
                self._init_adam_parameters()
                self.weight_decay = weight_decay

        for epoch in tqdm(range(epochs)):
            self.train(X, Y_g, batch_size, optimizer, X_val, Y_g_val)
            self.track_epoch(X, Y_g, X_val, Y_g_val)
    

    def calc_stats(self, Y:np.array, Y_g:np.array, Y_val:np.array = None, Y_g_val:np.array = None):
        """
        for classification problems -> Y_g needs to binary
        """
        # if Y is None:
        #     self.forward_prop(X)
        #     Y = self.O[-1].T
        if len(Y_g.shape) == 1:
            Y_g = Y_g.reshape(-1, 1)
        if not Y_g.shape == Y.shape:
            raise Exception('Y_g and Y shapes do not match')

        Y_one_hot = softmax2one_hot(Y)
        acc = (Y_g == Y_one_hot).all(axis=1).sum() / Y.shape[0]
        f1_scores = [None] * Y.shape[1]
        conf_matrix = calc_confusion_matrix(Y_one_hot, Y_g)
        for klasse in range(Y.shape[1]):
            f1_scores[klasse] = f1_score(conf_matrix, klasse)

        if not Y_val is None:
            Y_val_one_hot = softmax2one_hot(Y_val)

            val_acc = (Y_g_val == Y_val_one_hot).all(axis=1).sum() / Y_val.shape[0] 

            f1_scores_val = [None] * Y.shape[1]
            conf_matrix = calc_confusion_matrix(Y_val_one_hot, Y_g_val)
            for klasse in range(Y.shape[1]):
                f1_scores_val[klasse] = f1_score(conf_matrix, klasse)

            return acc, val_acc, f1_scores, f1_scores_val
        
        return acc, f1_scores

    def plot_stats(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].plot(self.loss_hist)
        axes[1].plot(self.acc_hist)
        # fig.show()
        return fig


    def calc_metrics(self, X: np.array, y_g: np.array):
        self.forward_prop(X)
        y = self.O[-1].T
        calc_metrics(y, y_g)

    def evaluate_model(self, X_train, y_train, X_val, y_val):
        evaluate_neural_net(self, X_train, y_train, X_val, y_val)
    

    def save_run(self, save_runs_folder_path:Path, run_group_name:str, author:str, 
                 data_file_name:str, lr:float, batch_size:int, epochs:int, num_samples:int, 
                 description:str=None, name:str=None):
        save_run(save_runs_folder_path, run_group_name, self, author, data_file_name, lr, batch_size, epochs, num_samples, description, name)
    
    @classmethod
    def load_run(cls, from_folder_path:Path) -> 'FCNN':
        # load an entire run with weights and all the meta data
        W, meta_data = load_run(from_folder_path)
        new_net = cls(meta_data.architecture[0], meta_data.architecture[1:], meta_data.bias_list, 
            meta_data.activation_functions, meta_data.loss_function, meta_data.lr, 
            StandardScaler.from_dict(meta_data.scaler), meta_data.loss_hist, meta_data.acc_hist, 
            meta_data.val_acc_hist, meta_data.f1_score_hist, meta_data.f1_score_val_hist)
        new_net.W = W
        print(f'Loaded run with epochs {meta_data.epochs}, batch_size {meta_data.batch_size}, num_samples {meta_data.num_samples}')
        return new_net
    

    def load_weights(self, from_folder_path:Path):
        # load all the weights without the meta data
        W, meta_data = load_run(from_folder_path)
        self.W = W

