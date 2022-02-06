# src of theorie: http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
import numpy as np

def check_gradient(func, gradient, x_test):
    """[summary]

    Args:
        func (callable): function of which the gradient should be checked
        gradient (callable): gradient of func
        x_test (np.array): point at which the gradient should be tested
    """
    f_x = func(x_test)
    g_x = gradient(x_test).reshape(1, -1)
    n = g_x.shape[1]
    num_grad = np.zeros_like(g_x).astype(np.float32)
    eps = 1e-4
    for i in range(n):
        num_grad[0, i] = (func(x_test + eps * np.identity(n)[:, i].reshape(-1, 1)) - func(x_test - eps * np.identity(n)[:, i].reshape(-1, 1))) / (2 * eps)
    
    return np.allclose(num_grad.round(4), g_x.round(4)), num_grad, g_x


def get_loss_func_only_dependent_one_weight_matrix(net, example_X, example_Y_g, w_index):
    def J(w):
        # put in w as a vector
        net.W[w_index] = w.reshape(*(net.W[w_index].shape))
        x = example_X
        y_g = example_Y_g
        net.forward_prop(x)
        net.calc_loss(y_g)
        return net.loss
    return J


def get_loss_gradient_only_dependent_one_weight_matrix(net, example_X, example_Y_g, w_index):
    def d_J(w):
        net.W[w_index] = w.reshape(*(net.W[w_index].shape))
        x = example_X
        y_g = example_Y_g
        net.forward_prop(x)
        net.backprop(y_g)
        return net.dW[w_index].flatten().reshape(-1, 1)
    return d_J


def print_gradients_to_compare(num_grads, net_grads):
    for net_grad, num_grad in zip(net_grads, num_grads):
        print(f"######\n{np.c_[net_grad.reshape(-1, 1), num_grad.reshape(-1, 1)]}")
    

def check_gradient_of_neural_net(net, example_X, example_Y_g, verbose=False):
        """check all the gradient matrices of a given neural net class, currently only for output > 1

        Args:
            net (FCNN): FCNN instantance with initiated weights
            example_X (np.array(1, n_0)): one sample data
            example_Y_g (np.array(1, n_(-1) )): y_g ground truth to the sample datum example_X
        """
        grad_bools = []
        num_grads = []
        net_grads = []
        for w_index in range(len(net.W)):

            J = get_loss_func_only_dependent_one_weight_matrix(net, example_X, example_Y_g, w_index)

            d_J = get_loss_gradient_only_dependent_one_weight_matrix(net, example_X, example_Y_g, w_index)
    
            w = net.W[w_index].flatten().reshape(-1, 1)

            grad_bool, num_grad, net_grad = check_gradient(J, d_J, w)
            grad_bools.append(grad_bool)
            num_grads.append(num_grad)
            net_grads.append(net_grad)
        
        if verbose: print_gradients_to_compare(num_grads, net_grads)
        
        return np.array([grad_bool]).all()


def grad_check_test():
    func = lambda x: 2*x[0]
    g_x = lambda x: np.array([2, 0])
    x_test = np.array([5, 0])
    check_gradient(func, g_x, x_test)


if __name__ == '__main__':
    # grad_check_test()
    pass
