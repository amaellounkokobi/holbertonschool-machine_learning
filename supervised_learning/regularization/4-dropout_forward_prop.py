#!/usr/bin/env python3
"""
This module contains :
A function that conducts forward propagation using Dropout


Function:
   def dropout_forward_prop(X, weights, L, keep_prob):
"""
import numpy as np

def softmax(x):
    return (np.exp(x) / np.sum(np.exp(x))) 


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Function that conducts forward propagation using Dropout:

    Args:
       X: is a numpy.ndarray of shape (nx, m) containing the
       input data for the network
          nx: is the number of input features
          m: is the number of data points
       weights: is a dictionary of the weights and biases of the neural network
       L: the number of layers in the network
       keep_prob: is the probability that a node will be kept

    Returns:
       a dictionary containing the outputs of each
       layer and the dropout mask used on each layer
       (see example for format)
    """
    cache = {}

    cache['A0'] = A = X

    for l_n in range(1, L):
        # set parameters W b A
        n_W = 'W{0}'.format(l_n)
        n_b = 'b{0}'.format(l_n)

        W = weights[n_W]
        b = weights[n_b]

        # Calculate Z    
        Z = np.matmul(W, A) + b

        A = np.tanh(Z)
        D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
        A = A * D / keep_prob

        #register D and Activation
        cache['A{0}'.format(l_n)] = A
        cache['D{0}'.format(l_n)] = D * 1 

    # Last layer activation
    l_n += 1

    n_W = 'W{0}'.format(l_n)
    n_b = 'b{0}'.format(l_n)

    W = weights[n_W]
    b = weights[n_b]

    # Calculate Z
    Z = np.matmul(W, A) + b
    A = softmax(Z)
    cache['A{0}'.format(l_n)] = A
    
    return cache
