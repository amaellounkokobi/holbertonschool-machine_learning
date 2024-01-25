#!/usr/bin/env python3
"""
This module contains :
A function that conducts forward propagation using Dropout


Function:
   def dropout_forward_prop(X, weights, L, keep_prob):
"""
import numpy as np

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

    cache['A0'] = X

    for l_n in range(1, L):
            n_W = 'W{0}'.format(l_n)
            n_b = 'b{0}'.format(l_n)
            p_A = 'A{0}'.format(l_n - 1)

            val_W = weights[n_W]
            val_b = weights[n_b]
            val_A = cache[p_A]

            activation = np.matmul(val_W, val_A) + val_b

            A = np.tanh(activation)

            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = (A * D) / keep_prob

            cache['A{0}'.format(l_n)] = A
            cache['D{0}'.format(l_n)] = D * 1

    return cache
