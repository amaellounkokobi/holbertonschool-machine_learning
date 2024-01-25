#!/usr/bin/env python3
"""
This module contains :
A function that conducts forward propagation using Dropout


Function:
   def dropout_forward_prop(X, weights, L, keep_prob):
"""
import numpy as np


def softmax(x):
    """
    Softmax function

    Args
       X
    """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


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
    A = X

    for lay in range(1, L):
        Z = np.matmul(
            weights['W{}'.format(lay)], A) + weights['b{}'.format(lay)]
        A = np.tanh(Z)

        D = np.random.rand(A.shape[0], A.shape[1])
        D = D < keep_prob

        A = A * D
        A = A / keep_prob

        cache['A{}'.format(lay)] = A
        cache['D{}'.format(lay)] = D * 1

    lay += 1

    Z = np.matmul(weights['W{}'.format(lay)], A) + weights['b{}'.format(lay)]
    A = softmax(Z)

    cache['A{}'.format(lay)] = A

    return cache
