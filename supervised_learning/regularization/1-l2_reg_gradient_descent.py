#!/usr/bin/env python3
"""
This module contains :
A function that updates the weights and biases of a neural network


Function:
   def l2_reg_gradient_descent(Y,
                               weights,
                               cache,
                               alpha,
                               lambtha,
                               L):
"""
import numpy as np


def l2_reg_gradient_descent(Y,
                            weights,
                            cache,
                            alpha,
                            lambtha,
                            L):
    """
    Function that updates the weights and biases of a neural network

    Args:
       Y: is a one-hot numpy.ndarray of shape (classes, m) that
       contains the correct labels for the data
          classes: is the number of classes
          m: is the number of data points

       weights: is a dictionary of the weights and biases
       of the neural network

       cache: is a dictionary of the outputs of each layer
       of the neural network

       alpha: is the learning rate

       lambtha: is the L2 regularization parameter

       L: is the number of layers of the network

    """
    N = Y.shape[1]
    c_A_n = 'A{}'.format(L)
    curr_A = cache[c_A_n]

    dZ_curr = curr_A - Y

    for l_n in range(L, 0, -1):

        c_W_n = 'W{}'.format(l_n)
        curr_W = weights[c_W_n]
        p_A_n = 'A{}'.format(l_n - 1)
        prev_A = cache[p_A_n]

        dW_curr = 1 / N * np.matmul(dZ_curr, prev_A.T) + (curr_W / N) * lambtha
        dB_curr = 1 / N * np.sum(dZ_curr, axis=1, keepdims=True)

        next_dtanh = 1 - (prev_A ** 2)

        dZ_curr = np.matmul(curr_W.T, dZ_curr) * next_dtanh

        c_b_n = 'b{}'.format(l_n)

        weights[c_W_n] = weights[c_W_n] - alpha * dW_curr
        weights[c_b_n] = weights[c_b_n] - alpha * dB_curr
