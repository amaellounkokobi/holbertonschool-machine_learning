#!/usr/bin/env python3
"""
This module contains :
a function def l2_reg that calculates the cost 
of a neural network with L2 regularization:


Function:
   def l2_reg_cost(cost, lambtha, weights, L, m):
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Function def l2_reg that calculates the cost                                                                                                                                                           
    of a neural network with L2 regularization:
    Args:
       cost: is the cost of the network without L2 regularization
       lambtha: is the regularization parameter
       weights: is a dictionary of the weights and biases
       (numpy.ndarrays) of the neural network
       L: is the number of layers in the neural network
       m: is the number of data points used

    Returns:
       the cost of the network accounting for L2 regularization


    """
    L2_reg = 0

    for key in range(1,L+1):
        label = 'W{}'.format(key)
        W = weights[label]
        W_squared = np.square(W)
        L2_reg = L2_reg +  np.sum(W_squared)

    L2_reg = L2_reg / (2 * m) * lambtha
    cost = cost + L2_reg

    return cost
