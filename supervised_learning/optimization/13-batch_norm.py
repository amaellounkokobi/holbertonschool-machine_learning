#!/usr/bin/env python3
"""
This module contains
A function that normalizes an unactivated output
of a neural network using batch normalization

Function:
   def batch_norm(Z, gamma, beta, epsilon):
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Function that normalizes an unactivated output
    of a neural network using batch normalization

    Args:
       Z: is a numpy.ndarray of shape (m, n) that should be normalized
       m: is the number of data points
       n: is the number of features in Z

       gamma: is a numpy.ndarray of shape (1, n)
       containing the scales used for batch normalization

       beta: is a numpy.ndarray of shape (1, n)
       containing the offsets used for batch normalization
       epsilon: is a small number used to avoid division by zero
    Returns:
       the normalized Z matrix

    """
    m = Z.shape[0]
    mu = 1 / m * np.sum(Z, axis=0)
    var = 1 / m * np.sum((Z - mu)**2, axis=0)
    Z_norm = (Z - mu) / (np.sqrt(var + epsilon))
    Z_tild = gamma * Z_norm + beta

    return Z_tild
