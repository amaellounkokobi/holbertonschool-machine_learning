#!/usr/bin/env python3
"""
This module contains
A function that shuffles the data points in two matrices the same way

Function:
   def shuffle_data(X, Y):
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Function that shuffles the data points in two matrices the same way

    Args:
       X: is the first numpy.ndarray of shape (m, nx) to shuffle
       m: is the number of data points
       nx: is the number of features in X
       Y: is the second numpy.ndarray of shape (m, ny) to shuffle
       m: is the same number of data points as in X
       ny: is the number of features in Y

    Returns:
      The shuffled X and Y matrices
    """
    nx = len(X)
    ny = len(Y)
    assert nx == ny
    permutation = np.random.permutation(nx)

    return X[permutation], Y[permutation]
