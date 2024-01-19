#!/usr/bin/env python3
"""
This module contains
A function that normalizes (standardizes) a matrix:

Function:
   def normalize(X, m, s):
"""
import numpy as np


def normalize(X, m, s):
    """
    Function that normalizes (standardizes) a matrix:

    Args:
       X: is the numpy.ndarray of shape (d, nx) to normalize
       d: is the number of data points
       nx: is the number of features
       m: is a numpy.ndarray of shape (nx,) that contains the mean of all features of X
       s: is a numpy.ndarray of shape (nx,) that contains the standard deviation of all features of X

    Returns:
       The normalized X matrix

    """
    norm_matrix = np.divide(np.subtract(X, m),np.sqrt(s**2 + 0.0000000001))

    return norm_matrix
