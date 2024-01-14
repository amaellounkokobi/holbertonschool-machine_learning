#!/usr/bin/env python3
"""
This module contains:
A function that converts a one-hot matrix into a vector of labels:
A function that converts a numeric label vector into a one-hot matrix

Function:
   def one_hot_encode(Y, classes):
   def one_hot_decode(one_hot):


"""
import numpy as np


def one_hot_decode(one_hot):
    """
     converts a one-hot matrix into a vector of labels

    Args:
       one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)

    """

    if isinstance(one_hot, np.ndarray) is False:
        return None
              
    one_hot_decode = np.argmax(one_hot.T, axis=1)

    return one_hot_decode
