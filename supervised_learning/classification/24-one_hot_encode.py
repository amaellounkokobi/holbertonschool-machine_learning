#!/usr/bin/env python3
"""
This module contains:
A function that converts a numeric label vector into a one-hot matrix

Function:

"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix:

    Args:
    Y: is a numpy.ndarray with shape (m,) containing numeric class labels
    m is the number of examples
    classes: is the maximum number of classes found in Y

    """
    one_hot = np.zeros((Y.size,classes))
    one_hot[Y, np.arange(classes)] = 1

    return one_hot
