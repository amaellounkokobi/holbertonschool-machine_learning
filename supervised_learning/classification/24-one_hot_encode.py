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

    if Y is not list:
        return "Y is not list Y is a {} ".format(type(Y))
    if classes is not int:
        return "classes is not int"
    if classes < 2:
        return "classes < 2"
    if classes < Y.max():
        return "classes < Y.max()"

    one_hot = np.eye(classes)[Y]

    return one_hot.T
