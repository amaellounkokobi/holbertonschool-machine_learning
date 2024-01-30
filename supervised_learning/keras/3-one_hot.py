#!/usr/bin/env python3
"""
This module contains :
A function that converts a label vector into a one-hot matrix:

Function:
   def one_hot(labels, classes=None):
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Function that converts a label vector into a one-hot matrix:

    Args:
       The last dimension of the one-hot matrix must be the number of classes
    Returns:
       the one-hot matrix
    """
    if classes is None:
        classes = max(labels) + 1

    one_hot = K.utils.to_categorical(labels, num_classes=classes)

    return one_hot
