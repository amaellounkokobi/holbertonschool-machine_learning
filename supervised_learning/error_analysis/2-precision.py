#!/usr/bin/env python3
"""
Thins module contains:
A function that calculates the precision for
each class in a confusion matrix

Function:
   def precision(confusion):
"""
import numpy as np


def precision(confusion):
    """
    Function that calculates the precision for
    each class in a confusion matrix:

    Args:
       confusion: is a confusion numpy.ndarray
       of shape (classes, classes)
          classes: is the number of classes

    Returns:
       a numpy.ndarray of shape (classes,)
       containing the precision of each class
    """

    diag_values = np.diag(confusion)
    predicted_ok = np.sum(confusion, axis=0)
    precision = diag_values / predicted_ok

    return precision
