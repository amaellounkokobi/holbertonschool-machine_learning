#!/usr/bin/env python3
"""
Thins module contains:
A function

Function:
   def f1_score(confusion):
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Fuction that calculates the F1 score of a
    confusion matrix

    Args:
       confusion: is a confusion numpy.ndarray of shape
       (classes, classes) where row indices represent
       the correct labels and column indices represent
       the predicted labels

          classes is the number of classes

    Returns:
       a numpy.ndarray of shape (classes,) containing
       the F1 score of each class

    """
    pre = precision(confusion)
    sen = sensitivity(confusion)

    F1 = 2 * ((pre * sen) / (pre + sen))

    return F1
