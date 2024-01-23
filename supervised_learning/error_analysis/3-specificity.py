#!/usr/bin/env python3
"""
Thins module contains:
A function that calculates the specificity
for each class in a confusion matrix

Function:
   def specificity(confusion):
"""
import numpy as np


def specificity(confusion):
    """
    Function that calculates the specificity
    for each class in a confusion matrix

    Args:
       confusion is a confusion numpy.ndarray of shape
       (classes, classes) where row indices represent
       the correct labels and column indices represent
       the predicted labels

    classes is the number of classes

    Returns:
       a numpy.ndarray of shape (classes,) containing
       the specificity of each class
    """
    diag_values = np.diag(confusion)
    specificity = np.array([])

    for value in range(len(diag_values)):
        FS = np.sum(confusion)
        ROW = np.sum(confusion[value, :])
        COL = np.sum(confusion[:, value])
        TP = diag_values[value]
        TN = (FS - ROW - COL + TP)
        FP = FS - TN - ROW
        specificity = np.append(specificity, TN / (TN + FP))

    return specificity
