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
    precision = np.array([])

    for value in range(len(diag_values)):
        true_pos = diag_values[value]
        full_matrix = np.sum(confusion)
        curr_col = np.sum(confusion[:, value])
        curr_row = np.sum(confusion[value, :])

        true_negatives = full_matrix - ((curr_col + curr_row) - true_pos)
        false_positives = full_matrix - curr_col - true_negatives
        precision = np.append(precision,
                              true_negatives /
                              (true_negatives + false_positives))

    return precision
