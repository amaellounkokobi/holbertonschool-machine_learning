#!/usr/bin/env python3
"""
Thins module contains:
A function that calculates the sensitivity for
each class in a confusion matrix

Function:
   def create_confusion_matrix(labels, logits):
"""
import numpy as np


def sensitivity(confusion):
    """
    Function that calculates the sensitivity
    for each class in a confusion matrix

    Args:
       confusion: is a confusion numpy.ndarray of shape
       (classes, classes) where row indices represent
       the correct labels and column indices represent
       the predicted labels
          classes is the number of classes

    Returns:
       a numpy.ndarray of shape (classes,) containing
       the sensitivity of each class

    """
    max_num_col = np.max(confusion, axis=1)
    sum_col = np.sum(confusion, axis=1)
    sensitivity = max_num_col / sum_col

    return sensitivity
