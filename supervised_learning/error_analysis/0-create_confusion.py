#!/usr/bin/env python3
"""
Thins module contains:
A function that creates a confusion matrix

Function:
   def create_confusion_matrix(labels, logits):
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Function that creates a confusion matrix

    Args:

       labels: is a one-hot numpy.ndarray of shape (m, classes)
       containing the correct labels for each data point
          m: is the number of data points
          classes: is the number of classes

       logits: is a one-hot numpy.ndarray of shape (m, classes)
       containing the predicted labels

    Returns:
       A confusion numpy.ndarray of shape (classes, classes) with row
       indices representing the correct labels and
       column indices representing the predicted labels

    """
    c_matrix = np.ndarray((labels.shape[1],labels.shape[1]))

    for class_col in range(labels.shape[1]):
        col_mat = np.array([])
        for class_row in range(labels.shape[1]):
            labels_class = np.argmax(labels, axis=1) == class_col
            logits_class = np.argmax(logits, axis=1) == class_row
            label_cmp_logits = np.equal(labels_class, logits_class)
            label_good_line = np.argmax(labels, axis=1) == class_col
            label_ok = np.logical_and(label_good_line, label_cmp_logits)
            result_mat = np.sum(label_ok)
            col_mat = np.append(col_mat, result_mat)

        c_matrix[class_col] = col_mat

    return c_matrix
