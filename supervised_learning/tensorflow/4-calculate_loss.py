#!/usr/bin/env python3
"""
This module contains:
A function that calculates the softmax cross-entropy loss of a prediction:

Function:
def calculate_loss(y, y_pred)

"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction:

    Args:
       y is a placeholder for the labels of the input data
       y_pred is a tensor containing the network’s predictions

    Returns:
       A tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    return loss
