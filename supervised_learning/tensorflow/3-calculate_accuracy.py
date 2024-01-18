#!/usr/bin/env python3
"""
This module contains:
a function that creates that calculates the accuracy of a prediction:
Function:
   def create_placeholders(nx, classes):

"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Function that  that calculates the accuracy of a prediction

    Args:
       y is a placeholder for the labels of the input data
       y_pred is a tensor containing the network’s predictions
       Returns: a tensor containing the decimal accuracy of the prediction
       hint: accuracy = correct_predictions / all_predictions

    """

    pred = tf.equal(y, y_pred)

    accuracy = tf.reduce_mean( tf.cast(pred, tf.float32))

    
    return accuracy
