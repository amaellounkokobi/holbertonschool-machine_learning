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
    
    true_false = tf.not_equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))

    loss = tf.reduce_mean(tf.cast(true_false, tf.float32))

    return loss


