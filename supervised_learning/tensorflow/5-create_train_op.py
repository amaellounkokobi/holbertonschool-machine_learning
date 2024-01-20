#!/usr/bin/env python3
"""
This module contains:
A function that creates the training operation for the network:

Function:
def create_train_op(loss, alpha):

"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    function that creates the training operation for the network

    Args:
       loss: is the loss of the network’s prediction
       alpha: is the learning rate
    Returns:
       An operation that trains the network using gradient descent

    """
    op = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    return op
