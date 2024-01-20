#!/usr/bin/env python3
"""
This module contains
A function that creates the training operation for a
neural network in tensorflow using the RMSProp 
optimization algorithm

Function:
   def create_RMSProp_op(loss, alpha, beta2, epsilon):
"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    function that creates the training operation for
    a neural network in tensorflow using the gradient
    descent with momentum optimization algorithm:

    Args:
       loss: is the loss of the network
       alpha: is the learning rate
       beta2: is the RMSProp weight
       epsilon: is a small number to avoid division by zero
    Returns:
       the RMSProp optimization operation
    """

    op = tf.train.RMSPropOptimizer(alpha, beta2, epsilon).minimize(loss)

    return op
