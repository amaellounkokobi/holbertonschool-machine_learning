#!/usr/bin/env python3
"""
This module contains
A function that creates the training operation for a neural network
in tensorflow using the gradient descent with momentum
optimization algorithm:

Function:
  def create_momentum_op(loss, alpha, beta1):
"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    function that creates the training operation for
    a neural network in tensorflow using the gradient
    descent with momentum optimization algorithm:

    Args:
       loss: is the loss of the network
       alpha: is the learning rate
       beta1: is the momentum weight
    Returns:
       the momentum optimization operation

    """

    op = tf.compat.v1.train.MomentumOptimizer(alpha, beta1).minimize(loss)

    return op
