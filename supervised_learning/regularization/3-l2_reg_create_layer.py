#!/usr/bin/env python3
"""
This module contains :
A function


Function:
   def l2_reg_create_layer(prev, n, activation, lambtha):
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Function

    Args:
       prev: is a tensor containing the output of the previous layer
       n: is the number of nodes the new layer should contain
       activation: is the activation function that should be used on the layer
       lambtha: is the L2 regularization parameter

    Returns:
       the output of the new layer

    """
    w_init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=w_init,
        bias_initializer=tf.compat.v1.zeros_initializer(),
        kernel_regularizer=tf.keras.regularizers.L2(lambtha),
        name="layer")

    return layer(prev)
