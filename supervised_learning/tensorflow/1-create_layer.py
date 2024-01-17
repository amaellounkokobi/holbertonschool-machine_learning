#!/usr/bin/env python3
"""
This module contains:
a function that creates a layer
Function:
   def create_placeholders(nx, classes):

"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Function that creates a layer

    Args:
      prev: is the tensor output of the previous layer
      n: is the number of nodes in the layer to create
      activation: is the activation function that the layer should use
      Returns: the tensor output of the layer
    """

    w_init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    name = activation.__name__.capitalize()

    layer = tf.layers.Dense(
            n,
            activation=activation,
            kernel_initializer=w_init,
            bias_initializer=tf.compat.v1.zeros_initializer(),
            name="layer",
    )

    return layer(prev)
