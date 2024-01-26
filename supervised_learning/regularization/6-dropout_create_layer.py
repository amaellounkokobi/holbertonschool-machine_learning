
#!/usr/bin/env python3
"""
This module contains :
A function that creates a layer of a neural network using dropout:


Function:
   def dropout_create_layer(prev, n, activation, keep_prob):
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
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
        name="layer")

    return layer(tf.layers.dropout(prev, rate=keep_prob))
