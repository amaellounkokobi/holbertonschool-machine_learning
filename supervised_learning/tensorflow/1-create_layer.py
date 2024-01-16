#!/usr/bin/env python3
"""
This module contains:
a function that returns two placeholders, x and y, for the neural network:

Function:
   def create_placeholders(nx, classes):

"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Function that returns two placeholders, x and y, for the neural network

    Args:
       nx: the number of feature columns in our data

       classes: the number of classes in our classifier

       Returns:
        placeholders named x and y, respectively
        x is the placeholder for the input data to the neural network
        y is the placeholder for the one-hot labels for the input data
    """

    w_init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    name = activation.__name__.capitalize()

    layer = tf.layers.dense(
            prev,
            n,
            activation=activation,
            kernel_initializer=w_init,
            bias_initializer=tf.compat.v1.zeros_initializer(),
            name=name,
    )

    return layer
