#!/usr/bin/env python3
"""
This module contains
A function that creates a batch normalization
layer for a neural network in tensorflow

Function:
   def create_batch_norm_layer(prev, n, activation):
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Function that creates a batch normalization
    layer for a neural network in tensorflow

    Args:
       prev: is the activated output of the previous layer

       n: is the number of nodes in the layer to be created

       activation: is the activation function that
       should be used on the output of the layer

    Returns:
       A tensor of the activated output for the layer

    """

    w_init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=w_init,
        bias_initializer=tf.compat.v1.zeros_initializer())

    z = layer(prev)

    mean = tf.reduce_mean(z, axis=1, keepdims=True)
    var = tf.reduce_mean((z - mean)**2, axis=1, keepdims=True)
    variance_epsilon =tf.constant(10**-8)

    z_norm = tf.nn.batch_normalization(
        x=z,
        mean=mean,
        variance=var,
        offset=tf.zeros([1, n]),
        scale=tf.ones([1, n]),
        variance_epsilon=variance_epsilon)

    return activation(z_norm)

