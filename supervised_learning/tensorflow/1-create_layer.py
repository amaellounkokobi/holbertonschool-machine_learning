#!/usr/bin/env python3
"""
This module contains:
a function that returns two placeholders, x and y, for the neural network:

Function:
   def create_placeholders(nx, classes):

"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

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

    layer = tf.compat.v1.layers.dense(
            prev,
            n,
            activation=activaion,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),
            bias_initializer=tf.compat.v1.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=activation.name,
            reuse=None
        )
    
    return layer
