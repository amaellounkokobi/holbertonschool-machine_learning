#!/usr/bin/env python3
"""
This module contains :
A function that builds a Dense block

Function:
   def dense_block(X, nb_filters, growth_rate, layers):
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a Dense block
    Args:
       X is the output from the previous layer
       nb_filters is an integer representing the number of filters in X
       growth_rate is the growth rate for the dense block
       layers is the number of layers in the dense block
    """
    # Init Kernel
    init = K.initializers.VarianceScaling(scale=2.0,
                                          mode='fan_in',
                                          distribution='truncated_normal',
                                          seed=None)

    for layer in range(layers):
        # Conv 1x1
        bn = K.layers.BatchNormalization()(X)
        relu = K.layers.ReLU()(bn)
        conv = K.layers.Conv2D(filters=4 * growth_rate,
                               kernel_size=(1, 1),
                               strides=1,
                               kernel_initializer=init)(relu)

        # Conv 3x3
        bn1 = K.layers.BatchNormalization()(conv)
        relu1 = K.layers.ReLU()(bn1)
        conv1 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3),
                                padding="same",
                                strides=1,
                                kernel_initializer=init)(relu1)

        # Concatenate
        concatenate = K.layers.Concatenate()([conv1, X])

        # Update X and filters
        X = concatenate
        nb_filters = nb_filters + growth_rate

    return concatenate, nb_filters
