#!/usr/bin/env python3

"""
This module contains :
A function that builds a transition layer

Function:
   def transition_layer(X, nb_filters, compression):
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer

    Args:
       X is the output from the previous layer
       nb_filters is an integer representing the number of filters in X
       compression is the compression factor for the transition layer

    """
    # Init Kernel
    init = K.initializers.VarianceScaling(scale=2.0,
                                          mode='fan_in',
                                          distribution='truncated_normal',
                                          seed=None)

    # Conv 1x1
    bn = K.layers.BatchNormalization()(X)
    relu = K.layers.ReLU()(bn)
    conv = K.layers.Conv2D(filters=nb_filters * compression,
                           kernel_size=(1, 1),
                           kernel_initializer=init)(relu)

    # Average pooling 2x2 stride 2
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                         strides=2)(conv)

    nb_filters = int(nb_filters * compression)
    return avg_pool, nb_filters
