#!/usr/bin/env python3
"""
This module contains :
A function that builds an identity block

Function:
   def identity_block(A_prev, filters):
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described

    Args:
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
       F11 is the number of filters in the first 1x1 convolution
       F3 is the number of filters in the 3x3 convolution
       F12 is the number of filters in the second 1x1 convolution
    """
    # Init kernels
    init = K.initializers.HeNormal()

    # Init Filters
    F11, F3, F12 = filters

    # 1x1 convolution
    conv_1_1 = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="valid",
                               kernel_initializer=init)(A_prev)

    # Batch Norm
    Bn1 = K.layers.BatchNormalization(axis=3)(conv_1_1)

    # Relu
    A_ReLu = K.layers.Activation(K.activations.relu)(Bn1)

    # 3x3 convolution
    conv_3_3 = K.layers.Conv2D(filters=F3,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               kernel_initializer=init)(A_ReLu)

    # Batch Norm
    Bn2 = K.layers.BatchNormalization(axis=3)(conv_3_3)

    # Relu
    A_ReLu_1 = K.layers.Activation(K.activations.relu)(Bn2)

    # 1x1 convolution
    conv_1_1_2 = K.layers.Conv2D(filters=F12,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="valid",
                                 kernel_initializer=init)(A_ReLu_1)

    # Batch Norm
    Bn3 = K.layers.BatchNormalization(axis=3)(conv_1_1_2)

    # Concat input/output
    output = K.layers.Add()([Bn3, A_prev])

    # Relu
    A_ReLu_3 = K.layers.Activation(K.activations.relu)(output)

    return A_ReLu_3
